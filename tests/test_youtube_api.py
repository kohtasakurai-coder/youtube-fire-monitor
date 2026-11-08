"""
youtube_api.py のテスト。
実際のAPIは呼ばず、requests.get を偽物に差し替えて動作を確かめる（APIキー・通信不要）。
"""
import pandas as pd
import pytest

from src import youtube_api
from src.preprocessing import load_comments_csv, save_comments_csv
from src.youtube_api import YouTubeAPIError, fetch_comments, get_youtube_comments

FAKE_KEY = "FAKE-KEY-123"


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def make_item(i):
    return {"snippet": {"topLevelComment": {"id": f"cid{i}", "snippet": {
        "textDisplay": f"コメント{i}",
        "publishedAt": f"2026-01-05T{i % 24:02d}:00:00Z",
        "likeCount": i,
    }}}}


def install_fake_api(monkeypatch, pages):
    """pages を順番に返す偽の requests.get を入れる。呼び出し時の params を記録する。"""
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(dict(params))
        return pages[len(calls) - 1]

    monkeypatch.setattr(youtube_api.requests, "get", fake_get)
    return calls


def test_pagination_and_columns(monkeypatch):
    pages = [
        FakeResponse(200, {"items": [make_item(i) for i in range(100)], "nextPageToken": "P2"}),
        FakeResponse(200, {"items": [make_item(i) for i in range(100, 130)]}),
    ]
    calls = install_fake_api(monkeypatch, pages)
    df = get_youtube_comments("Thc7JIJk6Co", FAKE_KEY, max_results=2000)
    assert len(df) == 130
    assert calls[1]["pageToken"] == "P2"
    assert list(df.columns) == youtube_api.OUTPUT_COLUMNS
    assert (df["data_source"] == "real").all()
    assert not df["is_reply"].any()


def test_max_results_stops_early(monkeypatch):
    pages = [FakeResponse(200, {"items": [make_item(i) for i in range(10)], "nextPageToken": "P2"})]
    calls = install_fake_api(monkeypatch, pages)
    df = get_youtube_comments("Thc7JIJk6Co", FAKE_KEY, max_results=10)
    assert len(df) == 10
    assert len(calls) == 1                  # 10件で止まり、次のページは取りに行かない
    assert calls[0]["maxResults"] == 10


def test_comments_disabled_message_without_key(monkeypatch):
    payload = {"error": {"errors": [{"reason": "commentsDisabled",
                                     "message": f"key={FAKE_KEY}"}]}}
    install_fake_api(monkeypatch, [FakeResponse(403, payload)])
    with pytest.raises(YouTubeAPIError) as info:
        get_youtube_comments("Thc7JIJk6Co", FAKE_KEY)
    assert info.value.reason == "commentsDisabled"
    assert "コメントが無効" in str(info.value)
    assert FAKE_KEY not in str(info.value)  # エラー表示にAPIキーが漏れない


def test_connection_error_hides_key(monkeypatch):
    def boom(url, params=None, timeout=None):
        raise youtube_api.requests.ConnectionError(f"failed url ...key={FAKE_KEY}")
    monkeypatch.setattr(youtube_api.requests, "get", boom)
    with pytest.raises(YouTubeAPIError) as info:
        get_youtube_comments("Thc7JIJk6Co", FAKE_KEY)
    assert FAKE_KEY not in str(info.value)
    assert info.value.__cause__ is None


def test_empty_video(monkeypatch):
    install_fake_api(monkeypatch, [FakeResponse(200, {"items": []})])
    df, warnings = fetch_comments("Thc7JIJk6Co", FAKE_KEY, max_results=10)
    assert df.empty and warnings


def test_saved_csv_is_readable_by_loader(monkeypatch, tmp_path):
    """取得 → CSV保存 → 既存ローダーで読み込み、件数と日時（JST）が一致すること。"""
    pages = [FakeResponse(200, {"items": [make_item(i) for i in range(10)], "nextPageToken": "P2"})]
    install_fake_api(monkeypatch, pages)
    df, warnings = fetch_comments("Thc7JIJk6Co", FAKE_KEY, max_results=10)
    assert any("取得上限" in w for w in warnings)

    path = save_comments_csv(df, tmp_path / "comments.csv")
    loaded, load_warnings = load_comments_csv(path)
    assert load_warnings == []              # 列不足・日時不明などの警告が出ない
    assert len(loaded) == 10
    assert str(loaded["published_at"].dt.tz) == "Asia/Tokyo"
    # UTC 00:00 → JST 09:00
    first = loaded.set_index("comment_id").loc["cid0", "published_at"]
    assert first == pd.Timestamp("2026-01-05 09:00", tz="Asia/Tokyo")


def test_cli_reads_key_from_env(monkeypatch, tmp_path):
    pages = [FakeResponse(200, {"items": [make_item(i) for i in range(3)]})]
    install_fake_api(monkeypatch, pages)
    monkeypatch.setenv("YOUTUBE_API_KEY", FAKE_KEY)
    out = tmp_path / "out.csv"
    assert youtube_api.main(["https://youtu.be/Thc7JIJk6Co", "--max", "10", "--out", str(out)]) == 0
    loaded, _ = load_comments_csv(out)
    assert len(loaded) == 3
    assert FAKE_KEY not in out.read_text(encoding="utf-8-sig")


# =====================================================================
# 返信コメントの取得（include_replies=True）
# =====================================================================
def make_reply(parent, j, hour=1):
    return {"id": f"{parent}.r{j}", "snippet": {
        "parentId": parent, "textDisplay": f"返信{j}",
        "publishedAt": f"2026-01-05T{hour:02d}:{j:02d}:00Z", "likeCount": j,
    }}


def make_thread(i, total_replies=0, embedded=0):
    item = make_item(i)
    item["snippet"]["totalReplyCount"] = total_replies
    if embedded:
        item["replies"] = {"comments": [make_reply(f"cid{i}", j) for j in range(embedded)]}
    return item


def install_routed_api(monkeypatch, thread_pages, reply_pages):
    """
    URLで振り分ける偽API。
    thread_pages: commentThreads の応答リスト（順番に返す）
    reply_pages : {親ID: [comments.list の応答, ...]}
    """
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append((url, dict(params)))
        if url == youtube_api.API_URL:
            n = sum(1 for u, _ in calls if u == youtube_api.API_URL)
            return thread_pages[n - 1]
        assert url == youtube_api.COMMENTS_URL
        parent = params["parentId"]
        n = sum(1 for u, p in calls if u == youtube_api.COMMENTS_URL and p["parentId"] == parent)
        return reply_pages[parent][n - 1]

    monkeypatch.setattr(youtube_api.requests, "get", fake_get)
    return calls


def test_default_does_not_request_replies(monkeypatch):
    """既定（include_replies=False）は従来どおり part=snippet で、comments.list を呼ばない。"""
    page = FakeResponse(200, {"items": [make_thread(0, total_replies=3, embedded=3)]})
    calls = install_routed_api(monkeypatch, [page], {})
    df = get_youtube_comments("Thc7JIJk6Co", FAKE_KEY, max_results=10)
    assert len(df) == 1 and not df["is_reply"].any()
    assert calls[0][1]["part"] == "snippet"
    assert all(u == youtube_api.API_URL for u, _ in calls)


def test_embedded_replies_are_used_without_extra_calls(monkeypatch):
    page = FakeResponse(200, {"items": [make_thread(0, total_replies=2, embedded=2),
                                        make_thread(1)]})
    calls = install_routed_api(monkeypatch, [page], {})
    df = get_youtube_comments("Thc7JIJk6Co", FAKE_KEY, include_replies=True)
    assert calls[0][1]["part"] == "snippet,replies"
    assert len(calls) == 1                                  # 追加の呼び出しなし
    replies = df[df["is_reply"]]
    assert list(replies["comment_id"]) == ["cid0.r0", "cid0.r1"]
    assert (replies["parent_id"] == "cid0").all()
    assert (df.loc[~df["is_reply"], "parent_id"] == "").all()


def test_many_replies_fetched_with_comments_list_and_pagination(monkeypatch):
    """返信が埋め込み分（5件）より多いときは comments.list でページを送って全件取る。"""
    page = FakeResponse(200, {"items": [make_thread(0, total_replies=130, embedded=5)]})
    reply_pages = {"cid0": [
        FakeResponse(200, {"items": [make_reply("cid0", j % 60, hour=1 + j // 60)
                                     for j in range(100)], "nextPageToken": "R2"}),
        FakeResponse(200, {"items": [make_reply("cid0", j % 60, hour=3 + j // 60)
                                     for j in range(100, 130)]}),
    ]}
    # IDが重複しないよう付け直す
    for k, resp in enumerate(reply_pages["cid0"]):
        for n, c in enumerate(resp._payload["items"]):
            c["id"] = f"cid0.p{k}.{n}"
    calls = install_routed_api(monkeypatch, [page], reply_pages)
    df = get_youtube_comments("Thc7JIJk6Co", FAKE_KEY, max_results=2000, include_replies=True)
    assert len(df) == 1 + 130
    assert df["is_reply"].sum() == 130
    reply_calls = [p for u, p in calls if u == youtube_api.COMMENTS_URL]
    assert len(reply_calls) == 2 and reply_calls[1]["pageToken"] == "R2"
    assert FAKE_KEY not in df.to_csv()


def test_limit_counts_parents_plus_replies(monkeypatch):
    """max_results はトップレベル＋返信の合計。上限で途中のスレッドの返信も打ち切る。"""
    page = FakeResponse(200, {"items": [
        make_thread(0, total_replies=3, embedded=3),   # 親1 + 返信3 = 4行
        make_thread(1, total_replies=4, embedded=4),   # 親1 + 返信4 → ここで合計9行
        make_thread(2, total_replies=5, embedded=5),   # 親1（=10行目）+ 返信は入らない
        make_thread(3),
    ], "nextPageToken": "NEXT"})
    calls = install_routed_api(monkeypatch, [page], {})
    df, warnings = fetch_comments("Thc7JIJk6Co", FAKE_KEY, max_results=10, include_replies=True)
    assert len(df) == 10
    assert (~df["is_reply"]).sum() == 3 and df["is_reply"].sum() == 7
    assert "cid3" not in set(df["comment_id"])
    assert len(calls) == 1                         # 上限に達したので次ページを取りに行かない
    assert any("トップレベル＋返信の合計" in w for w in warnings)


def test_partial_replies_when_limit_hits_inside_thread(monkeypatch):
    page = FakeResponse(200, {"items": [make_thread(0, total_replies=5, embedded=5)]})
    install_routed_api(monkeypatch, [page], {})
    df = get_youtube_comments("Thc7JIJk6Co", FAKE_KEY, max_results=3, include_replies=True)
    assert list(df["comment_id"]) == ["cid0", "cid0.r0", "cid0.r1"]   # 親 → 古い返信から2件


def test_reply_error_is_reported_safely(monkeypatch):
    page = FakeResponse(200, {"items": [make_thread(0, total_replies=50, embedded=5)]})
    err = FakeResponse(403, {"error": {"errors": [{"reason": "quotaExceeded"}]}})
    install_routed_api(monkeypatch, [page], {"cid0": [err]})
    with pytest.raises(YouTubeAPIError) as info:
        get_youtube_comments("Thc7JIJk6Co", FAKE_KEY, include_replies=True)
    assert info.value.reason == "quotaExceeded"
    assert FAKE_KEY not in str(info.value)


def test_replies_roundtrip_through_csv_loader(monkeypatch, tmp_path):
    """返信を含めて保存したCSVを既存ローダーで読むと、parent_id と is_reply が保たれる。"""
    page = FakeResponse(200, {"items": [make_thread(0, total_replies=2, embedded=2),
                                        make_thread(1)]})
    install_routed_api(monkeypatch, [page], {})
    df, _ = fetch_comments("Thc7JIJk6Co", FAKE_KEY, include_replies=True)
    path = save_comments_csv(df, tmp_path / "with_replies.csv")
    loaded, load_warnings = load_comments_csv(path)
    assert load_warnings == []
    assert len(loaded) == 4
    r = loaded.set_index("comment_id")
    assert bool(r.loc["cid0.r1", "is_reply"]) is True
    assert r.loc["cid0.r1", "parent_id"] == "cid0"
    assert bool(r.loc["cid1", "is_reply"]) is False and r.loc["cid1", "parent_id"] == ""
    # 返信の親は必ず同じCSV内に存在する
    parents = loaded.loc[loaded["is_reply"], "parent_id"]
    assert parents.isin(loaded["comment_id"]).all()


def test_cli_replies_flag(monkeypatch, tmp_path, capsys):
    page = FakeResponse(200, {"items": [make_thread(0, total_replies=1, embedded=1)]})
    install_routed_api(monkeypatch, [page], {})
    monkeypatch.setenv("YOUTUBE_API_KEY", FAKE_KEY)
    out = tmp_path / "out.csv"
    assert youtube_api.main(["Thc7JIJk6Co", "--max", "10", "--replies", "--out", str(out)]) == 0
    assert "返信 1件" in capsys.readouterr().out
