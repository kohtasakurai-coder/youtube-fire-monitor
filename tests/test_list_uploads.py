"""チャンネルの公開動画一覧（src/list_uploads.py）。API は呼ばず、応答を差し替えて確かめる。"""
import pandas as pd

import src.list_uploads as lu


def _fake_api(pages):
    calls = []

    def fake(url, params, api_key, timeout):
        calls.append((url, dict(params)))
        if url == lu.VIDEOS_URL and params["part"] == "snippet":
            return {"items": [{"snippet": {"channelId": "UC" + "a" * 22}}]}
        if url == lu.CHANNELS_URL:
            return {"items": [{"contentDetails": {"relatedPlaylists": {"uploads": "UUxyz"}}}]}
        if url == lu.PLAYLIST_ITEMS_URL:
            return pages[params.get("pageToken", "p0")]
        if url == lu.VIDEOS_URL and params["part"].startswith("statistics"):
            ids = params["id"].split(",")
            return {"items": [{"id": v, "statistics": ({} if v == "vidOFF00000" else {"commentCount": "120"}),
                               "contentDetails": {"duration": "PT1M5S"}} for v in ids]}
        raise AssertionError(url)
    return fake, calls


def _item(vid, published):
    return {"contentDetails": {"videoId": vid, "videoPublishedAt": published}, "snippet": {"title": vid}}


def test_list_uploads_filters_period_and_stops_paging(monkeypatch):
    pages = {
        "p0": {"items": [_item("vidNEW00000", "2025-09-20T03:00:00Z"),      # 期間より後
                         _item("vidIN100000", "2025-09-12T11:00:00Z"),      # 期間内（JST 20:00）
                         _item("vidOFF00000", "2025-09-07T11:01:00Z")],     # 期間内・コメント無効
               "nextPageToken": "p1"},
        "p1": {"items": [_item("vidOLD00000", "2025-08-01T00:00:00Z")], "nextPageToken": "p2"},
        "p2": {"items": [_item("never000000", "2025-07-01T00:00:00Z")]},
    }
    fake, calls = _fake_api(pages)
    monkeypatch.setattr(lu, "_get_json", fake)
    df = lu.list_uploads("KrkclH7G7S4", "KEY", "2025-09-01", "2025-09-19")

    assert list(df["video_id"]) == ["vidIN100000", "vidOFF00000"]               # 新しい順・期間内だけ
    assert df["published_at_jst"].iloc[0] == pd.Timestamp("2025-09-12 20:00", tz="Asia/Tokyo")
    assert df["comment_count"].iloc[0] == 120 and pd.isna(df["comment_count"].iloc[1])
    # p1 で期間より古い動画が出たので p2 は読まない
    assert not any(c[1].get("pageToken") == "p2" for c in calls if c[0] == lu.PLAYLIST_ITEMS_URL)


def test_channel_id_is_used_directly(monkeypatch):
    fake, calls = _fake_api({"p0": {"items": []}})
    monkeypatch.setattr(lu, "_get_json", fake)
    df = lu.list_uploads("UC" + "b" * 22, "KEY", "2025-09-01", "2025-09-19")
    assert df.empty
    assert not any(c[0] == lu.VIDEOS_URL for c in calls)                       # 動画情報は引かない


def test_parse_duration():
    assert lu.parse_duration("PT1H2M3S") == 3723.0
    assert lu.parse_duration("PT45S") == 45.0
    assert lu.parse_duration("P1DT1S") == 86401.0
