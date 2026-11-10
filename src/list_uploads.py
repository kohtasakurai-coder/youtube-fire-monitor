"""
チャンネルの公開動画を期間で一覧にする（本評価の動画選定用。コメントは取得しない）。

用途
  - 炎上候補：きっかけ（報道・公式発表・きっかけ動画の公開）より24時間以上前に公開された動画を探す
  - 「騒動の時点で最新だった動画」を特定する（報道が「最新動画のコメント欄」と書いている場合）
  - 比較候補：同じチャンネルの平常期や、別チャンネルの同時期の動画を探す

使う API（コメントは読まないので、クォータは少ない）
  videos.list（チャンネルIDを調べる・コメント数）／channels.list（アップロード一覧のID）／
  playlistItems.list（50件ごとに1単位）。合計は数単位〜十数単位

注意
  - 削除・非公開になった動画は一覧に出ない（チョコプラのきっかけ動画のように、後で非公開になったものは別の資料で確認する）
  - comment_count が空の動画はコメントが無効（取得できない）
  - 公開日時は YouTube の動画情報（コメントとは独立した資料）

使い方（プロジェクトのルートで。APIキーは環境変数 YOUTUBE_API_KEY、なければ入力を求める）:
    python -m src.list_uploads KrkclH7G7S4 --start 2025-09-01 --end 2025-09-19
    python -m src.list_uploads UCxxxxxxxxxxxxxxxxxxxxxx --start 2025-06-01 --end 2025-07-01
    → data/research/uploads_<チャンネルID>_<開始>_<終了>.csv に保存
"""
from __future__ import annotations

import argparse
import getpass
import re
import sys
from pathlib import Path

import pandas as pd

from src.preprocessing import extract_video_id
from src.youtube_api import VIDEOS_URL, YouTubeAPIError, _get_json, get_api_key

CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"
SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
PLAYLIST_ITEMS_URL = "https://www.googleapis.com/youtube/v3/playlistItems"
TZ = "Asia/Tokyo"


def _ts(value: str) -> pd.Timestamp:
    t = pd.Timestamp(value)
    return (t.tz_localize(TZ) if t.tzinfo is None else t).tz_convert(TZ)


def parse_duration(text: str) -> float:
    """ISO 8601 の動画の長さ（例 PT1H2M3S）を秒にする。"""
    m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", text or "")
    if not m:
        return float("nan")
    d, h, mi, se = (int(x) if x else 0 for x in m.groups())
    return float(d * 86400 + h * 3600 + mi * 60 + se)


def resolve_channel_id(video_or_channel: str, api_key: str, timeout: float = 30) -> str:
    """チャンネルID（UC…）ならそのまま、動画ID/URLならその動画のチャンネルIDを返す。"""
    s = video_or_channel.strip()
    if s.startswith("UC") and len(s) == 24:
        return s
    if s.startswith("@"):                       # ハンドル（@xxxx）
        data = _get_json(CHANNELS_URL, {"key": api_key, "part": "id", "forHandle": s}, api_key, timeout)
        if not data.get("items"):
            raise YouTubeAPIError(f"ハンドル {s} のチャンネルが見つかりません。", reason="channelNotFound")
        return data["items"][0]["id"]
    if s.startswith("search:"):                 # チャンネル名で検索（100単位。最上位の結果を使う）
        data = _get_json(SEARCH_URL, {"key": api_key, "part": "snippet", "type": "channel",
                                      "q": s[len("search:"):], "maxResults": 1}, api_key, timeout)
        if not data.get("items"):
            raise YouTubeAPIError(f"{s} に当たるチャンネルが見つかりません。", reason="channelNotFound")
        return data["items"][0]["snippet"]["channelId"]
    vid = extract_video_id(s)
    data = _get_json(VIDEOS_URL, {"key": api_key, "part": "snippet", "id": vid}, api_key, timeout)
    items = data.get("items", [])
    if not items:
        raise YouTubeAPIError("動画が見つかりません（削除・非公開・IDの誤り）。", reason="videoNotFound")
    return items[0]["snippet"]["channelId"]


def list_uploads(video_or_channel: str, api_key: str, start: str, end: str,
                 timeout: float = 30, max_pages: int = 40) -> pd.DataFrame:
    """
    期間 [start, end) に公開された動画の一覧（新しい順）。
    列: video_id, title, published_at_jst, comment_count（空ならコメント無効）, channel_id
    """
    t0, t1 = _ts(start), _ts(end)
    channel_id = resolve_channel_id(video_or_channel, api_key, timeout)
    ch = _get_json(CHANNELS_URL, {"key": api_key, "part": "contentDetails", "id": channel_id},
                   api_key, timeout)
    if not ch.get("items"):
        raise YouTubeAPIError("チャンネルが見つかりません。", reason="channelNotFound")
    uploads = ch["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

    rows: list[dict] = []
    params = {"key": api_key, "part": "contentDetails,snippet", "playlistId": uploads, "maxResults": 50}
    for _ in range(max_pages):
        data = _get_json(PLAYLIST_ITEMS_URL, params, api_key, timeout)
        page_oldest = None
        for it in data.get("items", []):
            vid = it["contentDetails"]["videoId"]
            published = it["contentDetails"].get("videoPublishedAt") or it["snippet"].get("publishedAt")
            if not published:
                continue
            t = _ts(published)
            page_oldest = t if page_oldest is None else min(page_oldest, t)
            if t0 <= t < t1:
                rows.append({"video_id": vid, "title": it["snippet"].get("title", ""), "published_at_jst": t})
        token = data.get("nextPageToken")
        # アップロード一覧は新しい順。このページの最も古い動画が期間より前なら、それ以上は見ない
        if not token or (page_oldest is not None and page_oldest < t0):
            break
        params["pageToken"] = token

    out = pd.DataFrame(rows, columns=["video_id", "title", "published_at_jst"])
    counts: dict[str, int | None] = {}
    durations: dict[str, float] = {}
    for i in range(0, len(out), 50):
        ids = ",".join(out["video_id"].iloc[i:i + 50])
        data = _get_json(VIDEOS_URL, {"key": api_key, "part": "statistics,contentDetails", "id": ids},
                         api_key, timeout)
        for it in data.get("items", []):
            c = it.get("statistics", {}).get("commentCount")
            counts[it["id"]] = int(c) if c is not None else None
            d = it.get("contentDetails", {}).get("duration")
            durations[it["id"]] = parse_duration(d) if d else float("nan")
    out["comment_count"] = out["video_id"].map(lambda v: counts.get(v)).astype("Int64")
    out["duration_sec"] = out["video_id"].map(lambda v: durations.get(v, float("nan")))
    out["channel_id"] = channel_id
    return out.sort_values("published_at_jst", ascending=False).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="チャンネルの公開動画を期間で一覧にする（コメントは取得しない）")
    p.add_argument("target", help="動画ID・動画URL・チャンネルID（UC…）・ハンドル（@…）・search:チャンネル名 のいずれか")
    p.add_argument("--start", required=True, help="例 2025-09-01（JST、この日時を含む）")
    p.add_argument("--end", required=True, help="例 2025-09-19（JST、この日時を含まない）")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    api_key = get_api_key() or getpass.getpass("YouTube APIキーを入力（表示されません）: ").strip()
    if not api_key:
        print("APIキーがありません。", file=sys.stderr)
        return 1
    try:
        df = list_uploads(args.target, api_key, args.start, args.end)
    except YouTubeAPIError as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1
    if df.empty:
        print("期間内に公開された動画はありません（削除・非公開の動画は表示されません）。")
        return 0
    ch = df["channel_id"].iloc[0]
    out = Path(args.out) if args.out else Path("data/research") / f"uploads_{ch}_{args.start}_{args.end}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    show = df.assign(published_at_jst=df["published_at_jst"].dt.strftime("%Y-%m-%d %H:%M"))
    print(show[["published_at_jst", "video_id", "comment_count", "duration_sec", "title"]].to_string(index=False))
    print(f"→ {out}（{len(df)}本。comment_count が空の動画はコメント無効）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
