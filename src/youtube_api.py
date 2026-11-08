"""
YouTube Data API v3 でコメントを取得するモジュール（STEP 2）。

- commentThreads.list（1回で最大100件、1リクエスト=1クォータ単位）でページを送って取得する
- 取得結果は src/preprocessing.py のCSVローダーと同じ列構成にそろえて返す
- APIキーはソースコードに書かない。環境変数 YOUTUBE_API_KEY または
  Streamlit Secrets（.streamlit/secrets.toml）から渡す
- 投稿者名・チャンネルIDは取得しない（研究に不要な個人情報を持たないため）

返信コメント：--replies（コード上は include_replies=True）を付けたときだけ取得する。
既定はトップレベルコメントのみ（STEP 2 前半と同じ動作）。

コマンドラインでの使い方（プロジェクトのルートで）:
    python -m src.youtube_api https://youtu.be/XXXXXXXXXXX --max 10
    python -m src.youtube_api https://youtu.be/XXXXXXXXXXX --max 50 --replies
    → APIキーは環境変数から読む。なければ画面に表示されない形で入力を求める
    → data/research/comments_<動画ID>_<取得日時>.csv に保存
"""
from __future__ import annotations

import argparse
import getpass
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from src.config import DATA_SOURCE_REAL, REQUIRED_COLUMNS
from src.preprocessing import extract_video_id, normalize_comments, save_comments_csv

API_URL = "https://www.googleapis.com/youtube/v3/commentThreads"
COMMENTS_URL = "https://www.googleapis.com/youtube/v3/comments"
VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
API_KEY_ENV = "YOUTUBE_API_KEY"

# APIが返すエラー理由（reason）→ 日本語の説明
ERROR_MESSAGES = {
    "commentsDisabled": "この動画はコメントが無効になっています。",
    "videoNotFound": "動画が見つかりません（削除・非公開・IDの誤り）。",
    "quotaExceeded": "APIの1日の利用上限（クォータ）に達しました。太平洋時間0時（日本時間16時または17時）以降に再実行してください。",
    "keyInvalid": "APIキーが無効です。キーを確認してください。",
    "badRequest": "APIキーまたはリクエストが不正です。",
    "forbidden": "アクセスが拒否されました（キーの制限設定を確認してください）。",
    "processingFailure": "YouTube側の一時的なエラーです。時間をおいて再実行してください。",
}

OUTPUT_COLUMNS = REQUIRED_COLUMNS + ["video_id", "data_source"]


class YouTubeAPIError(RuntimeError):
    """API取得の失敗。reason にAPIのエラー理由（例: commentsDisabled）が入る。"""

    def __init__(self, message: str, reason: str | None = None, status: int | None = None):
        super().__init__(message)
        self.reason = reason
        self.status = status


def get_api_key() -> str | None:
    """環境変数 YOUTUBE_API_KEY からAPIキーを読む（なければ None）。"""
    key = os.environ.get(API_KEY_ENV, "").strip()
    return key or None


def _safe_reason(response: requests.Response, api_key: str) -> str | None:
    """
    エラー応答から理由（reason）だけを取り出す。
    メッセージ本文はそのまま表示しない（APIキーなどが含まれる可能性があるため）。
    """
    try:
        error = response.json()["error"]
        reasons = [e.get("reason") for e in error.get("details", [])]
        reasons += [e.get("reason") for e in error.get("errors", [])]
    except (ValueError, KeyError, TypeError, AttributeError):
        return None
    return next(
        (r for r in reasons
         if isinstance(r, str)
         and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", r)
         and api_key not in r),
        None,
    )


def _get_json(url: str, params: dict, api_key: str, timeout: float) -> dict:
    """APIを1回呼んでJSONを返す。失敗時は YouTubeAPIError（APIキーは表示しない）。"""
    try:
        response = requests.get(url, params=params, timeout=timeout)
    except requests.RequestException:
        # 例外メッセージにはURL（=APIキー）が含まれるので表示しない
        raise YouTubeAPIError("YouTube APIに接続できませんでした。") from None

    if response.status_code != 200:
        reason = _safe_reason(response, api_key)
        message = ERROR_MESSAGES.get(reason, "YouTube APIの取得に失敗しました。")
        detail = f"（HTTP {response.status_code}" + (f", {reason}）" if reason else "）")
        raise YouTubeAPIError(message + detail, reason=reason, status=response.status_code)
    return response.json()


def _comment_row(comment: dict, video_id: str, parent_id: str = "") -> dict:
    """APIのコメント1件（comment リソース）をCSVの1行に変換する。"""
    snippet = comment["snippet"]
    return {
        "comment_id": comment["id"],
        "published_at": snippet["publishedAt"],
        "text": snippet.get("textDisplay", ""),
        "like_count": snippet.get("likeCount", 0),
        "parent_id": parent_id,
        "is_reply": bool(parent_id),
        "video_id": video_id,
        "data_source": DATA_SOURCE_REAL,
    }


def _fetch_all_replies(parent_id: str, video_id: str, api_key: str, limit: int,
                       timeout: float) -> list[dict]:
    """
    comments.list（parentId指定、1回最大100件・1クォータ単位）で
    1つの親コメントへの返信を最大 limit 件取得する。
    """
    params = {"key": api_key, "part": "snippet", "parentId": parent_id,
              "textFormat": "plainText"}
    replies: list[dict] = []
    while len(replies) < limit:
        params["maxResults"] = min(100, limit - len(replies))
        data = _get_json(COMMENTS_URL, params, api_key, timeout)
        for c in data.get("items", [])[: limit - len(replies)]:
            replies.append(_comment_row(c, video_id, c["snippet"].get("parentId") or parent_id))
        token = data.get("nextPageToken")
        if not token:
            break
        params["pageToken"] = token
    return replies


def get_youtube_comments(video_id: str, api_key: str, max_results: int = 2000,
                         timeout: float = 30, include_replies: bool = False
                         ) -> pd.DataFrame:
    """
    動画のコメントを取得する。

    include_replies=False（既定）: トップレベルコメントだけを最大 max_results 件。
    include_replies=True : トップレベルコメントと返信を取得する。

    【取得上限の数え方】max_results は「CSVの行数」＝トップレベル＋返信の合計件数。
      - 新しい順のスレッド（トップレベルコメント）ごとに、親 → その返信 の順で数える
      - 合計が max_results に達した時点で打ち切る。最後のスレッドの返信は途中までになることがある
      - 例: max_results=10 で、1件目の親に返信が3件 → 親1＋返信3＋次の親… で合計10行
    返信の取り方:
      - commentThreads の replies には返信の一部（最大5件程度）しか入らないため、
        返信数（totalReplyCount）がそれより多いスレッドは comments.list で全件を取り直す
        （その分クォータを1回100件ごとに1単位消費する）

    戻り値の列（CSVローダーと同じ）:
        comment_id, published_at, text, like_count, parent_id, is_reply,
        video_id, data_source
    返信行は parent_id に親コメントのID、is_reply に True が入る。
    published_at はAPIの値（UTC、例 "2026-01-05T09:00:00Z"）のまま。
    日本時間への変換は fetch_comments() / CSVローダーで行う。
    """
    if not video_id or not api_key:
        raise ValueError("video_id と api_key が必要です。")
    if max_results < 0:
        raise ValueError("max_results は0以上にしてください。")

    params = {
        "key": api_key,
        "textFormat": "plainText",
        "part": "snippet,replies" if include_replies else "snippet",
        "videoId": video_id,
        "order": "time",   # 新しい順（APIの既定と同じ。明示しておく）
    }
    rows: list[dict] = []

    while len(rows) < max_results:
        params["maxResults"] = min(100, max_results - len(rows))
        data = _get_json(API_URL, params, api_key, timeout)

        for item in data.get("items", []):
            if len(rows) >= max_results:
                break
            top = item["snippet"]["topLevelComment"]
            rows.append(_comment_row(top, video_id))
            if not include_replies:
                continue

            remaining = max_results - len(rows)
            total = int(item["snippet"].get("totalReplyCount", 0) or 0)
            embedded = item.get("replies", {}).get("comments", [])
            if remaining <= 0 or total == 0:
                continue
            if len(embedded) >= total:
                # スレッドに全返信が入っている → 追加のAPI呼び出しは不要
                replies = [_comment_row(c, video_id, top["id"]) for c in embedded]
            else:
                replies = _fetch_all_replies(top["id"], video_id, api_key,
                                             min(remaining, total), timeout)
            # 返信は古い順にそろえる（行の並びは後でローダーが投稿日時順に並べ直す）
            replies.sort(key=lambda r: r["published_at"])
            rows.extend(replies[:remaining])

        token = data.get("nextPageToken")
        if not token:
            break
        params["pageToken"] = token

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def get_video_info(video_id: str, api_key: str, timeout: float = 30) -> dict:
    """
    動画の公開日時・タイトル・総コメント数を取得する（videos.list、1クォータ単位）。
    総コメント数は YouTube が表示する値で、削除済みなどにより実際に取得できる件数と一致しないことがある。
    """
    data = _get_json(VIDEOS_URL, {"key": api_key, "part": "snippet,statistics", "id": video_id},
                     api_key, timeout)
    items = data.get("items", [])
    if not items:
        raise YouTubeAPIError(ERROR_MESSAGES["videoNotFound"], reason="videoNotFound")
    sn, stats = items[0].get("snippet", {}), items[0].get("statistics", {})
    published = pd.Timestamp(sn["publishedAt"]).tz_convert("Asia/Tokyo") if sn.get("publishedAt") else None
    count = stats.get("commentCount")
    return {"video_id": video_id, "title": sn.get("title", ""), "channel": sn.get("channelTitle", ""),
            "published_at_jst": published, "comment_count": int(count) if count is not None else None}


def fetch_comments(video_id: str, api_key: str, max_results: int = 2000,
                   include_replies: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """
    取得して、CSVローダーと同じ整形（JST変換・型そろえ・重複除去）までを行う。
    戻り値: (整形済みの表, 警告メッセージのリスト)
    """
    raw = get_youtube_comments(video_id, api_key, max_results=max_results,
                               include_replies=include_replies)
    df, warnings = normalize_comments(raw)
    if len(raw) >= max_results:
        what = "トップレベル＋返信の合計" if include_replies else "トップレベルコメント"
        warnings.append(
            f"取得上限（{what}{max_results}件）に達しました。APIは新しい順に返すため、"
            "古いコメント（公開直後など）が含まれていない可能性があります。"
            + ("最後のスレッドの返信は途中までの可能性があります。" if include_replies else "")
        )
    return df, warnings


def default_output_path(video_id: str, out_dir: str | Path = "data/research") -> Path:
    """保存先: data/research/comments_<動画ID>_<取得日時>.csv（Git管理外のフォルダ）"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(out_dir) / f"comments_{video_id}_{stamp}.csv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="YouTubeコメントを取得してCSVに保存する")
    parser.add_argument("video", help="動画のURLまたは動画ID")
    parser.add_argument("--max", type=int, default=2000, help="最大取得件数（既定 2000）")
    parser.add_argument("--replies", action="store_true",
                        help="返信コメントも取得する（--max は返信を含めた合計件数）")
    parser.add_argument("--out", default=None, help="保存先CSV（省略時は data/research/ に自動命名）")
    args = parser.parse_args(argv)

    video_id = extract_video_id(args.video)
    api_key = get_api_key()
    if api_key is None:
        # 入力した文字は画面にもPowerShellの履歴にも残らない
        api_key = getpass.getpass("YouTube APIキーを入力（表示されません）: ").strip()
    if not api_key:
        print("APIキーがありません。", file=sys.stderr)
        return 1

    try:
        info = get_video_info(video_id, api_key)
        df, warnings = fetch_comments(video_id, api_key, max_results=args.max,
                                      include_replies=args.replies)
    except YouTubeAPIError as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1
    print(f"動画：{info['title']}（{info['channel']}）")
    print(f"公開日時（JST）：{info['published_at_jst']}　総コメント数（YouTubeの表示）：{info['comment_count']}")

    out = Path(args.out) if args.out else default_output_path(video_id)
    save_comments_csv(df, out)
    for w in warnings:
        print(f"注意: {w}")
    n_reply = int(df["is_reply"].sum()) if len(df) else 0
    print(f"{len(df)} comments fetched（トップレベル {len(df) - n_reply}件・返信 {n_reply}件） → {out}")
    if len(df):
        print(f"投稿日時（JST）: {df['published_at'].min()} 〜 {df['published_at'].max()}")
        if info["comment_count"]:
            print(f"取得率（取得件数 / 総コメント数）：{len(df) / info['comment_count']:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
