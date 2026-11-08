"""
前処理モジュール

- YouTube動画URLから動画IDを取り出す
- コメントCSVを読み込み、列・型・タイムゾーンをそろえる
- コメントCSVを保存する

ここではYouTube APIを一切使わない。
（APIキーがなくても、CSVだけで分析を最後まで動かせるようにするため）
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import IO, Union
from urllib.parse import parse_qs, urlparse

import pandas as pd

from src.config import (
    DATA_SOURCE_REAL,
    DATA_SOURCE_SYNTHETIC,
    JST,
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
)

# YouTubeの動画IDは「英数字・ハイフン・アンダースコア」の11文字
_VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")

# 動画ページとして認めるホスト名
_YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
}
_SHORT_HOSTS = {"youtu.be", "www.youtu.be"}

# /shorts/ID, /embed/ID, /live/ID, /v/ID の形式
_PATH_PREFIXES = ("shorts", "embed", "live", "v")


class InvalidYouTubeURLError(ValueError):
    """URLから動画IDを取り出せなかったときのエラー。"""


class CommentCSVError(ValueError):
    """コメントCSVの形式が不正なときのエラー。"""


# ---------------------------------------------------------------------------
# 1. URL → 動画ID
# ---------------------------------------------------------------------------
def extract_video_id(url: str) -> str:
    """
    YouTubeのURL（または動画IDそのもの）から11文字の動画IDを返す。

    対応している形式の例:
        https://www.youtube.com/watch?v=dQw4w9WgXcQ
        https://youtu.be/dQw4w9WgXcQ?si=xxxx
        https://www.youtube.com/shorts/dQw4w9WgXcQ
        https://m.youtube.com/watch?v=dQw4w9WgXcQ&t=30s
        https://www.youtube.com/embed/dQw4w9WgXcQ
        https://www.youtube.com/live/dQw4w9WgXcQ
        dQw4w9WgXcQ（IDだけ）

    取り出せない場合は InvalidYouTubeURLError を投げる。
    """
    if url is None:
        raise InvalidYouTubeURLError("URLが入力されていません。")

    text = str(url).strip()
    if not text:
        raise InvalidYouTubeURLError("URLが入力されていません。")

    # IDだけが入力された場合
    if _VIDEO_ID_PATTERN.match(text):
        return text

    # "www.youtube.com/..." のようにスキームが省略された入力にも対応
    if not re.match(r"^https?://", text, flags=re.IGNORECASE):
        text = "https://" + text

    parsed = urlparse(text)
    host = (parsed.hostname or "").lower()
    path_parts = [p for p in parsed.path.split("/") if p]

    candidate = None
    if host in _SHORT_HOSTS:
        # https://youtu.be/ID
        if path_parts:
            candidate = path_parts[0]
    elif host in _YOUTUBE_HOSTS:
        if path_parts[:1] == ["watch"]:
            # https://www.youtube.com/watch?v=ID
            candidate = parse_qs(parsed.query).get("v", [None])[0]
        elif len(path_parts) >= 2 and path_parts[0] in _PATH_PREFIXES:
            # https://www.youtube.com/shorts/ID など
            candidate = path_parts[1]
    else:
        raise InvalidYouTubeURLError(
            f"YouTubeのURLではありません（ホスト名: {host or '不明'}）。"
        )

    if candidate and _VIDEO_ID_PATTERN.match(candidate):
        return candidate

    raise InvalidYouTubeURLError(
        "URLから動画IDを取り出せませんでした。"
        "通常の動画URL・短縮URL（youtu.be）・ShortsのURLを入力してください。"
    )


# ---------------------------------------------------------------------------
# 2. コメントCSVの読み込み
# ---------------------------------------------------------------------------
CSVSource = Union[str, Path, IO[bytes], IO[str], bytes]


def _read_csv_with_fallback(source: CSVSource) -> pd.DataFrame:
    """
    文字コードを自動で切り替えてCSVを読む。
    Windows の Excel で保存したCSVは Shift_JIS（cp932）のことが多いため、
    UTF-8 → cp932 の順に試す。
    """
    # Streamlitのアップロードファイルなどは一度バイト列にしておく
    if hasattr(source, "read"):
        raw = source.read()
    elif isinstance(source, bytes):
        raw = source
    else:
        raw = Path(source).read_bytes()

    if isinstance(raw, str):
        raw = raw.encode("utf-8")

    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "cp932"):
        try:
            return pd.read_csv(
                io.BytesIO(raw),
                encoding=encoding,
                dtype={"comment_id": str, "parent_id": str, "video_id": str},
                keep_default_na=True,
            )
        except UnicodeDecodeError as e:
            last_error = e
        except pd.errors.EmptyDataError:
            # 空ファイル → 列だけある空の表を返す
            return pd.DataFrame(columns=REQUIRED_COLUMNS)
    raise CommentCSVError(f"CSVの文字コードを判別できませんでした: {last_error}")


def _to_bool(value) -> bool:
    """'True', 'true', '1', 1, True などを真偽値にそろえる。"""
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def empty_comments_frame() -> pd.DataFrame:
    """列と型がそろった空のコメント表を返す。"""
    df = pd.DataFrame({c: pd.Series(dtype="object") for c in REQUIRED_COLUMNS})
    df["published_at"] = pd.Series(dtype=f"datetime64[ns, {JST}]")
    df["like_count"] = pd.Series(dtype="int64")
    df["is_reply"] = pd.Series(dtype="bool")
    df["data_source"] = pd.Series(dtype="object")
    return df


def normalize_comments(
    df: pd.DataFrame, naive_timezone: str = "UTC"
) -> tuple[pd.DataFrame, list[str]]:
    """
    コメント表の列・型をそろえる。

    戻り値: (整形後の表, 警告メッセージのリスト)

    - published_at は日本時間（JST）のタイムゾーン付き日時に変換する
    - タイムゾーン情報のない日時は naive_timezone の時刻とみなす
      （YouTube APIの生データはUTCなので初期値は "UTC"）
    - 日時を解釈できない行・コメントIDが重複した行は除外して警告に記録する
    """
    warnings: list[str] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise CommentCSVError(
            "必須の列がありません: " + ", ".join(missing)
            + "（必要な列: " + ", ".join(REQUIRED_COLUMNS) + "）"
        )

    if df.empty:
        warnings.append("コメントが0件です。")
        return empty_comments_frame(), warnings

    df = df.copy()
    keep = REQUIRED_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in df.columns]
    df = df[keep]

    # --- 投稿日時 → JST ---
    raw_times = df["published_at"].astype(str).str.strip()
    has_tz = raw_times.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True)
    parsed = pd.Series(pd.NaT, index=df.index, dtype=f"datetime64[ns, {JST}]")

    if has_tz.any():
        aware = pd.to_datetime(raw_times[has_tz], errors="coerce", utc=True, format="mixed")
        parsed.loc[has_tz] = aware.dt.tz_convert(JST)
    if (~has_tz).any():
        naive = pd.to_datetime(raw_times[~has_tz], errors="coerce", format="mixed")
        localized = naive.dt.tz_localize(naive_timezone, ambiguous="NaT", nonexistent="NaT")
        parsed.loc[~has_tz] = localized.dt.tz_convert(JST)
        n_naive = int(naive.notna().sum())
        if n_naive:
            warnings.append(
                f"{n_naive}件の投稿日時にタイムゾーン情報がないため、"
                f"{naive_timezone} の時刻とみなして日本時間に変換しました。"
            )
    df["published_at"] = parsed

    bad_time = df["published_at"].isna()
    if bad_time.any():
        warnings.append(f"投稿日時を解釈できない{int(bad_time.sum())}件を除外しました。")
        df = df[~bad_time]

    # --- その他の列の型 ---
    df["comment_id"] = df["comment_id"].astype(str).str.strip()
    df["text"] = df["text"].fillna("").astype(str)
    df["like_count"] = (
        pd.to_numeric(df["like_count"], errors="coerce").fillna(0).clip(lower=0).astype("int64")
    )
    df["parent_id"] = df["parent_id"].where(df["parent_id"].notna(), "").astype(str)
    df["parent_id"] = df["parent_id"].replace({"nan": "", "None": ""})
    df["is_reply"] = df["is_reply"].map(_to_bool).astype(bool)

    # --- 重複除去 ---
    dup = df["comment_id"].duplicated(keep="first")
    if dup.any():
        warnings.append(f"コメントIDが重複している{int(dup.sum())}件を除外しました。")
        df = df[~dup]

    # --- データの出所（架空 / 実データ） ---
    if "data_source" not in df.columns:
        df["data_source"] = DATA_SOURCE_REAL
        warnings.append(
            "data_source列がないため、実データ（real）として扱います。"
            "架空データの場合は data_source=synthetic を付けてください。"
        )
    df["data_source"] = df["data_source"].fillna(DATA_SOURCE_REAL).astype(str)

    df = df.sort_values("published_at", kind="stable").reset_index(drop=True)
    if df.empty:
        warnings.append("有効なコメントが0件になりました。")
    return df, warnings


def load_comments_csv(
    source: CSVSource, naive_timezone: str = "UTC"
) -> tuple[pd.DataFrame, list[str]]:
    """
    コメントCSVを読み込んで整形する（YouTube APIは使わない）。

    source にはファイルパス、Streamlitのアップロードファイル、バイト列を渡せる。
    戻り値: (整形済みの表, 警告メッセージのリスト)
    """
    raw_df = _read_csv_with_fallback(source)
    return normalize_comments(raw_df, naive_timezone=naive_timezone)


def describe_data_source(df: pd.DataFrame) -> str:
    """表全体が架空データか実データか（混在か）を返す。"""
    if df.empty or "data_source" not in df.columns:
        return "unknown"
    sources = set(df["data_source"].unique())
    if sources == {DATA_SOURCE_SYNTHETIC}:
        return DATA_SOURCE_SYNTHETIC
    if sources == {DATA_SOURCE_REAL}:
        return DATA_SOURCE_REAL
    return "mixed"


# ---------------------------------------------------------------------------
# 3. コメントCSVの保存
# ---------------------------------------------------------------------------
def comments_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    ダウンロード用のCSVバイト列を作る。
    Windows の Excel でも文字化けしないよう BOM付きUTF-8 で書き出す。
    日時はタイムゾーン付きISO形式（例: 2026-01-01T09:00:00+09:00）で保存する。
    """
    out = df.copy()
    if "published_at" in out.columns and len(out):
        out["published_at"] = out["published_at"].map(lambda t: t.isoformat())
    return out.to_csv(index=False).encode("utf-8-sig")


def save_comments_csv(df: pd.DataFrame, path: Union[str, Path]) -> Path:
    """コメント表をCSVファイルに保存する。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(comments_to_csv_bytes(df))
    return path
