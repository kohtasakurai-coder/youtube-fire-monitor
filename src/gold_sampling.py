"""
感情分析の人手評価（正解ラベル）用に、判定するコメントの候補を選ぶ。

方針（docs の「コメント判定基準」に沿う）
  - 辞書の予測を見ずに判定できるよう、出力には本文と出所だけを入れる
    （否定スコア・感情語・判定結果などの列は入れない）
  - 同一の comment_id、同一の本文（NFKC正規化・前後空白除去後）は1件として数える
  - 複数の動画・スレッドから集める：動画ごとの上限、スレッド（親コメントとその返信）ごとの上限を設ける
  - 済んでいるラベル（--exclude）と同じ comment_id は選ばない
  - 選び方は「seed と comment_id から作るハッシュ値の小さい順」。
    あとから動画を追加しても、上限を変えなければ既に選んだ候補は変わらない（再現性）
"""
from __future__ import annotations

import hashlib
import unicodedata

import pandas as pd

GOLD_COLUMNS = ["order", "comment_id", "video_id", "is_reply", "parent_id", "text",
                "gold_negative", "hold_reason", "annotator", "notes"]


def _rank(seed: str, comment_id: str) -> str:
    return hashlib.sha256(f"{seed}:{comment_id}".encode("utf-8")).hexdigest()


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text)).strip()


def sample_gold_candidates(comments: pd.DataFrame, n: int, per_video_cap: int,
                           per_thread_cap: int = 3, exclude_ids: set | None = None,
                           seed: str = "gold-2026", min_chars: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    戻り値: (候補の表［GOLD_COLUMNS］, 動画ごとの内訳)
    comments は load_comments_csv の出力（複数動画を連結してよい）。
    """
    exclude_ids = {str(x) for x in (exclude_ids or set())}
    df = comments.copy()
    df["comment_id"] = df["comment_id"].astype(str)
    df["video_id"] = df["video_id"].astype(str) if "video_id" in df.columns else "unknown"
    df["_norm"] = df["text"].map(_norm)
    df = df[df["_norm"].str.len() >= min_chars]
    df = df.drop_duplicates("comment_id")
    df = df[~df["comment_id"].isin(exclude_ids)]
    df["_rank"] = [_rank(seed, c) for c in df["comment_id"]]
    df = df.sort_values("_rank")
    df = df.drop_duplicates("_norm", keep="first")            # 同じ本文は1件
    # スレッド：返信は親コメントのID、トップレベルは自分のID
    df["_thread"] = df["parent_id"].where(df["is_reply"].astype(bool) & (df["parent_id"].astype(str) != ""),
                                          df["comment_id"])
    df["_thread_rank"] = df.groupby(["video_id", "_thread"]).cumcount()
    df = df[df["_thread_rank"] < per_thread_cap]
    df["_video_rank"] = df.groupby("video_id").cumcount()
    df = df[df["_video_rank"] < per_video_cap]
    picked = df.head(n).copy()

    out = pd.DataFrame({
        "order": range(1, len(picked) + 1),
        "comment_id": picked["comment_id"].values,
        "video_id": picked["video_id"].values,
        "is_reply": picked["is_reply"].astype(bool).values,
        "parent_id": picked["parent_id"].fillna("").astype(str).values,
        "text": picked["text"].values,
    })
    for c in ("gold_negative", "hold_reason", "annotator", "notes"):
        out[c] = ""
    breakdown = (out.groupby("video_id")
                 .agg(件数=("comment_id", "size"), 返信=("is_reply", "sum"))
                 .assign(トップレベル=lambda d: d["件数"] - d["返信"]).reset_index())
    return out[GOLD_COLUMNS], breakdown
