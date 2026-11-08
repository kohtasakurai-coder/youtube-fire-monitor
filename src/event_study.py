"""
予備評価：基準時刻（例：記者会見の開始）の前後でコメントと検知を比べる。

これは本評価（正解ラベルの「炎上開始時刻」を使う評価）とは別の分析。
基準時刻は炎上開始のラベルではなく、「その出来事の前後で反応が変わったか」
「出来事から何時間で検知したか」を見るための観測の基準として扱う。

■ 確認すること
  1. 取得範囲：最初・最後のコメント時刻が、観測期間［基準−before, 基準+after）を覆っているか
     （YouTube の API は新しい順に返すため、上限に達すると古いコメントが欠ける）
  2. 判定できた時間帯：観測期間の前半・後半それぞれで、判定条件を満たした時間帯の数
  3. 検知：前半（基準より前）と後半（基準以後）に分けて記録。
     基準から検知までの時間 = 最初の検知の時間帯の開始 − 基準（時間）
     ※ その時間帯の集計が確定するのは時間帯の終わりなので、「確定時刻ベース」は +1ビン
"""
from __future__ import annotations

import pandas as pd

from src.anomaly_detection import LEVEL_ALERT, _infer_freq
from src.evaluation import detector_flags

EVENT_DETECTORS = ["A", "B", "C", "警戒", "注意以上"]


def _bin_hours(table: pd.DataFrame) -> float:
    return pd.Timedelta(pd.tseries.frequencies.to_offset(_infer_freq(table))) / pd.Timedelta(hours=1)


def coverage(comments: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """コメントの取得範囲が観測期間を覆っているか。"""
    t = comments["published_at"]
    first, last = (t.min(), t.max()) if len(t) else (None, None)
    in_window = t[(t >= start) & (t < end)]
    return {
        "first_comment": first, "last_comment": last,
        "covers_start": bool(first is not None and first <= start),
        "covers_end": bool(last is not None and last >= end),
        "n_comments_in_window": int(len(in_window)),
    }


def event_window(table: pd.DataFrame, comments: pd.DataFrame, event_time: pd.Timestamp,
                 before_hours: float = 24, after_hours: float = 24) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    戻り値: (要約の辞書, 検知器ごとの前後比較の表, 観測期間の時間帯ごとの表)
    table は run_detection().table、comments は感情分析済み（または生）のコメント表。
    """
    event_time = pd.Timestamp(event_time)
    event_time = event_time.tz_localize("Asia/Tokyo") if event_time.tzinfo is None else event_time.tz_convert("Asia/Tokyo")
    start = event_time - pd.Timedelta(hours=before_hours)
    end = event_time + pd.Timedelta(hours=after_hours)
    cov = coverage(comments, start, end)

    window = table[(table.index >= start) & (table.index < end)]
    pre, post = window[window.index < event_time], window[window.index >= event_time]
    bin_h = _bin_hours(table) if len(table) >= 2 else 1.0
    summary = {
        "event_time": event_time, "window_start": start, "window_end": end, **cov,
        "n_bins_pre": len(pre), "n_bins_post": len(post),
        "n_comments_pre": int(pre["n_comments"].sum()) if len(pre) else 0,
        "n_comments_post": int(post["n_comments"].sum()) if len(post) else 0,
        "n_judged_bins_pre": int(pre["judged"].sum()) if len(pre) else 0,
        "n_judged_bins_post": int(post["judged"].sum()) if len(post) else 0,
        "neg_rate_pre": (pre["n_negative"].sum() / pre["n_comments"].sum()) if len(pre) and pre["n_comments"].sum() else float("nan"),
        "neg_rate_post": (post["n_negative"].sum() / post["n_comments"].sum()) if len(post) and post["n_comments"].sum() else float("nan"),
        "n_alert_bins_post": int((post["level"] == LEVEL_ALERT).sum()) if len(post) else 0,
        "bin_hours": bin_h,
    }

    rows = []
    flags = detector_flags(window) if len(window) else None
    for det in EVENT_DETECTORS:
        hits = [] if flags is None else list(flags.index[flags[det].to_numpy()])
        pre_hits = [t for t in hits if t < event_time]
        post_hits = [t for t in hits if t >= event_time]
        first = min(post_hits) if post_hits else None
        delay = (first - event_time) / pd.Timedelta(hours=1) if first is not None else float("nan")
        rows.append({
            "detector": det,
            "基準より前の検知（時間帯数）": len(pre_hits),
            "基準より前の最初の検知": min(pre_hits) if pre_hits else None,
            "基準以後の検知（時間帯数）": len(post_hits),
            "基準以後の最初の検知": first,
            "基準から検知まで(h)": delay,
            "基準から検知まで_集計確定時刻ベース(h)": delay + bin_h if first is not None else float("nan"),
        })
    cols = ["n_comments", "n_negative", "neg_rate", "judged", "judged_methods",
            "a_alarm", "b_alarm", "c_alarm", "level"]
    return summary, pd.DataFrame(rows), window[[c for c in cols if c in window.columns]]
