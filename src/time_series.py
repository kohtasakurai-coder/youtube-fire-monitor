"""
時系列集計モジュール（STEP 3）

感情分析済みのコメント表を、一定の時間幅（初期値1時間）ごとに集計する。

■ 各時間ビン t で計算する指標
  n_comments        コメント投稿数
  n_judged          感情を判定できたコメント数（辞書に一致した語があるもの）
  n_negative        否定的コメント数
  neg_rate          否定的コメント率 N_t = n_negative / n_comments（依頼仕様の定義）
  neg_rate_judged   判定できたコメントだけを分母にした否定的コメント率（比較用）
  coverage          判定できた割合 = n_judged / n_comments
  anger_mean など   判定できたコメントの感情スコアの平均
  n_comments_ma     コメント数の移動平均（過去 ma_window ビン。未来のビンは使わない）
  low_reliability   コメント数が min_comments 未満なら True（比率の信頼性が低い）

■ ルール
  - コメントが0件の時間帯も行として残す
  - 0件のビンでは比率・平均を 0 ではなく NaN（欠損）にする
  - 時刻は日本時間（JST）。ビンの時刻は「その時間帯の開始時刻」
  - 1本の動画ごとに集計する（複数動画が混ざっているとエラー）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import EMOTIONS, JST

TS_COLUMNS = (
    ["n_comments", "n_judged", "n_negative", "neg_rate", "neg_rate_judged", "coverage"]
    + [f"{e}_mean" for e in EMOTIONS]
    + ["n_comments_ma", "low_reliability"]
)


def _empty_frame(freq: str) -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz=JST, name="bin_start", freq=None)
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in TS_COLUMNS}, index=idx)


def aggregate_time_series(analyzed: pd.DataFrame, freq: str = "1h", ma_window: int = 6,
                          min_comments: int = 10, include_replies: bool = True) -> pd.DataFrame:
    """
    感情分析済みのコメント表 → 時間ビンごとの時系列表（index は bin_start）。

    analyzed には published_at（JSTの日時）, is_negative（1/0/NaN）と
    5感情のスコア列が必要（src.sentiment_analysis.SentimentAnalyzer.analyze の出力）。
    """
    required = {"published_at", "is_negative", *EMOTIONS}
    missing = required - set(analyzed.columns)
    if missing:
        raise ValueError("感情分析の列がありません: " + ", ".join(sorted(missing))
                         + "（先に感情分析を実行してください）")
    if "video_id" in analyzed.columns and analyzed["video_id"].nunique() > 1:
        raise ValueError("複数の動画が混ざっています。動画ごとに集計してください。")
    if ma_window < 1:
        raise ValueError("移動平均の期間は1以上にしてください。")

    df = analyzed if include_replies else analyzed[~analyzed["is_reply"].astype(bool)]
    if df.empty:
        return _empty_frame(freq)

    times = df["published_at"].dt.tz_convert(JST)
    bins = times.dt.floor(freq)
    # .values を使うとタイムゾーンが外れるので、Series のまま組み立てる
    work = pd.DataFrame({
        "bin": bins,
        "judged": df["is_negative"].notna(),
        "negative": df["is_negative"] == 1,
        **{e: df[e] for e in EMOTIONS},
    })
    grouped = work.groupby("bin")
    agg = pd.DataFrame({
        "n_comments": grouped.size(),
        "n_judged": grouped["judged"].sum(),
        "n_negative": grouped["negative"].sum(),
        **{f"{e}_mean": grouped[e].mean() for e in EMOTIONS},   # NaN（判定不能）は平均から除く
    })

    # コメントのない時間帯も含めた連続した時間軸にそろえる
    full_index = pd.date_range(bins.min(), bins.max(), freq=freq, tz=JST, name="bin_start")
    ts = agg.reindex(full_index)
    for c in ("n_comments", "n_judged", "n_negative"):
        ts[c] = ts[c].fillna(0).astype("int64")

    n = ts["n_comments"].astype("float64")
    nj = ts["n_judged"].astype("float64")
    ts["neg_rate"] = np.where(n > 0, ts["n_negative"] / n.where(n > 0), np.nan)
    ts["neg_rate_judged"] = np.where(nj > 0, ts["n_negative"] / nj.where(nj > 0), np.nan)
    ts["coverage"] = np.where(n > 0, nj / n.where(n > 0), np.nan)

    # 移動平均：時刻 t までの過去 ma_window ビンだけを使う（center=False ＝ 未来を使わない）
    ts["n_comments_ma"] = n.rolling(window=ma_window, min_periods=1).mean()
    ts["low_reliability"] = ts["n_comments"] < min_comments
    return ts[TS_COLUMNS]


def time_series_to_csv_bytes(ts: pd.DataFrame) -> bytes:
    """時系列表をCSV（Excelで開けるBOM付きUTF-8、時刻はISO形式）にする。"""
    out = ts.copy()
    out.index = [t.isoformat() for t in out.index]
    out.index.name = "bin_start"
    return out.to_csv().encode("utf-8-sig")
