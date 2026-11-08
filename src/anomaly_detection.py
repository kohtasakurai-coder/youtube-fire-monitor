"""
異常検知モジュール（STEP 4）：手法A（否定的コメント率）・手法B（Zスコア）・警戒レベル

すべて「時刻 t の判定には、時刻 t までに観測できたデータだけを使う」オンライン方式。
基準期間 B_t は t 自身を含まない直前 W ビン（初期値 24時間）。

■ 手法A：否定的コメント率（動画自身の過去との比較）
    N_t   = n_t^neg / n_t
    N̄_t  = Σ_{s∈B_t} n_s^neg / Σ_{s∈B_t} n_s      （件数で重み付けした直前24時間の否定率）
    検知  : n_t ≥ n_min かつ Σ_{s∈B_t} n_s ≥ n_min_base かつ N_t − N̄_t ≥ δ
    （オプション）さらに二項検定 P(X ≥ n_t^neg | X〜Bin(n_t, N̄_t)) ≤ α も要求できる
    → 否定率が元から高いニュース動画でも、「その動画のいつもの水準」からの上昇だけを見る

■ 手法B：コメント数のZスコア
    μ_t, σ_t = B_t のコメント数の平均・標本標準偏差
    Z_t = (x_t − μ_t) / max(σ_t, σ_floor)        （σ=0 でもゼロ除算しない）
    検知  : Z_t > z_threshold かつ x_t ≥ z_min_comments

■ 判定できない時間帯（status 列に理由を記録し、検知にはしない）
    履歴不足     : データの最初から W ビン経っていない（直前24時間がそろわない）
    件数不足     : その時間帯のコメント数が下限未満
    基準件数不足 : 直前24時間のコメント合計が下限未満（手法Aのみ）

■ 警戒レベル（直近 L ビン＝初期値3時間に検知した手法の数 k_t）
    正常（緑）: k_t = 0   注意（黄）: k_t ≥ 1   警戒（赤）: k_t ≥ red_min（初期値2）
    ※ 研究上の暫定的な判定基準であり、実際に炎上していることを保証しない
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binom

from src.config import AnalysisConfig

STATUS_OK = "判定"
STATUS_NO_HISTORY = "履歴不足"
STATUS_LOW_COUNT = "件数不足"
STATUS_LOW_BASELINE = "基準件数不足"

LEVEL_NORMAL = "正常"
LEVEL_CAUTION = "注意"
LEVEL_ALERT = "警戒"
LEVEL_COLORS = {LEVEL_NORMAL: "green", LEVEL_CAUTION: "orange", LEVEL_ALERT: "red"}

METHOD_NAMES = {
    "A": "手法A 否定率の上昇",
    "B": "手法B コメント数のZスコア",
    "C": "手法C 変化点（オンラインPELT）",
}


def bins_for_hours(hours: float, freq: str) -> int:
    """「○時間」を、集計時間幅 freq のビン数に直す（例：24時間・30分幅 → 48ビン）。"""
    bin_hours = pd.Timedelta(pd.tseries.frequencies.to_offset(freq)) / pd.Timedelta(hours=1)
    return max(1, int(round(hours / bin_hours)))


def _infer_freq(ts: pd.DataFrame) -> str:
    if ts.index.freq is not None:
        return pd.tseries.frequencies.to_offset(ts.index.freq).freqstr
    if len(ts.index) >= 2:
        return pd.tseries.frequencies.to_offset(ts.index[1] - ts.index[0]).freqstr
    return "1h"


# ---------------------------------------------------------------------------
# 手法A
# ---------------------------------------------------------------------------
def detect_negative_rate(ts: pd.DataFrame, cfg: AnalysisConfig | None = None,
                         freq: str | None = None) -> pd.DataFrame:
    """手法A。戻り値は ts と同じ index の表（a_ で始まる列）。"""
    cfg = cfg or AnalysisConfig()
    freq = freq or _infer_freq(ts)
    w = bins_for_hours(cfg.neg_baseline_hours, freq)

    n = ts["n_comments"].astype("float64")
    neg = ts["n_negative"].astype("float64")
    # shift(1) で t 自身を外し、直前 w ビンだけで合計する（未来も現在も使わない）
    base_n = n.shift(1).rolling(w, min_periods=w).sum()
    base_neg = neg.shift(1).rolling(w, min_periods=w).sum()
    base_rate = base_neg / base_n.where(base_n > 0)
    rate = ts["neg_rate"]
    delta = rate - base_rate

    status = pd.Series(STATUS_OK, index=ts.index, dtype="object")
    status[base_n.isna()] = STATUS_NO_HISTORY
    status[(status == STATUS_OK) & (n < cfg.neg_min_comments)] = STATUS_LOW_COUNT
    status[(status == STATUS_OK) & (base_n < cfg.neg_min_baseline_comments)] = STATUS_LOW_BASELINE
    alarm = (status == STATUS_OK) & (delta >= cfg.neg_rate_increase)

    # 二項検定の片側p値：直前24時間の否定率が続いているとしたら、今の件数以上が否定的になる確率
    p0 = base_rate.clip(0, 1).fillna(0).to_numpy()
    pvalue = pd.Series(binom.sf(neg.to_numpy() - 1, n.to_numpy().astype(int), p0), index=ts.index)
    pvalue[status != STATUS_OK] = np.nan
    if cfg.neg_binom_alpha is not None:
        alarm = alarm & (pvalue <= cfg.neg_binom_alpha)

    return pd.DataFrame({
        "a_rate": rate,
        "a_baseline_rate": base_rate,
        "a_baseline_n": base_n,
        "a_delta": delta,
        "a_pvalue": pvalue,
        "a_status": status,
        "a_alarm": alarm.fillna(False).astype(bool),
    }, index=ts.index)


# ---------------------------------------------------------------------------
# 手法B
# ---------------------------------------------------------------------------
def detect_zscore(ts: pd.DataFrame, cfg: AnalysisConfig | None = None,
                  freq: str | None = None) -> pd.DataFrame:
    """手法B。戻り値は ts と同じ index の表（b_ で始まる列）。"""
    cfg = cfg or AnalysisConfig()
    freq = freq or _infer_freq(ts)
    w = bins_for_hours(cfg.z_baseline_hours, freq)

    count = ts["n_comments"].astype("float64")
    if cfg.z_transform == "log1p":
        x = np.log1p(count)
        floor = np.log1p(cfg.z_sigma_floor)   # 変換後の尺度に合わせる
    elif cfg.z_transform == "none":
        x = count
        floor = cfg.z_sigma_floor
    else:
        raise ValueError(f"z_transform は none / log1p のどちらかです: {cfg.z_transform}")

    past = x.shift(1).rolling(w, min_periods=w)
    mu = past.mean()
    sigma = past.std(ddof=1)
    z = (x - mu) / np.maximum(sigma, floor)

    status = pd.Series(STATUS_OK, index=ts.index, dtype="object")
    status[mu.isna()] = STATUS_NO_HISTORY
    status[(status == STATUS_OK) & (count < cfg.z_min_comments)] = STATUS_LOW_COUNT
    alarm = (status == STATUS_OK) & (z > cfg.z_threshold)

    return pd.DataFrame({
        "b_count": count,
        "b_mu": mu,
        "b_sigma": sigma,
        "b_z": z,
        "b_status": status,
        "b_alarm": alarm.fillna(False).astype(bool),
    }, index=ts.index)


# ---------------------------------------------------------------------------
# 警戒レベルと検知の一覧
# ---------------------------------------------------------------------------
def alert_levels(alarms: pd.DataFrame, cfg: AnalysisConfig | None = None,
                 freq: str | None = None) -> pd.DataFrame:
    """
    alarms: 列 A, B, C（True/False）を持つ表。
    警戒は同じビンで複数手法が検知したときに出す。注意には直近 L ビンの検知を反映する。
    """
    cfg = cfg or AnalysisConfig()
    lookback = bins_for_hours(cfg.alert_lookback_hours, freq or _infer_freq(alarms))
    recent = alarms.astype(int).rolling(lookback, min_periods=1).max()
    k = recent.sum(axis=1).astype(int)
    simultaneous = alarms.astype(bool).sum(axis=1)
    level = np.where(simultaneous >= cfg.red_min_methods, LEVEL_ALERT,
                     np.where(k >= 1, LEVEL_CAUTION, LEVEL_NORMAL))
    methods = recent.apply(lambda r: "+".join(c for c in alarms.columns if r[c]), axis=1)
    return pd.DataFrame({"n_methods": k, "level": level, "active_methods": methods},
                        index=alarms.index)


def _fmt_pct(x: float) -> str:
    return "—" if pd.isna(x) else f"{x:.0%}"


def explain_a(row: pd.Series, cfg: AnalysisConfig) -> str:
    return (f"否定率 {_fmt_pct(row['a_rate'])}（{int(row['n_comments'])}件中 {int(row['n_negative'])}件）が"
            f"直前{cfg.neg_baseline_hours}時間の否定率 {_fmt_pct(row['a_baseline_rate'])}"
            f"（{int(row['a_baseline_n'])}件）より {row['a_delta'] * 100:+.0f}ポイント高い"
            f"（閾値 +{cfg.neg_rate_increase * 100:.0f}ポイント"
            + (f"、二項検定 p = {row['a_pvalue']:.3g} ≤ {cfg.neg_binom_alpha:g}"
               if cfg.neg_binom_alpha is not None else "") + "）")


def explain_b(row: pd.Series, cfg: AnalysisConfig) -> str:
    if cfg.z_transform == "log1p":
        return (f"コメント数 {int(row['b_count'])}件（log(1+件数)で計算）が直前{cfg.z_baseline_hours}"
                f"時間の水準から Z = {row['b_z']:.1f}（閾値 {cfg.z_threshold:g} を超過）")
    used = max(row["b_sigma"], cfg.z_sigma_floor)
    return (f"コメント数 {int(row['b_count'])}件が直前{cfg.z_baseline_hours}時間の平均 "
            f"{row['b_mu']:.1f}件（標準偏差 {used:.1f}）から Z = {row['b_z']:.1f}"
            f"（閾値 {cfg.z_threshold:g} を超過）")


def build_events(result: pd.DataFrame, cp_events: pd.DataFrame, cfg: AnalysisConfig) -> pd.DataFrame:
    """検知時刻・手法・根拠の一覧（時刻順）。"""
    rows = []
    for t, row in result[result["a_alarm"]].iterrows():
        rows.append({"detected_at": t, "method": METHOD_NAMES["A"], "reason": explain_a(row, cfg)})
    for t, row in result[result["b_alarm"]].iterrows():
        rows.append({"detected_at": t, "method": METHOD_NAMES["B"], "reason": explain_b(row, cfg)})
    for _, ev in cp_events.iterrows():
        rows.append({"detected_at": ev["detected_at"], "method": METHOD_NAMES["C"],
                     "reason": ev["reason"]})
    events = pd.DataFrame(rows, columns=["detected_at", "method", "reason"])
    if events.empty:
        events["level"] = pd.Series(dtype="object")
        return events
    events = events.sort_values(["detected_at", "method"]).reset_index(drop=True)
    events["level"] = result.loc[events["detected_at"], "level"].values
    return events
