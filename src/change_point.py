"""
変化点検出モジュール（STEP 4）：手法C（PELT）

■ オンライン検知（評価・警戒レベルに使うのはこちら）
  各時刻 t で、t までに観測できたデータだけを使って PELT を計算し直す。
    1. 直近 L ビン（初期値72時間）のうち、コメント数が下限（10件）以上の
       「信頼できるビン」の否定的コメント率 N_s（s ≤ t）だけを時刻順に並べる
       （件数不足のビンは除外する。設計書では直前値で埋める案だったが、
        件数不足が長く続くと同じ値が何十個も並ぶ人工的な平坦区間ができ、
        PELTの結果がゆがむことを架空データで確認したため、除外に変更した）
    2. その系列で PELT を実行（窓 L は計算量の上限を決めるため）
    3. 推定された変化点 τ が直近 k ビン（初期値3時間）以内にあり、
       変化後の平均 − 変化前の平均 ≥ 最小上昇幅 なら「時刻 t に検知」
    4. 同じ変化点（すでに報告した τ から k ビン以内）は二度報告しない
  ※ 「変化点の位置 τ」と「検知できた時刻 t」は別物。検知遅延 = t − τ（≥ 0）。
     評価（検知の速さ）には t を使う。

  計算の省略：時刻 t のビンに信頼できる新しい観測（コメント数 ≥ 下限）がないときは、
  前の時刻から入力が増えていないので計算しない（結果は変わらず、未来も使わない）。

■ 事後分析（全期間を一度に使う。画面で「事後分析用」と明示し、評価には使わない）
  全期間のデータで PELT を1回だけ実行した結果。未来のデータを使っているので、
  これをリアルタイムの検知結果として扱ってはいけない（データリーク）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.anomaly_detection import _infer_freq, bins_for_hours
from src.config import AnalysisConfig

CP_EVENT_COLUMNS = ["detected_at", "change_point_at", "delay_bins", "pre_mean", "post_mean",
                    "reason"]


def _import_ruptures():
    try:
        import ruptures as rpt
    except ImportError as e:  # pragma: no cover
        raise ImportError("ruptures がインストールされていません。"
                          "`pip install -r requirements.txt` を実行してください。") from e
    return rpt


def reliable_positions(ts: pd.DataFrame, min_comments: int) -> np.ndarray:
    """コメント数が下限以上のビンの位置（0始まり）。PELTにはこの位置の否定率だけを入れる。"""
    ok = (ts["n_comments"] >= min_comments) & ts["neg_rate"].notna()
    return np.flatnonzero(ok.to_numpy())


def _pelt_breakpoints(y: np.ndarray, cfg: AnalysisConfig) -> list[int]:
    """PELTの変化点（区間の切れ目の位置。最後の len(y) は除く）を返す。"""
    rpt = _import_ruptures()
    if len(y) < 2 * cfg.pelt_min_size or np.ptp(y) == 0:
        return []                     # 短すぎる・一定の系列には変化点なし
    algo = rpt.Pelt(model=cfg.pelt_model, min_size=cfg.pelt_min_size, jump=1).fit(y)
    return [b for b in algo.predict(pen=cfg.pelt_penalty) if b < len(y)]


def detect_change_points_online(ts: pd.DataFrame, cfg: AnalysisConfig | None = None,
                                freq: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    手法C（オンライン）。
    戻り値: (ビンごとの表［c_alarm, c_evaluated］, 検知イベントの表)
    """
    cfg = cfg or AnalysisConfig()
    freq = freq or _infer_freq(ts)
    window = bins_for_hours(cfg.pelt_window_hours, freq)
    recent = bins_for_hours(cfg.pelt_recent_hours, freq)
    history = bins_for_hours(cfg.neg_baseline_hours, freq)   # 手法Aと同じく24時間分の履歴を要求

    rate = ts["neg_rate"].to_numpy(dtype="float64")
    positions = reliable_positions(ts, cfg.neg_min_comments)
    reliable_now = np.zeros(len(ts), dtype=bool)
    reliable_now[positions] = True
    n = len(ts)
    alarm = np.zeros(n, dtype=bool)
    evaluated = np.zeros(n, dtype=bool)
    reported: list[int] = []          # 報告済みの変化点の位置（絶対位置）
    events = []

    for t in range(n):
        if t < history or not reliable_now[t]:
            continue
        # ← 時刻 t までの、直近 window ビン内の信頼できるビンだけ（未来は使わない）
        idx = positions[(positions <= t) & (positions > t - window)]
        if len(idx) < 2 * cfg.pelt_min_size:
            continue
        y = rate[idx]
        evaluated[t] = True

        bkps = _pelt_breakpoints(y, cfg)
        for i, b in enumerate(bkps):
            tau = int(idx[b])                  # 変化点の位置（新しい区間の最初のビン）
            if t - tau >= recent:
                continue                       # 古い変化点は「今の検知」ではない
            if any(abs(tau - r) < recent for r in reported):
                continue                       # すでに報告した変化点
            prev = bkps[i - 1] if i > 0 else 0
            nxt = bkps[i + 1] if i + 1 < len(bkps) else len(y)
            pre_mean, post_mean = float(np.mean(y[prev:b])), float(np.mean(y[b:nxt]))
            if post_mean - pre_mean < cfg.pelt_min_increase:
                continue                       # 上昇でない・上昇が小さい変化は対象外
            reported.append(tau)
            alarm[t] = True
            events.append({
                "detected_at": ts.index[t],
                "change_point_at": ts.index[tau],
                "delay_bins": t - tau,
                "pre_mean": pre_mean,
                "post_mean": post_mean,
                "reason": (f"否定率の水準が {ts.index[tau]:%Y/%m/%d %H:%M} から "
                           f"{pre_mean:.0%} → {post_mean:.0%}（{(post_mean - pre_mean) * 100:+.0f}ポイント）"
                           f"に変化したと、{ts.index[t]:%Y/%m/%d %H:%M} までのデータで推定"
                           f"（PELT model={cfg.pelt_model}, penalty={cfg.pelt_penalty:g}）"),
            })

    per_bin = pd.DataFrame({"c_alarm": alarm, "c_evaluated": evaluated}, index=ts.index)
    return per_bin, pd.DataFrame(events, columns=CP_EVENT_COLUMNS)


def detect_change_points_offline(ts: pd.DataFrame, cfg: AnalysisConfig | None = None
                                 ) -> pd.DataFrame:
    """
    事後分析用：全期間のデータで1回だけ PELT を実行した変化点。
    未来のデータを使っているため、検知の評価や警戒レベルには使わないこと。
    """
    cfg = cfg or AnalysisConfig()
    idx = reliable_positions(ts, cfg.neg_min_comments)
    cols = ["change_point_at", "pre_mean", "post_mean"]
    if len(idx) < 2 * cfg.pelt_min_size:
        return pd.DataFrame(columns=cols)
    y = ts["neg_rate"].to_numpy(dtype="float64")[idx]
    bkps = _pelt_breakpoints(y, cfg)
    rows = []
    for i, b in enumerate(bkps):
        prev = bkps[i - 1] if i > 0 else 0
        nxt = bkps[i + 1] if i + 1 < len(bkps) else len(y)
        rows.append({"change_point_at": ts.index[idx[b]],
                     "pre_mean": float(np.mean(y[prev:b])),
                     "post_mean": float(np.mean(y[b:nxt]))})
    return pd.DataFrame(rows, columns=cols)
