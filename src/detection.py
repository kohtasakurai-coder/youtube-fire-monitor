"""
検知処理のまとめ役：時系列表 → 手法A・B・C → 警戒レベル → 検知イベント（根拠付き）

    result = run_detection(ts, cfg)
    result.table      … ビンごとの全指標と判定（a_*, b_*, c_*, level）
    result.events     … 検知時刻・手法・根拠・警戒レベルの一覧
    result.cp_events  … 手法C（オンライン）の検知（変化点の位置と検知時刻を区別）
    result.offline_cps… 全期間を使った事後分析の変化点（評価には使わない）
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from src.anomaly_detection import (
    _infer_freq,
    alert_levels,
    build_events,
    detect_negative_rate,
    detect_zscore,
)
from src.change_point import detect_change_points_offline, detect_change_points_online
from src.config import AnalysisConfig


@dataclass
class DetectionResult:
    table: pd.DataFrame
    events: pd.DataFrame
    cp_events: pd.DataFrame
    offline_cps: pd.DataFrame
    seconds: dict = field(default_factory=dict)   # 手法ごとの計算時間（比較実験用）

    def first_detection(self, method_col: str):
        hits = self.table.index[self.table[method_col]]
        return hits[0] if len(hits) else None

    def summary(self) -> dict:
        t = self.table
        return {
            "n_bins": len(t),
            "n_alarm_A": int(t["a_alarm"].sum()),
            "n_alarm_B": int(t["b_alarm"].sum()),
            "n_alarm_C": int(t["c_alarm"].sum()),
            "first_A": self.first_detection("a_alarm"),
            "first_B": self.first_detection("b_alarm"),
            "first_C": self.first_detection("c_alarm"),
            "max_level": (t["level"].map({"正常": 0, "注意": 1, "警戒": 2}).max()
                          if len(t) else 0),
            "n_offline_cps": len(self.offline_cps),
        }


def _cp_cache_key(cfg: AnalysisConfig, freq: str) -> tuple:
    """手法Cの結果に影響する設定だけを並べたキー（評価のグリッド計算で再利用するため）。"""
    return (freq, cfg.pelt_model, cfg.pelt_penalty, cfg.pelt_window_hours, cfg.pelt_recent_hours,
            cfg.pelt_min_increase, cfg.pelt_min_size, cfg.neg_min_comments, cfg.neg_baseline_hours)


def run_detection(ts: pd.DataFrame, cfg: AnalysisConfig | None = None,
                  include_offline: bool = True, cp_cache: dict | None = None,
                  cache_prefix: tuple = ()) -> DetectionResult:
    """
    cp_cache を渡すと、同じ動画・同じ手法Cの設定の計算結果を使い回す（結果は同じ）。
    cache_prefix には動画IDなど、時系列を区別する値を入れる。
    """
    cfg = cfg or AnalysisConfig()
    if ts.empty:
        empty = ts.copy()
        for c in ("a_alarm", "b_alarm", "c_alarm"):
            empty[c] = pd.Series(dtype=bool)
        empty["level"] = pd.Series(dtype=object)
        return DetectionResult(empty, pd.DataFrame(columns=["detected_at", "method", "reason", "level"]),
                               pd.DataFrame(), pd.DataFrame())
    freq = _infer_freq(ts)
    seconds = {}

    t0 = time.perf_counter()
    a = detect_negative_rate(ts, cfg, freq)
    seconds["A"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    b = detect_zscore(ts, cfg, freq)
    seconds["B"] = time.perf_counter() - t0

    key = cache_prefix + _cp_cache_key(cfg, freq)
    if cp_cache is not None and key in cp_cache:
        c, cp_events, seconds["C"] = cp_cache[key]
    else:
        t0 = time.perf_counter()
        c, cp_events = detect_change_points_online(ts, cfg, freq)
        seconds["C"] = time.perf_counter() - t0
        if cp_cache is not None:
            cp_cache[key] = (c, cp_events, seconds["C"])

    table = pd.concat([ts, a, b, c], axis=1)
    alarms = pd.DataFrame({"A": table["a_alarm"], "B": table["b_alarm"], "C": table["c_alarm"]})
    table = pd.concat([table, alert_levels(alarms, cfg, freq)], axis=1)

    offline = detect_change_points_offline(ts, cfg) if include_offline else pd.DataFrame()
    events = build_events(table, cp_events, cfg)
    return DetectionResult(table, events, cp_events, offline, seconds)
