"""
異常検知（手法A・B・C、警戒レベル）のテスト。

いちばん重要なのは「未来のデータを足しても、過去の時刻のオンライン検知結果が変わらない」こと
（データリークがないこと）の確認。手法Cを含むテストは ruptures が必要。
"""
import numpy as np
import pandas as pd
import pytest

from src.anomaly_detection import (
    LEVEL_ALERT,
    LEVEL_CAUTION,
    LEVEL_NORMAL,
    STATUS_LOW_BASELINE,
    STATUS_LOW_COUNT,
    STATUS_NO_HISTORY,
    STATUS_OK,
    alert_levels,
    bins_for_hours,
    detect_negative_rate,
    detect_zscore,
)
from src.config import AnalysisConfig, EMOTIONS


def make_ts(counts, negs, start="2026-01-01 00:00", freq="1h"):
    """コメント数と否定的コメント数の列から、time_series と同じ形の表を作る。"""
    counts = np.asarray(counts, dtype=int)
    negs = np.asarray(negs, dtype=int)
    idx = pd.date_range(start, periods=len(counts), freq=freq, tz="Asia/Tokyo", name="bin_start")
    ts = pd.DataFrame({"n_comments": counts, "n_judged": counts, "n_negative": negs}, index=idx)
    with np.errstate(invalid="ignore", divide="ignore"):
        ts["neg_rate"] = np.where(counts > 0, negs / np.where(counts > 0, counts, 1), np.nan)
    ts["neg_rate_judged"] = ts["neg_rate"]
    ts["coverage"] = np.where(counts > 0, 1.0, np.nan)
    for e in EMOTIONS:
        ts[f"{e}_mean"] = np.nan
    ts["n_comments_ma"] = ts["n_comments"].rolling(6, min_periods=1).mean()
    ts["low_reliability"] = ts["n_comments"] < 10
    return ts


CFG = AnalysisConfig()


# ---------------------------------------------------------------- 共通
def test_bins_for_hours():
    assert bins_for_hours(24, "1h") == 24
    assert bins_for_hours(24, "h") == 24
    assert bins_for_hours(24, "30min") == 48
    assert bins_for_hours(24, "3h") == 8
    assert bins_for_hours(1, "3h") == 1          # 最低1ビン


# ---------------------------------------------------------------- 手法A
def test_method_a_detects_rise_against_own_past():
    counts = [20] * 24 + [20]
    negs = [2] * 24 + [10]                        # 10% → 50%
    a = detect_negative_rate(make_ts(counts, negs), CFG)
    assert (a["a_status"].iloc[:24] == STATUS_NO_HISTORY).all()   # 直前24時間がそろわない
    last = a.iloc[-1]
    assert last["a_status"] == STATUS_OK
    assert last["a_baseline_rate"] == pytest.approx(0.10)
    assert last["a_delta"] == pytest.approx(0.40)
    assert bool(last["a_alarm"]) is True


def test_method_a_constant_high_negativity_is_not_fire():
    """否定的な話題で、最初から否定率が60%のまま → 検知しない。"""
    a = detect_negative_rate(make_ts([30] * 60, [18] * 60), CFG)
    assert not a["a_alarm"].any()


def test_method_a_baseline_is_weighted_by_counts():
    # 直前24時間：1件だけ（100%否定）の時間が1つ、残りは20件中2件 → 重み付きでは約10%
    counts = [1] + [20] * 23 + [20]
    negs = [1] + [2] * 23 + [2]
    a = detect_negative_rate(make_ts(counts, negs), CFG).iloc[-1]
    assert a["a_baseline_rate"] == pytest.approx(47 / 461)
    assert not a["a_alarm"]


def test_method_a_low_counts_are_not_judged():
    counts = [20] * 24 + [5]
    negs = [0] * 24 + [5]                          # 5件中5件（100%）でも件数不足
    a = detect_negative_rate(make_ts(counts, negs), CFG).iloc[-1]
    assert a["a_status"] == STATUS_LOW_COUNT and not a["a_alarm"]

    counts = [0] * 24 + [30]                       # 直前24時間が0件 → 基準件数不足
    a = detect_negative_rate(make_ts(counts, [0] * 24 + [30]), CFG).iloc[-1]
    assert a["a_status"] == STATUS_LOW_BASELINE and not a["a_alarm"]


def test_method_a_zero_comment_bins_are_nan_not_zero():
    a = detect_negative_rate(make_ts([20] * 24 + [0], [2] * 24 + [0]), CFG).iloc[-1]
    assert np.isnan(a["a_rate"]) and not a["a_alarm"]


def test_method_a_binomial_option_removes_small_sample_noise():
    counts = [400] * 24 + [13]
    negs = [220] * 24 + [10]                       # 55% → 77%（+22pt）だが13件だけ
    base = detect_negative_rate(make_ts(counts, negs), CFG).iloc[-1]
    assert base["a_alarm"]                          # 設計書どおりの設定では検知
    strict = detect_negative_rate(make_ts(counts, negs),
                                  AnalysisConfig(neg_binom_alpha=0.01)).iloc[-1]
    assert strict["a_pvalue"] > 0.01 and not strict["a_alarm"]


# ---------------------------------------------------------------- 手法B
def test_method_b_zscore_and_threshold():
    counts = [10, 12] * 12 + [60]
    b = detect_zscore(make_ts(counts, [0] * 25), CFG).iloc[-1]
    mu, sd = np.mean([10, 12] * 12), np.std([10, 12] * 12, ddof=1)
    assert b["b_z"] == pytest.approx((60 - mu) / max(sd, 1.0))
    assert b["b_alarm"]


def test_method_b_sigma_zero_does_not_divide_by_zero():
    b = detect_zscore(make_ts([20] * 24 + [25], [0] * 25), CFG).iloc[-1]
    assert b["b_sigma"] == 0
    assert b["b_z"] == pytest.approx(5.0)          # σ_floor=1 を使う
    assert np.isfinite(b["b_z"])


def test_method_b_requires_history_and_min_count():
    b = detect_zscore(make_ts([0] * 24 + [9], [0] * 25), CFG)
    assert (b["b_status"].iloc[:24] == STATUS_NO_HISTORY).all()
    assert b["b_status"].iloc[-1] == STATUS_LOW_COUNT and not b["b_alarm"].iloc[-1]


def test_method_b_threshold_is_strict():
    b = detect_zscore(make_ts([20] * 24 + [23], [0] * 25), CFG).iloc[-1]
    assert b["b_z"] == pytest.approx(3.0) and not b["b_alarm"]   # 3を「超えた」ときだけ


# ---------------------------------------------------------------- 警戒レベル
def test_alert_levels():
    idx = pd.date_range("2026-01-01", periods=6, freq="1h", tz="Asia/Tokyo")
    alarms = pd.DataFrame({"A": [0, 1, 0, 0, 0, 0], "B": [0, 0, 1, 0, 0, 0],
                           "C": [0, 0, 1, 0, 0, 0]}, index=idx).astype(bool)
    lv = alert_levels(alarms, CFG, "1h")
    assert list(lv["level"]) == [LEVEL_NORMAL, LEVEL_CAUTION, LEVEL_ALERT, LEVEL_CAUTION,
                                 LEVEL_CAUTION, LEVEL_NORMAL]
    assert lv["active_methods"].iloc[2] == "A+B+C"
    assert alert_levels(alarms.assign(C=False), CFG, "1h")["level"].iloc[2] == LEVEL_CAUTION


# ---------------------------------------------------------------- 手法C（PELT）
rpt = pytest.importorskip("ruptures")
from src.change_point import detect_change_points_offline, detect_change_points_online  # noqa: E402
from src.detection import run_detection  # noqa: E402


def step_ts(before=40, after=12, p0=0.1, p1=0.7, n=30, seed=0):
    rng = np.random.default_rng(seed)
    counts = [n] * (before + after)
    negs = list(rng.binomial(n, p0, before)) + list(rng.binomial(n, p1, after))
    return make_ts(counts, negs)


def test_online_pelt_separates_change_point_and_detection_time():
    ts = step_ts()
    per_bin, events = detect_change_points_online(ts, CFG)
    assert len(events) == 1
    ev = events.iloc[0]
    assert ev["change_point_at"] == ts.index[40]          # 変化点の位置
    assert ev["detected_at"] > ev["change_point_at"]      # 検知できた時刻はそれより後
    assert ev["delay_bins"] == (ts.index.get_loc(ev["detected_at"]) - 40)
    assert per_bin["c_alarm"].sum() == 1
    assert ev["post_mean"] - ev["pre_mean"] >= CFG.pelt_min_increase


def test_online_pelt_ignores_decrease_and_constant():
    _, ev_down = detect_change_points_online(step_ts(p0=0.7, p1=0.1), CFG)
    assert ev_down.empty
    _, ev_flat = detect_change_points_online(make_ts([30] * 60, [9] * 60), CFG)
    assert ev_flat.empty


def test_online_pelt_skips_low_count_bins():
    counts = [30] * 30 + [2] * 10 + [30] * 10
    negs = [3] * 30 + [2] * 10 + [3] * 10          # 2件中2件（100%）の時間帯は信頼できない
    _, events = detect_change_points_online(make_ts(counts, negs), CFG)
    assert events.empty


def test_offline_uses_future_and_is_reported_separately():
    ts = step_ts()
    offline = detect_change_points_offline(ts, CFG)
    assert ts.index[40] in set(offline["change_point_at"])
    # 変化直後の時刻までしか見ていないオンライン検知では、まだ検知されていない
    _, early = detect_change_points_online(ts.iloc[:41], CFG)
    assert early.empty


# ---------------------------------------------------------------- データリークがないこと
def _comparable(table):
    cols = ["a_rate", "a_baseline_rate", "a_delta", "a_status", "a_alarm",
            "b_mu", "b_sigma", "b_z", "b_status", "b_alarm", "c_alarm", "c_evaluated",
            "n_methods", "level"]
    return table[cols]


@pytest.mark.parametrize("cut", [30, 41, 42, 43, 45, 50])
def test_future_data_does_not_change_past_online_results(cut):
    """全期間で計算した結果と、時刻 cut までのデータだけで計算した結果が、cut までで一致する。"""
    ts = step_ts(before=40, after=20)
    full = run_detection(ts, CFG)
    past = run_detection(ts.iloc[:cut], CFG)
    pd.testing.assert_frame_equal(_comparable(full.table).iloc[:cut], _comparable(past.table))
    full_ev = full.events[full.events["detected_at"] < ts.index[cut]].reset_index(drop=True)
    pd.testing.assert_frame_equal(full_ev, past.events)


def test_future_comments_do_not_change_past_results_end_to_end():
    """コメント単位：未来のコメント（大量の否定的コメント）を足しても過去の判定は同じ。"""
    from src.synthetic_data import _NEGATIVE, generate_sample_dataset
    from src.time_series import aggregate_time_series

    comments, _ = generate_sample_dataset(seed=3)
    df = comments[comments["video_id"] == "SYN_FIRE01"].copy()
    df["published_at"] = pd.to_datetime(df["published_at"])
    neg_texts = {f"【架空】{t}" for t in _NEGATIVE}
    df["is_negative"] = df["text"].isin(neg_texts).astype(float)   # Janomeを使わない簡易判定
    for e in EMOTIONS:
        df[e] = df["is_negative"] if e == "anger" else 0.0

    cutoff = df["published_at"].min().floor("1h") + pd.Timedelta(hours=58)
    observed = df[df["published_at"] < cutoff]
    future = df[df["published_at"] >= cutoff].copy()
    future["is_negative"] = 1.0                       # 未来をすべて否定的に書き換えても
    with_future = pd.concat([observed, future], ignore_index=True)

    r_past = run_detection(aggregate_time_series(observed), CFG)
    r_full = run_detection(aggregate_time_series(with_future), CFG)
    n = len(r_past.table)
    pd.testing.assert_frame_equal(_comparable(r_full.table).iloc[:n], _comparable(r_past.table))


# ---------------------------------------------------------------- 判定対象外と「正常」の区別
from src.anomaly_detection import LEVEL_UNJUDGED  # noqa: E402


def test_alert_levels_unjudged():
    idx = pd.date_range("2026-01-01", periods=5, freq="1h", tz="Asia/Tokyo")
    alarms = pd.DataFrame({"A": [0, 1, 0, 0, 0], "B": [0] * 5, "C": [0] * 5}, index=idx).astype(bool)
    judged = pd.Series([False, True, False, True, False], index=idx)
    lv = alert_levels(alarms, CFG, "1h", judged)
    # 判定なし・検知なし → 判定対象外、直近3時間に検知 → 注意（判定できない時間帯でも）、判定あり → 正常
    assert list(lv["level"]) == [LEVEL_UNJUDGED, LEVEL_CAUTION, LEVEL_CAUTION, LEVEL_CAUTION, LEVEL_UNJUDGED]
    lv2 = alert_levels(alarms.iloc[:1], CFG, "1h", judged.iloc[:1] | True)
    assert list(lv2["level"]) == [LEVEL_NORMAL]


def test_history_only_data_is_unjudged_not_normal():
    """履歴不足だけのデータ（20時間分）：すべて判定対象外で、「正常」は1つもない。"""
    r = run_detection(make_ts([30] * 20, [3] * 20), CFG)
    assert not r.table["judged"].any()
    assert (r.table["level"] == LEVEL_UNJUDGED).all()
    s = r.summary()
    assert s["n_judged_bins"] == 0 and s["max_level"] == -1


def test_low_count_burst_in_76_seconds_is_unjudged():
    """1スレッドの短時間に集中したコメント（1時間に50件、前後なし）も判定対象外。"""
    r = run_detection(make_ts([50], [5]), CFG)
    assert list(r.table["level"]) == [LEVEL_UNJUDGED]
    assert r.summary()["n_judged_bins"] == 0


def test_non_fire_data_with_judged_bins_is_normal():
    """判定できる非炎上データ：最初の24時間は判定対象外、その後は判定して正常。"""
    r = run_detection(make_ts([30] * 60, [9] * 60), CFG)
    t = r.table
    assert (t["level"].iloc[:24] == LEVEL_UNJUDGED).all()
    assert (t["level"].iloc[24:] == LEVEL_NORMAL).all()
    assert t["judged"].iloc[24:].all()
    assert set(t["judged_methods"].iloc[30:]) == {"A+B+C"}
    s = r.summary()
    assert s["n_judged_bins"] == 36 and s["max_level"] == 0 and s["n_alert_bins"] == 0


def test_existing_detection_unchanged_by_unjudged_level():
    """判定対象外の追加で、警戒（同じ時間帯に2手法以上）・各手法の検知は変わらない。"""
    ts = step_ts()
    r = run_detection(ts, CFG)
    both = (r.table[["a_alarm", "b_alarm", "c_alarm"]].sum(axis=1) >= 2)
    assert ((r.table["level"] == LEVEL_ALERT) == both).all()
    assert not ((r.table["level"] == LEVEL_UNJUDGED) & r.table[["a_alarm", "b_alarm", "c_alarm"]].any(axis=1)).any()
