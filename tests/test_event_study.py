"""予備評価（基準時刻の前後比較）のテスト。正解ラベル（炎上開始）は使わない。"""
import pandas as pd
import pytest

pytest.importorskip("ruptures")
from src.config import AnalysisConfig  # noqa: E402
from src.detection import run_detection  # noqa: E402
from src.event_study import coverage, event_window  # noqa: E402
from tests.test_anomaly import step_ts  # noqa: E402


def comments_at(times):
    return pd.DataFrame({"published_at": pd.DatetimeIndex(times)})


def test_coverage_flags():
    t0 = pd.Timestamp("2025-01-26 16:00", tz="Asia/Tokyo")
    t1 = pd.Timestamp("2025-01-28 16:00", tz="Asia/Tokyo")
    c = coverage(comments_at([t0 - pd.Timedelta(hours=1), t0 + pd.Timedelta(hours=3), t1]), t0, t1)
    assert c["covers_start"] and c["covers_end"] and c["n_comments_in_window"] == 1
    c2 = coverage(comments_at([t0 + pd.Timedelta(hours=5)]), t0, t1)
    assert not c2["covers_start"] and not c2["covers_end"]


def test_event_window_before_and_after():
    ts = step_ts(before=40, after=20)                  # 40ビン目から否定率が上がる
    res = run_detection(ts, AnalysisConfig())
    event = ts.index[40]
    comments = comments_at(ts.index.repeat(ts["n_comments"]))
    summary, det, window = event_window(res.table, comments, event, before_hours=12, after_hours=12)
    assert summary["n_bins_pre"] == 12 and summary["n_bins_post"] == 12
    assert summary["covers_start"] and summary["n_judged_bins_pre"] == 12
    assert summary["neg_rate_post"] > summary["neg_rate_pre"]
    a = det.set_index("detector").loc["A"]
    assert a["基準より前の検知（時間帯数）"] == 0
    assert a["基準から検知まで(h)"] == 0.0 and a["基準から検知まで_集計確定時刻ベース(h)"] == 1.0
    c = det.set_index("detector").loc["C"]
    assert c["基準から検知まで(h)"] >= 1.0                 # PELT は変化の後に検知（検知時刻で数える）
    assert len(window) == 24 and "judged" in window.columns


def test_observation_summary_separates_unjudged_from_no_alert():
    from src.event_study import OBSERVATION_CATEGORIES, observation_summary
    from tests.test_anomaly import make_ts
    # 履歴不足だけ：すべて「判定対象外（検知なし）」で、「警戒なし（正常）」は0
    r = run_detection(make_ts([30] * 20, [3] * 20), AnalysisConfig())
    s = observation_summary(r.table)
    assert s["判定できた時間帯"] == 0 and s["判定対象外（検知なし）"] == 20
    assert s["判定した：警戒なし（正常）"] == 0
    # 判定できる非炎上データ：24時間後から「判定した：警戒なし（正常）」
    r2 = run_detection(make_ts([30] * 60, [9] * 60), AnalysisConfig())
    s2 = observation_summary(r2.table)
    assert s2["判定対象外（検知なし）"] == 24 and s2["判定した：警戒なし（正常）"] == 36
    assert sum(s2[c] for c in OBSERVATION_CATEGORIES) == s2["時間帯数"] == 60
    # 期間で絞り込み
    t0 = r2.table.index[30]
    s3 = observation_summary(r2.table, start=t0, end=t0 + pd.Timedelta(hours=5))
    assert s3["時間帯数"] == 5 and s3["判定した：警戒なし（正常）"] == 5
