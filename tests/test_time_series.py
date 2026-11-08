"""時系列集計のテスト（Janome不要。感情分析済みの表を直接作って渡す）。"""
import numpy as np
import pandas as pd
import pytest

from src.config import EMOTIONS
from src.time_series import aggregate_time_series, time_series_to_csv_bytes

NAN = np.nan


def analyzed_frame(rows):
    """rows: (時刻文字列, is_negative, anger, is_reply) のリスト"""
    df = pd.DataFrame(rows, columns=["published_at", "is_negative", "anger", "is_reply"])
    df["published_at"] = pd.to_datetime(df["published_at"]).dt.tz_localize("Asia/Tokyo")
    for e in EMOTIONS:
        if e != "anger":
            df[e] = df["anger"].where(df["anger"].isna(), 0.0)
    df["video_id"] = "V1"
    return df


ROWS = [
    ("2026-01-01 10:05", 1.0, 1.0, False),
    ("2026-01-01 10:40", 0.0, 0.0, False),
    ("2026-01-01 10:59", NAN, NAN, True),    # 判定不能
    # 11時台は0件
    ("2026-01-01 12:30", 1.0, 0.5, False),
]


def test_hourly_bins_include_empty_hours():
    ts = aggregate_time_series(analyzed_frame(ROWS), min_comments=2)
    assert list(ts.index.hour) == [10, 11, 12]
    assert str(ts.index.tz) == "Asia/Tokyo"
    assert list(ts["n_comments"]) == [3, 0, 1]
    assert ts["n_comments"].sum() == len(ROWS)


def test_rates_and_nan_for_empty_bins():
    ts = aggregate_time_series(analyzed_frame(ROWS), min_comments=2)
    h10, h11, h12 = ts.iloc[0], ts.iloc[1], ts.iloc[2]
    assert h10["neg_rate"] == pytest.approx(1 / 3)          # 分母＝全コメント
    assert h10["neg_rate_judged"] == pytest.approx(1 / 2)   # 分母＝判定できたコメント
    assert h10["coverage"] == pytest.approx(2 / 3)
    assert h10["anger_mean"] == pytest.approx(0.5)          # 判定不能は平均に入れない
    assert np.isnan(h11["neg_rate"]) and np.isnan(h11["anger_mean"])   # 0件 → 0%ではなく欠損
    assert h11["n_negative"] == 0
    assert list(ts["low_reliability"]) == [False, True, True]
    assert h12["neg_rate"] == 1.0


def test_moving_average_uses_only_past():
    rows = [(f"2026-01-01 {h:02d}:00", 0.0, 0.0, False) for h in range(6) for _ in range(h + 1)]
    df = analyzed_frame(rows)
    ts1 = aggregate_time_series(df, ma_window=3)
    assert ts1["n_comments_ma"].iloc[2] == pytest.approx((1 + 2 + 3) / 3)
    assert ts1["n_comments_ma"].iloc[0] == 1.0
    # 未来のビン（5時）のコメントを増やしても、過去の移動平均は変わらない（データリークなし）
    extra = analyzed_frame([("2026-01-01 05:10", 0.0, 0.0, False)] * 50)
    ts2 = aggregate_time_series(pd.concat([df, extra], ignore_index=True), ma_window=3)
    pd.testing.assert_series_equal(ts1["n_comments_ma"].iloc[:5], ts2["n_comments_ma"].iloc[:5])


# 3時間ビンは日本時間 0,3,6,9,12… 時区切り → 10:05 と 12:30 は別のビン
@pytest.mark.parametrize("freq, n_bins", [("30min", 6), ("1h", 3), ("3h", 2)])
def test_other_frequencies(freq, n_bins):
    ts = aggregate_time_series(analyzed_frame(ROWS), freq=freq)
    assert len(ts) == n_bins and ts["n_comments"].sum() == len(ROWS)
    assert all(t.hour % 3 == 0 for t in ts.index) or freq != "3h"


def test_exclude_replies():
    ts = aggregate_time_series(analyzed_frame(ROWS), include_replies=False)
    assert ts["n_comments"].sum() == 3


def test_empty_input_does_not_crash():
    ts = aggregate_time_series(analyzed_frame(ROWS).iloc[0:0])
    assert ts.empty and "neg_rate" in ts.columns


def test_errors():
    df = analyzed_frame(ROWS)
    with pytest.raises(ValueError):
        aggregate_time_series(df.drop(columns=["is_negative"]))
    two = df.copy()
    two.loc[0, "video_id"] = "V2"
    with pytest.raises(ValueError):
        aggregate_time_series(two)


def test_csv_export():
    text = time_series_to_csv_bytes(aggregate_time_series(analyzed_frame(ROWS))).decode("utf-8-sig")
    assert text.splitlines()[0].startswith("bin_start,n_comments")
    assert "2026-01-01T10:00:00+09:00" in text
