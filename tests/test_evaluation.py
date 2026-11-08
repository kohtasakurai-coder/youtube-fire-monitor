"""
評価処理（STEP 6）のテスト。

- 検知の遅れは「検知できた時刻」で数え、炎上開始前の警報は正解にしない
- 事後分析（全期間PELT）は指標に入らない
- dev / test の分割は事件単位、test の結果を見てからの再調整は手順で止める
"""
import json

import numpy as np
import pandas as pd
import pytest

from src.config import AnalysisConfig, EMOTIONS
from src.evaluation import (
    DETECTOR_ORDER,
    EvalConfig,
    assign_splits,
    evaluate_offline_reference,
    evaluate_video,
    mcnemar_tests,
    merge_events,
    summarize,
    validate_labels,
    wilson_interval,
)

T0 = pd.Timestamp("2026-01-01 00:00", tz="Asia/Tokyo")


def table_with(alarms_a=(), alarms_b=(), alarms_c=(), n=100, red=()):
    """検知結果の表（run_detection().table と同じ列）を手で作る。数字はビン番号（1時間）。"""
    idx = pd.date_range(T0, periods=n, freq="1h", name="bin_start")
    t = pd.DataFrame(index=idx)
    for col, hits in (("a_alarm", alarms_a), ("b_alarm", alarms_b), ("c_alarm", alarms_c)):
        t[col] = False
        t.loc[idx[list(hits)], col] = True
    any_alarm = t["a_alarm"] | t["b_alarm"] | t["c_alarm"]
    t["level"] = np.where(any_alarm, "注意", "正常")
    t.loc[idx[list(red)], "level"] = "警戒"
    return t


def label(video="V1", fire=True, start_bin=50, split="test", incident="I1"):
    return pd.Series({"video_id": video, "incident_id": incident, "is_fire": int(fire),
                      "fire_start": T0 + pd.Timedelta(hours=start_bin) if fire and start_bin is not None
                      else pd.NaT, "split": split})


def row(df, det):
    return df[df["detector"] == det].iloc[0]


# ---------------------------------------------------------------- 1本の動画の判定
def test_merge_events():
    times = [T0 + pd.Timedelta(hours=h) for h in (0, 1, 5, 12, 13, 30)]
    assert merge_events(times, 6) == [times[0], times[3], times[5]]


def test_alarm_before_start_is_not_a_hit():
    """開始の1時間前の警報は正解にしない（誤警戒として数える）。開始後の検知がなければ見逃し。"""
    r = row(evaluate_video(table_with(alarms_a=[49]), label()), "A")
    assert r["outcome"] == "FN"
    assert r["false_alarm_events"] == 1
    assert r["monitored_hours"] == 50.0                 # 開始前の50時間


def test_delay_uses_detection_time():
    """手法Cは検知できた時刻（c_alarm の時間帯）で遅れを数える。"""
    ev = evaluate_video(table_with(alarms_a=[50], alarms_c=[53]), label())
    assert row(ev, "A")["delay_hours"] == 0.0
    assert row(ev, "C")["delay_hours"] == 3.0
    assert row(ev, "C")["first_detection"] == T0 + pd.Timedelta(hours=53)


def test_detection_after_window_is_a_miss():
    r = row(evaluate_video(table_with(alarms_b=[50 + 49]), label(), EvalConfig(window_after_hours=48)), "B")
    assert r["outcome"] == "FN"
    r = row(evaluate_video(table_with(alarms_b=[98]), label(), EvalConfig(window_after_hours=48)), "B")
    assert r["outcome"] == "TP" and r["delay_hours"] == 48.0


def test_non_fire_video():
    ev = evaluate_video(table_with(alarms_b=[10, 11, 40]), label(fire=False, start_bin=None))
    r = row(ev, "B")
    assert r["outcome"] == "FP" and r["false_alarm_events"] == 2 and r["monitored_hours"] == 100.0
    assert row(ev, "A")["outcome"] == "TN"


def test_combined_detectors_and_levels():
    ev = evaluate_video(table_with(alarms_a=[50, 60], alarms_b=[60], red=[60]), label())
    assert row(ev, "B∧A")["delay_hours"] == 10.0
    assert row(ev, "B∨A")["delay_hours"] == 0.0
    assert row(ev, "警戒")["delay_hours"] == 10.0
    assert row(ev, "注意以上")["delay_hours"] == 0.0
    assert list(ev["detector"]) == DETECTOR_ORDER


def test_unknown_start_is_counted_separately():
    r = row(evaluate_video(table_with(alarms_a=[30]), label(start_bin=None)), "A")
    assert r["outcome"] == "TP" and r["start_unknown"] and np.isnan(r["delay_hours"])


# ---------------------------------------------------------------- 集計
def test_summary_numbers():
    per_video = pd.concat([
        evaluate_video(table_with(alarms_a=[52]), label("F1", incident="I1")),
        evaluate_video(table_with(), label("F2", incident="I2")),
        evaluate_video(table_with(alarms_a=[10]), label("N1", fire=False, start_bin=None, incident="I3")),
        evaluate_video(table_with(), label("N2", fire=False, start_bin=None, incident="I4")),
    ], ignore_index=True)
    s = summarize(per_video, n_boot=50).set_index("detector").loc["A"]
    assert (s["検知(TP)"], s["見逃し(FN)"], s["誤警戒動画(FP)"], s["沈黙(TN)"]) == (1, 1, 1, 1)
    assert s["Precision"] == 0.5 and s["Recall"] == 0.5 and s["F1"] == 0.5
    assert s["監視時間(h)"] == 50 + 50 + 100 + 100
    assert s["誤警戒_100時間あたり"] == pytest.approx(1 / 300 * 100)
    assert s["遅れ_中央値(h)"] == 2.0


def test_wilson_interval():
    lo, hi = wilson_interval(0, 5)
    assert lo == 0.0 and hi == pytest.approx(0.4345, abs=1e-3)
    assert np.isnan(wilson_interval(0, 0)[0])


def test_mcnemar():
    rows = []
    for i in range(6):   # B は全部見逃し、B∨A は全部検知
        lab = label(f"F{i}", incident=f"I{i}")
        rows.append(evaluate_video(table_with(alarms_a=[55]), lab))
    m = mcnemar_tests(pd.concat(rows), pairs=[("B", "B∨A")]).iloc[0]
    assert m["後者だけ正しい動画数"] == 6 and m["前者だけ正しい動画数"] == 0
    assert m["p値(McNemar正確検定)"] == pytest.approx(0.03125)


def test_offline_reference_is_separate():
    off = pd.DataFrame({"change_point_at": [T0 + pd.Timedelta(hours=47), T0 + pd.Timedelta(hours=80)]})
    ref = evaluate_offline_reference(off, label())
    assert ref["nearest_cp_to_start_hours"] == -3.0
    assert "nearest_cp_to_start_hours" not in summarize(
        evaluate_video(table_with(), label()), n_boot=10).columns


# ---------------------------------------------------------------- ラベルと分割
def labels_df(rows):
    return pd.DataFrame(rows, columns=["video_id", "incident_id", "is_fire", "fire_start_jst",
                                       "label_source", "split"]).astype(str)


def test_validate_labels_errors():
    good = labels_df([["V1", "I1", "1", "2026-01-01T10:00:00+09:00", "news", "dev"],
                      ["V2", "I2", "0", "", "manual", "test"]])
    out = validate_labels(good)
    assert out.loc[0, "fire_start"] == pd.Timestamp("2026-01-01 10:00", tz="Asia/Tokyo")
    with pytest.raises(ValueError, match="dev と test"):
        validate_labels(labels_df([["V1", "I1", "1", "", "", "dev"], ["V2", "I1", "0", "", "", "test"]]))
    with pytest.raises(ValueError, match="非炎上動画に炎上開始時刻"):
        validate_labels(labels_df([["V1", "I1", "0", "2026-01-01T10:00:00+09:00", "", "dev"]]))
    with pytest.raises(ValueError, match="重複"):
        validate_labels(labels_df([["V1", "I1", "0", "", "", "dev"], ["V1", "I2", "0", "", "", "dev"]]))
    naive = validate_labels(labels_df([["V1", "I1", "1", "2026-01-01 10:00", "", "dev"]]))
    assert str(naive.loc[0, "fire_start"].tz) == "Asia/Tokyo"   # タイムゾーンなしは JST とみなす


def test_assign_splits_keeps_incidents_together():
    rows = [[f"V{i}", f"I{i // 2}", str(int(i < 8)), "", "", ""] for i in range(16)]
    out = assign_splits(labels_df(rows), dev_fraction=0.4, seed=1)
    assert out.groupby("incident_id")["split"].nunique().max() == 1
    assert set(out["split"]) == {"dev", "test"}
    fire_inc = out[out["is_fire"] == "1"].groupby("incident_id")["split"].first()
    assert (fire_inc == "dev").sum() == 2                      # 炎上4事件の40%（切り上げ）
    with pytest.raises(ValueError, match="すでに設定"):
        assign_splits(out)


# ---------------------------------------------------------------- 実行と手順
rpt = pytest.importorskip("ruptures")
from src.detection import run_detection  # noqa: E402
from src.experiment import (  # noqa: E402
    Protocol,
    ProtocolError,
    evaluate_config,
    grid_configs,
    prepare,
    run_grid,
    select_config,
)


class TextRuleAnalyzer:
    """Janome を使わないテスト用の分析器：架空データの否定的な定型文なら否定的。"""

    def analyze(self, df):
        from src.synthetic_data import _NEGATIVE
        neg = {f"【架空】{t}" for t in _NEGATIVE}
        out = df.copy()
        out["is_negative"] = out["text"].isin(neg).astype(float)
        for e in EMOTIONS:
            out[e] = out["is_negative"] if e == "anger" else 0.0
        return out


@pytest.fixture(scope="module")
def synthetic():
    from src.synthetic_data import generate_evaluation_dataset
    comments, labels = generate_evaluation_dataset(seed=7)
    labels = validate_labels(assign_splits(labels.astype(str), dev_fraction=0.4, seed=7))
    by_video = {v: g for v, g in comments.groupby("video_id")}
    return labels, by_video


def test_cache_gives_same_result():
    from src.synthetic_data import generate_sample_dataset
    from src.time_series import aggregate_time_series
    comments, _ = generate_sample_dataset()
    ts = aggregate_time_series(TextRuleAnalyzer().analyze(comments[comments["video_id"] == "SYN_FIRE01"]))
    cache = {}
    a = run_detection(ts, AnalysisConfig())
    b = run_detection(ts, AnalysisConfig(z_threshold=2.0), cp_cache=cache, cache_prefix=("v",))
    c = run_detection(ts, AnalysisConfig(), cp_cache=cache, cache_prefix=("v",))   # キャッシュ利用
    assert len(cache) == 1
    pd.testing.assert_frame_equal(a.table, c.table)
    pd.testing.assert_frame_equal(a.events, c.events)


def test_tune_uses_only_dev_and_test_runs_once(synthetic, tmp_path):
    labels, by_video = synthetic
    dev = labels[labels["split"] == "dev"]
    test = labels[labels["split"] == "test"]
    assert set(dev["incident_id"]).isdisjoint(set(test["incident_id"]))

    data_dev = prepare(dev, {v: by_video[v] for v in dev["video_id"]}, TextRuleAnalyzer())
    assert set(data_dev.analyzed) == set(dev["video_id"])           # test の動画は読み込まない
    grid = {"freq": ["1h"], "z_threshold": [2.5, 3.0], "pelt_penalty": [3.0]}
    summary = run_grid(data_dev, grid, n_boot=20)
    assert set(summary["config_id"]) == {0, 1}
    cid = select_config(summary)
    params, cfg = grid_configs(grid)[cid]

    # 手順のガード
    labels_path = tmp_path / "labels.csv"
    labels.drop(columns=["fire_start"]).to_csv(labels_path, index=False, encoding="utf-8-sig")
    proto = Protocol(tmp_path / "protocol")
    with pytest.raises(ProtocolError):
        proto.check_labels(labels_path)                              # 固定前
    h = proto.freeze_labels(labels_path)
    proto.save_frozen_config({"config_id": cid, "freq": "1h", "analysis_config": cfg.to_dict(),
                              "eval_config": {"window_after_hours": 48.0, "merge_gap_hours": 6.0}}, h)
    frozen = proto.start_test(h)
    data_test = prepare(test, {v: by_video[v] for v in test["video_id"]}, TextRuleAnalyzer())
    per_video, offline, _ = evaluate_config(data_test, cfg, "1h")
    assert set(per_video["video_id"]) == set(test["video_id"])
    assert "nearest_cp_to_start_hours" in offline.columns           # 事後分析は別の表
    proto.finish_test(h, frozen)

    with pytest.raises(ProtocolError, match="すでに1回"):
        proto.start_test(h)
    proto.start_test(h, rerun_reason="バグ修正のため")              # 理由があれば可能（記録される）
    with pytest.raises(ProtocolError, match="test の評価後"):
        proto.save_frozen_config({"config_id": 0}, h)
    labels_path.write_text(labels_path.read_text(encoding="utf-8-sig") + "\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="変更されています"):
        proto.check_labels(labels_path)
    log = pd.read_csv(proto.log_file)
    assert list(log["action"]) == ["freeze-labels", "tune（設定を固定）", "test"]
    assert json.loads(proto.test_done_file.read_text(encoding="utf-8"))["runs"][0]["config_id"] == cid


def test_unjudged_video_is_excluded_not_tn():
    """判定できた時間帯がない非炎上動画は TN ではなく「対象外」。注意以上にも判定対象外を含めない。"""
    t = table_with()
    t["judged"] = False
    t["level"] = "判定対象外"
    ev = evaluate_video(t, label("N0", fire=False, start_bin=None, incident="I9"))
    assert set(ev["outcome"]) == {"対象外"}
    assert row(ev, "注意以上")["n_alarm_bins"] == 0
    judged_tn = evaluate_video(table_with(), label("N1", fire=False, start_bin=None, incident="I8"))
    s = summarize(pd.concat([ev, judged_tn]), n_boot=10).set_index("detector").loc["A"]
    assert s["沈黙(TN)"] == 1 and s["非炎上動画数"] == 1 and s["判定対象外で除外した動画数"] == 1


def test_monitored_hours_count_only_judged_bins():
    t = table_with(alarms_b=[60])
    t["judged"] = [i >= 24 for i in range(100)]
    r = row(evaluate_video(t, label(fire=False, start_bin=None)), "B")
    assert r["monitored_hours"] == 76.0 and r["outcome"] == "FP"
