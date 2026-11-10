"""本評価の動画選定（experiments/prepare_main_eval.py）。API は呼ばず、架空データで流れを確かめる。"""
import pandas as pd

import experiments.prepare_main_eval as pm
from src.preprocessing import load_comments_csv
from src.evaluation import validate_labels

SAMPLE = pm.PROJECT_ROOT / "data" / "sample" / "sample_comments.csv"
TZ = "Asia/Tokyo"


def test_prepare_selects_by_rule_and_writes_labels(tmp_path, monkeypatch):
    (tmp_path / "data" / "research").mkdir(parents=True)
    monkeypatch.setattr(pm, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pm, "CALM_GAP_DAYS", 0)
    monkeypatch.setenv("YOUTUBE_API_KEY", "AIza" + "x" * 35)
    sample, _ = load_comments_csv(SAMPLE)
    t = lambda s: pd.Timestamp(s, tz=TZ)   # noqa: E731

    uploads = pd.DataFrame([
        # 新しい順に試す：最初の動画（ショート）は除外、次の SYN_FIRE01 を炎上として採用
        {"video_id": "SHORT000000", "title": "short", "published_at_jst": t("2026-01-06 12:00"),
         "comment_count": 50, "duration_sec": 40.0},
        {"video_id": "SYN_FIRE01", "title": "a", "published_at_jst": t("2026-01-05 18:00"),
         "comment_count": 2713, "duration_sec": 600.0},
        {"video_id": "SYN_NORMAL01", "title": "b", "published_at_jst": t("2026-01-01 06:00"),
         "comment_count": 2041, "duration_sec": 600.0},
        {"video_id": "OFF00000000", "title": "c", "published_at_jst": t("2025-12-20 06:00"),
         "comment_count": None, "duration_sec": 600.0},
    ])
    uploads["comment_count"] = uploads["comment_count"].astype("Int64")
    monkeypatch.setattr(pm, "resolve_channel_id", lambda c, k: "UC" + "x" * 22)
    monkeypatch.setattr(pm, "list_uploads", lambda *a, **k: uploads)
    monkeypatch.setattr(pm, "_get_json", lambda *a, **k: {"items": [{"snippet": {"title": "テスト"}}]})
    fetched = []

    def fake_fetch(vid, key, max_results, include_replies):
        fetched.append(vid)
        return sample[sample["video_id"] == vid].reset_index(drop=True), []
    monkeypatch.setattr(pm, "fetch_comments", fake_fetch)

    inc = tmp_path / "inc.csv"
    pd.DataFrame([{"incident_id": "I1", "騒動": "テスト", "channel": "x", "ref_time_jst": "2026-01-08T06:00:00+09:00",
                   "ref_upload_title": "", "ref_upload_date": "", "label_source": "報道"}]
                 ).to_csv(inc, index=False, encoding="utf-8-sig")
    assert pm.main(["--incidents", str(inc), "--dictionary", "simple",
                    "--labels-out", str(tmp_path / "data" / "research" / "labels.csv")]) == 0

    assert fetched == ["SYN_FIRE01", "SYN_NORMAL01"]          # ショートとコメント無効は取得しない
    lab = pd.read_csv(tmp_path / "data" / "research" / "labels.csv", dtype=str).fillna("")
    v = validate_labels(lab)
    fire = v[v["is_fire"] == 1].iloc[0]
    assert fire["video_id"] == "SYN_FIRE01" and fire["fire_start"] == t("2026-01-08 06:00")
    assert fire["eval_end"] - fire["eval_start"] == pd.Timedelta(hours=120)
    calm = v[v["is_fire"] == 0].iloc[0]
    assert calm["video_id"] == "SYN_NORMAL01" and calm["incident_id"] == "I1_CALM"
    assert calm["eval_start"] == t("2026-01-02 06:00")
    log = pd.read_csv(tmp_path / "results" / "prepare_main_eval.csv")
    assert set(log["status"]) == {"採用"}
    assert not any("警戒" in c for c in log.columns)            # 選定の記録に検知結果は入れない

    # 2回目は既存の labels.csv を上書きしない
    assert pm.main(["--incidents", str(inc), "--dictionary", "simple",
                    "--labels-out", str(tmp_path / "data" / "research" / "labels.csv")]) == 0
    assert (tmp_path / "data" / "research" / "labels_draft.csv").exists()
    assert fetched == ["SYN_FIRE01", "SYN_NORMAL01"]          # 取得済みのCSVを再利用


def test_placeholder_key_stops_before_any_request(monkeypatch, capsys):
    monkeypatch.setenv("YOUTUBE_API_KEY", "（手元のキー）")
    monkeypatch.setattr(pm, "resolve_channel_id", lambda *a: (_ for _ in ()).throw(AssertionError("called")))
    assert pm.main([]) == 1
    assert "APIキーの形式ではありません" in capsys.readouterr().err


def test_calm_exclusion_skips_windows_near_other_incidents(tmp_path, monkeypatch):
    """同じチャンネルの別の騒動の近くの期間は比較（平常期）に使わない。"""
    (tmp_path / "data" / "research").mkdir(parents=True)
    monkeypatch.setattr(pm, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pm, "CALM_GAP_DAYS", 0)
    monkeypatch.setenv("YOUTUBE_API_KEY", "AIza" + "x" * 35)
    sample, _ = load_comments_csv(SAMPLE)
    t = lambda s: pd.Timestamp(s, tz=TZ)   # noqa: E731
    uploads = pd.DataFrame([
        {"video_id": "SYN_FIRE01", "title": "a", "published_at_jst": t("2026-01-05 18:00"),
         "comment_count": 2713, "duration_sec": 600.0},
        {"video_id": "SYN_NORMAL01", "title": "b", "published_at_jst": t("2026-01-01 06:00"),
         "comment_count": 2041, "duration_sec": 600.0},
    ])
    uploads["comment_count"] = uploads["comment_count"].astype("Int64")
    monkeypatch.setattr(pm, "resolve_channel_id", lambda c, k: "UC" + "x" * 22)
    monkeypatch.setattr(pm, "list_uploads", lambda *a, **k: uploads)
    monkeypatch.setattr(pm, "_get_json", lambda *a, **k: {"items": [{"snippet": {"title": "テスト"}}]})
    fetched = []
    monkeypatch.setattr(pm, "fetch_comments", lambda vid, *a, **k: (fetched.append(vid) or
                        (sample[sample["video_id"] == vid].reset_index(drop=True), [])))
    inc = tmp_path / "inc.csv"
    pd.DataFrame([{"incident_id": "I1", "騒動": "x", "channel": "x", "ref_time_jst": "2026-01-08T06:00:00+09:00",
                   "ref_upload_title": "", "ref_upload_date": "", "label_source": "報道",
                   "calm_exclude_dates": "2026-01-03"}]).to_csv(inc, index=False, encoding="utf-8-sig")
    assert pm.main(["--incidents", str(inc), "--dictionary", "simple"]) == 0
    assert fetched == ["SYN_FIRE01"]                          # 比較候補は別の騒動に近いので試さない
    lab = pd.read_csv(tmp_path / "data" / "research" / "labels.csv", dtype=str)
    assert list(lab["is_fire"]) == ["1"]


def test_ref_from_upload_date_without_title():
    t = lambda s: pd.Timestamp(s, tz=TZ)   # noqa: E731
    ups = pd.DataFrame([
        {"video_id": "S", "title": "short", "published_at_jst": t("2026-10-23 12:00"), "duration_sec": 30.0},
        {"video_id": "V", "title": "説明", "published_at_jst": t("2026-10-23 19:00"), "duration_sec": 1200.0},
        {"video_id": "W", "title": "翌日", "published_at_jst": t("2026-10-24 19:00"), "duration_sec": 1200.0},
    ])
    ref, note = pm.resolve_ref(pd.Series({"ref_time_jst": "", "ref_upload_title": "",
                                          "ref_upload_date": "2026-10-23"}), ups)
    assert ref == t("2026-10-23 19:00") and "V" in note
