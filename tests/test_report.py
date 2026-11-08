"""論文用の表と図（src/report.py）のテスト。数値がCSVから変わらずに転記されることを確かめる。"""
import pandas as pd
import pytest

from src.evaluation import evaluate_video, mcnemar_tests, summarize
from src.report import main_table, sensitivity_table, to_markdown
from tests.test_evaluation import label, table_with


def _per_video():
    return pd.concat([
        evaluate_video(table_with(alarms_a=[52], alarms_c=[54]), label("SYN_F1", incident="I1")),
        evaluate_video(table_with(alarms_b=[10]), label("SYN_N1", fire=False, start_bin=None, incident="I2")),
    ], ignore_index=True)


def test_main_table_copies_numbers():
    summary = summarize(_per_video(), n_boot=20)
    t = main_table(summary).set_index("検知器")
    a = t.loc["手法A 否定率の上昇"]
    assert a["検知/炎上"] == "1/1" and a["誤警戒/非炎上"] == "0/1"
    assert a["遅れ 中央値[25%, 75%](h)"].startswith("2.0")
    assert t.loc["手法C オンラインPELT（検知時刻）", "遅れ 中央値[25%, 75%](h)"].startswith("4.0")
    md = to_markdown(main_table(summary))
    assert md.splitlines()[1].startswith("|---")


def test_sensitivity_table():
    grid = pd.DataFrame({"config_id": [1, 0, 0], "detector": ["警戒", "警戒", "A"],
                         "z_threshold": [3.0, 2.0, 2.0], "F1": [0.5, 0.8, 0.1]})
    s = sensitivity_table(grid)
    assert list(s["config_id"]) == [0, 1] and list(s["F1"]) == [0.8, 0.5]


def test_build_report_files(tmp_path):
    pytest.importorskip("plotly")
    from src.report import build_report
    pv = _per_video()
    (tmp_path / "test").mkdir()
    summarize(pv, n_boot=20).to_csv(tmp_path / "test" / "test_summary.csv", index=False, encoding="utf-8-sig")
    pv.to_csv(tmp_path / "test" / "test_per_video.csv", index=False, encoding="utf-8-sig")
    mcnemar_tests(pv).to_csv(tmp_path / "test" / "test_mcnemar.csv", index=False, encoding="utf-8-sig")
    out = build_report(tmp_path)
    names = {p.name for p in out.iterdir()}
    assert {"table1_test_main.md", "table1_test_main.csv", "table2_mcnemar.md", "fig2_test_delay.html"} <= names
    assert "架空データ" in (out / "table1_test_main.md").read_text(encoding="utf-8")   # SYN_ の動画
