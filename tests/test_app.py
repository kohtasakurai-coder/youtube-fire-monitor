"""
画面のテスト（Streamlit AppTest）。Streamlit と Janome が入っている環境でだけ実行される。
架空データを読み込み → 「感情分析を実行」→ 指標と集計表が表示されることを確かめる。
"""
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("janome")
from streamlit.testing.v1 import AppTest  # noqa: E402


def click(at, label):
    next(b for b in at.button if b.label == label).click().run()


def test_sentiment_section_on_sample_data():
    at = AppTest.from_file("../app.py", default_timeout=120)
    at.run()
    assert not at.exception

    click(at, "サンプル（架空データ）を読み込む")
    assert not at.exception
    assert any("架空データ" in w.value for w in at.warning)      # 架空データの警告

    click(at, "感情分析を実行")
    assert not at.exception
    labels = {m.label: m.value for m in at.metric}
    assert "否定的コメント率（全コメント中）" in labels
    assert "判定できた割合（カバー率）" in labels
    assert labels["信頼性の低い時間帯"].endswith("/ 120")       # SYN_FIRE01 は120時間分


def test_dashboard_detection_on_sample_data():
    """架空データの読み込み → 分析ダッシュボード → 「検知を実行」→ 比較表と根拠が表示される。"""
    pytest.importorskip("ruptures")
    pytest.importorskip("plotly")
    at = AppTest.from_file("../app.py", default_timeout=300)
    at.run()
    click(at, "サンプル（架空データ）を読み込む")
    at.sidebar.radio[0].set_value("分析ダッシュボード").run()
    assert not at.exception
    click(at, "検知を実行")
    assert not at.exception

    comparison = at.dataframe[0].value.set_index("動画")
    fire = comparison.loc["SYN_FIRE01"]
    assert fire["最初の警戒（赤）"] == "2026/01/08 06:00"      # 架空の炎上開始と同じ時刻
    assert fire["手法C 初検知"] == "2026/01/08 08:00"          # 変化点 06:00 を 2時間後に検知
    negnews = comparison.loc["SYN_NEGNEWS1"]
    assert negnews["最初の警戒（赤）"] == "—"             # 否定的な話題でも警戒（赤）にならない
    assert negnews["手法C 初検知"] == "—"
    assert comparison.loc["SYN_NORMAL01", "最初の警戒（赤）"] == "—"

    labels = {m.label: m.value for m in at.metric}
    assert "現在の警戒レベル" in labels and "最新の異常スコア（Z）" in labels
    events = at.dataframe[1].value                        # 詳細（SYN_FIRE01）の検知一覧
    assert "根拠" in events.columns and len(events) > 0


def test_comparison_and_data_pages():
    """検知の実行後、アルゴリズム比較（架空の正解での評価）とデータ確認が表示される。"""
    pytest.importorskip("ruptures")
    pytest.importorskip("plotly")
    at = AppTest.from_file("../app.py", default_timeout=300)
    at.run()
    click(at, "サンプル（架空データ）を読み込む")
    at.sidebar.radio[0].set_value("分析ダッシュボード").run()
    click(at, "検知を実行")

    at.sidebar.radio[0].set_value("アルゴリズム比較").run()
    assert not at.exception
    detail = at.dataframe[0].value
    assert {"検知した時間帯の数", "最初の検知", "計算時間(秒)"} <= set(detail.columns)
    summary = at.dataframe[1].value.set_index("detector")
    alert = summary.loc["警戒"]
    assert (alert["検知(TP)"], alert["見逃し(FN)"], alert["誤警戒動画(FP)"]) == (1, 0, 0)
    assert summary.loc["C", "遅れ_中央値(h)"] == 2.0          # 検知できた時刻（08:00）で計算
    assert any("実動画での性能ではありません" in w.value for w in at.warning)

    at.sidebar.radio[0].set_value("データ確認").run()
    assert not at.exception
    table = at.dataframe[0].value
    assert {"投稿日時（JST）", "コメント本文", "否定的コメント判定", "いいね数"} <= set(table.columns)
    assert len(table) == 2713


def test_dashboard_shows_unjudged_for_short_burst():
    """76秒に集中した50件（1スレッド）のようなデータでは「正常」ではなく判定対象外と表示する。"""
    pytest.importorskip("ruptures")
    pytest.importorskip("plotly")
    import pandas as pd
    from src.preprocessing import normalize_comments

    times = pd.date_range("2020-05-17 15:07:00", periods=50, freq="1500ms", tz="Asia/Tokyo")
    raw = pd.DataFrame({
        "comment_id": [f"c{i}" for i in range(50)], "published_at": [t.isoformat() for t in times],
        "text": ["おめでとう"] * 50, "like_count": 0, "parent_id": [""] + ["c0"] * 49,
        "is_reply": [False] + [True] * 49, "video_id": "SHORTBURST", "data_source": "real"})
    df, _ = normalize_comments(raw)
    at = AppTest.from_file("../app.py", default_timeout=300)
    at.session_state["comments"] = df
    at.session_state["data_label"] = "短時間に集中したデータ"
    at.session_state["load_warnings"] = []
    at.run()
    at.sidebar.radio[0].set_value("分析ダッシュボード").run()
    click(at, "検知を実行")
    assert not at.exception
    comparison = at.dataframe[0].value.set_index("動画")
    assert comparison.loc["SHORTBURST", "判定結果（期間中の最高レベル）"].startswith("⚪ 判定対象外")
    assert comparison.loc["SHORTBURST", "判定できた時間帯"] == "0 / 1"
    assert any("判定対象外" in w.value for w in at.warning)
    labels = {m.label: m.value for m in at.metric}
    assert labels["現在の警戒レベル"] == "⚪ 判定対象外"
