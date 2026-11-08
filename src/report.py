"""
論文用の表と図を、評価結果のCSVから作る（STEP 8 の準備）。

入力（experiments/compare_methods.py の出力）:
  <results>/dev_grid.csv               … dev でのパラメータごとの結果（実験3）
  <results>/test/test_summary.csv      … test での手法ごとの指標（実験1・2・4）
  <results>/test/test_per_video.csv    … test の動画ごとの判定
  <results>/test/test_mcnemar.csv      … 対応のある比較（実験4）
出力（<results>/report/）:
  table1_test_main.md / .csv           … 表：手法ごとの検知性能（test）
  table2_mcnemar.md                    … 表：McNemar 検定
  table3_dev_sensitivity.csv           … 表：パラメータごとの性能（dev、検知器「警戒」）
  fig1_dev_sensitivity.html            … 図：Zスコア閾値・ペナルティとF1（dev）
  fig2_test_delay.html                 … 図：検知までの遅れ（test、動画ごと）
HTMLの図はブラウザで開き、右上のカメラのアイコンからPNGで保存できる。

数値はCSVからそのまま転記する（丸め以外の加工はしない）。架空データの結果には必ずその旨を書く。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import DETECTOR_LABELS, DETECTOR_ORDER


def _num(x, digits=2) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


def _is_synthetic(per_video: pd.DataFrame) -> bool:
    return per_video["video_id"].astype(str).str.startswith("SYN_").any()


def to_markdown(df: pd.DataFrame) -> str:
    """tabulate を使わない簡単な Markdown 表。"""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")
    return "\n".join(lines) + "\n"


def main_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for det in DETECTOR_ORDER:
        s = summary[summary["detector"] == det]
        if s.empty:
            continue
        s = s.iloc[0]
        rows.append({
            "検知器": DETECTOR_LABELS[det],
            "検知/炎上": f"{int(s['検知(TP)'])}/{int(s['炎上動画数'])}",
            "誤警戒/非炎上": f"{int(s['誤警戒動画(FP)'])}/{int(s['非炎上動画数'])}",
            "Precision": _num(s["Precision"]),
            "Recall [95%CI]": f"{_num(s['Recall'])} {s['Recall_95CI']}",
            "F1 [95%CI]": f"{_num(s['F1'])} {s['F1_95CI(bootstrap)']}",
            "誤警戒/100h": _num(s["誤警戒_100時間あたり"]),
            "遅れ 中央値[25%, 75%](h)": (f"{_num(s['遅れ_中央値(h)'], 1)} "
                                       f"[{_num(s['遅れ_25%(h)'], 1)}, {_num(s['遅れ_75%(h)'], 1)}]"),
        })
    return pd.DataFrame(rows)


def sensitivity_table(dev_grid: pd.DataFrame, detector: str = "警戒") -> pd.DataFrame:
    d = dev_grid[dev_grid["detector"] == detector].copy()
    params = [c for c in ("freq", "neg_rate_increase", "neg_binom_alpha", "z_threshold", "pelt_penalty")
              if c in d.columns]
    keep = ["config_id"] + params + ["F1", "Precision", "Recall", "誤警戒_100時間あたり", "遅れ_中央値(h)",
                                     "計算時間C(s)"]
    return d[[c for c in keep if c in d.columns]].sort_values("config_id").reset_index(drop=True)


def fig_sensitivity(sens: pd.DataFrame, note: str):
    import plotly.graph_objects as go

    fig = go.Figure()
    group_cols = [c for c in ("freq", "pelt_penalty") if c in sens.columns]
    for key, g in sens.groupby(group_cols):
        key = key if isinstance(key, tuple) else (key,)
        best = g.groupby("z_threshold")["F1"].max().reset_index()   # 他のパラメータでの最大
        name = "、".join(f"{c}={v}" for c, v in zip(group_cols, key))
        fig.add_trace(go.Scatter(x=best["z_threshold"], y=best["F1"], mode="lines+markers", name=name))
    fig.update_layout(title=f"図1　Zスコア閾値とF1（dev、検知器「警戒」、他のパラメータでの最大）{note}",
                      xaxis_title="Zスコアの閾値", yaxis_title="F1", yaxis_range=[0, 1.05], height=420)
    return fig


def fig_delay(per_video: pd.DataFrame, note: str):
    import plotly.graph_objects as go

    d = per_video[(per_video["is_fire"] == 1) & (per_video["outcome"] == "TP")]
    fig = go.Figure()
    for det in DETECTOR_ORDER:
        g = d[d["detector"] == det]
        if g.empty:
            continue
        fig.add_trace(go.Box(y=g["delay_hours"], name=det, boxpoints="all", jitter=0.3,
                             text=g["video_id"], hovertemplate="%{text}: %{y}時間"))
    fig.update_layout(title=f"図2　炎上開始から検知までの時間（test、検知できた動画のみ）{note}",
                      yaxis_title="遅れ（時間）＝検知できた時刻 − 炎上開始", showlegend=False, height=420)
    return fig


def build_report(results_dir: str | Path) -> Path:
    results_dir = Path(results_dir)
    out = results_dir / "report"
    out.mkdir(parents=True, exist_ok=True)
    test_dir = results_dir / "test"
    summary = pd.read_csv(test_dir / "test_summary.csv", encoding="utf-8-sig")
    per_video = pd.read_csv(test_dir / "test_per_video.csv", encoding="utf-8-sig")
    synthetic = _is_synthetic(per_video)
    note = "【架空データ・実動画の性能ではない】" if synthetic else ""
    header = (f"# 評価結果の表{note}\n\n"
              f"test の動画 {per_video['video_id'].nunique()} 本"
              f"（炎上 {per_video.loc[per_video['is_fire'] == 1, 'video_id'].nunique()} 本）。"
              "炎上開始〜48時間後の検知のみ正解、開始前の警報は誤警戒。遅れは検知できた時刻で計算。\n\n")

    t1 = main_table(summary)
    t1.to_csv(out / "table1_test_main.csv", index=False, encoding="utf-8-sig")
    (out / "table1_test_main.md").write_text(header + "## 表1　手法ごとの検知性能（test）\n\n"
                                             + to_markdown(t1), encoding="utf-8")
    mc_path = test_dir / "test_mcnemar.csv"
    if mc_path.exists():
        mc = pd.read_csv(mc_path, encoding="utf-8-sig")
        mc["p値(McNemar正確検定)"] = mc["p値(McNemar正確検定)"].map(lambda p: _num(p, 3))
        (out / "table2_mcnemar.md").write_text(f"## 表2　対応のある比較（McNemar正確検定、test）{note}\n\n"
                                               + to_markdown(mc), encoding="utf-8")
    fig_delay(per_video, note).write_html(out / "fig2_test_delay.html", include_plotlyjs=True)

    dev_path = results_dir / "dev_grid.csv"
    if dev_path.exists():
        sens = sensitivity_table(pd.read_csv(dev_path, encoding="utf-8-sig"))
        sens.to_csv(out / "table3_dev_sensitivity.csv", index=False, encoding="utf-8-sig")
        fig_sensitivity(sens, note).write_html(out / "fig1_dev_sensitivity.html", include_plotlyjs=True)
    return out
