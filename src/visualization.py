"""
グラフ（Plotly）。拡大・縮小・ホバーで数値を確認できる。

- 時刻は日本時間（JST）の時計の値で表示する（Plotlyのタイムゾーン処理の違いを避けるため、
  表示直前にタイムゾーン情報だけを外す）
- 検知した時刻には赤い縦線を引く
- 手法Cは「変化点の位置（オレンジ破線）」と「検知できた時刻（赤線）」を分けて描く
- 全期間を使った事後分析の変化点は灰色の点線で描き、凡例に「事後分析・評価に使わない」と書く
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from src.config import EMOTIONS

EMOTION_LABELS = {"anger": "怒り", "fear": "恐怖", "sadness": "悲しみ", "joy": "喜び", "trust": "信頼"}
RED = "#d62728"
ORANGE = "#ff7f0e"
GRAY = "#7f7f7f"


def _naive(index) -> pd.DatetimeIndex:
    """タイムゾーン付きの時刻 → JSTの時計の値（タイムゾーン情報なし）。"""
    idx = pd.DatetimeIndex(index)
    return idx.tz_convert("Asia/Tokyo").tz_localize(None) if idx.tz is not None else idx


def _one(t) -> pd.Timestamp:
    return _naive([t])[0]


def _vlines(fig: go.Figure, times, color: str, dash: str = "solid", width: float = 1.5,
            name: str | None = None) -> None:
    """縦線を引き、凡例用に見えない線を1本追加する。"""
    times = list(times)
    for t in times:
        fig.add_shape(type="line", x0=_one(t), x1=_one(t), y0=0, y1=1, yref="paper",
                      line=dict(color=color, dash=dash, width=width), layer="below")
    if times and name:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=name,
                                 line=dict(color=color, dash=dash, width=width)))


def _layout(fig: go.Figure, title: str, ytitle: str, height: int = 320) -> go.Figure:
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=40, b=10),
                      hovermode="x unified", legend=dict(orientation="h", y=-0.2),
                      yaxis_title=ytitle)
    return fig


def _reference(fig: go.Figure, reference_time) -> None:
    if reference_time is not None:
        _vlines(fig, [reference_time], "#2ca02c", dash="dot", width=2,
                name="【架空】炎上開始（生成時に設定した正解）")


def fig_counts(table: pd.DataFrame, reference_time=None) -> go.Figure:
    """グラフ1：コメント投稿数・移動平均と、手法Bの検知時刻。"""
    x = _naive(table.index)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=table["n_comments"], name="コメント数", marker_color="#9ecae1"))
    fig.add_trace(go.Scatter(x=x, y=table["n_comments_ma"], name="移動平均", mode="lines",
                             line=dict(color="#08519c")))
    fig.add_trace(go.Scatter(x=x, y=table["b_mu"], name="直前24時間の平均 μ", mode="lines",
                             line=dict(color=GRAY, dash="dot")))
    _vlines(fig, table.index[table["b_alarm"]], RED, name="手法Bの検知")
    _reference(fig, reference_time)
    return _layout(fig, "グラフ1　コメント投稿数の推移", "件数")


def fig_negative_rate(table: pd.DataFrame, reference_time=None) -> go.Figure:
    """グラフ2：否定的コメント率・直前24時間の否定率と、手法Aの検知時刻。"""
    x = _naive(table.index)
    reliable = (~table["low_reliability"].astype(bool)).to_numpy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x[reliable], y=table["neg_rate"].to_numpy()[reliable], name="否定的コメント率", mode="lines+markers",
        line=dict(color=RED), customdata=table["n_comments"].to_numpy()[reliable],
        hovertemplate="%{y:.0%}（%{customdata}件）"))
    fig.add_trace(go.Scatter(
        x=x[~reliable], y=table["neg_rate"].to_numpy()[~reliable], name="件数が少ない時間帯（参考）",
        mode="markers", marker=dict(color="#bbbbbb", size=5),
        customdata=table["n_comments"].to_numpy()[~reliable], hovertemplate="%{y:.0%}（%{customdata}件）"))
    fig.add_trace(go.Scatter(x=x, y=table["a_baseline_rate"], name="直前24時間の否定率",
                             mode="lines", line=dict(color=GRAY, dash="dot")))
    _vlines(fig, table.index[table["a_alarm"]], RED, name="手法Aの検知")
    _reference(fig, reference_time)
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    return _layout(fig, "グラフ2　否定的コメント率の推移（0件の時間帯は欠損）", "否定率")


def fig_emotions(table: pd.DataFrame) -> go.Figure:
    """グラフ3：5種類の感情スコア（判定できたコメントの平均）。"""
    x = _naive(table.index)
    fig = go.Figure()
    for e in EMOTIONS:
        fig.add_trace(go.Scatter(x=x, y=table[f"{e}_mean"], name=EMOTION_LABELS[e], mode="lines",
                                 connectgaps=False))
    fig.update_yaxes(range=[0, 1])
    return _layout(fig, "グラフ3　感情スコアの推移", "平均スコア")


def fig_zscore(table: pd.DataFrame, threshold: float) -> go.Figure:
    """グラフ4：Zスコアと閾値、検知時刻（手法A・B・Cすべて）と警戒レベル。"""
    x = _naive(table.index)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=table["b_z"], name="Zスコア", mode="lines+markers",
                             line=dict(color="#6a51a3"), marker=dict(size=4)))
    fig.add_hline(y=threshold, line=dict(color=RED, dash="dash"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=f"閾値 {threshold:g}",
                             line=dict(color=RED, dash="dash")))
    level_y = {"注意": "orange", "警戒": RED}
    for level, color in level_y.items():
        hit = (table["level"] == level).to_numpy()
        if hit.any():
            fig.add_trace(go.Scatter(x=x[hit], y=[0] * int(hit.sum()), mode="markers",
                                     name=f"警戒レベル：{level}",
                                     marker=dict(color=color, symbol="square", size=8)))
    any_alarm = table["a_alarm"] | table["b_alarm"] | table["c_alarm"]
    _vlines(fig, table.index[any_alarm], RED, name="いずれかの手法で検知")
    return _layout(fig, "グラフ4　異常検知の結果と検知時刻", "Zスコア")


def fig_change_points(table: pd.DataFrame, cp_events: pd.DataFrame,
                      offline_cps: pd.DataFrame, reference_time=None) -> go.Figure:
    """グラフ5：変化点。オンラインの「変化点の位置」と「検知時刻」、事後分析を区別する。"""
    x = _naive(table.index)
    reliable = (~table["low_reliability"].astype(bool)).to_numpy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x[reliable], y=table["neg_rate"].to_numpy()[reliable], mode="lines+markers",
                             name="否定的コメント率（PELTへの入力）", line=dict(color="#444444"),
                             marker=dict(size=4)))
    if len(cp_events):
        _vlines(fig, cp_events["change_point_at"], ORANGE, dash="dash",
                name="オンライン：推定された変化点の位置")
        _vlines(fig, cp_events["detected_at"], RED, name="オンライン：検知できた時刻")
    if len(offline_cps):
        _vlines(fig, offline_cps["change_point_at"], GRAY, dash="dot", width=2,
                name="事後分析（全期間を使用・評価には使わない）")
    _reference(fig, reference_time)
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    return _layout(fig, "グラフ5　変化点検出（PELT）", "否定率")
