"""
評価モジュール（STEP 6）：正解ラベル付きの動画で、手法ごとに見逃し・誤警戒・検知までの遅れを測る。

■ 評価に使う「検知」
  オンライン検知のフラグだけを使う（各時刻までのデータで計算した結果）。
  手法Cは「変化点の位置」ではなく「検知できた時刻」（c_alarm が立った時間帯）を使う。
  全期間を使う事後分析の変化点は evaluate_offline_reference() で別に扱い、指標には入れない。

■ 評価する検知器（DETECTORS）
  A, B, C            … 各手法
  B∧A, B∨A          … 実験4用（投稿数だけ vs 投稿数＋感情。同じ時間帯での組み合わせ）
  警戒               … 警戒レベルが「警戒」（同じ時間帯で設定数以上の手法が検知）
  注意以上           … 警戒レベルが「注意」または「警戒」

■ 1本の動画の判定（炎上開始時刻 s、評価窓 W＝初期値48時間）
  炎上動画：s ≤ 検知時刻 ≤ s+W の検知があれば「検知（TP）」、なければ「見逃し（FN）」
            s より前の検知は正解として数えず、誤警戒として数える（開始前の許容幅は 0）
            検知遅れ = 最初の正しい検知時刻 − s（時間）。検知時刻は検知した時間帯の開始時刻。
            その時間帯の集計が確定するのは時間帯の終わりなので、確定時刻ベースの遅れ
            （delay_hours_bin_end ＝ 遅れ＋1ビン）も動画ごとの表に出す
  非炎上動画：検知が1回でもあれば「誤警戒（FP）」、なければ「正しく沈黙（TN）」
  炎上開始時刻が不明な炎上動画：遅れと開始前の誤警戒は計算せず、件数を別に数える
  判定対象外：評価する期間（炎上動画は開始〜開始+W、非炎上動画は全期間）に、どの手法も
            判定できた時間帯が1つもない動画は「対象外」とし、TP/FN/FP/TN に入れず件数だけ数える
            （判定していないのに「正しく沈黙」や「見逃し」と数えないため）

■ 集計（動画単位）
  再現率 Recall = TP / 炎上動画数
  適合率 Precision = TP / (TP + 誤警戒の出た非炎上動画数)
  F1 = 2PR / (P + R)
  誤警戒率① = 誤警戒の出た非炎上動画数 / 非炎上動画数
  誤警戒率② = 誤警戒イベント数 / 監視時間 × 100（100時間あたり）
      監視時間 = 非炎上動画の判定できた時間帯 ＋ 炎上動画の開始前の判定できた時間帯
      誤警戒イベント = 誤警戒の検知を、間隔 merge_gap（初期値6時間）以内ならまとめた回数
  比率には Wilson の95%信頼区間、F1には動画単位のブートストラップ95%区間を付ける
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.anomaly_detection import LEVEL_ALERT, LEVEL_CAUTION, _infer_freq

DETECTOR_ORDER = ["A", "B", "C", "B∧A", "B∨A", "警戒", "注意以上"]
DETECTOR_LABELS = {
    "A": "手法A 否定率の上昇",
    "B": "手法B 投稿数のZスコア",
    "C": "手法C オンラインPELT（検知時刻）",
    "B∧A": "実験4 投稿数かつ否定率（同じ時間帯）",
    "B∨A": "実験4 投稿数または否定率",
    "警戒": "警戒レベル「警戒」",
    "注意以上": "警戒レベル「注意」以上",
}
LABEL_COLUMNS = ["video_id", "incident_id", "is_fire", "fire_start_jst", "label_source", "split"]
SPLITS = {"dev", "test"}


@dataclass
class EvalConfig:
    window_after_hours: float = 48.0   # 炎上開始から何時間以内の検知を「検知できた」とみなすか
    merge_gap_hours: float = 6.0       # これ以内の間隔の検知は1回の検知イベントにまとめる
    # 炎上開始前の許容幅は 0 で固定（開始前の警報は正解として数えない）


# ---------------------------------------------------------------------------
# 検知器
# ---------------------------------------------------------------------------
def detector_flags(table: pd.DataFrame) -> pd.DataFrame:
    """検知結果の表（run_detection().table）から、検知器ごとの True/False 列を作る。"""
    a, b, c = table["a_alarm"].astype(bool), table["b_alarm"].astype(bool), table["c_alarm"].astype(bool)
    return pd.DataFrame({
        "A": a, "B": b, "C": c,
        "B∧A": a & b, "B∨A": a | b,
        "警戒": table["level"] == LEVEL_ALERT,
        "注意以上": table["level"].isin([LEVEL_CAUTION, LEVEL_ALERT]),   # 判定対象外は含めない
    }, index=table.index)[DETECTOR_ORDER]


def merge_events(times: Iterable[pd.Timestamp], gap_hours: float) -> list[pd.Timestamp]:
    """検知時刻の列を、間隔 gap 以内ならまとめて、各イベントの最初の時刻のリストにする。"""
    starts, prev = [], None
    for t in sorted(times):
        if prev is None or (t - prev) > pd.Timedelta(hours=gap_hours):
            starts.append(t)
        prev = t
    return starts


# ---------------------------------------------------------------------------
# 正解ラベル
# ---------------------------------------------------------------------------
def load_labels(path) -> pd.DataFrame:
    """正解ラベルCSVを読み込み、形式を確認する（問題があれば ValueError）。"""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    return validate_labels(df)


def validate_labels(df: pd.DataFrame, require_split: bool = False) -> pd.DataFrame:
    missing = [c for c in LABEL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("正解ラベルに必要な列がありません: " + ", ".join(missing))
    df = df.copy()
    errors = []
    if df["video_id"].duplicated().any():
        errors.append("video_id が重複しています: " + ", ".join(df.loc[df["video_id"].duplicated(), "video_id"]))
    bad = ~df["is_fire"].isin(["0", "1"])
    if bad.any():
        errors.append("is_fire は 0 か 1 にしてください: " + ", ".join(df.loc[bad, "video_id"]))
    starts = []
    for _, row in df.iterrows():
        text = str(row["fire_start_jst"]).strip()
        if not text:
            starts.append(pd.NaT)
            continue
        try:
            t = pd.Timestamp(text)
            t = t.tz_localize("Asia/Tokyo") if t.tzinfo is None else t.tz_convert("Asia/Tokyo")
            starts.append(t)
        except (ValueError, TypeError):
            errors.append(f"fire_start_jst を日時として読めません: {row['video_id']}（{text}）")
            starts.append(pd.NaT)
        if row["is_fire"] == "0" and text:
            errors.append(f"非炎上動画に炎上開始時刻があります: {row['video_id']}")
    df["fire_start"] = pd.Series(starts, index=df.index, dtype="datetime64[ns, Asia/Tokyo]")
    df["is_fire"] = pd.to_numeric(df["is_fire"], errors="coerce").fillna(-1).astype(int)

    splits = df["split"].str.strip()
    if require_split and not splits.isin(SPLITS).all():
        errors.append("split が dev / test になっていない動画があります: "
                      + ", ".join(df.loc[~splits.isin(SPLITS), "video_id"]))
    both = df[splits.isin(SPLITS)].groupby("incident_id")["split"].nunique()
    if (both > 1).any():
        errors.append("同じ事件（incident_id）が dev と test の両方にあります: "
                      + ", ".join(both[both > 1].index))
    if errors:
        raise ValueError("正解ラベルの問題:\n- " + "\n- ".join(errors))
    return df


def assign_splits(labels: pd.DataFrame, dev_fraction: float = 0.4, seed: int = 0,
                  overwrite: bool = False) -> pd.DataFrame:
    """
    事件（incident_id）単位で dev（調整用）と test（評価用）に分ける。
    炎上を含む事件と含まない事件それぞれで、dev_fraction の割合（切り上げ）を dev にする。
    すでに split が入っている場合は、overwrite=True でない限り変更しない。
    """
    df = labels.copy()
    if (df["split"].astype(str).str.strip() != "").any() and not overwrite:
        raise ValueError("split はすでに設定されています。分け直すと評価の手順が崩れるため、"
                         "やり直す場合は理由を記録したうえで overwrite=True にしてください。")
    rng = np.random.default_rng(seed)
    incidents = df.groupby("incident_id")["is_fire"].apply(
        lambda s: int((pd.to_numeric(s, errors="coerce") == 1).any()))
    split_of = {}
    for stratum in (1, 0):
        ids = sorted(incidents[incidents == stratum].index)
        rng.shuffle(ids)
        n_dev = math.ceil(len(ids) * dev_fraction) if ids else 0
        for i, inc in enumerate(ids):
            split_of[inc] = "dev" if i < n_dev else "test"
    df["split"] = df["incident_id"].map(split_of)
    return df


# ---------------------------------------------------------------------------
# 1本の動画の評価
# ---------------------------------------------------------------------------
def _bin_hours(table: pd.DataFrame) -> float:
    return pd.Timedelta(pd.tseries.frequencies.to_offset(_infer_freq(table))) / pd.Timedelta(hours=1)


def evaluate_video(table: pd.DataFrame, label: pd.Series, cfg: EvalConfig | None = None
                   ) -> pd.DataFrame:
    """1本の動画について、検知器ごとの判定（1行＝1検知器）を返す。"""
    cfg = cfg or EvalConfig()
    flags = detector_flags(table)
    bin_h = _bin_hours(table)
    is_fire = int(label["is_fire"]) == 1
    start = label.get("fire_start")
    start = None if start is None or pd.isna(start) else start
    judged = (table["judged"].astype(bool) if "judged" in table.columns
              else pd.Series(True, index=table.index))
    if is_fire and start is not None:
        end_ = start + pd.Timedelta(hours=cfg.window_after_hours)
        n_judged_eval = int(judged[(table.index >= start) & (table.index <= end_)].sum())
    else:
        n_judged_eval = int(judged.sum())
    rows = []
    for det in DETECTOR_ORDER:
        times = list(flags.index[flags[det].to_numpy()])
        row = {"video_id": label["video_id"], "incident_id": label["incident_id"],
               "split": label.get("split", ""), "is_fire": int(is_fire), "detector": det,
               "n_alarm_bins": len(times), "n_judged_bins": int(judged.sum()),
               "n_judged_bins_in_eval": n_judged_eval}
        if n_judged_eval == 0:   # 判定対象外：指標には入れない
            row.update({"outcome": "対象外", "first_detection": None, "delay_hours": np.nan,
                        "false_alarm_events": np.nan, "monitored_hours": 0.0,
                        "first_alarm_any": min(times) if times else None})
        elif is_fire and start is not None:
            end = start + pd.Timedelta(hours=cfg.window_after_hours)
            hits = [t for t in times if start <= t <= end]
            pre = [t for t in times if t < start]
            first = min(hits) if hits else None
            row.update({
                "outcome": "TP" if hits else "FN",
                "first_detection": first,
                "delay_hours": (first - start) / pd.Timedelta(hours=1) if first is not None else np.nan,
                "false_alarm_events": len(merge_events(pre, cfg.merge_gap_hours)),
                "monitored_hours": float((judged & (table.index < start)).sum()) * bin_h,
                "first_alarm_any": min(times) if times else None,
            })
        elif is_fire:   # 開始時刻不明：検知の有無だけ
            row.update({"outcome": "TP" if times else "FN", "first_detection": min(times) if times else None,
                        "delay_hours": np.nan, "false_alarm_events": np.nan, "monitored_hours": 0.0,
                        "first_alarm_any": min(times) if times else None, "start_unknown": True})
        else:
            events = merge_events(times, cfg.merge_gap_hours)
            row.update({"outcome": "FP" if times else "TN", "first_detection": None,
                        "delay_hours": np.nan, "false_alarm_events": len(events),
                        "monitored_hours": float(judged.sum()) * bin_h,
                        "first_alarm_any": min(times) if times else None})
        rows.append(row)
    out = pd.DataFrame(rows)
    # 参考：その時間帯の集計が確定する時刻（時間帯の終わり）で数えた遅れ（＝遅れ＋1ビン）
    out["delay_hours_bin_end"] = out["delay_hours"] + bin_h
    if "start_unknown" not in out.columns:
        out["start_unknown"] = False
    out["start_unknown"] = out["start_unknown"].fillna(False).astype(bool)
    return out


# ---------------------------------------------------------------------------
# 集計
# ---------------------------------------------------------------------------
def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """比率 k/n の Wilson 95%信頼区間。n=0 なら (NaN, NaN)。"""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else np.nan
    r = tp / (tp + fn) if tp + fn else np.nan
    f = 2 * p * r / (p + r) if (tp and not np.isnan(p) and not np.isnan(r)) else (0.0 if tp + fp + fn else np.nan)
    return p, r, f


def bootstrap_f1(per_video: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """動画単位で復元抽出したときの F1 の95%区間。"""
    if per_video.empty:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    outcomes = per_video["outcome"].to_numpy()
    vals = []
    for _ in range(n_boot):
        s = outcomes[rng.integers(0, len(outcomes), len(outcomes))]
        _, _, f = _prf(int((s == "TP").sum()), int((s == "FP").sum()), int((s == "FN").sum()))
        vals.append(f)
    vals = np.array(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if len(vals) else (np.nan, np.nan)


def summarize(per_video: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """検知器ごとの評価指標の表。"""
    rows = []
    for det in DETECTOR_ORDER:
        d = per_video[per_video["detector"] == det]
        if d.empty:
            continue
        evaluable = d[d["outcome"] != "対象外"]
        fire, non = evaluable[evaluable["is_fire"] == 1], evaluable[evaluable["is_fire"] == 0]
        tp, fn = int((fire["outcome"] == "TP").sum()), int((fire["outcome"] == "FN").sum())
        fp, tn = int((non["outcome"] == "FP").sum()), int((non["outcome"] == "TN").sum())
        p, r, f = _prf(tp, fp, fn)
        r_lo, r_hi = wilson_interval(tp, tp + fn)
        fa_lo, fa_hi = wilson_interval(fp, fp + tn)
        f_lo, f_hi = bootstrap_f1(evaluable, n_boot, seed)
        known = fire[~fire["start_unknown"]]
        delays = known["delay_hours"].dropna()
        hours = d["monitored_hours"].sum()
        events = d["false_alarm_events"].fillna(0).sum()
        rows.append({
            "detector": det, "説明": DETECTOR_LABELS[det],
            "炎上動画数": len(fire), "検知(TP)": tp, "見逃し(FN)": fn,
            "非炎上動画数": len(non), "誤警戒動画(FP)": fp, "沈黙(TN)": tn,
            "Precision": p, "Recall": r, "Recall_95CI": _ci(r_lo, r_hi),
            "F1": f, "F1_95CI(bootstrap)": _ci(f_lo, f_hi),
            "誤警戒率_動画": fp / (fp + tn) if fp + tn else np.nan, "誤警戒率_動画_95CI": _ci(fa_lo, fa_hi),
            "誤警戒イベント数": int(events), "監視時間(h)": hours,
            "誤警戒_100時間あたり": events / hours * 100 if hours else np.nan,
            "遅れ_中央値(h)": float(delays.median()) if len(delays) else np.nan,
            "遅れ_25%(h)": float(delays.quantile(0.25)) if len(delays) else np.nan,
            "遅れ_75%(h)": float(delays.quantile(0.75)) if len(delays) else np.nan,
            "遅れを測れた動画数": len(delays),
            "開始時刻不明の炎上動画数": int(fire["start_unknown"].sum()),
            "判定対象外で除外した動画数": int((d["outcome"] == "対象外").sum()),
        })
    return pd.DataFrame(rows)


def _ci(lo, hi) -> str:
    return "—" if (lo is None or np.isnan(lo)) else f"[{lo:.2f}, {hi:.2f}]"


def mcnemar_tests(per_video: pd.DataFrame,
                  pairs: Iterable[tuple[str, str]] = (("B", "B∧A"), ("B", "B∨A"), ("A", "B"), ("A", "C"))
                  ) -> pd.DataFrame:
    """
    同じ動画での正誤（炎上→TP、非炎上→TN を「正しい」とする）を2つの検知器で比べる
    McNemar 正確検定（二項検定）。実験4などの対応のある比較に使う。
    """
    per_video = per_video[per_video["outcome"] != "対象外"]   # 判定していない動画は比べない
    correct = per_video.assign(ok=per_video["outcome"].isin(["TP", "TN"])).pivot(
        index="video_id", columns="detector", values="ok")
    rows = []
    for x, y in pairs:
        if x not in correct or y not in correct:
            continue
        only_x = int((correct[x] & ~correct[y]).sum())
        only_y = int((~correct[x] & correct[y]).sum())
        n = only_x + only_y
        p = binomtest(only_x, n, 0.5).pvalue if n else np.nan
        rows.append({"比較": f"{x} vs {y}", "前者": x, "後者": y,
                     "前者だけ正しい動画数": only_x, "後者だけ正しい動画数": only_y,
                     "不一致の動画数": n, "p値(McNemar正確検定)": p})
    return pd.DataFrame(rows)


def evaluate_offline_reference(offline_cps: pd.DataFrame, label: pd.Series) -> dict:
    """
    事後分析（全期間PELT）の参考値：炎上開始に最も近い変化点との時間差。
    未来のデータを使っているため、検知性能の指標には入れない。
    """
    start = label.get("fire_start")
    if int(label["is_fire"]) != 1 or start is None or pd.isna(start) or offline_cps.empty:
        return {"video_id": label["video_id"], "offline_n_cps": len(offline_cps),
                "nearest_cp_to_start_hours": np.nan}
    diffs = [(t - start) / pd.Timedelta(hours=1) for t in offline_cps["change_point_at"]]
    nearest = min(diffs, key=abs)
    return {"video_id": label["video_id"], "offline_n_cps": len(offline_cps),
            "nearest_cp_to_start_hours": nearest}
