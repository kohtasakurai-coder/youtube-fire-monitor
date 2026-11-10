"""
本評価の動画を、事前に決めた規則で選び、コメントを取得し、判定できる時間帯を確認して、
正解ラベルの下書き（data/research/labels.csv）を作る。お手元のPC（APIキーがある環境）で実行する。

選び方（動画の情報だけで決める。検知の結果は見ない）
  基準時刻 ref：incidents_main.csv の ref_time_jst（掲載時刻つき報道・公式発表）、
               または ref_upload_title を含む動画（ref_upload_date に公開）の公開時刻
  炎上動画   ：同じチャンネルで ref の24時間以上前に公開された、コメントが有効で、長さ3分超（ショートを除く）の動画を、
               新しい順に試す（騒動の時点で最新だった動画にコメントが集まりやすいため）
  比較動画   ：同じチャンネルで ref の30日以上前に評価期間が終わる動画を、新しい順に試す（平常期）。
               calm_exclude_dates（同じチャンネルの別の騒動の報道日）の7日前〜14日後にかかる期間は除く
  評価期間   ：どちらも120時間。炎上は [ref−72時間, ref+48時間)、比較は [公開+24時間, 公開+144時間)
  採用条件   ：評価期間に判定できる時間帯があること（炎上は ref〜ref+48時間 に1つ以上、比較は期間内に1つ以上）。
               満たさなければ次に新しい動画を試す（各3本まで）。判定できたかどうか以外（警戒の有無など）は見ない
  ラベル     ：炎上動画は is_fire=1、fire_start_jst=ref。比較動画は is_fire=0。
               比較動画の期間に騒動の報道がないかは、ユーチュラのチャンネル別ニュースなどで別途確認して label_source に追記する

使い方（プロジェクトのルートで）:
    $env:YOUTUBE_API_KEY = "（手元のキー）"
    .venv\\Scripts\\python.exe -m experiments.prepare_main_eval
    → data/research/labels.csv（下書き）と results/prepare_main_eval.csv（選定の記録）
    → そのあと experiments\\run_main_evaluation.ps1
"""
from __future__ import annotations

import argparse
import getpass
import re
import sys
from pathlib import Path

import pandas as pd

from src.config import AnalysisConfig
from src.experiment import PROJECT_ROOT
from src.list_uploads import CHANNELS_URL, _get_json, list_uploads, resolve_channel_id
from src.youtube_api import YouTubeAPIError, fetch_comments, get_api_key
from src.preprocessing import save_comments_csv, load_comments_csv

TZ = "Asia/Tokyo"
H = pd.Timedelta(hours=1)
MIN_DURATION_SEC = 180          # これ以下はショート動画として除く
MAX_TRIES = 3
FIRE_BEFORE, FIRE_AFTER = 72, 48
CALM_GAP_DAYS = 30


def resolve_ref(row, uploads: pd.DataFrame) -> tuple[pd.Timestamp | None, str]:
    if str(row.get("ref_time_jst", "")).strip():
        return pd.Timestamp(row["ref_time_jst"]).tz_convert(TZ), ""
    title, day = str(row.get("ref_upload_title", "")).strip(), str(row.get("ref_upload_date", "")).strip()
    if day:
        d0 = pd.Timestamp(day).tz_localize(TZ)
        hit = uploads[(uploads["published_at_jst"] >= d0) & (uploads["published_at_jst"] < d0 + pd.Timedelta(days=1))]
        # タイトルの一部が分かればそれで、分からなければその日最初の通常動画（ショートを除く）をきっかけ動画とする
        hit = (hit[hit["title"].str.contains(title, regex=False)] if title
               else hit[hit["duration_sec"].fillna(0) > MIN_DURATION_SEC])
        if len(hit):
            h = hit.sort_values("published_at_jst").iloc[0]
            return h["published_at_jst"], f"（きっかけ動画 {h['video_id']}「{h['title']}」）"
    return None, "基準時刻が決まらない（ref_time_jst か、きっかけ動画のタイトルと公開日を入れてください）"


def existing_csv(video_id: str, folder: Path) -> Path | None:
    files = sorted(folder.glob(f"comments_{video_id}_*.csv"))
    return files[-1] if files else None


def judged_counts(csv_path: Path, analyzer, cfg: AnalysisConfig, freq: str, windows: dict) -> dict:
    """期間ごとの「判定できた時間帯の数」だけを返す（警戒の有無などの結果は返さない）。"""
    from src.detection import run_detection
    from src.time_series import aggregate_time_series

    comments, _ = load_comments_csv(csv_path)
    ts = aggregate_time_series(analyzer.analyze(comments), freq=freq, ma_window=cfg.moving_average_window,
                               min_comments=cfg.min_comments_reliable)
    t = run_detection(ts, cfg, include_offline=False).table
    out = {"n_comments": len(comments)}
    for k, (a, b) in windows.items():
        sel = (t.index >= a) & (t.index < b)
        out[k] = int(t.loc[sel, "judged"].astype(bool).sum())
    return out


def main(argv=None) -> int:
    from experiments.compare_methods import make_analyzer

    p = argparse.ArgumentParser(description="本評価の動画を規則で選び、取得・観測確認し、ラベルの下書きを作る")
    p.add_argument("--incidents", default=str(PROJECT_ROOT / "experiments" / "incidents_main.csv"))
    p.add_argument("--labels-out", default=str(PROJECT_ROOT / "data" / "research" / "labels.csv"))
    p.add_argument("--freq", default="3h")
    p.add_argument("--max-comments", type=int, default=30000)
    p.add_argument("--dictionary", default="auto")
    p.add_argument("--overwrite", action="store_true",
                   help="既存の labels.csv を上書きする（ラベルを固定済み＝protocol に記録があるときは上書きしない）")
    args = p.parse_args(argv)

    api_key = get_api_key() or getpass.getpass("YouTube APIキーを入力（表示されません）: ").strip()
    if not re.fullmatch(r"AIza[0-9A-Za-z_-]{35}", api_key or ""):
        print("APIキーの形式ではありません（AIza で始まる39文字）。Google Cloud でキーをコピーし直してください。",
              file=sys.stderr)
        return 1
    analyzer, info = make_analyzer(args.dictionary)
    cfg = AnalysisConfig()
    folder = PROJECT_ROOT / "data" / "research"
    incidents = pd.read_csv(args.incidents, encoding="utf-8-sig", dtype=str).fillna("")
    log, labels = [], []

    for _, inc in incidents.iterrows():
        iid = inc["incident_id"]
        print(f"\n=== {iid}：{inc['騒動']} ===")
        try:
            ch = resolve_channel_id(inc["channel"], api_key)
            ref_hint = inc["ref_time_jst"] or inc["ref_upload_date"]
            ref0 = pd.Timestamp(ref_hint).tz_convert(TZ) if "T" in ref_hint else pd.Timestamp(ref_hint).tz_localize(TZ)
            ups = list_uploads(ch, api_key, (ref0 - pd.Timedelta(days=150)).isoformat(),
                               (ref0 + pd.Timedelta(days=2)).isoformat())
        except YouTubeAPIError as e:
            print(f"  チャンネル・動画一覧の取得に失敗: {e}")
            log.append({"incident_id": iid, "role": "-", "status": f"失敗: {e}"})
            continue
        ref, note = resolve_ref(inc, ups)
        if ref is None:
            print("  " + note)
            log.append({"incident_id": iid, "role": "-", "channel_id": ch, "status": note})
            continue
        meta = _get_json(CHANNELS_URL, {"key": api_key, "part": "snippet,statistics", "id": ch}, api_key, 30)
        ch_title = meta["items"][0]["snippet"]["title"] if meta.get("items") else ""
        print(f"  チャンネル「{ch_title}」（{ch}）　基準時刻 {ref:%Y-%m-%d %H:%M} {note}")
        usable = ups[ups["comment_count"].notna() & (ups["comment_count"] > 0)
                     & (ups["duration_sec"].fillna(0) > MIN_DURATION_SEC)]
        calm = usable[usable["published_at_jst"] + 144 * H <= ref - pd.Timedelta(days=CALM_GAP_DAYS)]
        # 同じチャンネルの別の騒動（報道日）の 7日前〜14日後 にかかる評価期間は平常期とみなさない
        for d in [x for x in str(inc.get("calm_exclude_dates", "")).split(";") if x.strip()]:
            d = pd.Timestamp(d.strip()).tz_localize(TZ)
            lo, hi = d - pd.Timedelta(days=7), d + pd.Timedelta(days=14)
            calm = calm[(calm["published_at_jst"] + 144 * H <= lo) | (calm["published_at_jst"] + 24 * H >= hi)]
        roles = {"炎上": usable[usable["published_at_jst"] <= ref - 24 * H], "比較": calm}
        for role, cands in roles.items():
            adopted = False
            for _, v in cands.sort_values("published_at_jst", ascending=False).head(MAX_TRIES).iterrows():
                vid, pub = v["video_id"], v["published_at_jst"]
                if role == "炎上":
                    win = (ref - FIRE_BEFORE * H, ref + FIRE_AFTER * H)
                    need = {"評価期間": win, "開始後48時間": (ref, ref + FIRE_AFTER * H)}
                else:
                    win = (pub + 24 * H, pub + 144 * H)
                    need = {"評価期間": win}
                path = existing_csv(vid, folder)
                if path is None:
                    print(f"  {role}候補 {vid}（{pub:%Y-%m-%d %H:%M}、表示コメント数 {v['comment_count']}）を取得中…")
                    try:
                        # 新しい順に取得するので、表示件数より多めに上限をとる（評価期間の古いコメントを落とさない）
                        limit = int(max(args.max_comments, float(v["comment_count"]) * 1.2 + 100))
                        df, warns = fetch_comments(vid, api_key, max_results=limit, include_replies=True)
                    except YouTubeAPIError as e:
                        log.append({"incident_id": iid, "role": role, "video_id": vid, "status": f"取得失敗: {e}"})
                        continue
                    path = folder / f"comments_{vid}_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv"
                    save_comments_csv(df, path)
                    for w in warns:
                        print(f"    注意: {w}")
                counts = judged_counts(path, analyzer, cfg, args.freq, need)
                ok = all(counts[k] > 0 for k in need)
                rec = {"incident_id": iid, "role": role, "video_id": vid, "title": v["title"], "channel_title": ch_title,
                       "published_at_jst": pub, "channel_id": ch, "ref_jst": ref,
                       "eval_start_jst": win[0], "eval_end_jst": win[1], "comments_file": str(path.relative_to(PROJECT_ROOT)),
                       "display_comment_count": v["comment_count"], **counts,
                       "status": "採用" if ok else "不採用（判定できる時間帯なし）"}
                log.append(rec)
                print(f"    {vid}: 取得 {counts['n_comments']}件、判定できた時間帯 "
                      + "・".join(f"{k} {counts[k]}" for k in need) + f" → {rec['status']}")
                if ok:
                    labels.append({
                        "video_id": vid, "incident_id": iid if role == "炎上" else f"{iid}_CALM",
                        "is_fire": "1" if role == "炎上" else "0",
                        "fire_start_jst": ref.isoformat() if role == "炎上" else "",
                        "label_source": (inc["label_source"] + note) if role == "炎上" else
                        f"同じチャンネルの平常期（基準時刻の{CALM_GAP_DAYS}日以上前に評価期間が終わる動画。"
                        f"除外した別の騒動の報道日：{inc.get('calm_exclude_dates', '') or 'なし'}）。期間内の騒動報道の有無：未確認",
                        "split": "", "comments_file": rec["comments_file"],
                        "eval_start_jst": win[0].isoformat(), "eval_end_jst": win[1].isoformat(),
                    })
                    adopted = True
                    break
            if not adopted:
                print(f"  {role}：採用できる動画なし")

    out_log = PROJECT_ROOT / "results" / "prepare_main_eval.csv"
    out_log.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log).to_csv(out_log, index=False, encoding="utf-8-sig")
    lab = pd.DataFrame(labels)
    labels_out = Path(args.labels_out)
    if len(lab):
        frozen = (PROJECT_ROOT / "experiments" / "protocol" / "labels_sha256.txt").exists()
        if labels_out.exists() and (not args.overwrite or frozen):   # 既存（固定済みかもしれない）は上書きしない
            labels_out = labels_out.with_name(labels_out.stem + "_draft.csv")
            print(f"既存の {args.labels_out} は上書きせず、{labels_out} に書きます。")
        labels_out.parent.mkdir(parents=True, exist_ok=True)
        lab.to_csv(labels_out, index=False, encoding="utf-8-sig")
    n_fire = int((lab["is_fire"] == "1").sum()) if len(lab) else 0
    print(f"\n辞書：{info['dictionary']}　集計幅：{args.freq}")
    print(f"採用：炎上 {n_fire} 本・比較 {len(lab) - n_fire} 本"
          + (f" → {labels_out}" if len(lab) else "（labels.csv は作っていません）"))
    print(f"選定の記録 → {out_log}")
    print("比較動画の期間に騒動の報道がないかを確認してから、experiments\\run_main_evaluation.ps1 を実行してください。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
