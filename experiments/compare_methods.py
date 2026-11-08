"""
評価実験のコマンド（STEP 6）。プロジェクトのルートで実行する。

  # 0. 架空データで手順全体を試す（研究結果ではない。results/demo_synthetic/ に出力）
  python -m experiments.compare_methods demo

  # 1. 事件単位で dev（調整用）/ test（評価用）に分ける（split 列に書き込む）
  python -m experiments.compare_methods split --labels data/research/labels.csv
  # 2. 正解ラベルを固定（SHA-256 を experiments/protocol/ に記録）
  python -m experiments.compare_methods freeze-labels --labels data/research/labels.csv
  # 3. dev だけでパラメータを選び、固定する
  python -m experiments.compare_methods tune --labels data/research/labels.csv
  # 4. 固定した設定で test を1回だけ評価する
  python -m experiments.compare_methods test --labels data/research/labels.csv

  # 感情分析の精度（人手の正解との比較）
  python -m experiments.compare_methods sentiment --gold data/research/sentiment_gold.csv

  # 予備評価：基準時刻（会見開始など）の前後比較。正解ラベル（炎上開始）とは別に扱う
  python -m experiments.compare_methods event --csv data/research/comments_TeRXuoLAX9o_….csv \
      --event 2025-01-27T16:00:00+09:00 --before 24 --after 24

  # 感情分析の人手評価用の候補（辞書の予測を含まない）。済んだラベルは --exclude で除く
  python -m experiments.compare_methods gold-sample --csv data/research/a.csv data/research/b.csv \
      --n 180 --per-video-cap 90 --exclude data/research/sentiment_gold_done20.csv

  # 論文用の表と図（results/report/ に出力。demo の結果なら --from results/demo_synthetic）
  python -m experiments.compare_methods report

辞書は --dictionary auto（既定：非公開フォルダのNRC辞書があればそれ、なければ簡易辞書）/
simple / ファイルのパス で選ぶ。結果のCSVは results/（Git管理外）に保存する。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

from src.evaluation import EvalConfig, assign_splits, load_labels, summarize
from src.experiment import (
    DEFAULT_GRID,
    PROJECT_ROOT,
    SELECTION_DETECTOR,
    Protocol,
    ProtocolError,
    comparisons,
    config_payload,
    evaluate_config,
    frozen_to_config,
    grid_configs,
    load_comments_for_labels,
    prepare,
    run_grid,
    select_config,
    sha256_file,
)



def make_analyzer(choice: str):
    """辞書を選んで感情分析器を作る。戻り値: (分析器, 辞書の情報)"""
    from src.sentiment_analysis import (
        SIMPLE_DICTIONARY_PATH,
        SentimentAnalyzer,
        find_private_dictionaries,
        load_nrc_dictionary,
        load_simple_dictionary,
    )
    if choice == "auto":
        private = find_private_dictionaries()
        path, kind = (private[0], "nrc") if private else (SIMPLE_DICTIONARY_PATH, "simple")
    elif choice == "simple":
        path, kind = SIMPLE_DICTIONARY_PATH, "simple"
    else:
        path, kind = Path(choice), "nrc"
    d = load_nrc_dictionary(path) if kind == "nrc" else load_simple_dictionary(path)
    try:   # 記録にはプロジェクトからの相対パスを書く（ユーザー名などを残さない）
        rel = Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        rel = Path(path).name
    info = {**d.stats(), "kind": kind, "file": Path(path).name, "path": rel,
            "sha256": sha256_file(path)}
    return SentimentAnalyzer(d), info


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  → {path}")


def show(summary: pd.DataFrame) -> None:
    cols = ["detector", "検知(TP)", "見逃し(FN)", "誤警戒動画(FP)", "沈黙(TN)", "Precision", "Recall",
            "F1", "誤警戒_100時間あたり", "遅れ_中央値(h)"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary[cols].round(3).to_string(index=False))


# ---------------------------------------------------------------------------
def cmd_split(args) -> None:
    labels = pd.read_csv(args.labels, encoding="utf-8-sig", dtype=str).fillna("")
    out = assign_splits(labels, dev_fraction=args.dev_fraction, seed=args.seed, overwrite=args.overwrite)
    out.to_csv(args.labels, index=False, encoding="utf-8-sig")
    print(out.groupby(["split", "is_fire"]).size().rename("動画数").to_string())
    Protocol(Path(args.protocol_dir)).log("split", "", f"dev_fraction={args.dev_fraction} seed={args.seed}"
                                         + (" overwrite" if args.overwrite else ""))


def cmd_freeze(args) -> None:
    load_labels(args.labels)                      # 形式の確認
    h = Protocol(Path(args.protocol_dir)).freeze_labels(args.labels)
    print(f"正解ラベルを固定しました（SHA-256: {h}）")


def _grid_from_args(args) -> dict:
    grid = dict(DEFAULT_GRID)
    if args.freq:
        grid["freq"] = args.freq.split(",")
    return grid


def cmd_tune(args) -> None:
    proto = Protocol(Path(args.protocol_dir))
    h = proto.check_labels(args.labels)
    if proto.test_done_file.exists() and not args.after_test_reason:
        raise ProtocolError("test の評価後に設定を選び直すことはできません"
                            "（どうしても必要なら理由を --after-test-reason に書いて記録します）。")
    labels = load_labels(args.labels)
    dev = labels[labels["split"] == "dev"]
    if dev.empty:
        raise ProtocolError("dev の動画がありません。先に split を実行してください。")
    print(f"dev の動画 {len(dev)} 本だけで調整します（test の動画は読み込みません）。")
    analyzer, dict_info = make_analyzer(args.dictionary)
    data = prepare(dev, load_comments_for_labels(dev), analyzer)
    grid = _grid_from_args(args)
    summary = run_grid(data, grid, EvalConfig())
    save(summary, Path(args.results_dir) / "dev_grid.csv")
    cid = select_config(summary)
    params, cfg = grid_configs(grid)[cid]
    row = summary[(summary["config_id"] == cid) & (summary["detector"] == SELECTION_DETECTOR)].iloc[0]
    proto.save_frozen_config(config_payload(cid, params, cfg, EvalConfig(), dict_info,
                                            row.to_dict(), grid), h, args.after_test_reason)
    print(f"選ばれた設定 config_id={cid}: {params}")
    show(summary[summary["config_id"] == cid])


def cmd_test(args) -> None:
    proto = Protocol(Path(args.protocol_dir))
    h = proto.check_labels(args.labels)
    frozen = proto.start_test(h, args.rerun_reason)
    labels = load_labels(args.labels)
    test = labels[labels["split"] == "test"]
    if test.empty:
        raise ProtocolError("test の動画がありません。")
    fd = frozen["dictionary"]
    analyzer, dict_info = make_analyzer(
        ("simple" if fd["kind"] == "simple" else str(PROJECT_ROOT / fd["path"]))
        if args.dictionary == "frozen"
        else args.dictionary)
    if dict_info["sha256"] != frozen["dictionary"]["sha256"]:
        raise ProtocolError("調整したときと違う感情辞書です。")
    cfg, freq, eval_cfg = frozen_to_config(frozen)
    data = prepare(test, load_comments_for_labels(test), analyzer)
    per_video, offline, seconds = evaluate_config(data, cfg, freq, eval_cfg)
    summary = summarize(per_video)
    out = Path(args.results_dir) / "test"
    save(summary, out / "test_summary.csv")
    save(per_video, out / "test_per_video.csv")
    save(comparisons(per_video), out / "test_mcnemar.csv")
    save(offline, out / "test_offline_reference（事後分析・評価に使わない）.csv")
    proto.finish_test(h, frozen, args.rerun_reason)
    print(f"test {len(test)} 本の評価（config_id={frozen['config_id']}）")
    show(summary)


def cmd_demo(args) -> None:
    """架空データで split → freeze → tune → test を一通り実行する（実際の protocol は使わない）。"""
    from src.preprocessing import save_comments_csv
    from src.synthetic_data import generate_evaluation_dataset

    out = Path(args.results_dir) / "demo_synthetic"
    if out.exists():
        shutil.rmtree(out)
    comments, labels = generate_evaluation_dataset(seed=args.seed)
    (out / "comments").mkdir(parents=True)
    for v, g in comments.groupby("video_id"):
        save_comments_csv(g, out / "comments" / f"{v}.csv")
    labels["comments_file"] = [str((out / "comments" / f"{v}.csv").resolve()) for v in labels["video_id"]]
    labels = assign_splits(labels, dev_fraction=args.dev_fraction, seed=args.seed)
    labels_path = out / "labels_synthetic.csv"
    labels.to_csv(labels_path, index=False, encoding="utf-8-sig")
    print("【架空データ】評価手順の動作確認です。実動画での性能ではありません。")
    print(labels.groupby(["split", "is_fire"]).size().rename("動画数").to_string())

    # demo 専用の protocol・results フォルダを使う（実データの手順の記録には触れない）
    ns = argparse.Namespace(labels=labels_path, protocol_dir=out / "protocol", results_dir=out,
                            dictionary="simple", freq=args.freq, after_test_reason="", rerun_reason="")
    cmd_freeze(ns)
    cmd_tune(ns)
    cmd_test(argparse.Namespace(**{**vars(ns), "dictionary": "frozen"}))
    print(f"【架空データ】の結果は {out} に保存しました。")


def cmd_sentiment(args) -> None:
    """人手の正解（gold_negative: 1/0）と感情分析の否定判定を比べる。"""
    from src.sentiment_analysis import evaluate_negative_detection

    gold = pd.read_csv(args.gold, encoding="utf-8-sig", dtype={"comment_id": str})
    for col in ("text", "gold_negative"):
        if col not in gold.columns:
            raise ValueError(f"{args.gold} に {col} 列が必要です。")
    analyzer, info = make_analyzer(args.dictionary)
    pred = analyzer.analyze(gold)
    m = evaluate_negative_detection(pred["is_negative"], gold["gold_negative"])
    m["n_hold"] = int(pd.to_numeric(gold["gold_negative"], errors="coerce").isna().sum())   # 保留（空欄）
    print(f"辞書：{info['dictionary']}（{info['n_words']}語）")
    for k, v in m.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    save(pd.DataFrame([{**m, "dictionary": info["dictionary"], "n_words": info["n_words"]}]),
         Path(args.results_dir) / "sentiment_eval.csv")


def cmd_event(args) -> None:
    """予備評価：基準時刻の前後で、取得範囲・判定できた時間帯・検知を比べる。"""
    from src.config import AnalysisConfig
    from src.detection import run_detection
    from src.event_study import event_window
    from src.preprocessing import load_comments_csv
    from src.time_series import aggregate_time_series

    comments, _ = load_comments_csv(args.csv)
    if args.video:
        comments = comments[comments["video_id"] == args.video]
    if comments.empty:
        raise ValueError("コメントがありません。")
    video = args.video or (comments["video_id"].iloc[0] if "video_id" in comments.columns else "video")
    analyzer, info = make_analyzer(args.dictionary)
    cfg = AnalysisConfig()
    analyzed = analyzer.analyze(comments)
    ts = aggregate_time_series(analyzed, freq=args.freq, ma_window=cfg.moving_average_window,
                               min_comments=cfg.min_comments_reliable)
    res = run_detection(ts, cfg, include_offline=False)
    summary, by_detector, window = event_window(res.table, analyzed, pd.Timestamp(args.event),
                                                args.before, args.after)
    out = Path(args.results_dir) / "event"
    stem = f"{video}_{pd.Timestamp(summary['event_time']):%Y%m%d_%H%M}"
    print("【予備評価】基準時刻の前後比較です。炎上開始の正解ラベルを使う本評価とは別に扱います。")
    print(f"辞書：{info['dictionary']}（{info['n_words']}語）　動画：{video}")
    print(f"観測期間：{summary['window_start']} 〜 {summary['window_end']}（基準 {summary['event_time']}）")
    print(f"取得範囲：{summary['first_comment']} 〜 {summary['last_comment']}　"
          f"開始を覆う：{'はい' if summary['covers_start'] else 'いいえ'}　終了を覆う：{'はい' if summary['covers_end'] else 'いいえ'}")
    print(f"期間内コメント：前 {summary['n_comments_pre']}件 / 後 {summary['n_comments_post']}件　"
          f"判定できた時間帯：前 {summary['n_judged_bins_pre']}/{summary['n_bins_pre']} ・ 後 {summary['n_judged_bins_post']}/{summary['n_bins_post']}")
    if not summary["covers_start"]:
        print("注意：観測期間の開始より前のコメントがありません。動画の公開が観測開始より後か、"
              "取得上限で古いコメントが欠けた可能性があります（取得時に表示される公開日時・取得率で確認）。")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(by_detector.to_string(index=False))
    save(pd.DataFrame([summary]), out / f"{stem}_summary.csv")
    save(by_detector, out / f"{stem}_detectors.csv")
    w = window.copy()
    w.index = [t.isoformat() for t in w.index]
    w.index.name = "bin_start"
    out.mkdir(parents=True, exist_ok=True)
    w.to_csv(out / f"{stem}_bins.csv", encoding="utf-8-sig")
    print(f"  → {out / (stem + '_bins.csv')}")


def cmd_gold_sample(args) -> None:
    from src.gold_sampling import sample_gold_candidates
    from src.preprocessing import load_comments_csv

    frames = [load_comments_csv(p)[0] for p in args.csv]
    comments = pd.concat(frames, ignore_index=True)
    exclude = set()
    for p in args.exclude:
        exclude |= set(pd.read_csv(p, encoding="utf-8-sig", dtype=str)["comment_id"].dropna())
    out, breakdown = sample_gold_candidates(comments, n=args.n, per_video_cap=args.per_video_cap,
                                            per_thread_cap=args.per_thread_cap, exclude_ids=exclude,
                                            seed=args.seed)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"候補 {len(out)} 件 / 目標 {args.n} 件（除外した既存ラベル {len(exclude)} 件）→ {path}")
    print(breakdown.to_string(index=False))
    if len(out) < args.n:
        print(f"不足 {args.n - len(out)} 件：動画を追加して同じ seed で再実行すると、既に選んだ候補は変わらずに追加されます。")


def cmd_report(args) -> None:
    from src.report import build_report

    src_dir = Path(args.source) if args.source else Path(args.results_dir)
    out = build_report(src_dir)
    print(f"論文用の表と図を作成しました: {out}")
    for f in sorted(out.iterdir()):
        print(f"  {f.name}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="炎上兆候検知の評価実験")
    p.add_argument("--protocol-dir", default=str(PROJECT_ROOT / "experiments" / "protocol"))
    p.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("split", help="事件単位で dev/test に分ける")
    s.add_argument("--labels", required=True)
    s.add_argument("--dev-fraction", type=float, default=0.4)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(func=cmd_split)

    s = sub.add_parser("freeze-labels", help="正解ラベルを固定（SHA-256を記録）")
    s.add_argument("--labels", required=True)
    s.set_defaults(func=cmd_freeze)

    s = sub.add_parser("tune", help="dev だけでパラメータを選んで固定")
    s.add_argument("--labels", required=True)
    s.add_argument("--dictionary", default="auto")
    s.add_argument("--freq", default="", help="例: 30min,1h,3h（既定 1h）")
    s.add_argument("--after-test-reason", default="")
    s.set_defaults(func=cmd_tune)

    s = sub.add_parser("test", help="固定した設定で test を1回だけ評価")
    s.add_argument("--labels", required=True)
    s.add_argument("--dictionary", default="frozen")
    s.add_argument("--rerun-reason", default="")
    s.set_defaults(func=cmd_test)

    s = sub.add_parser("demo", help="架空データで手順全体を試す")
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--dev-fraction", type=float, default=0.4)
    s.add_argument("--freq", default="")
    s.set_defaults(func=cmd_demo)

    s = sub.add_parser("sentiment", help="感情分析の精度（人手の正解との比較）")
    s.add_argument("--gold", required=True)
    s.add_argument("--dictionary", default="auto")
    s.set_defaults(func=cmd_sentiment)

    s = sub.add_parser("event", help="予備評価：基準時刻の前後比較（正解ラベルとは別）")
    s.add_argument("--csv", required=True, help="コメントCSV")
    s.add_argument("--video", default="", help="CSVに複数動画がある場合の動画ID")
    s.add_argument("--event", required=True, help="基準時刻（例 2025-01-27T16:00:00+09:00）")
    s.add_argument("--before", type=float, default=24.0)
    s.add_argument("--after", type=float, default=24.0)
    s.add_argument("--freq", default="1h")
    s.add_argument("--dictionary", default="auto")
    s.set_defaults(func=cmd_event)

    s = sub.add_parser("gold-sample", help="感情分析の人手評価用の候補を作る（予測を含まない）")
    s.add_argument("--csv", nargs="+", required=True, help="コメントCSV（複数可）")
    s.add_argument("--n", type=int, default=180)
    s.add_argument("--per-video-cap", type=int, default=90, help="1動画から選ぶ最大件数")
    s.add_argument("--per-thread-cap", type=int, default=3, help="1スレッド（親＋返信）から選ぶ最大件数")
    s.add_argument("--exclude", nargs="*", default=[], help="済んだラベルのCSV（comment_id 列）")
    s.add_argument("--seed", default="gold-2026")
    s.add_argument("--out", default=str(PROJECT_ROOT / "data" / "research" / "sentiment_gold_candidates.csv"))
    s.set_defaults(func=cmd_gold_sample)

    s = sub.add_parser("report", help="論文用の表と図を作る")
    s.add_argument("--from", dest="source", default="", help="結果のフォルダ（既定 results/）")
    s.set_defaults(func=cmd_report)

    args = p.parse_args(argv)
    try:
        args.func(args)
    except (ProtocolError, ValueError, FileNotFoundError) as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
