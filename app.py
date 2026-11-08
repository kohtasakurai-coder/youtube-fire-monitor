"""
YouTube炎上早期検知Webアプリ（Streamlit）

起動方法（プロジェクトのルートで）:
    streamlit run app.py

現在の実装状況：STEP 4
  - 実装済み : ホーム画面（URLからのコメント取得、CSV読み込み、架空データの読み込み、データの概要表示）、
               感情分析（Janome＋感情辞書）と時間ごとの集計、
               分析ダッシュボード（手法A・B・Cの検知、検知時刻と根拠、動画ごとの比較）
               アルゴリズム比較（正解ラベルがある動画での評価）、データ確認（判定結果の表とCSV保存）
  - 評価実験（dev/test の手順）は experiments/compare_methods.py で実行する
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src import visualization as viz
from src.access import api_fetch_allowed, is_public_mode, read_setting
from src.anomaly_detection import LEVEL_ALERT, LEVEL_CAUTION, LEVEL_NORMAL, LEVEL_UNJUDGED
from src.config import DATA_SOURCE_SYNTHETIC, AnalysisConfig
from src.detection import run_detection
from src.evaluation import (
    DETECTOR_LABELS,
    detector_flags,
    evaluate_offline_reference,
    evaluate_video,
    merge_events,
    summarize,
    validate_labels,
)
from src.preprocessing import (
    CommentCSVError,
    InvalidYouTubeURLError,
    comments_to_csv_bytes,
    describe_data_source,
    extract_video_id,
    load_comments_csv,
)
from src.sentiment_analysis import (
    SIMPLE_DICTIONARY_PATH,
    SentimentAnalyzer,
    find_private_dictionaries,
    load_nrc_dictionary,
    load_simple_dictionary,
    summarize_sentiment,
)
from src.time_series import aggregate_time_series, time_series_to_csv_bytes
from src.youtube_api import API_KEY_ENV, YouTubeAPIError, fetch_comments, get_api_key

APP_TITLE = "YouTube炎上早期検知システム（研究用プロトタイプ）"
SAMPLE_CSV = Path(__file__).parent / "data" / "sample" / "sample_comments.csv"
SAMPLE_LABELS = Path(__file__).parent / "data" / "sample" / "sample_labels.csv"
RESEARCH_LABELS = Path(__file__).parent / "data" / "research" / "labels.csv"
LEVEL_BADGE = {LEVEL_NORMAL: "🟢 正常", LEVEL_CAUTION: "🟡 注意", LEVEL_ALERT: "🔴 警戒",
               LEVEL_UNJUDGED: "⚪ 判定対象外"}
RANK_TO_LEVEL = {-1: LEVEL_UNJUDGED, 0: LEVEL_NORMAL, 1: LEVEL_CAUTION, 2: LEVEL_ALERT}


def overall_status(summary: dict) -> str:
    """動画全体の判定結果。判定できた時間帯が0なら「正常」ではなく判定対象外と表示する。"""
    if summary["n_judged_bins"] == 0 and summary["max_level"] < 1:
        return "⚪ 判定対象外（判定できた時間帯なし）"
    return LEVEL_BADGE[RANK_TO_LEVEL[int(summary["max_level"])]]

st.set_page_config(page_title="炎上早期検知", page_icon="📈", layout="wide")


# ---------------------------------------------------------------------------
# 共通部品
# ---------------------------------------------------------------------------
def show_data_source_banner(df: pd.DataFrame) -> None:
    """架空データか実データかを、画面上部ではっきり表示する（取り違え防止）。"""
    source = describe_data_source(df)
    if source == DATA_SOURCE_SYNTHETIC:
        st.warning("⚠️ 表示中のデータは **動作確認用の架空データ** です。研究結果には使用できません。")
    elif source == "mixed":
        st.error("架空データと実データが混在しています。分析前に分けてください。")
    elif source == "real":
        st.info("実データ（YouTubeから収集したデータ）を表示しています。")


def load_api_key() -> str | None:
    """APIキーを 環境変数 → Streamlit Secrets の順に探す（画面には表示しない）。"""
    key = get_api_key()
    if key:
        return key
    try:
        key = str(st.secrets.get(API_KEY_ENV, "")).strip()
    except Exception:  # secrets.toml がない場合など
        key = ""
    return key or None


def set_comments(df: pd.DataFrame, warnings: list[str], label: str) -> None:
    """読み込んだデータをセッション（画面をまたいで保持する領域）に保存する。"""
    st.session_state["comments"] = df
    st.session_state["load_warnings"] = warnings
    st.session_state["data_label"] = label
    st.session_state.pop("analysis", None)   # 前のデータの分析結果を消す
    st.session_state.pop("detection", None)


@st.cache_resource(show_spinner="感情辞書とJanomeを準備中…")
def get_analyzer(dictionary_path: str, kind: str, neg_threshold: float) -> SentimentAnalyzer:
    """Janomeの初期化に数秒かかるため、同じ条件なら作った分析器を使い回す。"""
    if kind == "nrc":
        dictionary = load_nrc_dictionary(dictionary_path)
    else:
        dictionary = load_simple_dictionary(dictionary_path)
    return SentimentAnalyzer(dictionary, neg_threshold=neg_threshold)


# ---------------------------------------------------------------------------
# 画面①：ホーム
# ---------------------------------------------------------------------------
def page_home() -> None:
    st.title(APP_TITLE)
    st.markdown(
        """
**研究概要**　YouTube動画のコメントを1時間ごとの時系列として集計し、
否定的感情の急増やコメント数の異常な増加を検出して、炎上の発生をできるだけ早く検知します。
3種類の異常検知手法（否定的コメント率・Zスコア・変化点検出）を比較・評価します。

> 本アプリの判定は研究上の暫定的な基準によるもので、実際に炎上していることを保証するものではありません。
"""
    )
    if is_public_mode(st.secrets):
        st.info("公開デモ版です。感情分析には自作の簡易辞書を使います。アップロードしたCSVは"
                "このアプリ上で処理するだけで、サーバーには保存しません。")

    col_url, col_csv = st.columns(2)

    # --- YouTube URL 入力 ---
    with col_url:
        st.subheader("YouTube動画URLから取得")
        url = st.text_input(
            "動画URL",
            placeholder="https://www.youtube.com/watch?v=... / https://youtu.be/... / Shorts のURL",
        )
        include_replies = st.checkbox(
            "返信コメントも取得する",
            value=False,
            help="オンにすると、最大取得件数は「トップレベル＋返信」の合計件数になります。"
                 "返信が多いスレッドは追加でAPIを呼ぶため、クォータを多く使います。",
        )
        max_comments = st.number_input(
            "最大取得件数（返信を含める場合は合計）" if include_replies else "最大取得件数",
            min_value=1, max_value=20000, value=2000, step=100)
        api_key = load_api_key()
        public = is_public_mode(st.secrets)
        entered = ""
        if public and read_setting("APP_PASSCODE", st.secrets):
            entered = st.text_input("合言葉（公開版でYouTubeから取得する場合）", type="password")
        allowed, reason = api_fetch_allowed(public, read_setting("APP_PASSCODE", st.secrets), entered)
        if not allowed:
            st.caption(reason)
        if api_key is None:
            st.caption("APIキーが未設定です（環境変数 YOUTUBE_API_KEY または "
                       ".streamlit/secrets.toml）。CSV読み込みは使えます。")
        if st.button("コメントを取得", disabled=not url or api_key is None or not allowed):
            try:
                video_id = extract_video_id(url)
                with st.spinner(f"動画 {video_id} のコメントを取得中…"):
                    df, warnings = fetch_comments(video_id, api_key, int(max_comments),
                                                  include_replies=include_replies)
                set_comments(df, warnings, f"YouTube取得：{video_id}")
                st.success(f"{len(df)}件を取得しました。下の「CSVでダウンロード」で保存できます。")
            except InvalidYouTubeURLError as e:
                st.error(str(e))
            except YouTubeAPIError as e:
                st.error(str(e))

    # --- CSV 読み込み ---
    with col_csv:
        st.subheader("CSVファイルから読み込み")
        uploaded = st.file_uploader("コメントCSV（APIキー不要）", type=["csv"])
        naive_tz = st.selectbox(
            "タイムゾーン情報のない日時の扱い",
            ["UTC", "Asia/Tokyo"],
            help="YouTube APIの生データはUTCです。Excelで手入力した日時なら Asia/Tokyo を選んでください。",
        )
        if uploaded is not None and st.button("CSVを読み込む"):
            try:
                df, warnings = load_comments_csv(uploaded, naive_timezone=naive_tz)
                set_comments(df, warnings, uploaded.name)
            except CommentCSVError as e:
                st.error(f"CSVを読み込めませんでした: {e}")

        if st.button("サンプル（架空データ）を読み込む"):
            if SAMPLE_CSV.exists():
                df, warnings = load_comments_csv(SAMPLE_CSV)
                set_comments(df, warnings, "架空サンプルデータ")
            else:
                st.error("サンプルCSVがありません。`python -m src.synthetic_data` で作成してください。")

    st.divider()
    df = show_loaded_summary()
    if df is not None and not df.empty:
        st.divider()
        show_sentiment_section(df)


def show_loaded_summary() -> pd.DataFrame | None:
    """読み込んだデータの概要を表示し、分析対象（1本の動画）の表を返す。"""
    df: pd.DataFrame | None = st.session_state.get("comments")
    if df is None:
        st.caption("まだデータが読み込まれていません。")
        return None

    st.subheader(f"読み込んだデータ：{st.session_state.get('data_label', '')}")
    show_data_source_banner(df)
    for w in st.session_state.get("load_warnings", []):
        st.caption(f"ℹ️ {w}")

    if df.empty:
        st.warning("分析できるコメントがありません。")
        return df

    # 動画が複数含まれている場合は1本を選ぶ
    if "video_id" in df.columns and df["video_id"].nunique() > 1:
        video = st.selectbox("分析する動画", sorted(df["video_id"].unique()))
        df = df[df["video_id"] == video]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("総コメント数", f"{len(df):,}")
    c2.metric("返信コメントの割合", f"{df['is_reply'].mean():.1%}")
    c3.metric("最初の投稿（JST）", df["published_at"].min().strftime("%Y/%m/%d %H:%M"))
    c4.metric("最後の投稿（JST）", df["published_at"].max().strftime("%Y/%m/%d %H:%M"))

    st.dataframe(
        df[["published_at", "text", "like_count", "is_reply"]].head(200),
        hide_index=True,
    )
    st.download_button(
        "このデータをCSVでダウンロード",
        data=comments_to_csv_bytes(df),
        file_name="comments.csv",
        mime="text/csv",
    )
    return df


def dictionary_options() -> dict[str, tuple[str, str]]:
    """使える辞書：公開デモ用の簡易辞書 ＋ 手元の非公開フォルダにある辞書（NRCなど）。"""
    options = {"研究用簡易辞書（自作・公開デモ用）": (str(SIMPLE_DICTIONARY_PATH), "simple")}
    if is_public_mode(st.secrets):
        return options          # 公開版では非公開の辞書（NRCなど）を使わない
    for p in find_private_dictionaries():
        options[f"非公開フォルダ：{p.name}（NRC形式）"] = (str(p), "nrc")
    return options


def show_sentiment_section(df: pd.DataFrame) -> None:
    """感情分析と時間ごとの集計（STEP 3）。"""
    st.subheader("感情分析と時系列集計")

    options = dictionary_options()
    c1, c2, c3, c4 = st.columns(4)
    dict_label = c1.selectbox("感情辞書", list(options.keys()))
    freq = c2.selectbox("集計時間幅", ["1h", "30min", "3h"], index=0)
    neg_threshold = c3.slider("否定判定の閾値 θ", 0.1, 1.0, 0.5, 0.05,
                              help="否定的な感情語の割合がθ以上なら否定的コメントと判定")
    min_comments = c4.number_input("信頼性の目安（最低コメント数）", 1, 100, 10)
    ma_window = st.number_input("移動平均の期間（ビン数）", 1, 48, 6)

    if st.button("感情分析を実行", type="primary"):
        path, kind = options[dict_label]
        try:
            analyzer = get_analyzer(path, kind, float(neg_threshold))
        except ImportError as e:
            st.error(str(e))
            return
        except (ValueError, OSError) as e:
            st.error(f"辞書を読み込めませんでした: {e}")
            return
        with st.spinner("感情分析中…"):
            analyzed = analyzer.analyze(df)
            ts = aggregate_time_series(analyzed, freq=freq, ma_window=int(ma_window),
                                       min_comments=int(min_comments))
        st.session_state["analysis"] = {
            "analyzed": analyzed, "ts": ts, "dict": analyzer.dictionary.stats(),
            "video": df["video_id"].iloc[0] if "video_id" in df.columns else "",
        }

    result = st.session_state.get("analysis")
    current_video = df["video_id"].iloc[0] if "video_id" in df.columns else ""
    if not result or result["video"] != current_video:
        st.caption("条件を選んで「感情分析を実行」を押してください。")
        return

    analyzed, ts, stats = result["analyzed"], result["ts"], result["dict"]
    summary = summarize_sentiment(analyzed)
    st.caption(f"辞書：{stats['dictionary']}（{stats['n_words']}語）")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("否定的コメント率（全コメント中）", f"{summary['neg_rate_all']:.1%}")
    m2.metric("判定できた割合（カバー率）", f"{summary['coverage']:.1%}")
    m3.metric("判定不能のコメント", f"{summary['n_unknown']:,}件")
    m4.metric("信頼性の低い時間帯", f"{int(ts['low_reliability'].sum())} / {len(ts)}")
    if summary["coverage"] < 0.5:
        st.warning("辞書に一致しないコメントが半数以上あります。判定不能のコメントは"
                   "否定的とも肯定的とも扱っていません。辞書の網羅性の限界に注意してください。")

    if len(ts) > 0:
        st.markdown("**コメント投稿数と移動平均**")
        st.line_chart(ts[["n_comments", "n_comments_ma"]])
        st.markdown("**否定的コメント率**（0件の時間帯は欠損として表示しない）")
        st.line_chart(ts[["neg_rate"]])
        st.caption("グラフの詳しい表示（Plotly・異常時刻の縦線など）は STEP 5 で実装します。")

    with st.expander("コメントごとの判定結果"):
        cols = ["published_at", "text", "sentiment_status", "neg_score",
                "anger", "fear", "sadness", "joy", "trust",
                "matched_words", "negated_words", "like_count"]
        st.dataframe(analyzed[cols].head(500), hide_index=True)
    with st.expander("時系列の集計表"):
        st.dataframe(ts)

    d1, d2 = st.columns(2)
    d1.download_button("判定結果付きコメントをCSVで保存", comments_to_csv_bytes(analyzed),
                       file_name="comments_with_sentiment.csv", mime="text/csv")
    d2.download_button("時系列データをCSVで保存", time_series_to_csv_bytes(ts),
                       file_name="time_series.csv", mime="text/csv")
    st.caption("限界：皮肉・文脈・比喩は判定できません。「最悪ではない」のような直後の否定だけを除外しています。")


# ---------------------------------------------------------------------------
# 画面②：分析ダッシュボード（STEP 4）
# ---------------------------------------------------------------------------
def detection_config_ui() -> AnalysisConfig:
    """検知の条件を画面で変えられるようにする（初期値は src/config.py）。"""
    d = AnalysisConfig()
    with st.expander("検知の条件（研究用パラメータ）"):
        c1, c2, c3 = st.columns(3)
        c1.markdown("**手法A 否定率の上昇**")
        delta = c1.number_input("直前24時間からの増加幅 δ（ポイント）", 1, 100,
                                int(round(d.neg_rate_increase * 100)))
        min_n = c1.number_input("最低コメント数（A・B・C共通）", 1, 500, d.neg_min_comments)
        binom = c1.checkbox("二項検定（α=0.01）も条件に加える", value=False,
                            help="件数が少ない時間帯の偶然の上振れを抑える比較用オプション。初期値はオフ（設計書どおり）")
        c2.markdown("**手法B Zスコア**")
        z_th = c2.number_input("Zスコアの閾値", 0.5, 20.0, float(d.z_threshold), 0.5)
        z_tr = c2.selectbox("コメント数の変換", ["none", "log1p"])
        c3.markdown("**手法C PELT・警戒レベル**")
        model = c3.selectbox("コストモデル", ["rbf", "l2", "l1"],
                             help="l2・l1 は否定率（0〜1）の差をそのまま測るので、ペナルティは 0.1〜0.5 程度が目安")
        pen = c3.number_input("ペナルティ", 0.01, 100.0, float(d.pelt_penalty), 0.5)
        red = c3.number_input("警戒（赤）にする同時検知の手法数", 1, 3, d.red_min_methods)
    return AnalysisConfig(
        neg_rate_increase=delta / 100, neg_min_comments=int(min_n),
        neg_min_baseline_comments=int(min_n), z_min_comments=int(min_n),
        min_comments_reliable=int(min_n), neg_binom_alpha=0.01 if binom else None,
        z_threshold=float(z_th), z_transform=z_tr,
        pelt_model=model, pelt_penalty=float(pen), red_min_methods=int(red),
    )


def load_reference_times(df: pd.DataFrame) -> dict:
    """架空データのときだけ、生成時に決めた炎上開始時刻（正解）を読み込む。"""
    if describe_data_source(df) != DATA_SOURCE_SYNTHETIC or not SAMPLE_LABELS.exists():
        return {}
    labels = pd.read_csv(SAMPLE_LABELS, encoding="utf-8-sig", dtype=str).fillna("")
    out = {}
    for _, row in labels.iterrows():
        out[row["video_id"]] = (pd.Timestamp(row["fire_start_jst"]).tz_convert("Asia/Tokyo")
                                if row["fire_start_jst"] else None)
    return out


def _fmt_time(t) -> str:
    return "—" if t is None or pd.isna(t) else pd.Timestamp(t).strftime("%Y/%m/%d %H:%M")


def page_dashboard() -> None:
    st.title("分析ダッシュボード")
    st.caption("警戒レベルは研究上の暫定的な判定基準であり、実際に炎上していることを保証するものではありません。")
    df_all: pd.DataFrame | None = st.session_state.get("comments")
    if df_all is None or df_all.empty:
        st.info("ホーム画面でデータを読み込んでください（「サンプル（架空データ）を読み込む」で試せます）。")
        return
    show_data_source_banner(df_all)

    options = dictionary_options()
    c1, c2 = st.columns(2)
    dict_label = c1.selectbox("感情辞書", list(options.keys()), key="dash_dict")
    freq = c2.selectbox("集計時間幅", ["1h", "30min", "3h"], key="dash_freq")
    cfg = detection_config_ui()

    if st.button("検知を実行", type="primary"):
        path, kind = options[dict_label]
        try:
            analyzer = get_analyzer(path, kind, 0.5)
        except (ImportError, ValueError, OSError) as e:
            st.error(f"感情分析の準備に失敗しました: {e}")
            return
        videos = (sorted(df_all["video_id"].unique()) if "video_id" in df_all.columns
                  else ["(動画IDなし)"])
        results = {}
        with st.spinner("感情分析と検知を実行中…（手法Cは各時刻でPELTを計算し直すので少し時間がかかります）"):
            for v in videos:
                df_v = df_all[df_all["video_id"] == v] if "video_id" in df_all.columns else df_all
                analyzed = analyzer.analyze(df_v)
                ts = aggregate_time_series(analyzed, freq=freq, ma_window=cfg.moving_average_window,
                                           min_comments=cfg.min_comments_reliable)
                try:
                    results[v] = {"analyzed": analyzed, "result": run_detection(ts, cfg)}
                except ImportError as e:
                    st.error(str(e))
                    return
        st.session_state["detection"] = {"videos": results, "cfg": cfg,
                                         "dict": analyzer.dictionary.stats()}

    det = st.session_state.get("detection")
    if not det:
        st.caption("条件を確認して「検知を実行」を押してください。")
        return
    cfg, results = det["cfg"], det["videos"]
    refs = load_reference_times(df_all)
    st.caption(f"辞書：{det['dict']['dictionary']}（{det['dict']['n_words']}語）")

    # --- 動画ごとの比較 ---
    st.subheader("動画ごとの比較")
    rows = []
    for v, r in results.items():
        s = r["result"].summary()
        ref = refs.get(v)
        first_any = min([t for t in (s["first_A"], s["first_B"], s["first_C"]) if t is not None],
                        default=None)
        rows.append({
            "動画": v,
            "正解（架空）": ("炎上（" + _fmt_time(ref) + "開始）") if ref is not None
                           else ("非炎上" if v in refs else "—"),
            "判定結果（期間中の最高レベル）": overall_status(s),
            "判定できた時間帯": f"{s['n_judged_bins']} / {s['n_bins']}",
            "警戒の時間帯数": s["n_alert_bins"],
            "最初の警戒（赤）": _fmt_time(r["result"].table.index[r["result"].table["level"] == LEVEL_ALERT][0]
                                       if (r["result"].table["level"] == LEVEL_ALERT).any() else None),
            "手法A 初検知": _fmt_time(s["first_A"]), "A 回数": s["n_alarm_A"],
            "手法B 初検知": _fmt_time(s["first_B"]), "B 回数": s["n_alarm_B"],
            "手法C 初検知": _fmt_time(s["first_C"]), "C 回数": s["n_alarm_C"],
            "最初の検知（いずれか）": _fmt_time(first_any),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("「判定できた時間帯」は、手法A・B・Cのいずれかが判定条件（24時間分の履歴、10件以上のコメント）を"
               "満たした時間帯の数です。判定できた時間帯が0の動画は「正常」ではなく判定対象外と表示します。"
               "回数は検知した時間帯の数（連続した検知をまとめる処理は評価実験〔STEP 6〕で行います）。"
               "架空データの正解は生成プログラムで決めた値で、研究の評価には使いません。")
    st.download_button("動画ごとの比較をCSVで保存", pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig"),
                       file_name="dashboard_comparison.csv", mime="text/csv")

    # --- 1本の動画の詳細 ---
    st.divider()
    video = st.selectbox("詳しく見る動画", list(results.keys()), key="dash_video")
    r = results[video]
    res, analyzed = r["result"], r["analyzed"]
    table = res.table
    ref = refs.get(video)
    if table.empty:
        st.warning("この動画には集計できるコメントがありません。")
        return

    last = table.iloc[-1]
    summary = res.summary()
    judged_idx = table.index[table["judged"]]
    if summary["n_judged_bins"] == 0:
        st.warning("この動画には判定条件を満たす時間帯がありません（すべて履歴不足・件数不足）。"
                   "検知がないのは「正常」と判定したためではなく、判定対象外だからです。")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("総コメント数", f"{len(analyzed):,}")
    neg_all = (analyzed["is_negative"] == 1).mean()
    m2.metric("否定的コメント率", f"{neg_all:.1%}")
    m3.metric("最新の異常スコア（Z）", "—" if pd.isna(last["b_z"]) else f"{last['b_z']:.1f}",
              help=f"最後の時間帯（{_fmt_time(table.index[-1])}）の手法BのZスコア。判定状態：{last['b_status']}")
    last_judged = (f"最後に判定できた時間帯：{_fmt_time(judged_idx[-1])}（{table.loc[judged_idx[-1], 'level']}）"
                   if len(judged_idx) else "判定できた時間帯はありません")
    m4.metric("現在の警戒レベル", LEVEL_BADGE[last["level"]],
              help="最後の時間帯のレベル。同じ時間帯に設定数以上の手法が検知すると警戒、直近3時間に検知があれば注意、"
                   "検知がなく判定もできなかった時間帯は判定対象外。" + last_judged)
    st.caption(f"期間中の判定結果：{overall_status(summary)}　／　判定できた時間帯 "
               f"{summary['n_judged_bins']} / {summary['n_bins']}　／　{last_judged}")

    a_counts = table["a_status"].value_counts()
    st.caption(
        "判定できた時間帯（手法A）：" + " ・ ".join(f"{k} {int(v)}" for k, v in a_counts.items())
        + f"（全{len(table)}）。履歴不足＝データの最初から{cfg.neg_baseline_hours}時間未満、"
        + f"件数不足＝コメント{cfg.neg_min_comments}件未満の時間帯は判定しません。")

    st.plotly_chart(viz.fig_counts(table, ref), key="g1")
    st.plotly_chart(viz.fig_negative_rate(table, ref), key="g2")
    st.plotly_chart(viz.fig_emotions(table), key="g3")
    st.plotly_chart(viz.fig_zscore(table, cfg.z_threshold), key="g4")
    st.plotly_chart(viz.fig_change_points(table, res.cp_events, res.offline_cps, ref), key="g5")

    st.subheader("検知の一覧と判定の根拠")
    if res.events.empty:
        st.success("この動画では、どの手法も検知しませんでした。")
    else:
        ev = res.events.copy()
        if ref is not None:
            ev["炎上開始からの時間"] = ((ev["detected_at"] - ref) / pd.Timedelta(hours=1)).map(
                lambda h: f"{h:+.0f}時間")
        ev["detected_at"] = ev["detected_at"].map(_fmt_time)
        ev = ev.rename(columns={"detected_at": "検知時刻", "method": "手法", "reason": "根拠",
                                "level": "その時点の警戒レベル"})
        st.dataframe(ev, hide_index=True)

    st.subheader("手法C：変化点の位置と検知時刻（オンライン）")
    st.caption("各時刻で、その時刻までのデータだけでPELTを計算し直した結果です。"
               "変化点の位置は後から振り返った推定値で、実際に気づけたのは「検知時刻」です。")
    if res.cp_events.empty:
        st.write("オンラインでの検知はありません。")
    else:
        cp = res.cp_events.copy()
        cp["change_point_at"] = cp["change_point_at"].map(_fmt_time)
        cp["detected_at"] = cp["detected_at"].map(_fmt_time)
        cp = cp.rename(columns={"change_point_at": "変化点の位置", "detected_at": "検知時刻",
                                "delay_bins": "遅れ（ビン数）", "pre_mean": "変化前の否定率",
                                "post_mean": "変化後の否定率"}).drop(columns=["reason"])
        st.dataframe(cp, hide_index=True)
    with st.expander("事後分析：全期間のデータを使ったPELT（評価には使わない）"):
        st.caption("未来のデータも使って求めた変化点です。リアルタイムの検知結果ではありません。")
        if res.offline_cps.empty:
            st.write("変化点はありません。")
        else:
            off = res.offline_cps.copy()
            off["change_point_at"] = off["change_point_at"].map(_fmt_time)
            st.dataframe(off.rename(columns={"change_point_at": "変化点の位置",
                                             "pre_mean": "変化前の否定率", "post_mean": "変化後の否定率"}),
                         hide_index=True)

    with st.expander("時間帯ごとの全指標（CSVで保存可）"):
        st.dataframe(table)
    st.download_button("検知結果（時間帯ごと）をCSVで保存", time_series_to_csv_bytes(table),
                       file_name=f"detection_{video}.csv", mime="text/csv")
    st.caption(f"計算時間：手法A {res.seconds['A']:.3f}秒、手法B {res.seconds['B']:.3f}秒、"
               f"手法C {res.seconds['C']:.2f}秒")


# ---------------------------------------------------------------------------
# 画面③：アルゴリズム比較（STEP 5〜6）
# ---------------------------------------------------------------------------
def need_detection() -> dict | None:
    det = st.session_state.get("detection")
    if not det:
        st.info("先に「分析ダッシュボード」で「検知を実行」を押してください（ここではその結果を使います）。")
    return det


def load_labels_for_app(df_all: pd.DataFrame) -> tuple[pd.DataFrame | None, str]:
    """架空データなら生成時の正解、実データなら data/research/labels.csv かアップロードしたファイル。"""
    if describe_data_source(df_all) == DATA_SOURCE_SYNTHETIC and SAMPLE_LABELS.exists():
        return pd.read_csv(SAMPLE_LABELS, encoding="utf-8-sig", dtype=str).fillna(""), "架空データの正解（生成時に設定）"
    uploaded = st.file_uploader("正解ラベルCSV（任意）", type=["csv"], key="labels_upload",
                                help="列：video_id, incident_id, is_fire, fire_start_jst, label_source, split")
    if uploaded is not None:
        return pd.read_csv(uploaded, encoding="utf-8-sig", dtype=str).fillna(""), uploaded.name
    if RESEARCH_LABELS.exists():
        return pd.read_csv(RESEARCH_LABELS, encoding="utf-8-sig", dtype=str).fillna(""), "data/research/labels.csv"
    return None, ""


def page_comparison() -> None:
    st.title("アルゴリズム比較")
    df_all: pd.DataFrame | None = st.session_state.get("comments")
    det = need_detection()
    if df_all is None or not det:
        return
    show_data_source_banner(df_all)
    cfg, results = det["cfg"], det["videos"]

    # --- 正解ラベルがなくても出せる比較：検知数・検知時刻・計算時間 ---
    st.subheader("手法ごとの検知結果")
    rows = []
    for v, r in results.items():
        tbl = r["result"].table
        flags = detector_flags(tbl)
        judged_count = {"A": int((tbl["a_status"] == "判定").sum()), "B": int((tbl["b_status"] == "判定").sum()),
                        "C": int(tbl["c_evaluated"].sum()), "警戒": int(tbl["judged"].sum())}
        for d in ["A", "B", "C", "警戒"]:
            times = list(flags.index[flags[d].to_numpy()])
            rows.append({
                "動画": v, "手法": DETECTOR_LABELS[d],
                "判定できた時間帯": f"{judged_count[d]} / {len(tbl)}",
                "結果": ("判定対象外" if judged_count[d] == 0 and not times
                         else ("検知あり" if times else "判定したが検知なし")),
                "検知した時間帯の数": len(times),
                "検知イベント数（6時間以内はまとめる）": len(merge_events(times, 6)),
                "最初の検知": _fmt_time(times[0]) if times else "—",
                "検知時刻（最大5件）": "、".join(_fmt_time(t) for t in times[:5]) + ("…" if len(times) > 5 else ""),
                "計算時間(秒)": r["result"].seconds.get(d, float("nan")) if d in ("A", "B", "C") else float("nan"),
            })
    detail = pd.DataFrame(rows)
    st.dataframe(detail, hide_index=True)
    total = {k: sum(r["result"].seconds.get(k, 0.0) for r in results.values()) for k in ("A", "B", "C")}
    st.caption(f"計算時間の合計：手法A {total['A']:.3f}秒、手法B {total['B']:.3f}秒、手法C {total['C']:.2f}秒"
               "（手法Cは各時刻でPELTを計算し直すため時間がかかります）")

    # --- 正解ラベルがある場合だけ精度指標 ---
    st.subheader("正解ラベルによる評価")
    labels_raw, source = load_labels_for_app(df_all)
    if labels_raw is None:
        st.info("正解ラベルがないため、Precision・Recall・F1・誤警戒率・検知の遅れは計算しません。")
        st.download_button("検知結果の比較をCSVで保存", detail.to_csv(index=False).encode("utf-8-sig"),
                           file_name="comparison_detections.csv", mime="text/csv")
        return
    try:
        labels = validate_labels(labels_raw)
    except ValueError as e:
        st.error(str(e))
        return
    labels = labels[labels["video_id"].isin(results.keys())]
    hidden = labels[labels["split"] == "test"]
    shown = labels[labels["split"] != "test"]
    st.caption(f"正解ラベル：{source}")
    if len(hidden):
        st.warning(f"評価用（test）の動画 {len(hidden)} 本はこの画面では評価しません。"
                   "閾値を決めたあとに `python -m experiments.compare_methods test` で1回だけ評価します。")
    if shown.empty:
        st.info("この画面で評価できる（dev または分割前の）ラベル付き動画がありません。")
        return
    if describe_data_source(df_all) == DATA_SOURCE_SYNTHETIC:
        st.warning("架空データでの結果です。評価処理の動作確認用で、実動画での性能ではありません。")

    per_video = pd.concat([evaluate_video(results[row["video_id"]]["result"].table, row)
                           for _, row in shown.iterrows()], ignore_index=True)
    n_excluded = per_video.loc[per_video["outcome"] == "対象外", "video_id"].nunique()
    if n_excluded:
        st.warning(f"判定できた時間帯がない動画 {n_excluded} 本は「判定対象外」として指標から除外しました。")
    summary = summarize(per_video, n_boot=200)
    cols = ["detector", "説明", "検知(TP)", "見逃し(FN)", "誤警戒動画(FP)", "沈黙(TN)", "Precision",
            "Recall", "Recall_95CI", "F1", "誤警戒率_動画", "誤警戒_100時間あたり",
            "遅れ_中央値(h)", "遅れを測れた動画数", "判定対象外で除外した動画数"]
    st.dataframe(summary[cols].round(3), hide_index=True)
    st.caption("炎上動画は「炎上開始 ≤ 検知時刻 ≤ 開始＋48時間」の検知だけを正解とし、開始前の警報は誤警戒として"
               "数えます。遅れは推定された変化点の位置ではなく、実際に検知できた時刻で計算しています。"
               f"評価した動画は {len(shown)} 本で、少数のため信頼区間が広くなります。")

    with st.expander("動画ごとの判定"):
        pv = per_video.copy()
        pv["first_detection"] = pv["first_detection"].map(_fmt_time)
        pv["first_alarm_any"] = pv["first_alarm_any"].map(_fmt_time)
        st.dataframe(pv, hide_index=True)
    with st.expander("参考：事後分析（全期間のPELT）。未来のデータを使うため評価指標には含めません"):
        ref = pd.DataFrame([evaluate_offline_reference(results[r["video_id"]]["result"].offline_cps, r)
                            for _, r in shown.iterrows()])
        st.dataframe(ref.rename(columns={"offline_n_cps": "変化点の数",
                                         "nearest_cp_to_start_hours": "炎上開始に最も近い変化点との差(時間)"}),
                     hide_index=True)
    c1, c2 = st.columns(2)
    c1.download_button("評価指標をCSVで保存", summary.to_csv(index=False).encode("utf-8-sig"),
                       file_name="comparison_summary.csv", mime="text/csv")
    c2.download_button("動画ごとの判定をCSVで保存", per_video.to_csv(index=False).encode("utf-8-sig"),
                       file_name="comparison_per_video.csv", mime="text/csv")


# ---------------------------------------------------------------------------
# 画面④：データ確認
# ---------------------------------------------------------------------------
def page_data() -> None:
    st.title("データ確認")
    df_all: pd.DataFrame | None = st.session_state.get("comments")
    det = need_detection()
    if df_all is None or not det:
        return
    show_data_source_banner(df_all)
    results = det["videos"]
    video = st.selectbox("動画", list(results.keys()), key="data_video")
    analyzed = results[video]["analyzed"]

    c1, c2, c3 = st.columns(3)
    statuses = c1.multiselect("否定的コメント判定", ["否定的", "非否定的", "判定不能"],
                              default=["否定的", "非否定的", "判定不能"])
    include_replies = c2.checkbox("返信コメントを含める", value=True)
    keyword = c3.text_input("本文で絞り込み（部分一致）", key="data_keyword")

    view = analyzed[analyzed["sentiment_status"].isin(statuses)]
    if not include_replies:
        view = view[~view["is_reply"]]
    if keyword:
        view = view[view["text"].str.contains(keyword, regex=False, na=False)]

    table = pd.DataFrame({
        "投稿日時（JST）": view["published_at"].dt.strftime("%Y/%m/%d %H:%M"),
        "コメント本文": view["text"],
        "否定的コメント判定": view["sentiment_status"],
        "否定スコア": view["neg_score"].round(2),
        **{f"{EMOTION_JA[e]}": view[e].round(2) for e in EMOTION_JA},
        "いいね数": view["like_count"],
        "返信": view["is_reply"],
        "一致した感情語": view["matched_words"],
        "否定で除外した語": view["negated_words"],
    })
    st.caption(f"{len(table):,}件 / {len(analyzed):,}件を表示（感情スコアは0〜1。判定不能のコメントは空欄）")
    st.dataframe(table, hide_index=True)
    st.download_button("表示中のコメントをCSVで保存", comments_to_csv_bytes(view),
                       file_name=f"comments_{video}_checked.csv", mime="text/csv")


EMOTION_JA = {"anger": "怒り", "fear": "恐怖", "sadness": "悲しみ", "joy": "喜び", "trust": "信頼"}


PAGES = {
    "ホーム": page_home,
    "分析ダッシュボード": page_dashboard,
    "アルゴリズム比較": page_comparison,
    "データ確認": page_data,
}

choice = st.sidebar.radio("画面", list(PAGES.keys()))
st.sidebar.caption("全画面実装済み（STEP 5〜7）")
PAGES[choice]()
