"""
感情分析のテスト。

前半：偽の形態素解析器（FakeTokenizer）で、照合・否定処理・スコア計算のルールだけを確かめる
後半：実際のJanomeで代表的な文を確かめる（Janomeがない環境では自動でスキップ）
"""
import math

import pandas as pd
import pytest

from src.config import EMOTIONS
from src.sentiment_analysis import (
    STATUS_NEGATIVE,
    STATUS_NOT_NEGATIVE,
    STATUS_UNKNOWN,
    EmotionDictionary,
    SentimentAnalyzer,
    Token,
    evaluate_negative_detection,
    find_private_dictionaries,
    load_nrc_dictionary,
    load_simple_dictionary,
    summarize_sentiment,
)


class FakeTokenizer:
    """「表層形/基本形/品詞」を空白区切りで書いた文字列をそのままトークンにする。"""

    def tokenize(self, text):
        out = []
        for part in text.split():
            surface, base, pos = part.split("/")
            out.append(Token(surface, base, pos))
        return out


def make_analyzer(entries, **kw):
    d = EmotionDictionary({w: frozenset(e) for w, e in entries.items()}, name="test")
    return SentimentAnalyzer(d, tokenizer=FakeTokenizer(), **kw)


DIC = {"最悪": {"anger"}, "怖い": {"fear"}, "嬉しい": {"joy"}, "応援": {"trust"},
       "腹が立つ": {"anger"}, "遺憾": {"anger", "sadness"}}


# ------------------------------------------------------------ ルールの確認
def test_no_match_is_unknown_not_positive():
    r = make_analyzer(DIC).analyze_text("今日/今日/名詞 は/は/助詞 晴れ/晴れ/名詞")
    assert r.status == STATUS_UNKNOWN
    assert math.isnan(r.is_negative)
    assert all(math.isnan(r.scores[e]) for e in EMOTIONS)


def test_scores_are_shares_between_0_and_1():
    r = make_analyzer(DIC).analyze_text("最悪/最悪/名詞 で/だ/助動詞 怖い/怖い/形容詞 けど/けど/助詞 応援/応援/名詞")
    assert r.scores["anger"] == pytest.approx(1 / 3)
    assert r.scores["fear"] == pytest.approx(1 / 3)
    assert r.scores["trust"] == pytest.approx(1 / 3)
    assert r.neg_score == pytest.approx(2 / 3)
    assert r.status == STATUS_NEGATIVE
    assert all(0 <= v <= 1 for v in r.scores.values())


def test_multi_emotion_word_counts_once_as_negative():
    r = make_analyzer(DIC).analyze_text("遺憾/遺憾/名詞 です/です/助動詞")
    assert r.scores["anger"] == 1.0 and r.scores["sadness"] == 1.0
    assert r.neg_score == 1.0


def test_threshold():
    text = "最悪/最悪/名詞 だ/だ/助動詞 けど/けど/助詞 嬉しい/嬉しい/形容詞"
    assert make_analyzer(DIC, neg_threshold=0.5).analyze_text(text).status == STATUS_NEGATIVE
    assert make_analyzer(DIC, neg_threshold=0.6).analyze_text(text).status == STATUS_NOT_NEGATIVE


def test_phrase_with_inflected_last_word():
    # 「腹が立った」→ 腹＋が＋立つ（最後の語だけ基本形）で辞書の「腹が立つ」に一致
    r = make_analyzer(DIC).analyze_text("腹/腹/名詞 が/が/助詞 立っ/立つ/動詞 た/た/助動詞")
    assert r.matched == ["腹が立つ"]


@pytest.mark.parametrize("text", [
    "最悪/最悪/名詞 で/で/助詞 は/は/助詞 ない/ない/形容詞",                    # 最悪ではない
    "怖く/怖い/形容詞 ない/ない/助動詞",                                         # 怖くない
    "怖く/怖い/形容詞 あり/ある/動詞 ませ/ます/助動詞 ん/ん/助動詞",              # 怖くありません
])
def test_negation_removes_word(text):
    r = make_analyzer(DIC).analyze_text(text)
    assert r.matched == [] and len(r.negated) == 1
    assert r.status == STATUS_UNKNOWN


def test_shika_nai_is_not_negation():
    # 「怖いしかない」の「しか…ない」は否定ではない
    r = make_analyzer(DIC).analyze_text("怖い/怖い/形容詞 しか/しか/助詞 ない/ない/形容詞")
    assert r.matched == ["怖い"]


def test_negation_does_not_jump_over_content_words():
    # 「怖くてもう使えない」の「ない」は「使える」を否定しているので「怖い」は残る
    r = make_analyzer(DIC).analyze_text(
        "怖く/怖い/形容詞 て/て/助詞 もう/もう/副詞 使え/使える/動詞 ない/ない/助動詞")
    assert r.matched == ["怖い"]


def test_empty_and_nan_text():
    a = make_analyzer(DIC)
    assert a.analyze_text("").status == STATUS_UNKNOWN
    assert a.analyze_text(None).status == STATUS_UNKNOWN


def test_analyze_dataframe_and_summary():
    a = make_analyzer(DIC)
    df = pd.DataFrame({"text": ["最悪/最悪/名詞", "嬉しい/嬉しい/形容詞", "晴れ/晴れ/名詞"]})
    out = a.analyze(df)
    assert list(out["sentiment_status"]) == [STATUS_NEGATIVE, STATUS_NOT_NEGATIVE, STATUS_UNKNOWN]
    assert "text" in df.columns and "anger" not in df.columns   # 元の表は変更しない
    s = summarize_sentiment(out)
    assert s["n_judged"] == 2 and s["n_unknown"] == 1
    assert s["neg_rate_all"] == pytest.approx(1 / 3)
    assert s["neg_rate_judged"] == pytest.approx(1 / 2)


def test_evaluate_negative_detection():
    pred = pd.Series([1.0, 0.0, float("nan"), 1.0])
    gold = pd.Series([1, 1, 1, 0])
    m = evaluate_negative_detection(pred, gold)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 2, 0)
    assert m["precision"] == 0.5 and m["recall"] == pytest.approx(1 / 3)
    assert m["n_unknown"] == 1


# ------------------------------------------------------------ 辞書ファイル
def test_simple_dictionary_file_is_valid():
    d = load_simple_dictionary()
    stats = d.stats()
    assert stats["n_words"] >= 200
    assert all(stats[f"n_{e}"] > 0 for e in EMOTIONS)
    raw = pd.read_csv("data/sentiment_dictionary.csv", encoding="utf-8-sig")
    assert set(raw["emotion"]) <= set(EMOTIONS)
    assert not raw.duplicated(["word", "emotion"]).any()


def test_load_nrc_wide_format(tmp_path):
    """NRC多言語版（横長・タブ区切り）の形式。英語列・disgust列などは無視する。"""
    p = tmp_path / "Japanese-NRC-EmoLex.txt"
    p.write_text(
        "English Word\tanger\tanticipation\tdisgust\tfear\tjoy\tnegative\tpositive\tsadness\tsurprise\ttrust\tJapanese Word\n"
        "abandon\t0\t0\t0\t1\t0\t1\t0\t1\t0\t0\t放棄\n"
        "happy\t0\t1\t0\t0\t1\t0\t1\t0\t0\t1\t幸せ\n"
        "gross\t0\t0\t1\t0\t0\t1\t0\t0\t0\t0\tグロス\n",
        encoding="utf-8")
    d = load_nrc_dictionary(p)
    assert d.entries["放棄"] == frozenset({"fear", "sadness"})
    assert d.entries["幸せ"] == frozenset({"joy", "trust"})
    assert "グロス" not in d.entries          # disgust だけの語は5感情に入らない


def test_load_nrc_long_format(tmp_path):
    p = tmp_path / "nrc_long.csv"
    p.write_text("word,emotion,association\n怒り,anger,1\n怒り,joy,0\n", encoding="utf-8")
    assert load_nrc_dictionary(p).entries == {"怒り": frozenset({"anger"})}


def test_load_nrc_rejects_unknown_format(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_nrc_dictionary(p)


def test_find_private_dictionaries(tmp_path):
    assert find_private_dictionaries(tmp_path / "なし") == []
    (tmp_path / "Japanese-NRC-EmoLex.txt").write_text("x", encoding="utf-8")
    (tmp_path / "readme.md").write_text("x", encoding="utf-8")
    assert [p.name for p in find_private_dictionaries(tmp_path)] == ["Japanese-NRC-EmoLex.txt"]


# ------------------------------------------------------------ 実際のJanome
@pytest.fixture(scope="module")
def janome_analyzer():
    pytest.importorskip("janome")
    return SentimentAnalyzer(load_simple_dictionary())


@pytest.mark.parametrize("text, status", [
    ("最悪だ", STATUS_NEGATIVE),
    ("最悪ではない", STATUS_UNKNOWN),         # 否定 → 数えない
    ("最悪じゃないけど微妙", STATUS_UNKNOWN),
    ("怖くありません", STATUS_UNKNOWN),
    ("許せない", STATUS_NEGATIVE),
    ("許せません", STATUS_NEGATIVE),
    ("腹が立った", STATUS_NEGATIVE),
    ("説明が不誠実で怒りしかない", STATUS_NEGATIVE),
    ("怖くてもう使えない", STATUS_NEGATIVE),
    ("信頼できない", STATUS_NEGATIVE),
    ("信頼できる会社だと思う", STATUS_NOT_NEGATIVE),
    ("嬉しいニュースですね", STATUS_NOT_NEGATIVE),
    ("今北産業", STATUS_UNKNOWN),             # 辞書に一致なし → 判定不能（肯定扱いしない）
    ("ＦＵＬＬ　ＷＩＤＴＨ　最悪", STATUS_NEGATIVE),   # 全角もNFKCで正規化
])
def test_janome_examples(janome_analyzer, text, status):
    assert janome_analyzer.analyze_text(text).status == status


def test_janome_known_limitations(janome_analyzer):
    """限界の記録：皮肉は判定不能、「怖いくらい早い」は否定的と誤判定する。"""
    assert janome_analyzer.analyze_text("さすがですね（笑）").status == STATUS_UNKNOWN
    assert janome_analyzer.analyze_text("怖いくらい対応が早い").status == STATUS_NEGATIVE


# ---------------------------------------------------------------------------
# 辞書の選択・感情別集計・辞書名の記録
# ---------------------------------------------------------------------------
def test_dictionary_choices_prefer_nrc_locally_and_simple_in_public(tmp_path):
    from src.sentiment_analysis import NRC_FILENAME, SIMPLE_LABEL, dictionary_choices, find_private_dictionaries
    (tmp_path / "README.md").write_text("x", encoding="utf-8")
    (tmp_path / "another.tsv").write_text("word\temotion\n怒り\tanger\n", encoding="utf-8")
    (tmp_path / NRC_FILENAME).write_text("x", encoding="utf-8")
    found = find_private_dictionaries(tmp_path)
    assert [p.name for p in found] == [NRC_FILENAME, "another.tsv"]   # README は除外、NRCが先頭

    local, idx = dictionary_choices(found, public_mode=False)
    assert local[idx].kind == "nrc" and local[idx].label == f"NRC日本語辞書（{NRC_FILENAME}）"
    assert SIMPLE_LABEL in [c.label for c in local]                    # 簡易辞書にも切り替えられる

    public, idx = dictionary_choices(found, public_mode=True)
    assert [c.label for c in public] == [SIMPLE_LABEL] and idx == 0    # クラウドはNRCを使わない

    none, idx = dictionary_choices([], public_mode=False)
    assert none[idx].label == SIMPLE_LABEL                             # NRCがなければ簡易辞書


def test_emotion_summary_uses_judged_comments_as_denominator():
    import math
    from src.sentiment_analysis import emotion_summary
    df = pd.DataFrame({
        "is_negative": [1.0, 0.0, float("nan"), 1.0],
        "anger": [1.0, 0.0, float("nan"), 0.5], "fear": [0.0, 0.0, float("nan"), 0.5],
        "sadness": [0.0, 0.0, float("nan"), 0.0], "joy": [0.0, 1.0, float("nan"), 0.0],
        "trust": [0.0, 0.0, float("nan"), 0.0],
    })
    out = emotion_summary(df).set_index("emotion")
    assert out.loc["anger", "該当コメント数"] == 2
    assert math.isclose(out.loc["anger", "判定できたコメントに占める割合"], 2 / 3)   # 判定不能1件は分母に入れない
    assert math.isclose(out.loc["anger", "平均スコア（判定できたコメント）"], 0.5)
    assert out.loc["joy", "該当コメント数"] == 1 and not out.loc["joy", "否定的感情"]


def test_analyze_records_dictionary_name():
    from src.sentiment_analysis import add_dictionary_column
    d = EmotionDictionary({"最悪": frozenset({"anger"})}, name="テスト辞書")
    analyzer = SentimentAnalyzer(d, tokenizer=FakeTokenizer())
    out = analyzer.analyze(pd.DataFrame({"text": ["最悪/最悪/形容動詞", "普通/普通/名詞"]}))
    assert (out["dictionary"] == "テスト辞書").all()
    assert list(out["sentiment_status"]) == ["否定的", "判定不能"]
    ts = add_dictionary_column(pd.DataFrame({"n": [1]}), "テスト辞書")
    assert ts["dictionary"].iloc[0] == "テスト辞書"
