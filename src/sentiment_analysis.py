"""
感情分析モジュール（STEP 3）

Janomeで形態素解析し、感情辞書と照合して、コメントごとに
anger / fear / sadness / joy / trust のスコア（0〜1）と否定的判定を付ける。

■ 計算方法（設計書 6章）
  1. 本文をNFKC正規化（全角英数→半角など）してJanomeで分かち書きする
  2. 連続する最大4語までを「前の語は表層形＋最後の語は基本形」でつなぎ、辞書と照合する
     （例：「腹が立った」→ 腹＋が＋立つ ＝「腹が立つ」。長い一致を優先）
  3. 一致した語の直後に否定（ない・ぬ・ん・ず）が来ていたら、その語は数えない
     （例：「最悪ではない」「怖くない」「怖くありません」）
     ただし「怒りしかない」の「しかない」は否定ではないので数える
  4. M ＝ 数えた感情語の数、m_e ＝ 感情eに該当する語の数として
        score_e = m_e / M          （0〜1。M=0なら NaN＝判定不能）
        neg_score = (anger・fear・sadness のどれかに該当する語の数) / M
        is_negative = 1（neg_score ≥ 閾値θ） / 0（それ以外） / NaN（M=0、判定不能）
     判定不能のコメントを「肯定的」や「否定的でない」とはみなさない

■ 限界（論文に明記すること）
  - 皮肉（「さすがですね（笑）」）、文脈依存、比喩、ネットスラングは判定できない
  - 否定の判定は直後の機能語だけを見る簡易ルール（「最悪では決してない」などは検出しない）
  - 簡易辞書は作者が作成した約200語で網羅性が低い（判定不能が多くなる）
"""
from __future__ import annotations

import csv
import io
import math
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol

import pandas as pd

from src.config import EMOTIONS, NEGATIVE_EMOTIONS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIMPLE_DICTIONARY_PATH = PROJECT_ROOT / "data" / "sentiment_dictionary.csv"
PRIVATE_DICTIONARY_DIR = PROJECT_ROOT / "data" / "dictionaries" / "private"
# ローカル研究版で優先して使うNRC日本語辞書のファイル名（再配布不可のため GitHub・クラウドには置かない）
NRC_FILENAME = "Japanese-NRC-EmoLex.txt"
NRC_DEFAULT_PATH = PRIVATE_DICTIONARY_DIR / NRC_FILENAME
SIMPLE_LABEL = "簡易辞書（自作・公開デモ用）"

MAX_PHRASE_TOKENS = 4          # 何語までの連続を1つの見出し語として照合するか
NEGATION_BASES = {"ない", "無い", "ぬ", "ん", "ず"}
# 感情語と否定語のあいだに挟まってもよい動詞（「怖くありません」「がっかりしない」）
BRIDGE_VERBS = {"ある", "する", "なる", "いる", "できる"}

STATUS_NEGATIVE = "否定的"
STATUS_NOT_NEGATIVE = "非否定的"
STATUS_UNKNOWN = "判定不能"


# ---------------------------------------------------------------------------
# 形態素解析
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Token:
    surface: str     # 表層形（文中の形）
    base_form: str   # 基本形（辞書の形）
    pos: str         # 品詞（名詞・動詞・助詞 など）


class Tokenizer(Protocol):
    def tokenize(self, text: str) -> list[Token]: ...


class JanomeTokenizer:
    """Janomeの薄いラッパー。初期化に数秒かかるので1回だけ作って使い回す。"""

    def __init__(self) -> None:
        try:
            from janome.tokenizer import Tokenizer as _JanomeTokenizer
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Janomeがインストールされていません。`pip install -r requirements.txt` を実行してください。"
            ) from e
        self._tokenizer = _JanomeTokenizer()

    def tokenize(self, text: str) -> list[Token]:
        tokens = []
        for t in self._tokenizer.tokenize(text):
            base = t.base_form if t.base_form and t.base_form != "*" else t.surface
            tokens.append(Token(t.surface, base, t.part_of_speech.split(",")[0]))
        return tokens


def normalize_text(text: str) -> str:
    """NFKC正規化（全角英数字・記号を半角に、半角カナを全角に）。"""
    return unicodedata.normalize("NFKC", str(text)).strip()


# ---------------------------------------------------------------------------
# 感情辞書
# ---------------------------------------------------------------------------
@dataclass
class EmotionDictionary:
    """見出し語 → 感情の集合。"""

    entries: dict[str, frozenset[str]]
    name: str
    source: str = ""

    def stats(self) -> dict:
        """論文に書く収録語数（実際に読み込んだ数）を返す。"""
        out = {"dictionary": self.name, "n_words": len(self.entries)}
        for e in EMOTIONS:
            out[f"n_{e}"] = sum(1 for emos in self.entries.values() if e in emos)
        return out


def _build(pairs: Iterable[tuple[str, str]], name: str, source: str) -> EmotionDictionary:
    entries: dict[str, set[str]] = {}
    for word, emotion in pairs:
        word = normalize_text(word)
        emotion = str(emotion).strip().lower()
        if not word or emotion not in EMOTIONS:
            continue   # 5感情以外（disgust, surprise など）は今回は使わない
        entries.setdefault(word, set()).add(emotion)
    if not entries:
        raise ValueError(f"辞書「{name}」に有効な見出し語がありません。")
    return EmotionDictionary({w: frozenset(e) for w, e in entries.items()}, name, source)


def load_simple_dictionary(path: str | Path = SIMPLE_DICTIONARY_PATH) -> EmotionDictionary:
    """
    研究用簡易辞書（公開デモ用・自作）を読み込む。
    形式: word,emotion,source の縦長CSV（1行＝1語×1感情）
    """
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    return _build(zip(df["word"], df["emotion"]), name="研究用簡易辞書（自作 v1）",
                  source=str(path))


def load_nrc_dictionary(path: str | Path) -> EmotionDictionary:
    """
    NRC Emotion Lexicon の日本語版を読み込む（再配布禁止のため手元の非公開フォルダに置く）。

    対応する形式:
      (1) 横長のタブ区切り（多言語版 OneFilePerLanguage の形式）
          列に「Japanese」を含む見出し語の列と、anger, fear, ... の0/1列がある
      (2) 縦長: word, emotion（, association）の列を持つCSV/TSV
    英語の列しかないファイル（英語版）は読み込めない。
    """
    path = Path(path)
    raw = path.read_bytes().decode("utf-8-sig")
    sep = "\t" if raw.split("\n", 1)[0].count("\t") >= 1 else ","
    df = pd.read_csv(io.StringIO(raw), sep=sep, dtype=str)
    cols = {c.strip().lower(): c for c in df.columns}

    word_col = next((c for c in df.columns if "japanese" in c.lower()), None)
    if word_col is not None:                       # (1) 横長
        pairs = []
        for e in EMOTIONS:
            if e not in cols:
                continue
            hit = df[pd.to_numeric(df[cols[e]], errors="coerce") == 1]
            pairs += [(w, e) for w in hit[word_col].dropna()]
    elif "word" in cols and "emotion" in cols:     # (2) 縦長
        sub = df
        if "association" in cols:
            sub = df[pd.to_numeric(df[cols["association"]], errors="coerce") == 1]
        pairs = list(zip(sub[cols["word"]], sub[cols["emotion"]]))
    else:
        raise ValueError(
            "NRC辞書の形式を判別できません。日本語の見出し語列（列名に Japanese を含む）"
            "または word,emotion 列が必要です。"
        )
    return _build(pairs, name=f"NRC Emotion Lexicon 日本語版（{path.name}）", source=str(path))


def find_private_dictionaries(folder: str | Path = PRIVATE_DICTIONARY_DIR) -> list[Path]:
    """
    非公開フォルダにある辞書ファイル（.txt / .tsv / .csv）を返す。なければ空。
    NRC日本語辞書（Japanese-NRC-EmoLex.txt）があれば先頭にする。README は辞書ではないので除く。
    """
    folder = Path(folder)
    if not folder.is_dir():
        return []
    found = [p for p in folder.iterdir()
             if p.is_file() and p.suffix.lower() in {".txt", ".tsv", ".csv"}
             and not p.stem.lower().startswith("readme")]
    return sorted(found, key=lambda p: (p.name != NRC_FILENAME, p.name))


@dataclass(frozen=True)
class DictionaryChoice:
    label: str     # 画面に出す名前
    path: str
    kind: str      # "nrc" / "simple"


def dictionary_choices(private_paths: Iterable[Path], public_mode: bool) -> tuple[list[DictionaryChoice], int]:
    """
    画面で選べる辞書と、初期選択の位置を返す。
      - 公開デモ（クラウド）：簡易辞書だけ（NRC辞書本体はクラウドに置かない）
      - ローカル研究版：非公開フォルダのNRC辞書があればそれを初期選択。簡易辞書にも切り替えられる
    """
    simple = DictionaryChoice(SIMPLE_LABEL, str(SIMPLE_DICTIONARY_PATH), "simple")
    if public_mode:
        return [simple], 0
    nrc = [DictionaryChoice(f"NRC日本語辞書（{p.name}）" if p.name == NRC_FILENAME
                            else f"非公開フォルダ：{p.name}（NRC形式）", str(p), "nrc")
           for p in private_paths]
    return nrc + [simple], 0


def add_dictionary_column(df: pd.DataFrame, dictionary_name: str) -> pd.DataFrame:
    """結果のCSVに、使った辞書の名前を残すための列を付ける（元の表は変更しない）。"""
    out = df.copy()
    out["dictionary"] = dictionary_name
    return out


# ---------------------------------------------------------------------------
# 感情分析
# ---------------------------------------------------------------------------
@dataclass
class TextResult:
    scores: dict[str, float]
    neg_score: float
    is_negative: float           # 1.0 / 0.0 / NaN（判定不能）
    n_tokens: int
    matched: list[str] = field(default_factory=list)
    negated: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if math.isnan(self.is_negative):
            return STATUS_UNKNOWN
        return STATUS_NEGATIVE if self.is_negative == 1.0 else STATUS_NOT_NEGATIVE


class SentimentAnalyzer:
    def __init__(self, dictionary: EmotionDictionary, tokenizer: Tokenizer | None = None,
                 neg_threshold: float = 0.5, negation_window: int = 3) -> None:
        self.dictionary = dictionary
        self.tokenizer = tokenizer if tokenizer is not None else JanomeTokenizer()
        self.neg_threshold = neg_threshold
        self.negation_window = negation_window
        self._cache: dict[str, TextResult] = {}

    # --- 内部処理 ---
    def _lookup(self, tokens: list[Token], i: int) -> tuple[int, str, frozenset[str]] | None:
        """位置iから始まる最長の見出し語を探す。戻り値: (語数, 見出し語, 感情)"""
        entries = self.dictionary.entries
        for length in range(min(MAX_PHRASE_TOKENS, len(tokens) - i), 0, -1):
            span = tokens[i:i + length]
            head = "".join(t.surface for t in span[:-1])
            for key in (head + span[-1].base_form, head + span[-1].surface):
                if key in entries:
                    return length, key, entries[key]
        return None

    def _is_negated(self, tokens: list[Token], j: int) -> bool:
        """位置j以降（感情語の直後）が否定になっているか。"""
        for t in tokens[j:j + self.negation_window]:
            if t.surface == "しか":                       # 「〜しかない」は否定ではない
                return False
            if t.base_form in NEGATION_BASES and t.pos in ("助動詞", "形容詞"):
                return True
            if t.pos in ("助詞", "助動詞"):
                continue
            if t.pos == "動詞" and t.base_form in BRIDGE_VERBS:
                continue
            return False                                   # 内容語や記号が来たら打ち切り
        return False

    # --- 公開メソッド ---
    def analyze_text(self, text: str) -> TextResult:
        text = normalize_text(text if isinstance(text, str) else "")
        if text in self._cache:
            return self._cache[text]

        tokens = self.tokenizer.tokenize(text) if text else []
        counts = {e: 0 for e in EMOTIONS}
        matched, negated = [], []
        n_neg_words = 0
        i = 0
        while i < len(tokens):
            hit = self._lookup(tokens, i)
            if hit is None:
                i += 1
                continue
            length, word, emotions = hit
            if self._is_negated(tokens, i + length):
                negated.append(word)
            else:
                matched.append(word)
                for e in emotions:
                    counts[e] += 1
                if emotions & set(NEGATIVE_EMOTIONS):
                    n_neg_words += 1
            i += length

        m = len(matched)
        if m == 0:
            nan = float("nan")
            result = TextResult({e: nan for e in EMOTIONS}, nan, nan, len(tokens), matched, negated)
        else:
            neg_score = n_neg_words / m
            result = TextResult(
                {e: counts[e] / m for e in EMOTIONS},
                neg_score,
                1.0 if neg_score >= self.neg_threshold else 0.0,
                len(tokens), matched, negated,
            )
        self._cache[text] = result
        return result

    def analyze(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """コメント表に感情分析の列を追加して返す（元の表は変更しない）。"""
        out = df.copy()
        results = [self.analyze_text(t) for t in out[text_column].tolist()]
        for e in EMOTIONS:
            out[e] = [r.scores[e] for r in results]
        out["neg_score"] = [r.neg_score for r in results]
        out["is_negative"] = pd.array([r.is_negative for r in results], dtype="float64")
        out["sentiment_status"] = [r.status for r in results]
        out["n_matched"] = [len(r.matched) for r in results]
        out["n_negated"] = [len(r.negated) for r in results]
        out["matched_words"] = [";".join(r.matched) for r in results]
        out["negated_words"] = [";".join(r.negated) for r in results]
        out["dictionary"] = self.dictionary.name          # どの辞書で判定したかをCSVに残す
        return out


def summarize_sentiment(analyzed: pd.DataFrame) -> dict:
    """全体の集計（カバー率＝判定できたコメントの割合 など）。"""
    n = len(analyzed)
    judged = analyzed["is_negative"].notna()
    n_judged = int(judged.sum())
    n_neg = int((analyzed["is_negative"] == 1).sum())
    return {
        "n_comments": n,
        "n_judged": n_judged,
        "n_unknown": n - n_judged,
        "n_negative": n_neg,
        "coverage": n_judged / n if n else float("nan"),
        "neg_rate_all": n_neg / n if n else float("nan"),          # 分母＝全コメント
        "neg_rate_judged": n_neg / n_judged if n_judged else float("nan"),  # 分母＝判定可能
        "n_negated_words": int(analyzed["n_negated"].sum()),
    }


EMOTION_LABELS_JA = {"anger": "怒り", "fear": "恐怖", "sadness": "悲しみ", "joy": "喜び", "trust": "信頼"}


def emotion_summary(analyzed: pd.DataFrame) -> pd.DataFrame:
    """
    感情別の集計。判定不能（辞書に一致する語なし）のコメントは分母に入れない。
      該当コメント数 ＝ その感情の語を1つ以上含むコメント数（score_e > 0）
      割合           ＝ 該当コメント数 ÷ 判定できたコメント数
      平均スコア     ＝ 判定できたコメントでの score_e の平均
    """
    judged = analyzed["is_negative"].notna()
    n_judged = int(judged.sum())
    rows = []
    for e in EMOTIONS:
        sc = pd.to_numeric(analyzed.loc[judged, e], errors="coerce")
        hit = int((sc > 0).sum())
        rows.append({
            "感情": EMOTION_LABELS_JA.get(e, e), "emotion": e,
            "否定的感情": e in NEGATIVE_EMOTIONS,
            "該当コメント数": hit,
            "判定できたコメントに占める割合": hit / n_judged if n_judged else float("nan"),
            "平均スコア（判定できたコメント）": float(sc.mean()) if n_judged else float("nan"),
        })
    return pd.DataFrame(rows)


def evaluate_negative_detection(pred: pd.Series, gold: pd.Series) -> dict:
    """
    手作業の正解（gold: 1=否定的, 0=否定的でない）と予測 is_negative を比べる。
    判定不能（NaN）は「否定的と予測しなかった」として数え、その件数も別に返す。
    """
    gold = pd.to_numeric(gold, errors="coerce")
    ok = gold.notna()
    pred, gold = pred[ok], gold[ok].astype(int)
    pred_pos = pred == 1
    tp = int((pred_pos & (gold == 1)).sum())
    fp = int((pred_pos & (gold == 0)).sum())
    fn = int((~pred_pos & (gold == 1)).sum())
    tn = int((~pred_pos & (gold == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if tp and not math.isnan(precision + recall) else float("nan"))
    return {"n": int(ok.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "n_unknown": int(pred.isna().sum()),
            "coverage": float(pred.notna().mean()) if len(pred) else float("nan")}


def dictionary_stats_csv(dictionaries: Iterable[EmotionDictionary]) -> str:
    """辞書の収録語数を表にしたCSV文字列（論文の表に使う）。"""
    rows = [d.stats() for d in dictionaries]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()
