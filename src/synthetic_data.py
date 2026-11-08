"""
動作確認用の【架空データ】を生成するモジュール。

※ ここで作るデータは実在の動画・企業・人物とは無関係の架空データです。
※ 研究の評価（卒論の実験結果）には使いません。アプリとテストの動作確認専用です。

生成する3本の架空動画:
  SYN_FIRE01   : 公開60時間後に「炎上」が起きる動画（正解ラベルあり）
  SYN_NORMAL01 : 炎上しない通常の動画
  SYN_NEGNEWS1 : 最初から否定的コメントが多いニュース動画。公開72時間後に続報で
                 コメント数が増えるが、否定的コメントの割合は変わらない（炎上ではない）
                 （「話題が否定的」「投稿数だけが増える」と「炎上」は別物であることを確認するため）

STEP 4 で、公開2日目以降の投稿数を1時間あたり約3件 → 約15件に増やした
（約3件だと、ほぼすべての時間帯が最低コメント数10件に届かず検知の比較ができないため）。

コメント数のモデル（非一様ポアソン過程）:
  λ(h) = a·exp(-h/τ) + b                   … 公開直後に多く、だんだん減る
       + c·exp(-(h-h0)/τ2)  (h ≥ h0 のとき) … 炎上による急増
  h は公開からの経過時間（時間）
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_SOURCE_SYNTHETIC, JST

# 架空の企業名（実在しない名前を使う）
_COMPANY = "架空商事"

_NEUTRAL = [
    "動画ありがとうございます", "今北産業", "最後まで見ました", "このあとどうなるんだろう",
    "初見です", "解説わかりやすい", "続報待ってます", "ニュースで見ました",
]
_POSITIVE = [
    "応援しています", "信頼できる会社だと思う", "嬉しいニュースですね", "安心しました",
    "素晴らしい対応", "楽しみにしています", "感動しました",
]
_NEGATIVE = [
    f"{_COMPANY}には本当に腹が立つ", "最悪の対応だ", "怖くてもう使えない", "悲しい気持ちになった",
    "許せない", "不安しかない", "説明が不誠実で怒りしかない", "がっかりした", "ひどすぎる",
]
# 感情分析の限界を確認するための例（否定表現・皮肉）
_TRICKY = ["最悪ではない", "さすがですね（笑）", "怖いくらい対応が早い"]


def _hourly_rate(hours: np.ndarray, a: float, tau: float, b: float,
                 fire_hour: float | None, c: float, tau2: float) -> np.ndarray:
    rate = a * np.exp(-hours / tau) + b
    if fire_hour is not None:
        after = hours >= fire_hour
        rate = rate + np.where(after, c * np.exp(-(hours - fire_hour) / tau2), 0.0)
    return rate


def _make_video(rng: np.random.Generator, video_id: str, upload_time: pd.Timestamp,
                n_hours: int, a: float, tau: float, b: float,
                fire_hour: float | None, c: float, tau2: float,
                p_neg_base: float, p_neg_fire: float) -> pd.DataFrame:
    """1本の架空動画のコメント表を作る。"""
    hours = np.arange(n_hours, dtype=float)
    counts = rng.poisson(_hourly_rate(hours, a, tau, b, fire_hour, c, tau2))

    rows = []
    top_level_ids: list[str] = []
    serial = 0
    for h, n in zip(hours.astype(int), counts):
        in_fire = fire_hour is not None and fire_hour <= h < fire_hour + 24
        p_neg = p_neg_fire if in_fire else p_neg_base
        for _ in range(n):
            serial += 1
            cid = f"{video_id}_{serial:05d}"
            r = rng.random()
            if r < p_neg:
                text = rng.choice(_NEGATIVE)
            elif r < p_neg + 0.03:
                text = rng.choice(_TRICKY)
            elif r < p_neg + 0.25:
                text = rng.choice(_POSITIVE)
            else:
                text = rng.choice(_NEUTRAL)

            # 2割程度を返信コメントにする（親は既存のトップレベルコメント）
            is_reply = bool(top_level_ids) and rng.random() < 0.2
            parent_id = rng.choice(top_level_ids) if is_reply else ""
            if not is_reply:
                top_level_ids.append(cid)

            ts = upload_time + pd.Timedelta(hours=int(h)) + pd.Timedelta(
                seconds=int(rng.integers(0, 3600)))
            rows.append({
                "comment_id": cid,
                "published_at": ts,
                "text": f"【架空】{text}",
                "like_count": int(rng.geometric(0.3) - 1),
                "parent_id": parent_id,
                "is_reply": is_reply,
                "video_id": video_id,
                "data_source": DATA_SOURCE_SYNTHETIC,
            })
    df = pd.DataFrame(rows)
    # 返信の投稿時刻が親コメントより前にならないようにそろえる。
    # 親は同じ時間帯かそれ以前に作られているので、同じ時間帯に収まり集計結果は変わらない
    if len(df):
        parent_time = df.set_index("comment_id")["published_at"]
        is_reply = df["is_reply"].astype(bool)
        parents = df.loc[is_reply, "parent_id"].map(parent_time)
        df.loc[is_reply, "published_at"] = np.maximum(df.loc[is_reply, "published_at"], parents)
    return df


def generate_sample_dataset(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    架空のコメントデータと、その正解ラベルを返す。
    seed を固定すれば毎回同じデータになる（再現性）。
    """
    rng = np.random.default_rng(seed)
    upload = pd.Timestamp("2026-01-05 18:00", tz=JST)

    fire = _make_video(rng, "SYN_FIRE01", upload, 120, a=40, tau=8, b=15,
                       fire_hour=60, c=60, tau2=10, p_neg_base=0.12, p_neg_fire=0.65)
    normal = _make_video(rng, "SYN_NORMAL01", upload, 120, a=35, tau=10, b=14,
                         fire_hour=None, c=0, tau2=1, p_neg_base=0.10, p_neg_fire=0.10)
    # 続報による投稿増（fire_hour=72）。否定率は増加の前後で同じ 0.55
    negnews = _make_video(rng, "SYN_NEGNEWS1", upload, 120, a=50, tau=8, b=16,
                          fire_hour=72, c=45, tau2=8, p_neg_base=0.55, p_neg_fire=0.55)
    comments = pd.concat([fire, normal, negnews], ignore_index=True)

    # 正解ラベル（架空データなので「生成時に決めた値」がそのまま正解になる）
    labels = pd.DataFrame([
        {"video_id": "SYN_FIRE01", "incident_id": "SYN_INC01", "is_fire": 1,
         "fire_start_jst": (upload + pd.Timedelta(hours=60)).isoformat(),
         "label_source": "synthetic_generator", "split": "demo"},
        {"video_id": "SYN_NORMAL01", "incident_id": "SYN_INC02", "is_fire": 0,
         "fire_start_jst": "", "label_source": "synthetic_generator", "split": "demo"},
        {"video_id": "SYN_NEGNEWS1", "incident_id": "SYN_INC03", "is_fire": 0,
         "fire_start_jst": "", "label_source": "synthetic_generator", "split": "demo"},
    ])
    return comments, labels


def generate_evaluation_dataset(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    【架空】評価手順の動作確認用のデータセット（16本）。研究結果には使わない。
      炎上 6本（うち2本は同じ事件を別チャンネルが報じた設定＝incident_id が同じ）
      通常 5本、否定的な話題のニュース 5本（うち3本は続報で投稿数だけ増える）
    炎上の大きさ・開始時刻・投稿数の水準を動画ごとに変えている。
    """
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2026-02-02 18:00", tz=JST)
    videos, labels = [], []
    specs = []
    for i in range(6):   # 炎上
        inc = "SYN_EV_INC_F1" if i in (0, 1) else f"SYN_EV_INC_F{i}"
        specs.append(dict(vid=f"SYN_EV_FIRE{i + 1:02d}", inc=inc, fire=True,
                          fire_hour=int(rng.integers(40, 90)), c=float(rng.uniform(20, 70)),
                          p_neg_base=float(rng.uniform(0.08, 0.2)), p_neg_fire=float(rng.uniform(0.45, 0.7))))
    for i in range(5):   # 通常
        specs.append(dict(vid=f"SYN_EV_NORM{i + 1:02d}", inc=f"SYN_EV_INC_N{i + 1}", fire=False,
                          fire_hour=None, c=0.0, p_neg_base=float(rng.uniform(0.05, 0.15)), p_neg_fire=0.0))
    for i in range(5):   # 否定的な話題（3本は続報で投稿数だけ増える）
        surge = i < 3
        p = float(rng.uniform(0.45, 0.6))
        specs.append(dict(vid=f"SYN_EV_NEGN{i + 1:02d}", inc=f"SYN_EV_INC_G{i + 1}", fire=False,
                          fire_hour=int(rng.integers(40, 90)) if surge else None,
                          c=float(rng.uniform(25, 50)) if surge else 0.0, p_neg_base=p, p_neg_fire=p))
    for k, sp in enumerate(specs):
        upload = base + pd.Timedelta(days=7 * k)
        v = _make_video(rng, sp["vid"], upload, 120, a=float(rng.uniform(30, 50)), tau=8,
                        b=float(rng.uniform(12, 18)), fire_hour=sp["fire_hour"], c=sp["c"], tau2=10,
                        p_neg_base=sp["p_neg_base"],
                        p_neg_fire=sp["p_neg_fire"] if sp["fire"] or sp["fire_hour"] is not None
                        else sp["p_neg_base"])
        videos.append(v)
        labels.append({"video_id": sp["vid"], "incident_id": sp["inc"], "is_fire": int(sp["fire"]),
                       "fire_start_jst": (upload + pd.Timedelta(hours=sp["fire_hour"])).isoformat()
                       if sp["fire"] else "",
                       "label_source": "synthetic_generator", "split": ""})
    return pd.concat(videos, ignore_index=True), pd.DataFrame(labels)


def main(out_dir: str | Path = "data/sample") -> None:
    """data/sample/ に架空データのCSVを書き出す。"""
    from src.preprocessing import save_comments_csv

    out_dir = Path(out_dir)
    comments, labels = generate_sample_dataset()
    save_comments_csv(comments, out_dir / "sample_comments.csv")
    labels.to_csv(out_dir / "sample_labels.csv", index=False, encoding="utf-8-sig")
    print(f"架空データを書き出しました: {len(comments)}件 → {out_dir}")
    print(comments.groupby("video_id").size().to_string())


if __name__ == "__main__":
    # 実行方法（プロジェクトのルートで）:  python -m src.synthetic_data
    main()
