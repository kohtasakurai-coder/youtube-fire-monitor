"""人手評価用の候補選び（src/gold_sampling.py）のテスト。"""
import pandas as pd

from src.gold_sampling import GOLD_COLUMNS, sample_gold_candidates


def comments(video, n, reply_parent=None, text=None):
    return pd.DataFrame({
        "comment_id": [f"{video}_{i}" for i in range(n)],
        "video_id": video,
        "text": [text or f"{video}のコメント{i}" for i in range(n)],
        "is_reply": reply_parent is not None,
        "parent_id": reply_parent or "",
    })


def test_no_prediction_columns_and_caps():
    df = pd.concat([comments("V1", 50), comments("V2", 50), comments("V3", 10, reply_parent="P")])
    out, breakdown = sample_gold_candidates(df, n=60, per_video_cap=25, per_thread_cap=3)
    assert list(out.columns) == GOLD_COLUMNS                       # 予測・スコアの列はない
    assert not {"is_negative", "neg_score", "anger", "matched_words"} & set(out.columns)
    counts = out["video_id"].value_counts()
    assert counts["V1"] == 25 and counts["V2"] == 25 and counts["V3"] == 3   # 1スレッドは3件まで
    assert len(out) == 53 and (out["gold_negative"] == "").all()


def test_exclude_duplicates_and_stability():
    base = pd.concat([comments("V1", 30), comments("V2", 5, text="同じ本文")])
    out1, _ = sample_gold_candidates(base, n=20, per_video_cap=10, exclude_ids={"V1_0", "V1_1"})
    assert not {"V1_0", "V1_1"} & set(out1["comment_id"])
    assert (out1["video_id"] == "V2").sum() == 1                   # 同じ本文は1件
    more = pd.concat([base, comments("V9", 30)])
    out2, _ = sample_gold_candidates(more, n=40, per_video_cap=10, exclude_ids={"V1_0", "V1_1"})
    # 動画を足しても、既に選んだ V1 の候補は変わらない
    assert set(out1.loc[out1.video_id == "V1", "comment_id"]) == set(out2.loc[out2.video_id == "V1", "comment_id"])
