"""
前処理（URL解析・CSV読み込み）と架空データ生成のテスト。
実行方法（プロジェクトのルートで）:  python -m pytest
"""
import pandas as pd
import pytest

from src.preprocessing import (
    CommentCSVError,
    InvalidYouTubeURLError,
    comments_to_csv_bytes,
    describe_data_source,
    extract_video_id,
    load_comments_csv,
)
from src.synthetic_data import generate_sample_dataset

VID = "dQw4w9WgXcQ"


# ---------------------------------------------------------------- URL解析
@pytest.mark.parametrize("url", [
    f"https://www.youtube.com/watch?v={VID}",
    f"https://youtube.com/watch?v={VID}&t=30s",
    f"https://m.youtube.com/watch?feature=share&v={VID}",
    f"https://youtu.be/{VID}",
    f"https://youtu.be/{VID}?si=abcdef",
    f"https://www.youtube.com/shorts/{VID}",
    f"https://youtube.com/shorts/{VID}?feature=share",
    f"https://www.youtube.com/embed/{VID}",
    f"https://www.youtube.com/live/{VID}",
    f"www.youtube.com/watch?v={VID}",   # https:// 省略
    f"  {VID}  ",                        # IDだけ（前後に空白）
])
def test_extract_video_id_valid(url):
    assert extract_video_id(url) == VID


@pytest.mark.parametrize("url", [
    "",
    None,
    "https://example.com/watch?v=dQw4w9WgXcQ",   # YouTube以外
    "https://www.youtube.com/watch?v=short",      # 11文字でない
    "https://www.youtube.com/channel/UCxxxx",     # チャンネルURL
    "https://youtu.be/",                          # IDなし
    "これはURLではありません",
])
def test_extract_video_id_invalid(url):
    with pytest.raises(InvalidYouTubeURLError):
        extract_video_id(url)


# ---------------------------------------------------------------- CSV読み込み
HEADER = "comment_id,published_at,text,like_count,parent_id,is_reply\n"


def test_load_csv_converts_utc_to_jst():
    csv = HEADER + "c1,2026-01-01T00:30:00Z,テスト,3,,False\n"
    df, _ = load_comments_csv(csv.encode("utf-8"))
    ts = df.loc[0, "published_at"]
    assert str(ts.tz) == "Asia/Tokyo"
    assert ts.hour == 9 and ts.minute == 30   # UTC 0:30 → JST 9:30


def test_load_csv_naive_time_warns():
    csv = HEADER + "c1,2026-01-01 00:30:00,テスト,0,,False\n"
    df, warnings = load_comments_csv(csv.encode("utf-8"), naive_timezone="Asia/Tokyo")
    assert df.loc[0, "published_at"].hour == 0
    assert any("タイムゾーン情報がない" in w for w in warnings)


def test_load_csv_cp932_excel_file():
    """WindowsのExcelで保存したShift_JIS（cp932）のCSVも読めること。"""
    csv = HEADER + "c1,2026-01-01T00:00:00+09:00,最悪だ,0,,False\n"
    df, _ = load_comments_csv(csv.encode("cp932"))
    assert df.loc[0, "text"] == "最悪だ"


def test_load_csv_missing_column_raises():
    with pytest.raises(CommentCSVError):
        load_comments_csv("comment_id,text\nc1,abc\n".encode("utf-8"))


def test_load_csv_empty_does_not_crash():
    df, warnings = load_comments_csv(HEADER.encode("utf-8"))
    assert df.empty
    assert warnings


def test_load_csv_cleans_bad_rows():
    csv = (HEADER
           + "c1,2026-01-01T00:00:00Z,a,x,,true\n"     # いいね数が数字でない → 0
           + "c1,2026-01-01T01:00:00Z,dup,1,,False\n"  # ID重複 → 除外
           + "c2,not-a-date,b,1,,False\n"              # 日時不正 → 除外
           + "c3,2026-01-01T02:00:00Z,,2,c1,1\n")      # 本文空欄 → ""
    df, warnings = load_comments_csv(csv.encode("utf-8"))
    assert list(df["comment_id"]) == ["c1", "c3"]
    assert df.loc[0, "like_count"] == 0
    assert bool(df.loc[0, "is_reply"]) is True
    assert df.loc[1, "text"] == ""
    assert len(warnings) >= 2


def test_csv_roundtrip_keeps_values():
    comments, _ = generate_sample_dataset(seed=1)
    df1, _ = load_comments_csv(comments_to_csv_bytes(comments))
    df2, _ = load_comments_csv(comments_to_csv_bytes(df1))
    pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------- 架空データ
def test_synthetic_data_is_reproducible_and_labeled():
    a, labels = generate_sample_dataset(seed=7)
    b, _ = generate_sample_dataset(seed=7)
    pd.testing.assert_frame_equal(a, b)                 # 同じseedなら同じデータ
    assert describe_data_source(a) == "synthetic"       # 架空データだと判別できる
    assert a["text"].str.startswith("【架空】").all()
    assert set(labels["video_id"]) == set(a["video_id"])
    # 返信コメントの親は、必ず同じ動画内に存在するトップレベルコメント
    replies = a[a["is_reply"]]
    top = a[~a["is_reply"]].set_index("comment_id")["video_id"]
    assert (replies["parent_id"].map(top) == replies["video_id"]).all()
    # 返信は親コメントより前の時刻にならない
    parent_time = a.set_index("comment_id")["published_at"]
    assert (replies["published_at"] >= replies["parent_id"].map(parent_time)).all()
