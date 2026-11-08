"""
公開（STEP 7）に関するテスト：
- 公開版で API 取得を合言葉なしに使えないこと
- 公開してはいけないもの（秘密情報・実データ・NRC辞書・実験結果）が .gitignore で除外されていること
"""
import fnmatch
from pathlib import Path

from src.access import api_fetch_allowed, is_public_mode, read_setting

ROOT = Path(__file__).resolve().parent.parent


def test_local_mode_allows_api():
    assert api_fetch_allowed(False, "", "") == (True, "")


def test_public_mode_requires_passcode():
    ok, msg = api_fetch_allowed(True, "", "")
    assert not ok and "無効" in msg                      # 合言葉が未設定なら使えない
    assert not api_fetch_allowed(True, "secret", "")[0]
    assert not api_fetch_allowed(True, "secret", "wrong")[0]
    assert api_fetch_allowed(True, "secret", "secret")[0]


def test_settings_from_env_and_secrets(monkeypatch):
    monkeypatch.delenv("PUBLIC_MODE", raising=False)
    assert not is_public_mode({})
    assert is_public_mode({"PUBLIC_MODE": "true"})
    monkeypatch.setenv("PUBLIC_MODE", "1")
    assert is_public_mode(None)
    assert read_setting("APP_PASSCODE", {"APP_PASSCODE": " abc "}) == "abc"

    class Broken:
        def get(self, *a):
            raise FileNotFoundError

    assert read_setting("APP_PASSCODE", Broken()) == ""


def _ignored(path: str, patterns: list[str]) -> bool:
    """.gitignore の簡易判定（このプロジェクトで使っている書き方だけに対応）。"""
    ignored = False
    for pat in patterns:
        neg = pat.startswith("!")
        p = pat[1:] if neg else pat
        hit = (fnmatch.fnmatch(path, p) or fnmatch.fnmatch(path, p.rstrip("/") + "/*")
               or path.startswith(p.rstrip("*")) and p.endswith(("/", "/*"))
               or fnmatch.fnmatch(Path(path).name, p))
        if hit:
            ignored = not neg
    return ignored


def test_private_files_are_not_published():
    patterns = [line.strip() for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")]
    must_ignore = [
        ".streamlit/secrets.toml", ".env",
        "data/research/comments_XXXXXXXXXXX_20261107_000000.csv", "data/research/labels.csv",
        "data/research/sentiment_gold.csv",
        "data/dictionaries/private/Japanese-NRC-EmoLex.txt",
        "results/test/test_summary.csv", ".venv/Lib/x.py",
    ]
    for path in must_ignore:
        assert _ignored(path, patterns), f"{path} が公開対象になっています"
    must_publish = ["app.py", "data/sample/sample_comments.csv", "data/sentiment_dictionary.csv",
                    "experiments/protocol/labels_sha256.txt", "data/research/.gitkeep"]
    for path in must_publish:
        assert not _ignored(path, patterns), f"{path} が除外されています"
