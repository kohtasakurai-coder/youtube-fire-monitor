"""
公開（Streamlit Community Cloud）用の安全設定。

公開版では、誰でもアプリを開けるため、次のようにする。
  - PUBLIC_MODE=true のとき、YouTube APIでの取得は「合言葉（APP_PASSCODE）」を入れた人だけが使える
    （APIキーの1日の利用上限を他人に使い切られないようにするため）
  - 合言葉が設定されていなければ、公開版ではAPI取得を使えない（CSVと架空データだけで動く）
  - 手元（PUBLIC_MODE なし）では今までどおり
設定は環境変数または .streamlit/secrets.toml（Cloud では Settings → Secrets）に書く。
"""
from __future__ import annotations

import hmac
import os


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def read_setting(name: str, secrets=None) -> str:
    """環境変数 → Streamlit Secrets の順に設定値を読む（なければ空文字）。"""
    value = os.environ.get(name, "")
    if value:
        return value.strip()
    if secrets is not None:
        try:
            return str(secrets.get(name, "")).strip()
        except Exception:   # secrets.toml がない場合など
            return ""
    return ""


def is_public_mode(secrets=None) -> bool:
    return _truthy(read_setting("PUBLIC_MODE", secrets))


def api_fetch_allowed(public_mode: bool, configured_passcode: str, entered: str) -> tuple[bool, str]:
    """
    API取得を許可するか。戻り値: (許可するか, 画面に出す説明)
    合言葉の比較は、文字の一致位置から推測されないよう hmac.compare_digest を使う。
    """
    if not public_mode:
        return True, ""
    if not configured_passcode:
        return False, "公開版ではYouTubeからの取得は無効です。CSVの読み込みか架空データでお試しください。"
    if not entered:
        return False, "公開版でYouTubeから取得するには合言葉が必要です。"
    if hmac.compare_digest(entered.encode("utf-8"), configured_passcode.encode("utf-8")):
        return True, ""
    return False, "合言葉が違います。"
