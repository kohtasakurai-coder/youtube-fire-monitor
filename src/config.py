"""
研究全体で共有する設定値（パラメータ）をまとめたモジュール。

卒論の再現性のため、「どの値で実験したか」を1か所で管理する。
アプリ画面や実験スクリプトで値を変更するときは、
このクラスのインスタンスを作り直して各処理に渡す。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

# 日本時間（JST）のタイムゾーン名。投稿日時はすべてこの時刻に変換して扱う。
JST = "Asia/Tokyo"

# コメントデータ（CSV）の必須列。YouTube APIから取得したデータも、
# 読み込んだCSVも、最終的にこの列構成にそろえる。
REQUIRED_COLUMNS = [
    "comment_id",    # コメントID（YouTube側の一意なID）
    "published_at",  # 投稿日時
    "text",          # コメント本文
    "like_count",    # いいね数
    "parent_id",     # 親コメントID（トップレベルコメントは空欄）
    "is_reply",      # 返信コメントかどうか（True/False）
]

# あってもなくてもよい列。
OPTIONAL_COLUMNS = [
    "video_id",      # 動画ID
    "data_source",   # "synthetic"（架空データ）または "real"（実データ）
]

# 架空データと実データを区別するためのラベル。
DATA_SOURCE_SYNTHETIC = "synthetic"
DATA_SOURCE_REAL = "real"

# 分析に使う5種類の感情。
EMOTIONS = ["anger", "fear", "sadness", "joy", "trust"]
NEGATIVE_EMOTIONS = ["anger", "fear", "sadness"]


@dataclass
class AnalysisConfig:
    """分析パラメータ一式（初期値は依頼仕様に合わせている）。"""

    # --- コメント取得 ---
    max_comments: int = 2000          # 取得するコメントの最大件数

    # --- 時系列 ---
    freq: str = "1h"                  # 集計時間幅（"30min", "1h", "3h" など）
    moving_average_window: int = 6    # 移動平均の期間（ビン数）
    min_comments_reliable: int = 10   # これ未満のビンは「信頼性が低い」と表示

    # --- 手法A：否定的コメント率 ---
    neg_baseline_hours: int = 24      # 比較に使う過去期間（時間）
    neg_rate_increase: float = 0.15   # 過去平均からの増加幅（0.15 = 15ポイント）
    neg_min_comments: int = 10        # 判定に必要な最低コメント数
    # [仮置き] 基準期間（直前24時間）の合計コメント数の下限。設計書の「Σn_s ≥ n_min」をそのまま採用
    neg_min_baseline_comments: int = 10
    # [追加オプション・初期値はオフ] 二項検定の有意水準。None なら使わない（設計書どおり）。
    # 例：0.01 にすると「直前24時間の否定率 p0 のもとで、n_t 件中 n_t^neg 件以上が否定的に
    # なる確率 ≤ 0.01」も検知の条件に加える。件数が少ない時間帯の偶然の上振れを抑えるための比較用
    neg_binom_alpha: float | None = None

    # --- 手法B：Zスコア ---
    z_baseline_hours: int = 24        # μ, σ を計算する過去期間（時間）
    z_threshold: float = 3.0          # Zスコアの閾値（Z がこれを「超えた」ら異常）
    z_sigma_floor: float = 1.0        # σ がこれより小さいときの代わりの値（σ=0 のゼロ除算回避）
    # [仮置き] 手法Bでも最低コメント数を要求する。深夜など0〜1件が続いたあとの数件で
    # Zが極端に大きくなる誤警報を防ぐため。手法Aと同じ10件にそろえた
    z_min_comments: int = 10
    z_transform: str = "none"         # "none" または "log1p"（公開直後の増加を和らげる比較用）

    # --- 手法C：変化点検出（PELT） ---
    pelt_model: str = "rbf"           # "rbf", "l2", "l1" など
    # [仮置き・設計書の10から変更] ペナルティ値（大きいほど変化点が減る）。
    # rbf では1区間のコストが区間の長さ以下なので、10だと10点以上の証拠がたまるまで
    # 変化点にならず検知が遅れる（架空データでは炎上後も検知しなかった）。3にすると
    # 架空の炎上動画で2時間後に検知し、非炎上の架空動画では検知なし。
    # 本番の値は調整用（dev）データで決めること（実験3）
    pelt_penalty: float = 3.0
    pelt_window_hours: int = 72       # 各時刻で使う直近データの長さ（計算量対策）
    pelt_recent_hours: int = 3        # 直近何時間以内の変化点を「今の検知」とみなすか
    # [仮置き] 変化前後の平均の差がこれ以上の「上昇」だけを検知にする。
    # 手法Aと同じ15ポイントにそろえ、ごく小さな水準変化を炎上の兆候とみなさない
    pelt_min_increase: float = 0.15
    pelt_min_size: int = 2            # 1区間の最小の長さ（ビン数。rupturesの min_size）

    # --- 警戒レベル ---
    alert_lookback_hours: int = 3     # 直近何時間の検知を警戒レベルに反映するか
    red_min_methods: int = 2          # 何手法が同時検知したら「警戒（赤）」にするか

    # 追加のメモ（実験条件の記録用）
    note: str = field(default="")

    def to_dict(self) -> dict:
        """実験結果CSVに条件を書き出すときに使う。"""
        return asdict(self)
