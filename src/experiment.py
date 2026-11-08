"""
評価実験の実行と手順（プロトコル）の管理（STEP 6）。

手順（評価用の結果を見てから閾値を調整しないため）:
  1. split          … 事件（incident_id）単位で dev（調整用）／test（評価用）に分ける
  2. freeze-labels  … 正解ラベルCSVの SHA-256 を記録する（以後ラベルを変えると検出できる）
  3. tune           … dev の動画だけでパラメータを総当たりし、事前に決めた規則で1つ選んで固定する
                      （test の動画は読み込みもしない）
  4. test           … 固定した設定で test の動画を1回だけ評価する
                      2回目以降・test 後の再調整は理由の記録なしには実行できない
すべての操作は experiments/protocol/protocol_log.csv に時刻とラベルのハッシュ付きで記録する。
このフォルダには実コメントは保存されない（設定値・ハッシュ・件数だけ）。

パラメータの選び方（事前に決めた規則）：
  検知器「警戒」の dev での F1 が最大 → 同点なら 100時間あたり誤警戒が少ない → 遅れの中央値が小さい
  → それでも同点ならグリッドの順番が早いもの
"""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import AnalysisConfig
from src.detection import run_detection
from src.evaluation import (
    EvalConfig,
    evaluate_offline_reference,
    evaluate_video,
    mcnemar_tests,
    summarize,
)
from src.time_series import aggregate_time_series

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_DIR = PROJECT_ROOT / "experiments" / "protocol"
SELECTION_DETECTOR = "警戒"

DEFAULT_GRID = {
    "freq": ["1h"],
    "neg_rate_increase": [0.10, 0.15, 0.20],
    "neg_binom_alpha": [None, 0.01],
    "z_threshold": [2.0, 2.5, 3.0],
    "pelt_penalty": [1.0, 3.0, 10.0],
}


class ProtocolError(RuntimeError):
    """評価の手順に反する操作（ラベルの変更、test 後の再調整など）。"""


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# パラメータの組み合わせ
# ---------------------------------------------------------------------------
def grid_configs(grid: dict, base: AnalysisConfig | None = None) -> list[tuple[dict, AnalysisConfig]]:
    """グリッド（辞書）→ [(パラメータの辞書, AnalysisConfig), ...]（順番は固定）。"""
    base = base or AnalysisConfig()
    keys = list(grid.keys())
    out = []
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))
        cfg_kwargs = {k: v for k, v in params.items() if k != "freq"}
        cfg = AnalysisConfig(**{**base.to_dict(), **cfg_kwargs})
        out.append((params, cfg))
    return out


# ---------------------------------------------------------------------------
# データの準備
# ---------------------------------------------------------------------------
@dataclass
class PreparedData:
    labels: pd.DataFrame
    analyzed: dict                       # video_id → 感情分析済みコメント表
    series: dict = field(default_factory=dict)   # (video_id, freq) → 時系列表

    def ts(self, video_id: str, freq: str, ma_window: int, min_comments: int) -> pd.DataFrame:
        key = (video_id, freq, ma_window, min_comments)
        if key not in self.series:
            self.series[key] = aggregate_time_series(self.analyzed[video_id], freq=freq,
                                                     ma_window=ma_window, min_comments=min_comments)
        return self.series[key]


def load_comments_for_labels(labels: pd.DataFrame, root: Path = PROJECT_ROOT) -> dict:
    """ラベルの comments_file 列のCSVを読み込み、動画IDごとのコメント表を返す。"""
    from src.preprocessing import load_comments_csv

    if "comments_file" not in labels.columns:
        raise ValueError("正解ラベルに comments_file 列（コメントCSVのパス）が必要です。")
    out = {}
    for _, row in labels.iterrows():
        path = Path(row["comments_file"])
        path = path if path.is_absolute() else root / path
        if not path.exists():
            raise FileNotFoundError(f"{row['video_id']} のコメントCSVがありません: {path}")
        df, _ = load_comments_csv(path)
        if "video_id" in df.columns and (df["video_id"] == row["video_id"]).any():
            df = df[df["video_id"] == row["video_id"]]
        out[row["video_id"]] = df
    return out


def prepare(labels: pd.DataFrame, comments_by_video: dict, analyzer) -> PreparedData:
    analyzed = {v: analyzer.analyze(comments_by_video[v]) for v in labels["video_id"]}
    return PreparedData(labels.reset_index(drop=True), analyzed)


# ---------------------------------------------------------------------------
# 評価の実行
# ---------------------------------------------------------------------------
def evaluate_config(data: PreparedData, cfg: AnalysisConfig, freq: str,
                    eval_cfg: EvalConfig | None = None, cp_cache: dict | None = None
                    ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """1つの設定で全動画を検知・評価する。戻り値: (動画×検知器の表, 事後分析の参考表, 計算時間)"""
    eval_cfg = eval_cfg or EvalConfig()
    per_video, offline, seconds = [], [], {"A": 0.0, "B": 0.0, "C": 0.0}
    for _, label in data.labels.iterrows():
        ts = data.ts(label["video_id"], freq, cfg.moving_average_window, cfg.min_comments_reliable)
        res = run_detection(ts, cfg, cp_cache=cp_cache, cache_prefix=(label["video_id"],))
        per_video.append(evaluate_video(res.table, label, eval_cfg))
        offline.append(evaluate_offline_reference(res.offline_cps, label))
        for k in seconds:
            seconds[k] += res.seconds.get(k, 0.0)
    return pd.concat(per_video, ignore_index=True), pd.DataFrame(offline), seconds


def run_grid(data: PreparedData, grid: dict, eval_cfg: EvalConfig | None = None,
             base: AnalysisConfig | None = None, n_boot: int = 200) -> pd.DataFrame:
    """グリッドの全設定を評価し、設定×検知器の要約表を返す（config_id で設定を区別）。"""
    cache: dict = {}
    rows = []
    for cid, (params, cfg) in enumerate(grid_configs(grid, base)):
        per_video, _, seconds = evaluate_config(data, cfg, params["freq"], eval_cfg, cache)
        summ = summarize(per_video, n_boot=n_boot)
        for k, v in params.items():
            summ[k] = "なし" if v is None else v
        summ.insert(0, "config_id", cid)
        summ["計算時間A(s)"], summ["計算時間B(s)"], summ["計算時間C(s)"] = (
            seconds["A"], seconds["B"], seconds["C"])
        rows.append(summ)
    return pd.concat(rows, ignore_index=True)


def select_config(grid_summary: pd.DataFrame, detector: str = SELECTION_DETECTOR) -> int:
    """事前に決めた規則で config_id を1つ選ぶ（docstring 参照）。"""
    d = grid_summary[grid_summary["detector"] == detector].copy()
    if d.empty:
        raise ValueError(f"検知器 {detector} の結果がありません。")
    d["_f1"] = d["F1"].fillna(-1)
    d["_fa"] = d["誤警戒_100時間あたり"].fillna(np.inf)
    d["_delay"] = d["遅れ_中央値(h)"].fillna(np.inf)
    d = d.sort_values(["_f1", "_fa", "_delay", "config_id"], ascending=[False, True, True, True])
    return int(d.iloc[0]["config_id"])


# ---------------------------------------------------------------------------
# 手順の記録とガード
# ---------------------------------------------------------------------------
class Protocol:
    def __init__(self, directory: Path = PROTOCOL_DIR):
        self.dir = Path(directory)
        self.labels_hash_file = self.dir / "labels_sha256.txt"
        self.frozen_file = self.dir / "frozen_config.json"
        self.test_done_file = self.dir / "test_done.json"
        self.log_file = self.dir / "protocol_log.csv"

    def log(self, action: str, labels_hash: str = "", note: str = "") -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        row = pd.DataFrame([{"timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                             "action": action, "labels_sha256": labels_hash, "note": note}])
        row.to_csv(self.log_file, mode="a", header=not self.log_file.exists(), index=False,
                   encoding="utf-8")

    def freeze_labels(self, labels_path) -> str:
        h = sha256_file(labels_path)
        if self.labels_hash_file.exists() and self.labels_hash_file.read_text().strip() != h:
            if self.test_done_file.exists():
                raise ProtocolError("test の評価後に正解ラベルを固定し直すことはできません。")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.labels_hash_file.write_text(h + "\n", encoding="utf-8")
        self.log("freeze-labels", h, str(labels_path))
        return h

    def check_labels(self, labels_path) -> str:
        if not self.labels_hash_file.exists():
            raise ProtocolError("先に freeze-labels で正解ラベルを固定してください。")
        h = sha256_file(labels_path)
        if h != self.labels_hash_file.read_text().strip():
            raise ProtocolError("正解ラベルが固定したときから変更されています（SHA-256 不一致）。"
                                "変更した理由を記録し、test 前であれば freeze-labels をやり直してください。")
        return h

    def save_frozen_config(self, payload: dict, labels_hash: str, after_test_reason: str = "") -> None:
        if self.test_done_file.exists() and not after_test_reason:
            raise ProtocolError("test の評価後に設定を選び直すことはできません"
                                "（どうしても必要なら理由を --after-test-reason に書いて記録します）。")
        payload = {**payload, "labels_sha256": labels_hash,
                   "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                   "tuned_after_test": bool(after_test_reason), "after_test_reason": after_test_reason}
        self.dir.mkdir(parents=True, exist_ok=True)
        self.frozen_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                                    encoding="utf-8")
        self.log("tune（設定を固定）", labels_hash,
                 f"config_id={payload.get('config_id')}" + (f" 理由:{after_test_reason}" if after_test_reason else ""))

    def load_frozen_config(self) -> dict:
        if not self.frozen_file.exists():
            raise ProtocolError("固定された設定がありません。先に tune を実行してください。")
        return json.loads(self.frozen_file.read_text(encoding="utf-8"))

    def start_test(self, labels_hash: str, rerun_reason: str = "") -> dict:
        frozen = self.load_frozen_config()
        if frozen["labels_sha256"] != labels_hash:
            raise ProtocolError("設定を固定したときと正解ラベルが違います。")
        if self.test_done_file.exists() and not rerun_reason:
            raise ProtocolError("test はすでに1回実行されています。再実行するなら理由を --rerun-reason に書いてください"
                                "（記録に残り、論文で報告する必要があります）。")
        return frozen

    def finish_test(self, labels_hash: str, frozen: dict, rerun_reason: str = "") -> None:
        record = {"finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                  "labels_sha256": labels_hash, "config_id": frozen.get("config_id"),
                  "rerun_reason": rerun_reason}
        history = []
        if self.test_done_file.exists():
            history = json.loads(self.test_done_file.read_text(encoding="utf-8")).get("runs", [])
        self.test_done_file.write_text(json.dumps({"runs": history + [record]}, ensure_ascii=False,
                                                  indent=2), encoding="utf-8")
        self.log("test", labels_hash, f"config_id={frozen.get('config_id')}"
                 + (f" 再実行理由:{rerun_reason}" if rerun_reason else ""))


def frozen_to_config(frozen: dict) -> tuple[AnalysisConfig, str, EvalConfig]:
    cfg = AnalysisConfig(**frozen["analysis_config"])
    return cfg, frozen["freq"], EvalConfig(**frozen["eval_config"])


def config_payload(config_id: int, params: dict, cfg: AnalysisConfig, eval_cfg: EvalConfig,
                   dictionary: dict, dev_row: dict, grid: dict) -> dict:
    return {"config_id": config_id, "params": params, "freq": params["freq"],
            "analysis_config": cfg.to_dict(), "eval_config": asdict(eval_cfg),
            "dictionary": dictionary, "selection_rule": f"検知器「{SELECTION_DETECTOR}」のdev F1最大"
            "→100時間あたり誤警戒が少ない→遅れ中央値が小さい→config_idが小さい",
            "dev_result": dev_row, "grid": grid}


def comparisons(per_video: pd.DataFrame) -> pd.DataFrame:
    return mcnemar_tests(per_video)
