# YouTube炎上早期検知システム（卒業研究用）

YouTubeコメントの時系列感情分析と異常検知を用いて、炎上の発生をできるだけ早く検知し、
複数の異常検知手法を比較・評価する研究用Webアプリです。

> ⚠️ 本アプリの判定は研究上の暫定的な基準であり、実際に炎上していることを保証するものではありません。

設計の詳細は [docs/STEP1_設計書.md](docs/STEP1_設計書.md) を参照してください。

## 実装状況

| STEP | 内容 | 状態 |
|---|---|---|
| 1 | システム設計・URL解析・CSV読込・架空データ・ホーム画面 | ✅ 完了 |
| 2 | YouTube APIによるコメント取得（返信を含む）・CSV保存 | ✅ 完了 |
| 3 | 感情分析（Janome＋感情辞書）・時系列集計 | ✅ 完了（ローカル研究版はNRC日本語辞書を初期選択、簡易辞書に切替可。公開デモは簡易辞書のみ） |
| 4 | 異常検知（否定率・Zスコア・オンラインPELT）・分析ダッシュボード | ✅ 完了（架空データで確認） |
| 5 | 画面（分析ダッシュボード・アルゴリズム比較・データ確認） | ✅ 完了（架空データで確認） |
| 6 | 評価実験 | 🔧 仕組みは完成・架空データで動作確認済み。実動画の正解ラベル待ち |
| 7 | クラウド公開 | 🔧 準備済み（公開モード・ライブラリのバージョン固定）。GitHub と Streamlit Community Cloud での操作待ち |
| 8 | 卒論作成支援 | 🔧 構成案（`docs/STEP8_卒論構成案.md`）と表・図の自動作成（`report`）。結果は実データの評価後 |

## セットアップ（Windows）

[Python 3.11](https://www.python.org/downloads/) をインストールしてから、PowerShellで実行します。

```powershell
cd youtube-fire-monitor
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1          # 実行できない場合: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements-dev.txt
```

macOS / Linux の場合は `python3.11 -m venv .venv` → `source .venv/bin/activate`。

## 実行

```powershell
python -m src.synthetic_data   # 架空データを data/sample/ に作成（同梱済みなので省略可）
streamlit run app.py           # ブラウザでアプリが開く
python -m pytest               # テスト
```

APIキーがなくても、ホーム画面の「サンプル（架空データ）を読み込む」またはCSVアップロードで動作確認できます。

## コメントの取得（YouTube API）

APIキーはソースコードにもチャットにも書かず、実行時に手元で設定します。

**方法1：コマンドで取得してCSVに保存（おすすめ）**

```powershell
python -m src.youtube_api https://www.youtube.com/watch?v=Thc7JIJk6Co --max 10
```

返信コメントも取得する場合は `--replies` を付けます。

```powershell
python -m src.youtube_api https://www.youtube.com/watch?v=Thc7JIJk6Co --max 50 --replies
```

**取得上限（`--max`）の数え方**

- 返信なし（既定）：トップレベルコメントの件数。
- `--replies` あり：CSVの行数、つまり**トップレベル＋返信の合計**。新しい順のスレッドごとに「親 → その返信（古い順）」の順で数え、合計が上限に達したところで止めます。最後のスレッドの返信は途中までになることがあります。
- 例：`--max 10` で、1件目の親に返信3件・2件目の親に返信4件なら、親1+返信3+親1+返信4+3件目の親1 = 10行。
- 返信行は `parent_id` に親コメントのID、`is_reply` に `True` が入ります。
- クォータ：commentThreads 1回（最大100スレッド）で1単位。返信が5件程度を超えるスレッドは comments.list で取り直すため、100件ごとに1単位が追加でかかります。

環境変数 `YOUTUBE_API_KEY` がなければ、キーの入力を求められます（入力した文字は画面にも履歴にも残りません）。
保存先は `data/research/comments_<動画ID>_<日時>.csv`（Git管理外）です。

同じPowerShell画面で何度も使う場合は、履歴に残らない方法で環境変数に入れられます。

```powershell
$env:YOUTUBE_API_KEY = (Read-Host "APIキー" -AsSecureString | ConvertFrom-SecureString -AsPlainText)
```

（`-AsPlainText` は PowerShell 7 以降。Windows PowerShell 5.1 では方法1の入力プロンプトを使ってください）

**方法2：Streamlitの画面から取得**

1. `.streamlit/secrets.toml.example` をコピーして `.streamlit/secrets.toml` を作る
2. `YOUTUBE_API_KEY = "..."` を書き込む（`.gitignore` 済みなので GitHub には上がりません）
3. ホーム画面でURLを入れ、必要なら「返信コメントも取得する」をオンにして「コメントを取得」→「CSVでダウンロード」

## コメントCSVの形式

| 列 | 必須 | 内容 |
|---|---|---|
| `comment_id` | ✅ | コメントID |
| `published_at` | ✅ | 投稿日時（例 `2026-01-05T18:23:33+09:00`。タイムゾーンなしはUTC扱い、画面で変更可） |
| `text` | ✅ | 本文 |
| `like_count` | ✅ | いいね数 |
| `parent_id` | ✅ | 親コメントID（トップレベルは空欄） |
| `is_reply` | ✅ | 返信なら True |
| `video_id` |  | 動画ID |
| `data_source` |  | `synthetic`（架空）/ `real`（実データ）。なければ real 扱い |

文字コードは UTF-8 と Shift_JIS（Excel保存）の両方に対応しています。

## データの扱い（重要）

| 場所 | 中身 | GitHub |
|---|---|---|
| `data/sample/` | **架空データ**（本文に【架空】、`data_source=synthetic`）。動作確認専用で研究結果には使わない | 公開 |
| `data/research/` | YouTubeから収集した**実データ**と正解ラベル | **公開しない**（.gitignore） |
| `data/dictionaries/private/` | NRC感情辞書など再配布禁止の辞書 | **公開しない**（.gitignore） |

投稿者名・チャンネルIDは取得しません。

## 感情分析と時系列集計（STEP 3）

ホーム画面でデータを読み込んだあと「感情分析を実行」を押します。判定結果付きコメントと時系列データはCSVで保存できます。

**計算方法**（詳細は `src/sentiment_analysis.py` の先頭の説明）

1. 本文をNFKC正規化してJanomeで分かち書き
2. 最大4語の連続（最後の語は基本形）を辞書と照合（例：「腹が立った」→「腹が立つ」）
3. 直後に否定（ない・ぬ・ん・ず）がある語は数えない（「最悪ではない」「怖くありません」）。「〜しかない」は否定とみなさない
4. 数えた感情語の数を M として、各感情スコア = 該当語数 / M（0〜1）。否定的 = (anger・fear・sadness の語数)/M ≥ θ（初期値0.5）
5. **M = 0 のコメントは「判定不能」**（NaN）。肯定的とも否定的でないとも扱わない

**時系列**（`src/time_series.py`）：1時間（30分・3時間も可）ごとに、コメント数、否定的コメント数、否定的コメント率（分母＝全コメント。判定できたものだけを分母にした率も併記）、カバー率、5感情の平均、コメント数の移動平均（過去のビンのみ）、信頼性フラグ（10件未満）を計算します。0件の時間帯も残し、比率は欠損（NaN）にします。

## 炎上兆候の検知（STEP 4）

ホーム画面でデータを読み込み、サイドバーの「分析ダッシュボード」で「検知を実行」を押します。

| 手法 | 見るもの | 検知の条件（初期値） |
|---|---|---|
| A 否定率の上昇 | 否定的コメント率 | 直前24時間の否定率（件数で重み付け）より **+15ポイント以上**。その時間帯10件以上・直前24時間の合計10件以上 |
| B Zスコア | コメント数 | 直前24時間の平均・標準偏差から **Z > 3**（σ<1 のときは1を使う）。その時間帯10件以上 |
| C 変化点（PELT） | 否定的コメント率 | 各時刻までのデータだけでPELT（rbf, penalty=3）を計算し、直近3時間以内に **+15ポイント以上の上昇** の変化点 |

- すべてオンライン方式（時刻 t の判定には t までのデータだけを使う）。最初の24時間は「履歴不足」として判定しません
- 警戒レベル：**同じ時間帯**に2手法以上が検知＝警戒、それ以外で直近3時間に検知あり＝注意、判定して検知なし＝正常、**どの手法も判定条件を満たさず検知もない＝判定対象外**（別々の時間帯の検知を組み合わせて警戒にはしない）
- 判定できた時間帯が1つもない動画（履歴不足・件数不足だけ）は「正常」ではなく「判定対象外」と表示し、比較表・CSV（`judged`, `judged_methods`, `level` 列）でも区別します。評価では TP/FN/FP/TN に入れず「判定対象外で除外した動画数」として数えます
- 手法Cは「変化点の位置」と「検知できた時刻」を分けて表示します。全期間を使った事後分析の変化点は別表示で、評価には使いません
- 画面では、検知時刻・手法・根拠（例：「否定率 61%（74件中45件）が直前24時間の否定率 12%（367件）より +49ポイント高い」）を一覧で確認できます
- 設計書から変更・仮置きした値と理由は `docs/STEP1_設計書.md` の「STEP 4 実装時に決めたこと」に記載しています

## 画面

| 画面 | 内容 |
|---|---|
| ホーム | URLからの取得、CSV・架空データの読み込み、感情分析と時系列集計 |
| 分析ダッシュボード | 「検知を実行」→ 指標カード、グラフ5種（検知時刻は赤線）、検知の一覧と根拠、手法Cの変化点の位置と検知時刻、動画ごとの比較 |
| アルゴリズム比較 | 手法ごとの検知数・検知時刻・計算時間。正解ラベルがあれば Precision・Recall・F1・誤警戒率・検知の遅れ（split=test の動画は表示しない） |
| データ確認 | コメントごとの投稿日時・本文・感情スコア・否定判定・いいね数。絞り込みとCSV保存 |

## 評価実験（STEP 6）

評価用（test）の結果を見てから閾値を調整しないため、次の順番でだけ実行します。各操作は `experiments/protocol/protocol_log.csv` に記録されます（実コメントは保存されません）。結果のCSVは `results/`（Git管理外）に出ます。

```powershell
# 架空データで手順全体を試す（研究結果ではありません。results\demo_synthetic\ に出力）
.venv\Scripts\python.exe -m experiments.compare_methods demo

# 実データ
.venv\Scripts\python.exe -m experiments.compare_methods split --labels data/research/labels.csv
.venv\Scripts\python.exe -m experiments.compare_methods freeze-labels --labels data/research/labels.csv
.venv\Scripts\python.exe -m experiments.compare_methods tune --labels data/research/labels.csv
.venv\Scripts\python.exe -m experiments.compare_methods test --labels data/research/labels.csv

# 感情分析の精度（人手の正解との比較）
.venv\Scripts\python.exe -m experiments.compare_methods sentiment --gold data/research/sentiment_gold.csv

# 予備評価：基準時刻（会見開始など）の前後比較。正解ラベル（炎上開始）とは別に扱う
.venv\Scripts\python.exe -m experiments.compare_methods event --csv data/research/comments_（動画ID）_（日時）.csv --event 2025-01-27T16:00:00+09:00 --before 24 --after 24

# 動画数が少ない場合：調整せず既定値で固定して全動画を1回だけ評価（split --dev-fraction 0 のあと）
.venv\Scripts\python.exe -m experiments.compare_methods fix-config --labels data/research/labels.csv --freq 3h

# 観測できる時間帯を数える（判定対象外と「判定したが警戒なし」を分けて集計）
.venv\Scripts\python.exe -m experiments.compare_methods observe --csv data\research\（CSV1） data\research\（CSV2） --start 2025-09-09T00:00:00+09:00 --end 2025-09-19T00:00:00+09:00

# 論文用の表と図（results\report\ に出力。架空データの結果なら --from results/demo_synthetic）
.venv\Scripts\python.exe -m experiments.compare_methods report
```

本評価をまとめて実行する場合は `experiments\run_main_evaluation.ps1`（採用条件の確認つき）、コメントの一括取得は `experiments\fetch_comments.ps1` を使います（`docs/本評価_動画選定手順.md` 6章）。動画の選定には、チャンネルの公開動画を期間で一覧にする `python -m src.list_uploads <動画IDかチャンネルID> --start 2025-09-01 --end 2025-09-19`（コメントは取らない）を使います（同 8章）。

- `split`：事件（incident_id）単位で dev／test に分けます（同じ事件の動画が両方に入らない）
- `freeze-labels`：正解ラベルの SHA-256 を記録します。以後ラベルを変えると `tune`・`test` が止まります。ハッシュの記録（`experiments/protocol/`）は Git にコミットしてください
- `tune`：dev の動画だけでパラメータを総当たりし、事前に決めた規則（検知器「警戒」の F1 最大など）で1つ選んで固定します
- `test`：固定した設定と同じ辞書で test を1回だけ評価します。再実行・test 後の再調整は理由の記録が必要です
- 判定のルール：炎上開始〜48時間後の検知だけを正解とし、開始前の警報は誤警戒。遅れは手法Cも「検知できた時刻」で計算。全期間の事後分析は別ファイル（評価には使わない）

### 用意する正解ラベル（`data/research/labels.csv`、Git管理外）

雛形：`docs/templates/labels_template.csv`

| 列 | 内容 |
|---|---|
| `video_id` | 動画ID（11文字） |
| `incident_id` | 事件ID。同じ事件を扱う動画は同じIDにする（dev/test の分割に使う） |
| `is_fire` | 炎上=1、非炎上=0 |
| `fire_start_jst` | 炎上開始時刻（例 `2026-03-01T14:00:00+09:00`）。非炎上は空欄。分からない炎上動画は空欄（遅れの評価から外れる） |
| `label_source` | 判断の根拠（報道記事のURLと掲載時刻など。コメント自体からは決めない） |
| `split` | 空欄のままにして `split` コマンドで埋める |
| `comments_file` | その動画のコメントCSVのパス（例 `data/research/comments_XXXXXXXXXXX_20261107_183806.csv`） |

### 用意する感情分析の正解（`data/research/sentiment_gold.csv`、Git管理外）

雛形：`docs/templates/sentiment_gold_template.csv`。列は `comment_id, text, gold_negative`（1/0。保留は空欄）。

判定する候補は次のコマンドで作ります。出力には本文と出所（動画ID・返信かどうか）だけが入り、辞書の予測は入りません。1動画・1スレッドあたりの上限があり、済んだラベルは `--exclude` で除けます。動画を追加して同じ seed で再実行しても、既に選んだ候補は変わりません。

```powershell
.venv\Scripts\python.exe -m experiments.compare_methods gold-sample --csv data\research\（CSV1） data\research\（CSV2） --n 180 --per-video-cap 90 --exclude data\research\（済んだ20件のCSV）
```

## クラウド公開（STEP 7）

公開版は「架空データ＋自作の簡易辞書」で動くデモです。NRC辞書・実コメント・正解ラベル・実験結果は `.gitignore` で除外しているため GitHub に上がりません（`tests/test_publish.py` で確認しています）。

1. GitHub で新しいリポジトリを作り、プロジェクトのフォルダで次を実行する
   ```powershell
   git init
   git add .
   git status            # data/research や data/dictionaries/private、.streamlit/secrets.toml が含まれていないことを確認
   git commit -m "YouTube炎上兆候検知アプリ"
   git branch -M main
   git remote add origin https://github.com/（ユーザー名）/（リポジトリ名）.git
   git push -u origin main
   ```
2. https://share.streamlit.io で「Create app」→ リポジトリ・ブランチ `main`・ファイル `app.py` を選ぶ
3. Advanced settings で Python **3.12** を選び、Secrets に次を書く（APIキーは書かなくてよい）
   ```toml
   PUBLIC_MODE = "true"
   ```
   自分だけ公開版からYouTube取得を使いたい場合は `YOUTUBE_API_KEY` と `APP_PASSCODE`（合言葉）も書く。合言葉がなければ、公開版ではYouTube取得は使えません
4. 公開されたURLを開き、「サンプル（架空データ）を読み込む」→「分析ダッシュボード」→「検知を実行」が動くことを確認する

## 感情辞書について

| 辞書 | 置き場所 | GitHub | 用途 |
|---|---|---|---|
| 研究用簡易辞書（自作 v1、222語：anger 79・fear 30・sadness 48・joy 41・trust 25） | `data/sentiment_dictionary.csv` | 公開 | 公開デモ・動作確認 |
| NRC Emotion Lexicon 日本語版 | `data/dictionaries/private/` | **公開しない** | 研究用（手元のみ） |

NRC辞書は非商用の研究目的なら無料ですが、**再配布は禁止**です。公式ページ（https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm ）からダウンロードし、zipの中の日本語版のファイル（多言語版の日本語ファイル）を `data/dictionaries/private/` に置いてください。アプリの「感情辞書」の選択肢に自動で表示されます。読み込める形式は次の2つです。

- 横長のタブ区切り：列名に `Japanese` を含む見出し語の列と、`anger`・`fear` などの0/1の列
- 縦長：`word,emotion`（任意で `association`）の列

論文に書く収録語数は、読み込み後に画面に表示される語数（実際に使った数）を使ってください。

### 辞書の選び方と画面の表示

- **ローカル研究版**：`data/dictionaries/private/Japanese-NRC-EmoLex.txt` があれば、NRC日本語辞書が初期選択になります。選択欄で簡易辞書にも切り替えられます。ファイルがない場合は、画面に設置場所が表示されます。
- **公開デモ版（`PUBLIC_MODE=true`、Streamlit Community Cloud）**：簡易辞書だけを使います。NRC辞書本体と実コメントは GitHub にもクラウドにも置きません。
- 感情分析と炎上検知ダッシュボードは**同じ辞書の選択**を使います（どちらで変えても、もう一方に反映）。結果の表示中に辞書を変えると、再実行するよう案内が出ます。
- 結果のCSV（判定結果付きコメント・時系列・感情別の集計・検知結果・動画ごとの比較・アルゴリズム比較）には `dictionary`（または「辞書」）列で辞書名を記録します。
- 画面には、使用した辞書名、感情別の集計、否定判定率（**分母＝判定できたコメント**。全コメントを分母にした値も併記）、判定できた割合（カバー率）、判定不能件数を表示します。
- 「判定不能」はコメント単位（辞書に一致する語がない）、「判定対象外」は時間帯単位（コメント数や直前24時間の履歴が足りず、検知の判定をしていない）で、別々に数えます。判定対象外の時間帯は「正常」に含めません。

### NRC日本語辞書を採用した理由（人手ラベル195件での比較）

同じ人手ラベル（200件のうち保留5件を除く195件）で比べると、NRC日本語辞書は判定カバー率 63.6%、否定の再現率 54.3%、F1 0.702 で、比較した2つの辞書（NRC・自作の簡易辞書）の中で最良でした（簡易辞書はカバー率 21.5%、再現率 14.1%、F1 0.248）。適合率は両者とも高く（簡易辞書 1.000、NRC 0.990）差は小さいため、判定できるコメントの多さと再現率を理由にNRCを採用しました。

評価の限界：非否定は11件しかなく、そのうち10件が同じスレッド（YouTubeテーマソング動画の1つの親コメントへの返信）に偏っています。このため、非否定を誤って否定とする傾向（特異度）はほとんど評価できていません。判定者は1名で、一致度は未測定です（詳細は `docs/STEP8_卒論構成案.md` 5.1節）。

簡易辞書の限界：作者が作った約200語のため網羅性が低く、実データでは判定不能が多くなります。皮肉・文脈・比喩は判定できません。

## ライセンス・引用

NRC Emotion Lexicon を使う場合は、公式ページの指示に従い関連論文を引用してください。
