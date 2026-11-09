# 本評価の候補動画のコメントをまとめて取得する（APIキーは環境変数 YOUTUBE_API_KEY で渡す。ここには書かない）
#
# 使い方（プロジェクトのルートで）:
#   $env:YOUTUBE_API_KEY = "（手元のキー）"     # このウィンドウだけ有効
#   powershell -ExecutionPolicy Bypass -File experiments\fetch_comments.ps1 -Ids qTnrEgRVEDk,KrkclH7G7S4
#
# 保存先は data/research/comments_<動画ID>_<取得日時>.csv（.gitignore で GitHub に上がらない）。
# 表示される「公開日時・コメント数・取得率」を data/research/labels_candidates.csv に記録してください。
# 1動画あたりの API 消費の目安：コメント100件につき約1ユニット（返信の追加取得を含めると増える）。
param(
    [Parameter(Mandatory = $true)][string[]]$Ids,
    [int]$Max = 30000
)
$ErrorActionPreference = "Stop"
$py = ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw ".venv が見つかりません。プロジェクトのルートで実行してください。" }
if (-not $env:YOUTUBE_API_KEY) { throw "環境変数 YOUTUBE_API_KEY が設定されていません（キーはスクリプトに書かないでください）。" }

$failed = @()
foreach ($id in $Ids) {
    Write-Host "=== $id ===" -ForegroundColor Cyan
    & $py -m src.youtube_api $id --max $Max --replies
    if ($LASTEXITCODE -ne 0) { $failed += $id }
}
if ($failed.Count) {
    Write-Host ("取得できなかった動画: " + ($failed -join ", ")) -ForegroundColor Yellow
    exit 1
}
Write-Host "すべて取得しました。次は observe で判定できる時間帯を確認します（docs/本評価_動画選定手順.md 6章）。"
