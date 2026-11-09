# 動画単位の本評価を、決めた手順どおりに1回で実行する（卒論の水準：2026-11-09 決定）
#
#   1. 採用条件の確認（炎上・比較とも5本以上、comments_file があること、炎上動画に fire_start_jst と label_source があること）
#   2. 全動画を test にする（split --dev-fraction 0）→ 正解ラベルを固定（freeze-labels）
#   3. 集計幅3時間（主）：既定値で固定（fix-config、調整なし）→ test を1回だけ → report
#   4. 集計幅1時間（比較）：別の protocol フォルダで同じ手順
#
# 使い方（プロジェクトのルートで）:
#   powershell -ExecutionPolicy Bypass -File experiments\run_main_evaluation.ps1
#   powershell -ExecutionPolicy Bypass -File experiments\run_main_evaluation.ps1 -Labels data\research\labels.csv -MinPerClass 5
#
# test は1回だけ実行する設計なので、途中で止まった場合は protocol フォルダの protocol_log.csv を確認してから再実行してください。
param(
    [string]$Labels = "data\research\labels.csv",
    [int]$MinPerClass = 5
)
$ErrorActionPreference = "Stop"
$py = ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw ".venv が見つかりません。プロジェクトのルートで実行してください。" }
if (-not (Test-Path $Labels)) { throw "$Labels がありません。docs\templates\labels_template.csv をもとに作成してください。" }

function Invoke-Step([string[]]$StepArgs) {
    Write-Host ("> compare_methods " + ($StepArgs -join " ")) -ForegroundColor Cyan
    & $py -m experiments.compare_methods @StepArgs
    if ($LASTEXITCODE -ne 0) { throw ("失敗しました: " + ($StepArgs -join " ")) }
}

# --- 1. 採用条件の確認（評価を始める前に、足りないものがあれば止める） ---
$rows = Import-Csv $Labels -Encoding UTF8
$fire = @($rows | Where-Object { $_.is_fire -eq "1" })
$normal = @($rows | Where-Object { $_.is_fire -eq "0" })
$problems = @()
if ($fire.Count -lt $MinPerClass) { $problems += "炎上動画が $($fire.Count) 本（$MinPerClass 本以上が必要）" }
if ($normal.Count -lt $MinPerClass) { $problems += "比較用動画が $($normal.Count) 本（$MinPerClass 本以上が必要）" }
foreach ($r in $rows) {
    if (-not $r.comments_file -or -not (Test-Path $r.comments_file)) { $problems += "$($r.video_id): comments_file がない" }
    if ($r.is_fire -eq "1" -and (-not $r.fire_start_jst -or -not $r.label_source)) {
        $problems += "$($r.video_id): 炎上動画なのに fire_start_jst か label_source が空"
    }
}
if ($problems.Count) {
    Write-Host "本評価を始められません:" -ForegroundColor Yellow
    $problems | ForEach-Object { Write-Host "  - $_" }
    exit 1
}
Write-Host "採用条件 OK：炎上 $($fire.Count) 本・比較 $($normal.Count) 本" -ForegroundColor Green

# --- 2. 全動画を test にして、正解ラベルを固定 ---
Invoke-Step @("split", "--labels", $Labels, "--dev-fraction", "0")

# --- 3. 集計幅3時間（主） ---
Invoke-Step @("freeze-labels", "--labels", $Labels)
Invoke-Step @("fix-config", "--labels", $Labels, "--freq", "3h")
Invoke-Step @("test", "--labels", $Labels)
Invoke-Step @("report")

# --- 4. 集計幅1時間（比較。別の protocol と結果フォルダ） ---
$p1 = @("--protocol-dir", "experiments\protocol_1h", "--results-dir", "results\eval_1h")
Invoke-Step ($p1 + @("freeze-labels", "--labels", $Labels))
Invoke-Step ($p1 + @("fix-config", "--labels", $Labels, "--freq", "1h"))
Invoke-Step ($p1 + @("test", "--labels", $Labels))
Invoke-Step ($p1 + @("report"))

Write-Host "完了：3時間集計の表と図は results\ 、1時間集計の比較は results\eval_1h\ にあります。卒論 5.2 節に転記してください。" -ForegroundColor Green
