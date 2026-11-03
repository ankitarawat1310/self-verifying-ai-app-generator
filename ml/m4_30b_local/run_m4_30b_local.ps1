# M4 repair experiment with your local Ollama qwen3-coder:30b, base vs fine-tuned (LoRA adapter).
# Run from this folder, after setup and check_adapter.py pass (see README_LOCAL.md).
# Prefer the .cmd launcher: it bypasses the execution policy that blocks .ps1 files.
#   .\run_m4_30b_local.cmd -Smoke      # 2 apps, fine-tuned only
#   .\run_m4_30b_local.cmd             # all 28 apps: fine-tuned first, then base
# Re-running resumes: finished cases are skipped. Results: <repo>\results\repair\repair_qwen30b_local_*\
param([string]$Repo = "D:\298B\svaga_repo", [string]$Python = "", [switch]$Smoke)
if (-not $Python) {
    if (Test-Path "$Repo\.venv\Scripts\python.exe") { $Python = "$Repo\.venv\Scripts\python.exe" } else { $Python = "python" }
}
$env:SVAGA_MODEL_ENDPOINTS = "svaga-qwen30b-ft=http://127.0.0.1:11434,svaga-qwen30b-base=http://127.0.0.1:11434"
Push-Location $Repo
try {
    if ($Smoke) {
        & $Python scripts\run_repair_eval.py --llm-model svaga-qwen30b-ft --name repair_qwen30b_local_smoke --limit 2
    } else {
        & $Python scripts\run_repair_eval.py --llm-model svaga-qwen30b-ft   --name repair_qwen30b_local_tuned
        & $Python scripts\run_repair_eval.py --llm-model svaga-qwen30b-base --name repair_qwen30b_local_base
        Write-Host "`n===== fine-tuned =====";  Get-Content results\repair\repair_qwen30b_local_tuned\summary.md
        Write-Host "`n===== base =====";        Get-Content results\repair\repair_qwen30b_local_base\summary.md
    }
} finally { Pop-Location }
