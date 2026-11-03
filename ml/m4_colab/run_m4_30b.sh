#!/usr/bin/env bash
# M4 repair experiment with Qwen3-Coder-30B-A3B (base and fine-tuned), same setup as the Granite run.
#   bash run_m4_30b.sh smoke  -> 2 rejected apps, fine-tuned only
#   bash run_m4_30b.sh full   -> all 28 rejected M1 apps: fine-tuned first, then base
# One server holds one 4-bit copy of the model and serves both names. Re-running resumes (finished cases are skipped).
set -euo pipefail
MODE=${1:-full}
KIT=$(cd "$(dirname "$0")" && pwd)
RES=${SVAGA_M4_RESULTS:-$KIT/m4_results_30b}
mkdir -p "$RES"
ln -sfn "$RES" "$KIT/svaga_repo/results/repair"
PORT=11600
if ! curl -sf "http://127.0.0.1:$PORT/svaga/health" >/dev/null 2>&1; then
  nohup python "$KIT/serve_30b.py" --adapter "$KIT/final_adapter" --port $PORT --log "$RES/serve_log_30b.jsonl" ${SVAGA_30B_ARGS:-} \
      > "$RES/server_30b.log" 2>&1 &
  echo "starting the 30B server (first start downloads about 61 GB; log: $RES/server_30b.log)"
fi
for i in $(seq 1 360); do
  if curl -sf "http://127.0.0.1:$PORT/svaga/health" >/dev/null 2>&1; then echo "server ready"; break; fi
  if ! pgrep -f serve_30b.py >/dev/null; then echo "server stopped; last lines of its log:"; tail -n 30 "$RES/server_30b.log"; exit 1; fi
  sleep 10
done
curl -sf "http://127.0.0.1:$PORT/svaga/health" >/dev/null || { echo "server not ready after 60 minutes"; tail -n 30 "$RES/server_30b.log"; exit 1; }

export SVAGA_MODEL_ENDPOINTS="qwen30b=http://127.0.0.1:$PORT,qwen30b-base=http://127.0.0.1:$PORT"
cd "$KIT/svaga_repo"
if [ "$MODE" = smoke ]; then
  python scripts/run_repair_eval.py --llm-model qwen30b --name repair_qwen30b_smoke --limit 2
else
  python scripts/run_repair_eval.py --llm-model qwen30b      --name repair_qwen30b_tuned 2>&1 | tee -a "$RES/eval_tuned.log"
  python scripts/run_repair_eval.py --llm-model qwen30b-base --name repair_qwen30b_base  2>&1 | tee -a "$RES/eval_base.log"
  echo; echo "===== fine-tuned ====="; cat "$RES/repair_qwen30b_tuned/summary.md"
  echo; echo "===== base ====="; cat "$RES/repair_qwen30b_base/summary.md"
fi
