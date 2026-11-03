#!/usr/bin/env bash
# M4 repair experiment on a GPU box (Colab or Linux).
#   bash run_m4.sh smoke   -> 2 rejected apps, fine-tuned Granite only (about 10 minutes)
#   bash run_m4.sh full    -> all 28 rejected M1 apps, base and fine-tuned Granite side by side
# Re-running the same command resumes: finished cases are skipped.
# Results go to $SVAGA_M4_RESULTS (default: ./m4_results). On Colab, point it at Google Drive so a disconnect loses nothing.
set -euo pipefail
MODE=${1:-full}
KIT=$(cd "$(dirname "$0")" && pwd)
RES=${SVAGA_M4_RESULTS:-$KIT/m4_results}
mkdir -p "$RES"
ln -sfn "$RES" "$KIT/svaga_repo/results/repair"

start_server() {   # port, then extra args
  local port=$1; shift
  if curl -sf "http://127.0.0.1:$port/svaga/health" >/dev/null 2>&1; then echo "server on $port already up"; return; fi
  nohup python "$KIT/svaga_plugin/serve_svaga_model.py" --run "$KIT/ftC_granite33_2b" --port "$port" \
      --unconstrained --log "$RES/serve_log_$port.jsonl" "$@" > "$RES/server_$port.log" 2>&1 &
  echo "starting model server on $port (log: $RES/server_$port.log)"
}
wait_server() {
  for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$1/svaga/health" >/dev/null 2>&1; then echo "server $1 ready"; return; fi
    sleep 5
  done
  echo "server $1 did not start; last lines of its log:"; tail -n 30 "$RES/server_$1.log"; exit 1
}

start_server 11500
if [ "$MODE" = full ]; then start_server 11501 --base_only; fi
wait_server 11500
if [ "$MODE" = full ]; then wait_server 11501; fi

export SVAGA_MODEL_ENDPOINTS="ftC_granite33_2b=http://127.0.0.1:11500,ftC_granite33_2b-base=http://127.0.0.1:11501"
cd "$KIT/svaga_repo"
if [ "$MODE" = smoke ]; then
  python scripts/run_repair_eval.py --llm-model ftC_granite33_2b --name repair_smoke --limit 2
else
  python scripts/run_repair_eval.py --llm-model ftC_granite33_2b-base --name repair_granite_base  > "$RES/eval_base.log"  2>&1 &
  B=$!
  python scripts/run_repair_eval.py --llm-model ftC_granite33_2b      --name repair_granite_tuned > "$RES/eval_tuned.log" 2>&1 &
  T=$!
  echo "running base (pid $B) and fine-tuned (pid $T); progress: tail -f $RES/eval_*.log"
  wait $B; wait $T
  echo; echo "===== base ====="; cat "$RES/repair_granite_base/summary.md"
  echo; echo "===== fine-tuned ====="; cat "$RES/repair_granite_tuned/summary.md"
fi
