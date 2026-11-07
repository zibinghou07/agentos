#!/usr/bin/env bash
# Serve a model with vLLM (Hermes tool calls), evaluate it in SearchEnv, stop the server.
#   MODEL=$HOME/autodl-tmp/models/Qwen2.5-1.5B-Instruct TAG=pre_rl bash scripts/rl_serve_and_eval.sh
#   MODEL=$HOME/autodl-tmp/merged/grpo_full TAG=post_rl_grpo_full SPLITS="test ood_2wiki ood_musique" bash scripts/rl_serve_and_eval.sh
#   EXTRA=--closed-book TAG=pre_rl_closed_book SPLITS=test ...   # extra flags for rl_eval.py
#   MODEL=<base model> LORA=<run>/lora_adapter TAG=post_rl ...       # serve base + LoRA as "policy"
# verl's model_merger writes the trained LoRA to lora_adapter/ and an unmerged base
# model.safetensors, so a trained policy is served the way verl rolled it out: base + LoRA.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:?path to a HF model directory}
TAG=${TAG:?result tag, e.g. pre_rl}
SPLITS=${SPLITS:-"dev test"}
SEEDS=${SEEDS:-"0 1 2"}
LIMIT=${LIMIT:-}
DENSE=${DENSE:-$HOME/autodl-tmp/models/bge-small-en-v1.5}
PORT=${PORT:-8000}
CONCURRENCY=${CONCURRENCY:-32}
EXTRA=${EXTRA:-}
LORA=${LORA:-}
LOG=results/rl_eval/$TAG/vllm.log
mkdir -p "results/rl_eval/$TAG"

SERVE_ARGS=(--served-model-name policy)
if [[ -n "$LORA" ]]; then
  RANK=$(python -c "import json, sys; print(json.load(open(sys.argv[1] + '/adapter_config.json'))['r'])" "$LORA")
  SERVE_ARGS=(--served-model-name base --enable-lora --max-lora-rank "$RANK" --lora-modules "policy=$LORA")
fi
vllm serve "$MODEL" "${SERVE_ARGS[@]}" --port "$PORT" \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-model-len 8192 --gpu-memory-utilization 0.85 >"$LOG" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT

for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; then break; fi
  if ! kill -0 $SERVER 2>/dev/null; then echo "vLLM exited; see $LOG"; exit 1; fi
  sleep 5
done

for SPLIT in $SPLITS; do
  python scripts/rl_eval.py --split "$SPLIT" ${LIMIT:+--limit $LIMIT} --model policy \
    --base-url "http://127.0.0.1:$PORT/v1" --api-key EMPTY --dense "$DENSE" \
    --seeds $SEEDS --concurrency "$CONCURRENCY" --tag "$TAG" $EXTRA
done
