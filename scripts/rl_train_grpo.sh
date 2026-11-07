#!/usr/bin/env bash
# Multi-turn GRPO with LoRA on one GPU (verl 0.9.1). Run from the repository root.
#
#   EXP=grpo_full bash scripts/rl_train_grpo.sh                     # main run
#   EXP=grpo_v11_no_cost REWARD_PRESET=v11_no_cost STEPS=80 bash scripts/rl_train_grpo.sh
#   EXP=dapo_dyn DYNAMIC_SAMPLING=1 bash scripts/rl_train_grpo.sh    # DAPO ablations
#   EXP=seqmean LOSS_AGG=seq-mean-token-mean bash scripts/rl_train_grpo.sh
#
# Every knob below can be overridden from the environment; the resolved values are
# written to results/rl_train/$EXP/run_config.env before training starts.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

EXP=${EXP:?set EXP to a run name, e.g. EXP=grpo_full}
MODEL_PATH=${MODEL_PATH:-$HOME/autodl-tmp/models/Qwen2.5-1.5B-Instruct}
DATA_DIR=${DATA_DIR:-$ROOT/data/rl}
DENSE=${DENSE:-$HOME/autodl-tmp/models/bge-small-en-v1.5}   # must match rl_build_dense.py --model
REWARD_PRESET=${REWARD_PRESET:-v11}
DYNAMIC_SAMPLING=${DYNAMIC_SAMPLING:-0}
LOSS_AGG=${LOSS_AGG:-token-mean}
STEPS=${STEPS:-150}
BATCH=${BATCH:-8}                 # questions per step
GROUP=${GROUP:-8}                 # rollouts per question (GRPO group size)
MAX_TURNS=${MAX_TURNS:-8}
SOFT_TURNS=${SOFT_TURNS:-6}
LR=${LR:-1e-5}
LORA_RANK=${LORA_RANK:-32}
KL_COEF=${KL_COEF:-0.001}
GPU_UTIL=${GPU_UTIL:-0.45}
SEED=${SEED:-1}
SAVE_FREQ=${SAVE_FREQ:-25}
MAX_CKPT=${MAX_CKPT:-3}           # keep the last 3 checkpoints so dev can pick the best step
ATTN=${ATTN:-flash_attention_2}   # use ATTN=sdpa when flash-attn is not installed
# FSDP1 flattens parameters, hiding LoRA names, so verl sent vLLM an empty adapter.
# FSDP2 keeps real parameter names. LORA_MERGE=1 syncs merged weights instead of an adapter.
FSDP_STRATEGY=${FSDP_STRATEGY:-fsdp2}
LORA_MERGE=${LORA_MERGE:-0}
# Samples per micro-batch for training and log-prob passes. Without flash-attn verl pads
# every rollout to prompt+response and computes the full 152K-vocabulary entropy in one
# go (chunking only exists on the padding-free path), so one sample at a time fits 24GB.
MICRO_BATCH=${MICRO_BATCH:-1}
TEST_FREQ=${TEST_FREQ:-25}

RUN_DIR=$ROOT/results/rl_train/$EXP
export TENSORBOARD_DIR=$RUN_DIR/tensorboard
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
CKPT_DIR=${CKPT_DIR:-$HOME/autodl-tmp/checkpoints/$EXP}
mkdir -p "$RUN_DIR" "$CKPT_DIR"

# Tool config with absolute paths: Ray workers do not share the launch directory.
cat > "$RUN_DIR/tools.yaml" <<YAML
tools:
YAML
for TOOL in bm25_search dense_search get_document; do
  cat >> "$RUN_DIR/tools.yaml" <<YAML
  - class_name: agentos.rl.verl_tools.SearchTool
    config: {type: native, tool_name: $TOOL, data_dir: "$DATA_DIR", dense: "$DENSE"}
YAML
done

( set -o posix; set ) | grep -E '^(EXP|MODEL_PATH|DATA_DIR|DENSE|REWARD_PRESET|DYNAMIC_SAMPLING|LOSS_AGG|STEPS|BATCH|GROUP|MAX_TURNS|SOFT_TURNS|LR|LORA_RANK|KL_COEF|GPU_UTIL|SEED|ATTN|FSDP_STRATEGY|LORA_MERGE|MICRO_BATCH)=' > "$RUN_DIR/run_config.env"
git rev-parse HEAD > "$RUN_DIR/git_commit.txt" 2>/dev/null || true

ATTN_ARGS=(actor_rollout_ref.model.use_remove_padding=True)
if [[ "$ATTN" != "flash_attention_2" ]]; then
  # Padding removal relies on flash-attn varlen kernels; turn it off with sdpa/eager.
  ATTN_ARGS=(actor_rollout_ref.model.use_remove_padding=False "++actor_rollout_ref.model.override_config.attn_implementation=$ATTN")
fi

MERGE_ARGS=()
if [[ "$LORA_MERGE" == "1" ]]; then
  MERGE_ARGS=(++actor_rollout_ref.model.lora.merge=True)
fi

DYN_ARGS=()
if [[ "$DYNAMIC_SAMPLING" == "1" ]]; then
  # DAPO dynamic sampling: drop groups whose rewards are all equal, regenerate up to 4 batches.
  DYN_ARGS=(algorithm.filter_groups.enable=True algorithm.filter_groups.metric=reward algorithm.filter_groups.max_num_gen_batches=4)
fi

# Single GPU: FSDP runs NO_SHARD, and layered_summon's fallback summons with
# offload_to_cpu=True, which PyTorch rejects for NO_SHARD. Plain summon works.
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  "${DYN_ARGS[@]}" \
  data.train_files="$DATA_DIR/verl/train.parquet" \
  data.val_files="$DATA_DIR/verl/dev.parquet" \
  data.train_batch_size=$BATCH \
  data.val_batch_size=100 \
  data.max_prompt_length=1024 \
  data.max_response_length=3072 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  data.seed=$SEED \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.lora_rank=$LORA_RANK \
  actor_rollout_ref.model.lora_alpha=$((LORA_RANK * 2)) \
  actor_rollout_ref.model.target_modules=all-linear \
  "${ATTN_ARGS[@]}" \
  "${MERGE_ARGS[@]}" \
  actor_rollout_ref.actor.strategy=$FSDP_STRATEGY \
  actor_rollout_ref.actor.fsdp_config.strategy=$FSDP_STRATEGY \
  actor_rollout_ref.ref.fsdp_config.strategy=$FSDP_STRATEGY \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=$LR \
  actor_rollout_ref.actor.ppo_mini_batch_size=$BATCH \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH \
  actor_rollout_ref.actor.entropy_checkpointing=True \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_UTIL \
  actor_rollout_ref.rollout.n=$GROUP \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=False \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$RUN_DIR/tools.yaml" \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$MAX_TURNS \
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1600 \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
  actor_rollout_ref.rollout.agent.num_workers=4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  reward.custom_reward_function.path="$ROOT/agentos/rl/verl_reward.py" \
  reward.custom_reward_function.name=compute_score \
  ++reward.custom_reward_function.reward_kwargs.preset=$REWARD_PRESET \
  ++reward.custom_reward_function.reward_kwargs.max_turns=$MAX_TURNS \
  ++reward.custom_reward_function.reward_kwargs.soft_turns=$SOFT_TURNS \
  trainer.logger='["console","tensorboard"]' \
  trainer.project_name=searchrl \
  trainer.experiment_name="$EXP" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_training_steps=$STEPS \
  trainer.save_freq=$SAVE_FREQ \
  trainer.test_freq=$TEST_FREQ \
  trainer.val_before_train=True \
  trainer.default_local_dir="$CKPT_DIR" \
  trainer.rollout_data_dir="$RUN_DIR/rollouts" \
  trainer.validation_data_dir="$RUN_DIR/validation" \
  trainer.resume_mode=auto \
  trainer.max_actor_ckpt_to_keep=$MAX_CKPT \
  "$@" 2>&1 | tee -a "$RUN_DIR/train.log"   # append: a resumed run keeps earlier steps
