#!/usr/bin/env bash
export GRASP_ROOT="${GRASP_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
export GRASP_CONFIG="${GRASP_CONFIG:-$GRASP_ROOT/profiles/workflow_3060.json}"
export PATH="${GRASP_VENV:-$GRASP_ROOT/venv}/bin:$PATH"
export PYTHONPATH="$GRASP_ROOT:${VAGEN_DIR:-$GRASP_ROOT/VAGEN-vagen-lite}:${VERL_DIR:-$GRASP_ROOT/verl-3fe0a29975e1b02ae2bd1dec249f7807dd7966f5}${PYTHONPATH:+:$PYTHONPATH}"
# Respect scheduler-provided visibility; single-node profile controls worker count.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled
export VLLM_NO_USAGE_STATS=1
export RAY_USAGE_STATS_ENABLED=0
export DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export HYDRA_FULL_ERROR=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_PLUGINS=grasp_qwen3_vl
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export C_INCLUDE_PATH="$GRASP_ROOT/sysroot/usr/include:$GRASP_ROOT/sysroot/usr/include/python3.10:$GRASP_ROOT/sysroot/usr/include/x86_64-linux-gnu/python3.10${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
