#!/bin/bash
#SBATCH --job-name=llava-lora
#SBATCH --partition=gpu_top
#SBATCH --account=cai_cv
#SBATCH --gres=gpu:1
#SBATCH --exclude=irvine
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/cluster/home/%u/.logs/slurm/%j/%x_%j.out
#SBATCH --error=/cluster/home/%u/.logs/slurm/%j/%x_%j.err
#
# Trains the LoRA adapters on the language model with the patch and node projectors for two epochs on the embeddings of the frozen d2g encoder.
# Arguments after the script go on to train_lora.py:
#
#   sbatch runs/llava-lora.sh
#   sbatch runs/llava-lora.sh --rank 128 --alpha 256
#
# irvine's GPUs are AMD; every other GPU in gpu_top runs this PyTorch build.
# The venv lives on the node's persistent scratch, built by the first job that lands there.

set -euo pipefail

export UV_PROJECT_ENVIRONMENT=/raid/persistent_scratch/$USER/llava-venv
export UV_CACHE_DIR=/raid/persistent_scratch/$USER/uv-cache

cd LLaVA
uv sync
uv run train_lora.py \
  --embeddings ../results/runs/floorplans-d2g-walls-first-by-kind-frozen-seed0-200k/embeddings-d2g-synthetic-floorplans-s-deep-20261031_112046-200000 \
  --patch-projector ../results/runs/floorplans-d2g-walls-first-by-kind-frozen-seed0-200k/llava/Qwen2.5-0.5B-Instruct/20261118_171735/patches-synthetic-floorplans-captioning \
  --node-projector ../results/runs/floorplans-d2g-walls-first-by-kind-frozen-seed0-200k/llava/Qwen2.5-0.5B-Instruct/20261123_180259/nodes-synthetic-floorplans-node-captioning \
  --epochs 2 \
  "$@"
