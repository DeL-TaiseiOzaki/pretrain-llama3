#!/bin/bash
#SBATCH --job-name=llama_pretrain
#SBATCH --output=logs/llama_pretrain_%j.out
#SBATCH --error=logs/llama_pretrain_%j.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --nodelist=isk-gpu05
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --partition=matsuolab

# 環境設定
module load cuda/12.5
module load miniconda/24.7.1-py311-pytorch

# HuggingFace tokenの設定
export HUGGING_FACE_HUB_TOKEN="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"

# Weights & Biases の設定
export WANDB_API_KEY="64bbb327e74281cbe4622d5b38868ce54e861967"
export WANDB_ENTITY="symonds6457"

# 学習の実行
srun python -m torch.distributed.run \
    --nproc_per_node=8 \
    --master_port=29500 \
    train.py