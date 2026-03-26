#!/bin/bash
#SBATCH -A shocher_prj
#SBATCH -p rtx6k-shocher
#SBATCH --qos=contrib
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --job-name=resnet50_celeba

set -euo pipefail

mkdir -p logs
cd /home/yamitehrlich/work/SPNN

source venv/bin/activate

python train_resnet_baseline.py \
    --batch_size 256 \
    --epoch 15 \
    --img_size 256 \
    --lr 2e-4 \
    --seed 556
