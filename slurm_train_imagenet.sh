#!/bin/bash
#SBATCH -A shocher_prj
#SBATCH -p rtx6k-shocher
#SBATCH --qos=contrib
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --job-name=spnn_imagenette_matexp

set -euo pipefail

mkdir -p logs
cd /home/yamitehrlich/work/SPNN

source venv/bin/activate

python train_imagenet.py imagenette2-320 \
    --num-classes 10 \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.1 \
    --mix-type householder \
    --scale-bound 1.0 \
    --checkpoint-dir check_points_cls_householder_sb1 \
    --wandb \
    --wandb-project spnn-imagenet \
    --wandb-run-name imagenette-householder-sb1
