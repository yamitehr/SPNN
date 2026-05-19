#!/bin/bash
#SBATCH -A shocher_prj
#SBATCH -p rtx6k-shocher
#SBATCH --qos=contrib
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --job-name=fcos_spnn

set -euo pipefail

# Unset SLURM vars that confuse torch distributed
unset SLURM_PROCID
unset RANK
unset WORLD_SIZE
unset LOCAL_RANK

mkdir -p logs
cd /home/yamitehrlich/work/SPNN/fcos_ref

source ../venv/bin/activate

# End-to-end invertible SPNN detector (for DDNM)
# Single scale, full pinv() support
python train_voc_spnn.py \
    --voc_root ../datasets/VOCdevkit/VOC2012 \
    --end_to_end \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.01 \
    --n_gpu 0,1,2,3 \
    --n_cpu 8 \
    --grad_clip 1.0 \
    --eval_freq 15 \
    --mix_type householder \
    --scale_bound 1.0 \
    --lambda_cycle 0.0 \
    --lambda_rec 0.0 \
    --freeze_backbone_epochs 0 \
    --save_dir ./checkpoint_spnn_e2e_householder \
    --wandb \
    --wandb_project spnn-fcos-voc \
    --wandb_run_name fcos_spnn_e2e_householder
