#!/bin/bash
#SBATCH -A shocher_prj
#SBATCH -p rtx6k-shocher
#SBATCH --qos=contrib
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --job-name=fcos_spnn_nette

set -euo pipefail

# Unset SLURM vars that confuse torch distributed
unset SLURM_PROCID
unset RANK
unset WORLD_SIZE
unset LOCAL_RANK

mkdir -p logs
cd /home/yamitehrlich/work/SPNN/fcos_ref

source ../venv/bin/activate

python train_voc_spnn.py \
    --voc_root ../datasets/VOCdevkit/VOC2012 \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001 \
    --n_gpu 0 \
    --n_cpu 8 \
    --grad_clip 1.0 \
    --eval_freq 10 \
    --lambda_cycle 10.0 \
    --lambda_rec 5.0 \
    --mix_type cayley \
    --pretrained_backbone ../check_points_cls_matexp_new_arch/checkpoint.pth.tar \
    --freeze_backbone_epochs 5 \
    --save_dir ./checkpoint_spnn_imagenette_matexp_new_arch \
    --wandb \
    --wandb_project spnn-fcos-voc \
    --wandb_run_name fcos_spnn_imagenette_backbone_new_arch
