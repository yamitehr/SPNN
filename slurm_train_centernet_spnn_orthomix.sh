#!/bin/bash
#SBATCH -A cycle1_iit_shocher_prj            # account
#SBATCH -p compute-gpu                       # partition
#SBATCH --qos=owner_95
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH -o /home/users/%u/logs/%x_%j.out
#SBATCH -e /home/users/%u/logs/%x_%j.err
#SBATCH --job-name=centernet_spnn_orthomix
#SBATCH --container-image=docker://nvcr.io/nvidia/pytorch:25.10-py3
#SBATCH --container-mounts=/shared/cycle1_iit_shocher_prj:/shared/cycle1_iit_shocher_prj,/home/users/yamitehrlich:/home/users/yamitehrlich

# Option 1B experiment: SPNN-CenterNet with a Householder-orthogonal 20×20
# channel mixer added between hmap_scale and hmap_bias. Head stays a
# bijection on the heatmap channels — DDNM/pinv compatibility intact, and
# Householder reflections give bit-exact orthogonality for clean round-trips.
#
# Same hyperparameters as slurm_train_centernet_spnn.sh — only delta is
# --head_mode orthogonal_mix and a new --log_name. Direct comparison
# against the existing 30.2% mAP affine baseline.
#
# To ablate Cayley vs Householder, change --head_mix_type below and rerun
# with a fresh --log_name.

set -euo pipefail

mkdir -p /home/users/$USER/logs

source /shared/cycle1_iit_shocher_prj/SPNN/.venv-cluster/bin/activate
cd /shared/cycle1_iit_shocher_prj/SPNN/centernet_ref

cleanup() {
    echo "[$(date -Is)] SIGTERM received; forwarding to PID=$PID"
    kill -TERM "$PID"
    wait "$PID"
    exit 0
}
trap cleanup SIGTERM

python train.py \
    --arch spnn \
    --head_mode orthogonal_mix \
    --head_mix_type householder \
    --head_mix_reflections 20 \
    --dataset pascal \
    --img_size 256 \
    --batch_size 64 \
    --num_epochs 200 \
    --lr 0.0005 \
    --lr_step 90,120 \
    --val_interval 5 \
    --log_interval 50 \
    --num_workers 12 \
    --data_dir ./data \
    --log_name spnn_centernet_orthomix_NOdistill \
    --spnn_backbone /shared/cycle1_iit_shocher_prj/SPNN/check_points_cls_imagenet1k_Apr29_more_blocks/best_model.pth \
    --hmap_init_scale 0.1 \
    --lambda_distill_hmap 0.0 \
    --lambda_distill_regs 0.0 \
    --lambda_distill_wh 0.0 &

PID=$!
wait $PID
