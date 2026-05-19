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
#SBATCH --job-name=centernet_spnn_deephead_mlptail
#SBATCH --container-image=docker://nvcr.io/nvidia/pytorch:25.10-py3
#SBATCH --container-mounts=/shared/cycle1_iit_shocher_prj:/shared/cycle1_iit_shocher_prj,/home/users/yamitehrlich:/home/users/yamitehrlich

# Deep-detector-head + MLP-tail residual experiment.
#
# Same orthomix head + deep U-Net s/t/r as run 1455 (mAP 36.26%), with one
# addition: a per-pixel MLP residual is appended after the U-Net inside each
# of the s/t/r networks of block 4.  Zero-init last conv keeps step-0 output
# bit-identical to the warm-start checkpoint, then training learns a
# nonlinear channel-direction correction on top.  Bijectivity preserved
# (the same residual is traversed by spnn.pinv).
#
# Targets the diagnosed bottleneck: at the dog cell, the dog-vs-horse
# contrast direction (W[dog,:] - W[horse,:]) was nearly orthogonal to the
# scaled raw feature (cos = 0.019).  The MLP tail can rotate the per-cell
# 24-d feature so that projection onto the per-class contrast directions
# becomes large.
#
# Warm-start: full state-dict from spnn_centernet_deephead_NOdistill_hidden256_GN
# loaded with strict=False; only s_tail/t_tail/r_tail params start from
# their constructed init (zero-last-conv).  Validation at epoch 0 will
# reproduce ~36.26% mAP.

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
    --deep_det_head \
    --deep_head_hidden 256 \
    --mlp_tail_hidden 128 \
    --warm_start_full /shared/cycle1_iit_shocher_prj/SPNN/centernet_ref/ckpt/spnn_centernet_deephead_NOdistill_hidden256_GN/checkpoint.t7 \
    --dataset pascal \
    --img_size 256 \
    --batch_size 64 \
    --num_epochs 80 \
    --lr 0.00025 \
    --lr_step 40,60 \
    --val_interval 5 \
    --log_interval 50 \
    --num_workers 12 \
    --data_dir ./data \
    --log_name spnn_centernet_deephead_NOdistill_hidden256_GN_mlptail128 \
    --wandb \
    --wandb_project spnn-centernet \
    --hmap_init_scale 0.1 \
    --lambda_distill_hmap 0.0 \
    --lambda_distill_regs 0.0 \
    --lambda_distill_wh 0.0 &

PID=$!
wait $PID
