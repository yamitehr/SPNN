#!/bin/bash
#SBATCH -A cycle1_iit_shocher_prj            # account
#SBATCH -p compute-gpu                       # partition
#SBATCH --qos=owner_95
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH -o /shared/cycle1_iit_shocher_prj/SPNN/logs/%x_%j.out
#SBATCH -e /shared/cycle1_iit_shocher_prj/SPNN/logs/%x_%j.err
#SBATCH --job-name=centernet_spnn_deephead_8gpu
#SBATCH --container-image=docker://nvcr.io/nvidia/pytorch:25.10-py3
#SBATCH --container-mounts=/shared/cycle1_iit_shocher_prj:/shared/cycle1_iit_shocher_prj,/home/users/yamitehrlich:/home/users/yamitehrlich

# Deep-detector-head experiment, multi-GPU (8) version.
#
# Same training run as the original 1-GPU deephead script: orthomix head,
# block 4 uses the deeper 3-level U-net t/s/r nets via _DeepHeadConvPINNBlock.
# Backbone (blocks 0-3) still uses models.py — bit-compatible with the
# pretrained classifier checkpoint, so backbone transfer is intact.
#
# Multi-GPU deltas vs the 1-GPU script:
#   --gres=gpu:8, --cpus-per-task=64, --mem=512G  (8x resources)
#   launch via `python -m torch.distributed.run` (uses the venv's python so
#     children inherit the same interpreter and have all deps; the system
#     `torchrun` would spawn /usr/bin/python which lacks cv2/torch).
#   --num_workers 4 (per rank → 32 workers total instead of 8x12=96).
#   --dist flag in train.py wires DDP + NCCL.
#
# Hyperparameters unchanged from the 1-GPU run:
#   global batch_size=64 → per-GPU bs=8 under DDP, gradients allreduce →
#   mathematically equivalent to 1-GPU bs=64.  Same lr, same schedule.
#   Net: identical optimization dynamics, ~8x wall-clock speedup.

set -euxo pipefail

# Force python stdout/stderr unbuffered so logs flush before any crash
export PYTHONUNBUFFERED=1

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

# Use `python -m torch.distributed.run` instead of bare `torchrun`: the
# container's /usr/local/bin/torchrun spawns children with /usr/bin/python
# (the system Python — no cv2/torch).  Going through the venv's python
# ensures children inherit sys.executable from the activated venv.
python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    train.py \
        --dist \
        --arch spnn \
        --deep_det_head \
        --deep_head_hidden 128 \
        --dataset pascal \
        --img_size 256 \
        --batch_size 64 \
        --num_epochs 300 \
        --lr 0.0005 \
        --lambda_img_rec 1.0 \
        --lr_step 90,120 \
        --val_interval 5 \
        --log_interval 50 \
        --num_workers 4 \
        --data_dir ./data \
        --log_name spnn_centernet_deephead_NOdistill_hidden128_GN_UnetLike_recLoss_v16 \
        --wandb \
        --wandb_project spnn-centernet \
        --spnn_backbone /shared/cycle1_iit_shocher_prj/SPNN/check_points_cls_imagenet1k_Apr29_more_blocks/checkpoint.pth.tar \
        --hmap_init_scale 0.5 \
        --lambda_distill_hmap 0.0 \
        --lambda_distill_regs 0.0 \
        --lambda_distill_wh 0.0 &

PID=$!
wait $PID
