#!/bin/bash
#SBATCH -A cycle1_iit_shocher_prj            # account
#SBATCH -p compute-gpu                       # partition
#SBATCH --qos=owner_95
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=1-00:00:00
#SBATCH -o /home/users/%u/logs/%x_%j.out
#SBATCH -e /home/users/%u/logs/%x_%j.err
#SBATCH --job-name=centernet_spnn_deephead_mlptail_fromscratch_8gpu
#SBATCH --container-image=docker://nvcr.io/nvidia/pytorch:25.10-py3
#SBATCH --container-mounts=/shared/cycle1_iit_shocher_prj:/shared/cycle1_iit_shocher_prj,/home/users/yamitehrlich:/home/users/yamitehrlich

# Deep-detector-head + MLP-tail residual, trained FROM SCRATCH (no
# --warm_start_full).  This is the clean head-to-head against
# slurm_train_centernet_spnn_deephead.sh: same schedule, same hyperparams,
# only delta is --mlp_tail_hidden 128.  Tells us whether the MLP tail
# pulls its weight when both runs see the same training budget.
#
# At step 0 the tail's last 1x1 conv is zero -> tail outputs 0 -> block 4
# behaves exactly like the no-tail deephead.  As the U-net last-conv inside
# s/t/r wakes up (a few steps in), the tail wakes up shortly after.
# find_unused_parameters=True in DDP handles the early-step zero-grad case.
#
# Multi-GPU: identical layout to slurm_train_centernet_spnn_deephead.sh
# (8 GPUs, python -m torch.distributed.run, num_workers=4 per rank,
# global batch_size=64 -> per-GPU bs=8 under DDP).

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

python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    train.py \
        --dist \
        --arch spnn \
        --deep_det_head \
        --deep_head_hidden 128 \
        --mlp_tail_hidden 128 \
        --dataset pascal \
        --img_size 256 \
        --batch_size 64 \
        --num_epochs 200 \
        --lr 0.0005 \
        --lr_step 90,120 \
        --val_interval 5 \
        --log_interval 50 \
        --num_workers 4 \
        --data_dir ./data \
        --log_name spnn_centernet_deephead_NOdistill_hidden128_GN_mlptail128_fromscratch \
        --wandb \
        --wandb_project spnn-centernet \
        --spnn_backbone /shared/cycle1_iit_shocher_prj/SPNN/check_points_cls_imagenet1k_Apr29_more_blocks/best_model.pth \
        --hmap_init_scale 0.1 \
        --lambda_distill_hmap 0.0 \
        --lambda_distill_regs 0.0 \
        --lambda_distill_wh 0.0 &

PID=$!
wait $PID
