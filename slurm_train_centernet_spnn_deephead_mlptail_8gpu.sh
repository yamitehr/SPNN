#!/bin/bash
#SBATCH -A cycle1_iit_shocher_prj            # account
#SBATCH -p compute-gpu                       # partition
#SBATCH --qos=owner_95
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH -o /home/users/%u/logs/%x_%j.out
#SBATCH -e /home/users/%u/logs/%x_%j.err
#SBATCH --job-name=centernet_spnn_deephead_mlptail_8gpu
#SBATCH --container-image=docker://nvcr.io/nvidia/pytorch:25.10-py3
#SBATCH --container-mounts=/shared/cycle1_iit_shocher_prj:/shared/cycle1_iit_shocher_prj,/home/users/yamitehrlich:/home/users/yamitehrlich

# Multi-GPU (8) version of slurm_train_centernet_spnn_deephead_mlptail.sh.
#
# Launch model: 1 process per GPU via torchrun, NCCL backend (set by --dist
# in train.py).  Per-rank reads LOCAL_RANK / WORLD_SIZE from env.  Validation
# is rank-0-only (race-free file writes).
#
# Effective vs single-GPU:
#   global batch_size = 64  -> each GPU sees bs=8 per step  (DDP allreduce
#   keeps the gradient identical to single-GPU bs=64).
#   LR = 2.5e-4 unchanged (no linear-scaling — same effective bs).
# Net: ~8x wall-clock speedup at identical optimization dynamics.
#
# To go for higher per-step throughput, raise batch_size to 256 (each GPU
# bs=32) AND scale LR ~4x to 1e-3 (linear scaling rule).  Skipped here to
# keep warm-start fine-tuning stable.

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

# Use `python -m torch.distributed.run` instead of the bare `torchrun` binary:
# in this container the system `torchrun` lives at /usr/local/bin/torchrun and
# spawns children with /usr/bin/python, which is NOT the activated venv (no
# cv2 / torch / etc. installed there).  Going through the venv's `python`
# ensures the spawned children inherit the same interpreter (sys.executable)
# and have all our dependencies available.
#
# --standalone picks an unused MASTER_PORT on localhost — fine for
# single-node multi-GPU, no rendezvous file needed.  Each spawned process
# reads its LOCAL_RANK / RANK / WORLD_SIZE from env vars.
python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    train.py \
        --dist \
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
        --num_workers 4 \
        --data_dir ./data \
        --log_name spnn_centernet_deephead_NOdistill_hidden256_GN_mlptail128_8gpu \
        --wandb \
        --wandb_project spnn-centernet \
        --hmap_init_scale 0.1 \
        --lambda_distill_hmap 0.0 \
        --lambda_distill_regs 0.0 \
        --lambda_distill_wh 0.0 &

PID=$!
wait $PID
