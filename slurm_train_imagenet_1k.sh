#!/bin/bash
#SBATCH -A cycle1_iit_shocher_prj            # account
#SBATCH -p compute-gpu                       # partition
#SBATCH --qos=owner_95
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=192G
#SBATCH --time=5-00:00:00
#SBATCH -o /home/users/%u/logs/%x_%j.out
#SBATCH -e /home/users/%u/logs/%x_%j.err
#SBATCH --job-name=imagenet_spnn
#SBATCH --container-image=docker://nvcr.io/nvidia/pytorch:25.10-py3
#SBATCH --container-mounts=/shared/cycle1_iit_shocher_prj:/shared/cycle1_iit_shocher_prj,/home/users/yamitehrlich:/home/users/yamitehrlich


set -euo pipefail

mkdir -p /home/users/$USER/logs

source /shared/cycle1_iit_shocher_prj/SPNN/.venv-cluster/bin/activate
cd /shared/cycle1_iit_shocher_prj/SPNN

# Cap BLAS thread libraries to avoid RLIMIT_NPROC explosion under DDP
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cleanup() {
    echo "[$(date -Is)] SIGTERM received; forwarding to PID=$PID"
    kill -TERM "$PID"
    wait "$PID"
    exit 0
}
trap cleanup SIGTERM

python train_imagenet.py datasets/imagenet \
    --num-classes 1000 \
    --epochs 150 \
    --batch-size 1024 \
    --lr 0.2 \
    --scheduler cosine \
    --warmup-epochs 8 \
    --scale-bound 1.0 \
    --lambda-cycle 0.0 \
    --lambda-rec 5.0 \
    --workers 64 \
    --print-freq 100 \
    --checkpoint-dir check_points_cls_imagenet1k_May19_b200_bf16 \
    --multiprocessing-distributed \
    --dist-url tcp://127.0.0.1:23456 \
    --world-size 1 \
    --rank 0 \
    --wandb \
    --wandb-project spnn-imagenet \
    --wandb-run-name imagenet1k-8gpu_May19_b200_bf16 &

PID=$!
wait $PID
