#!/bin/bash
#SBATCH -A cycle1_iit_shocher_prj
#SBATCH -p compute-gpu
#SBATCH --qos=owner_95
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH -o /home/users/%u/logs/%x_%j.out
#SBATCH -e /home/users/%u/logs/%x_%j.err
#SBATCH --job-name=centernet_resnet18
#SBATCH --container-image=docker://nvcr.io/nvidia/pytorch:25.10-py3
#SBATCH --container-mounts=/shared/cycle1_iit_shocher_prj:/shared/cycle1_iit_shocher_prj,/home/users/yamitehrlich:/home/users/yamitehrlich

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

# Train CenterNet ResNet-18 (no DCN) on VOC, img_size=256
# Uses ImageNet pretrained ResNet-18 weights automatically
python train.py \
    --arch resnet_18 \
    --dataset pascal \
    --img_size 256 \
    --batch_size 32 \
    --num_epochs 70 \
    --lr 1.25e-4 \
    --lr_step 45,60 \
    --val_interval 5 \
    --log_interval 50 \
    --num_workers 4 \
    --data_dir ./data \
    --log_name centernet_resnet18_voc_256 &

PID=$!
wait $PID
