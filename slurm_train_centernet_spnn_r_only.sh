#!/bin/bash
#SBATCH -A cycle1_iit_shocher_prj            # account
#SBATCH -p compute-gpu                       # partition
#SBATCH --qos=owner_95
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH -o /shared/cycle1_iit_shocher_prj/SPNN/logs/%x_%j.out
#SBATCH -e /shared/cycle1_iit_shocher_prj/SPNN/logs/%x_%j.err
#SBATCH --job-name=centernet_spnn_r_only_8gpu
#SBATCH --container-image=docker://nvcr.io/nvidia/pytorch:25.10-py3
#SBATCH --container-mounts=/shared/cycle1_iit_shocher_prj:/shared/cycle1_iit_shocher_prj,/home/users/yamitehrlich:/home/users/yamitehrlich

# r-net-only fine-tune phase, multi-GPU (8) version.
#
# Loads a detection-trained ckpt (--pretrain_name), freezes the entire
# forward path (s/t/mix/internal-affine/backbone), and trains ONLY the .r
# submodules of every ConvPINNBlock with two losses:
#     lambda_r_norm  * ||z(pinv(forward(x))) - z(zeros)||²    (G-norm)
#     lambda_r_rec   * ||pinv(forward(x))    - x||²            (image rec)
#
# Detection performance is preserved exactly because the forward path is
# frozen. r-net quality after this phase directly improves DDNM's
# Ap(y, latents=None) reconstruction and the NLBP correction stability.
#
# Assumptions (assumed by the r-opt code, not configurable):
#   - ckpt was trained with --no_hmap_scale --no_hmap_bias --internal_head_affine
#   - head_mode is INFERRED from the ckpt (presence of hmap_mix.* keys)
#
# Set PRETRAIN_NAME below to the detection-trained run you want to fine-tune;
# OUT_NAME is the new ckpt directory under centernet_ref/ckpt/.

set -euxo pipefail
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

PRETRAIN_NAME=${PRETRAIN_NAME:-spnn_centernet_deephead_NOdistill_hidden128_GN_UnetLike_recLoss_v16}
OUT_NAME=${OUT_NAME:-${PRETRAIN_NAME}_r_only}

python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    train_copy.py \
        --dist \
        --arch spnn \
        --only_train_r \
        --pretrain_name "$PRETRAIN_NAME" \
        --deep_det_head \
        --deep_head_hidden 128 \
        --dataset pascal \
        --img_size 256 \
        --batch_size 64 \
        --num_epochs 150 \
        --lr 0.001 \
        --num_workers 4 \
        --no_hmap_scale \
        --no_hmap_bias \
        --internal_head_affine \
        --lambda_r_norm 1.0 \
        --lambda_r_rec 10.0 \
        --data_dir ./data \
        --log_name "$OUT_NAME" \
        --wandb \
        --wandb_project spnn-centernet &

PID=$!
wait $PID
