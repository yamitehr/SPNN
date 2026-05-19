#!/bin/bash
#SBATCH -A cycle1_iit_shocher_prj
#SBATCH -p compute-gpu
#SBATCH --qos=owner_95
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH -o /shared/cycle1_iit_shocher_prj/SPNN/logs/%x_%A_%a.out
#SBATCH -e /shared/cycle1_iit_shocher_prj/SPNN/logs/%x_%A_%a.err
#SBATCH --array=0-31
#SBATCH --container-image=docker://nvcr.io/nvidia/pytorch:25.10-py3
#SBATCH --container-mounts=/shared/cycle1_iit_shocher_prj:/shared/cycle1_iit_shocher_prj,/home/users/yamitehrlich:/home/users/yamitehrlich

# Usage:
#   sbatch --job-name=ddnm_imagenet_10k slurm_eval_imagenet_10k.sh
#
# Runs DDNM ImageNet classifier reconstruction on the 10k val subset
# (imagenet_val_10k.txt) split across 32 GPU array tasks.

set -euxo pipefail
export PYTHONUNBUFFERED=1

mkdir -p /shared/cycle1_iit_shocher_prj/SPNN/logs

TOTAL_IMAGES=10000
NUM_JOBS=32
IMAGES_PER_JOB=$(( (TOTAL_IMAGES + NUM_JOBS - 1) / NUM_JOBS ))   # ceil = 313
START=$(( SLURM_ARRAY_TASK_ID * IMAGES_PER_JOB ))
END=$(( START + IMAGES_PER_JOB ))
if [ "${END}" -gt "${TOTAL_IMAGES}" ]; then END=${TOTAL_IMAGES}; fi

RESULTS_NAME="imagenet1k_ddnm_10k"
SPNN_CKPT="../check_points_cls_imagenet1k_Apr27_2003/checkpoint.pth.tar"

source /shared/cycle1_iit_shocher_prj/SPNN/.venv-cluster/bin/activate
cd /shared/cycle1_iit_shocher_prj/SPNN/DDNM

echo "Job ${SLURM_ARRAY_TASK_ID}: images ${START}-${END} (total=${TOTAL_IMAGES})"

python main.py --ni \
    --config imagenet_256.yml \
    --path_y imagenet \
    --spnn_ckpt "${SPNN_CKPT}" \
    --spnn_num_classes 1000 \
    --spnn_mix_type householder \
    --spnn_scale_bound 1.0 \
    --subset_start ${START} \
    --subset_end ${END} \
    --seed 1234 \
    -i "${RESULTS_NAME}/part_${SLURM_ARRAY_TASK_ID}"
