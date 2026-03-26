"""
Train a vanilla ResNet-50 classifier on CelebA-HQ 40 binary attributes.

This serves as a baseline classifier for DPS-style guidance comparison.
Training setup mirrors train_celeba.py: same data, same splits, same BCE loss,
same optimizer, same hyperparameters — only the architecture differs.

Usage:
    python train_resnet_baseline.py --dataset_path /path/to/CelebAMask-HQ
    python train_resnet_baseline.py  # auto-download via kagglehub
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm
from torchvision.models import resnet50

from data_loader import get_local_celebahq_loaders
from logger import setup_logger
from pytorch_optimization import get_linear_schedule_with_warmup


class ResNet50CelebA(nn.Module):
    """Standard ResNet-50 with a 40-output head for multi-label classification."""
    def __init__(self, num_classes=40, pretrained=True):
        super().__init__()
        self.backbone = resnet50(weights="IMAGENET1K_V1" if pretrained else None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def train(args):
    logger = setup_logger(args.checkpoint_dir, logfile_name=args.log_file, logger_name='resnet_baseline')

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # Dataset — same loader as SPNN training
    if args.dataset_path is not None:
        path = args.dataset_path
    else:
        print("No --dataset_path provided. Downloading via kagglehub...")
        import kagglehub
        path = kagglehub.dataset_download("liusonghua/celebamaskhq")
        path = os.path.join(path, "CelebAMask-HQ")

    print(f"Loading CelebA-HQ from: {path}")
    train_loader, dev_loader, test_loader = get_local_celebahq_loaders(
        root=path, batch_size=args.batch_size, img_size=args.img_size
    )

    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = ResNet50CelebA(num_classes=40, pretrained=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"ResNet-50 parameters: {total_params:,}")

    # Optimizer — same as SPNN training
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    total_steps = int(args.epoch * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_iters,
        fix_steps=int(args.fix_epoch * total_steps),
        num_training_steps=total_steps,
    )

    # Training loop
    best_val_loss = float("inf")
    current_step = 0

    logger.info(f"Start training ResNet-50 baseline for {args.epoch} epochs")
    logger.info(f"Total steps: {total_steps}, lr: {args.lr}")

    for epoch in range(args.epoch):
        model.train()
        show_loss = 0

        for train_img, train_labels in train_loader:
            train_img = train_img.to(device, non_blocking=True)
            train_labels = train_labels.to(device, dtype=torch.float32, non_blocking=True)
            if train_labels.min().item() < 0:
                train_labels = (train_labels + 1) / 2

            optimizer.zero_grad(set_to_none=True)

            logits = model(train_img)
            loss = nn.BCEWithLogitsLoss()(logits, train_labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
            scheduler.step()

            show_loss += loss.detach().item()
            current_step += 1

            if current_step % args.print_freq == 0:
                avg_loss = show_loss / args.print_freq
                logger.info(f'[epoch:{epoch}/{args.epoch}, step:{current_step}/{total_steps}, '
                            f'lr:{scheduler.get_lr()[0]:.3e}, bce:{avg_loss:.5f}]')
                show_loss = 0

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        att_wrong = np.zeros(40)

        with torch.no_grad():
            for val_img, val_labels in tqdm(dev_loader, desc=f"Val epoch {epoch}"):
                val_img = val_img.to(device, non_blocking=True)
                val_labels_t = val_labels.to(device, dtype=torch.float32, non_blocking=True)
                if val_labels_t.min().item() < 0:
                    val_labels_t = (val_labels_t + 1) / 2

                logits = model(val_img)
                loss = nn.BCEWithLogitsLoss()(logits, val_labels_t)
                val_loss_sum += loss.item()
                val_batches += 1

                # Per-attribute accuracy
                logits_np = logits.detach().cpu().numpy()
                val_labels_np = val_labels.numpy()
                if val_labels_np.min() < 0:
                    val_labels_np = (val_labels_np + 1) / 2
                preds = (logits_np > 0).astype(float)
                att_wrong += np.abs(preds - val_labels_np).sum(axis=0)

        att_wrong /= len(dev_loader.dataset)
        att_acc = 1 - att_wrong
        val_mean_acc = np.mean(att_acc)
        val_loss = val_loss_sum / val_batches

        logger.info(f'[epoch:{epoch} end] val_loss:{val_loss:.5f} mean_acc:{val_mean_acc:.5f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'resnet50_celeba_best.pth'))
            logger.info(f'[BEST] epoch:{epoch} val_loss:{best_val_loss:.5f}')

    # Test
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'resnet50_celeba_best.pth'),
                                     map_location=device, weights_only=True))
    model.eval()
    att_wrong = np.zeros(40)

    with torch.no_grad():
        for test_img, test_labels in tqdm(test_loader, desc="Test"):
            logits = model(test_img.to(device, non_blocking=True))
            test_labels_np = test_labels.numpy()
            if test_labels_np.min() < 0:
                test_labels_np = (test_labels_np + 1) / 2
            preds = (logits.detach().cpu().numpy() > 0).astype(float)
            att_wrong += np.abs(preds - test_labels_np).sum(axis=0)

    att_wrong /= len(test_loader.dataset)
    att_acc = 1 - att_wrong
    test_mean_acc = np.mean(att_acc)
    logger.info(f'[TEST] mean_acc:{test_mean_acc:.5f}')
    print(f"\nTest accuracy: {test_mean_acc:.4f}")
    print(f"Checkpoint saved to: {os.path.join(args.checkpoint_dir, 'resnet50_celeba_best.pth')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--fix_epoch', type=float, default=0.4)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--clip_norm', type=float, default=1.0)
    parser.add_argument('--warmup_iters', type=int, default=200)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--seed', type=int, default=556)
    parser.add_argument('--checkpoint_dir', type=str, default='check_points_resnet')
    parser.add_argument('--log_file', type=str, default='log.txt')
    parser.add_argument('--dataset_path', type=str, default=None)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train(args)


if __name__ == '__main__':
    main()
