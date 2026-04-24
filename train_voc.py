"""
Training entry point for SPNN object detection on Pascal VOC.

Uses HuggingFace Accelerate for multi-GPU (DDP).

Usage (single GPU):
  python train_voc.py --dataset_path datasets --is_forward_train --is_r_opt

Usage (multi-GPU via accelerate):
  accelerate launch --num_processes=N train_voc.py --dataset_path datasets --is_forward_train --is_r_opt
"""

import os
import argparse
import torch
import numpy as np
import random

from accelerate import Accelerator
from models import SPNN, ConvPINNBlock, PixelUnshuffleBlock
from logger import setup_logger
from train_detection import DetectionTrainer
from dataset_voc import get_voc_loaders
from diagnostics import PenroseChecker, GinvNormCalculator


def build_detection_spnn(grid_size=8, num_det_classes=20, num_boxes=2,
                          hidden=256, mix_type="cayley"):
    """Build a detection SPNN with the deeper backbone architecture.

    Backbone (4 ConvPINNBlocks, shared with classification):
      PixelUnshuffle(4) → ConvPINNBlock(48→24) → ConvPINNBlock(24→12) →
      PixelUnshuffle(4) → ConvPINNBlock(192→96) → ConvPINNBlock(96→48)

    Detection head (1 ConvPINNBlock):
      PixelUnshuffle(2) → ConvPINNBlock(192→out_ch)
    """
    out_ch = num_det_classes + num_boxes * 5  # C + B*5 = 20 + 10 = 30

    layer_channels = [
        # Backbone (shared with classification pretrained model)
        (PixelUnshuffleBlock, {"r": 4}),
        (ConvPINNBlock, {"in_ch": 48, "out_ch": 24, "hidden": hidden,
                         "scale_bound": 2.0, "feat_size": 64, "mix_type": mix_type}),
        (ConvPINNBlock, {"in_ch": 24, "out_ch": 12, "hidden": hidden,
                         "scale_bound": 2.0, "feat_size": 64, "mix_type": mix_type}),
        (PixelUnshuffleBlock, {"r": 4}),
        (ConvPINNBlock, {"in_ch": 192, "out_ch": 96, "hidden": hidden,
                         "scale_bound": 2.0, "feat_size": 16, "mix_type": mix_type}),
        (ConvPINNBlock, {"in_ch": 96, "out_ch": 48, "hidden": hidden,
                         "scale_bound": 2.0, "feat_size": 16, "mix_type": mix_type}),
        # Detection head
        (PixelUnshuffleBlock, {"r": 2}),
        (ConvPINNBlock, {"in_ch": 192, "out_ch": out_ch, "hidden": hidden,
                         "scale_bound": 2.0, "feat_size": grid_size, "mix_type": mix_type}),
    ]

    return SPNN(
        img_ch=3,
        num_classes=out_ch,
        img_size=256,
        layer_channels=layer_channels,
        output_spatial_size=(grid_size, grid_size),
    )


def transfer_backbone_weights(cls_checkpoint_path, det_model):
    """Copy backbone ConvPINNBlock weights from classification checkpoint to detection model.

    Backbone blocks are pinn.blocks[0..5] in both models:
      0: PixelUnshuffle(4)   — no weights
      1: ConvPINNBlock(48→24) — B1
      2: ConvPINNBlock(24→12) — B2
      3: PixelUnshuffle(4)   — no weights
      4: ConvPINNBlock(192→96) — B3
      5: ConvPINNBlock(96→48) — B4

    Returns number of transferred parameter tensors.
    """
    raw = torch.load(cls_checkpoint_path, map_location="cpu", weights_only=True)
    # Handle both formats: raw state_dict or nested checkpoint dict
    cls_state = raw.get("state_dict", raw) if isinstance(raw, dict) and "state_dict" in raw else raw
    det_state = det_model.state_dict()

    transferred = 0
    for key, val in cls_state.items():
        if any(key.startswith(f"pinn.blocks.{i}.") for i in range(6)):
            if key in det_state and det_state[key].shape == val.shape:
                det_state[key] = val
                transferred += 1

    det_model.load_state_dict(det_state)
    print(f"[transfer] Copied {transferred} backbone parameter tensors from classification checkpoint")
    return transferred


def main():
    parser = argparse.ArgumentParser(description="Train SPNN for object detection on Pascal VOC")

    # Data
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--voc_year', type=str, default='2012', choices=['2007', '2012'])
    parser.add_argument('--download', action='store_true')

    # Architecture
    parser.add_argument('--grid_size', type=int, default=8)
    parser.add_argument('--num_det_classes', type=int, default=20)
    parser.add_argument('--mix_type', type=str, default='cayley', choices=['cayley', 'householder'])

    # Training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--fix_epoch', type=float, default=0.4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--clip_norm', type=float, default=1.0)
    parser.add_argument('--warmup_iters', type=int, default=200)
    parser.add_argument('--seed', type=int, default=556)

    # Loss weights
    parser.add_argument('--lambda_det', type=float, default=1.0)
    parser.add_argument('--lambda_right_inverse', type=float, default=40.0)
    parser.add_argument('--lambda_img_rec', type=float, default=40.0)

    # r-optimization
    parser.add_argument('--lambda_r_norm', type=float, default=0.1)
    parser.add_argument('--lambda_r_rec', type=float, default=40.0)
    parser.add_argument('--lambda_r_cycle', type=float, default=1.0)
    parser.add_argument('--r_opt_epochs', type=int, default=50)
    parser.add_argument('--r_opt_lr', type=float, default=1e-4)

    # Backbone pretraining
    parser.add_argument('--pretrained_backbone', type=str, default=None,
                        help='Path to ImageNet classification checkpoint for backbone transfer')
    parser.add_argument('--freeze_backbone_epochs', type=int, default=0,
                        help='Freeze backbone for first N epochs (0 = no freezing)')

    # Evaluation
    parser.add_argument('--eval_freq', type=int, default=10,
                        help='Compute mAP every N epochs (default: 10). Val loss is computed every epoch.')

    # Flags
    parser.add_argument('--checkpoint_dir', type=str, default='check_points_det')
    parser.add_argument('--log_file', type=str, default='log_det.txt')
    parser.add_argument('--is_forward_train', action='store_true')
    parser.add_argument('--is_r_opt', action='store_true')

    # Wandb
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='spnn-detection')
    parser.add_argument('--wandb_run_name', type=str, default=None)

    args = parser.parse_args()

    # Accelerator handles device placement and DDP
    accelerator = Accelerator()
    device = accelerator.device
    is_main = accelerator.is_main_process

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logger = setup_logger(args.checkpoint_dir, logfile_name=args.log_file, logger_name='det')

    if is_main and args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )
    args.use_wandb = args.wandb

    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Data loaders
    if is_main:
        print(f"Loading Pascal VOC {args.voc_year} from: {args.dataset_path}")
    train_loader, val_loader = get_voc_loaders(
        root=args.dataset_path,
        batch_size=args.batch_size,
        img_size=256,
        S=args.grid_size,
        C=args.num_det_classes,
        year=args.voc_year,
        download=args.download,
    )
    if is_main:
        print(f"Train: {len(train_loader.dataset)} images, Val: {len(val_loader.dataset)} images")

    # Model (moved to device, but NOT wrapped yet — trainer does accelerator.prepare)
    model = build_detection_spnn(
        grid_size=args.grid_size,
        num_det_classes=args.num_det_classes,
        mix_type=args.mix_type,
    ).to(device)

    if is_main:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Device: {device}  |  Num processes: {accelerator.num_processes}")
        print(f"Model total_params: {total_params:,}")

    # Transfer pretrained backbone weights
    if args.pretrained_backbone is not None:
        if is_main:
            print(f"Transferring backbone from: {args.pretrained_backbone}")
        transfer_backbone_weights(args.pretrained_backbone, model)

    # Train
    trainer = DetectionTrainer(args, model, train_loader, val_loader, accelerator, logger)

    if args.is_forward_train:
        trainer.train()

    # Diagnostics & r-opt (main process only)
    if is_main:
        ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pth")
        if os.path.exists(ckpt_path):
            penrose_checker = PenroseChecker(logger)
            ginv_calculator = GinvNormCalculator(logger)

            out_ch = args.num_det_classes + 2 * 5
            hidden = 256
            model_kwargs = dict(
                img_ch=3, num_classes=out_ch, img_size=256,
                layer_channels=[
                    (PixelUnshuffleBlock, {"r": 4}),
                    (ConvPINNBlock, {"in_ch": 48, "out_ch": 24, "hidden": hidden,
                                     "scale_bound": 2.0, "feat_size": 64, "mix_type": args.mix_type}),
                    (ConvPINNBlock, {"in_ch": 24, "out_ch": 12, "hidden": hidden,
                                     "scale_bound": 2.0, "feat_size": 64, "mix_type": args.mix_type}),
                    (PixelUnshuffleBlock, {"r": 4}),
                    (ConvPINNBlock, {"in_ch": 192, "out_ch": 96, "hidden": hidden,
                                     "scale_bound": 2.0, "feat_size": 16, "mix_type": args.mix_type}),
                    (ConvPINNBlock, {"in_ch": 96, "out_ch": 48, "hidden": hidden,
                                     "scale_bound": 2.0, "feat_size": 16, "mix_type": args.mix_type}),
                    (PixelUnshuffleBlock, {"r": 2}),
                    (ConvPINNBlock, {"in_ch": 192, "out_ch": out_ch, "hidden": hidden,
                                     "scale_bound": 2.0, "feat_size": args.grid_size, "mix_type": args.mix_type}),
                ],
                output_spatial_size=(args.grid_size, args.grid_size),
            )

            penrose_metrics = penrose_checker.run_penrose_batched(
                checkpoint_path=ckpt_path, test_loader=val_loader,
                device=device, model_cls=SPNN, model_kwargs=model_kwargs,
            )
            print("[Before r-opt] Penrose metrics:")
            for k, v in penrose_metrics.items():
                print(f"  {k}: {v}")

            ginv_norm = ginv_calculator.run(
                checkpoint_path=ckpt_path, loader=val_loader,
                device=device, model_cls=SPNN, model_kwargs=model_kwargs,
            )
            print(f"  ||g'(g(x))||^2: {float(ginv_norm)}")

            if args.is_r_opt:
                r_opt_ckpt = os.path.join(args.checkpoint_dir, "best_model_r_opt.pth")
                trainer.train_r_opt(
                    checkpoint_path=ckpt_path,
                    device=device,
                    loader=train_loader,
                    epochs=args.r_opt_epochs,
                    lr=args.r_opt_lr,
                    out_checkpoint_path=r_opt_ckpt,
                )

                penrose_after = penrose_checker.run_penrose_batched(
                    checkpoint_path=r_opt_ckpt, test_loader=val_loader,
                    device=device, model_cls=SPNN, model_kwargs=model_kwargs,
                )
                print("[After r-opt] Penrose metrics:")
                for k, v in penrose_after.items():
                    print(f"  {k}: {v}")
        else:
            print(f"No checkpoint found at {ckpt_path}, skipping diagnostics.")

    if is_main and args.wandb:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    main()
