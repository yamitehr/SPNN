"""
Train FCOS with SPNN backbone on Pascal VOC.

Uses the FCOS repo's tested loss, target generation, and evaluation code.
Only the backbone is replaced with SPNN.

Usage:
  python train_voc_spnn.py --voc_root /path/to/VOCdevkit/VOC2012 --epochs 50
  python train_voc_spnn.py --voc_root /path/to/VOCdevkit/VOC2012 --pretrained_backbone /path/to/imagenet_ckpt.pth
"""

import os
import sys
import time
import math
import argparse
import random

import torch
import torch.backends.cudnn as cudnn
import numpy as np

# Add project root to path for models.py
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from model.fcos import FCOSDetector, FCOSDetectorE2E
from model.config import SPNNConfig, SPNNE2EConfig
from dataset.VOC_dataset import VOCDataset
from dataset.augment import Transforms
from eval_voc import eval_ap_2d, sort_by_score


def parse_args():
    parser = argparse.ArgumentParser(description="Train FCOS-SPNN on VOC")
    parser.add_argument("--voc_root", type=str, required=True,
                        help="Path to VOC root (e.g., datasets/VOCdevkit/VOC2012)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="Initial learning rate (SGD)")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--warmup_steps", type=int, default=501)
    parser.add_argument("--n_cpu", type=int, default=4)
    parser.add_argument("--n_gpu", type=str, default='0',
                        help="GPU IDs (comma-separated)")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_freq", type=int, default=5,
                        help="Evaluate mAP every N epochs")
    parser.add_argument("--save_dir", type=str, default="./checkpoint_spnn")
    parser.add_argument("--seed", type=int, default=0)

    # SPNN-specific
    parser.add_argument("--mix_type", type=str, default="householder",
                        choices=["cayley", "householder"])
    parser.add_argument("--scale_bound", type=float, default=2.0,
                        help="Scale bound for s-network: s in [exp(-b), exp(b)]")
    parser.add_argument("--pretrained_backbone", type=str, default=None,
                        help="Path to ImageNet classification checkpoint for backbone transfer")
    parser.add_argument("--freeze_backbone_epochs", type=int, default=0,
                        help="Freeze SPNN backbone for first N epochs")

    parser.add_argument("--end_to_end", action="store_true",
                        help="Use end-to-end invertible SPNN (for DDNM). "
                             "Single scale, no FPN/head, full pinv() support.")

    # SPNN cycle/reconstruction losses
    parser.add_argument("--lambda_cycle", type=float, default=0.0,
                        help="Weight for right-inverse cycle loss || f(f'(y)) - y ||^2")
    parser.add_argument("--lambda_rec", type=float, default=0.0,
                        help="Weight for image reconstruction loss || f'(f(x)) - x ||^2")

    # Wandb
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="spnn-fcos-voc")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    return parser.parse_args()


class VOCDataset256(VOCDataset):
    """VOCDataset that resizes to 256x256 (required by SPNN)."""

    def __init__(self, root_dir, split='trainval', use_difficult=False,
                 is_train=True, augment=None):
        super().__init__(
            root_dir=root_dir,
            resize_size=[256, 256],  # will be overridden
            split=split,
            use_difficult=use_difficult,
            is_train=is_train,
            augment=augment,
        )

    def preprocess_img_boxes(self, image, boxes, input_ksize):
        """Resize to exactly 256x256 (no aspect ratio preservation)."""
        import cv2
        h, w, _ = image.shape
        target_h, target_w = 256, 256

        image_resized = cv2.resize(image, (target_w, target_h))

        if boxes is None:
            return image_resized

        # Scale boxes
        scale_w = target_w / w
        scale_h = target_h / h
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_w
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_h
        return image_resized, boxes


@torch.no_grad()
def evaluate(model, eval_loader, num_classes):
    """Run mAP evaluation using the FCOS repo's eval code."""
    model.eval()
    gt_boxes = []
    gt_classes = []
    pred_boxes = []
    pred_classes = []
    pred_scores = []

    for img, boxes, classes in eval_loader:
        out = model(img.cuda())
        # out = (scores, classes, boxes) from inference mode
        pred_boxes.append(out[2][0].cpu().numpy())
        pred_classes.append(out[1][0].cpu().numpy())
        pred_scores.append(out[0][0].cpu().numpy())
        gt_boxes.append(boxes[0].numpy())
        gt_classes.append(classes[0].numpy())

    pred_boxes, pred_classes, pred_scores = sort_by_score(
        pred_boxes, pred_classes, pred_scores
    )
    all_ap = eval_ap_2d(
        gt_boxes, gt_classes, pred_boxes, pred_classes, pred_scores,
        iou_thread=0.5, num_cls=num_classes + 1,  # +1 for background
    )

    mAP = 0.0
    for class_id, ap in all_ap.items():
        mAP += float(ap)
    if len(all_ap) > 0:
        mAP /= num_classes
    return mAP, all_ap


def main():
    opt = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.n_gpu
    os.makedirs(opt.save_dir, exist_ok=True)

    # Seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    cudnn.benchmark = True
    cudnn.deterministic = False

    # Wandb
    use_wandb = opt.wandb
    if use_wandb:
        import wandb
        wandb.init(project=opt.wandb_project, name=opt.wandb_run_name, config=vars(opt))

    # Config
    if opt.end_to_end:
        config = SPNNE2EConfig()
    else:
        config = SPNNConfig()
    config.spnn_mix_type = opt.mix_type
    config.spnn_scale_bound = opt.scale_bound
    config.spnn_pretrained = opt.pretrained_backbone

    # Dataset
    transform = Transforms()
    train_dataset = VOCDataset256(
        root_dir=opt.voc_root, split='trainval',
        use_difficult=False, is_train=True, augment=transform,
    )
    eval_dataset = VOCDataset256(
        root_dir=opt.voc_root, split='val' if os.path.exists(
            os.path.join(opt.voc_root, "ImageSets", "Main", "val.txt")
        ) else 'test',
        use_difficult=False, is_train=False, augment=None,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=opt.n_cpu, worker_init_fn=np.random.seed(opt.seed),
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=1, shuffle=False,
        collate_fn=eval_dataset.collate_fn, num_workers=opt.n_cpu,
    )

    print(f"Train: {len(train_dataset)} images, Eval: {len(eval_dataset)} images")

    # Model — training mode
    if opt.end_to_end:
        model_train = FCOSDetectorE2E(mode="training", config=config).cuda()
    else:
        model_train = FCOSDetector(mode="training", config=config).cuda()
    model_train = torch.nn.DataParallel(model_train)

    total_params = sum(p.numel() for p in model_train.parameters())
    trainable_params = sum(p.numel() for p in model_train.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}  Trainable: {trainable_params:,}")

    # Optimizer
    optimizer = torch.optim.SGD(
        model_train.parameters(), lr=opt.lr,
        momentum=opt.momentum, weight_decay=opt.weight_decay,
    )

    steps_per_epoch = len(train_dataset) // opt.batch_size
    TOTAL_STEPS = steps_per_epoch * opt.epochs
    LR_INIT = opt.lr
    LR_END = opt.lr * 0.01
    GLOBAL_STEPS = 1

    best_mAP = 0.0
    model_train.train()

    for epoch in range(opt.epochs):
        # Freeze/unfreeze backbone
        if opt.freeze_backbone_epochs > 0:
            if epoch < opt.freeze_backbone_epochs:
                if epoch == 0:
                    for name, param in model_train.named_parameters():
                        if 'backbone.spnn' in name:
                            param.requires_grad = False
                    print(f"INFO===>Backbone frozen for first {opt.freeze_backbone_epochs} epochs")
            elif epoch == opt.freeze_backbone_epochs:
                for name, param in model_train.named_parameters():
                    if 'backbone.spnn' in name:
                        param.requires_grad = True
                print("INFO===>Backbone unfrozen")

        for epoch_step, data in enumerate(train_loader):
            batch_imgs, batch_boxes, batch_classes = data
            batch_imgs = batch_imgs.cuda()
            batch_boxes = batch_boxes.cuda()
            batch_classes = batch_classes.cuda()

            # Learning rate schedule: warmup + step decay
            if GLOBAL_STEPS < opt.warmup_steps:
                lr = float(GLOBAL_STEPS / opt.warmup_steps * LR_INIT)
                for param in optimizer.param_groups:
                    param['lr'] = lr
            elif GLOBAL_STEPS == int(TOTAL_STEPS * 0.667):
                lr = LR_INIT * 0.1
                for param in optimizer.param_groups:
                    param['lr'] = lr
                print(f"INFO===>LR reduced to {lr:.6e} at step {GLOBAL_STEPS}")
            elif GLOBAL_STEPS == int(TOTAL_STEPS * 0.889):
                lr = LR_INIT * 0.01
                for param in optimizer.param_groups:
                    param['lr'] = lr
                print(f"INFO===>LR reduced to {lr:.6e} at step {GLOBAL_STEPS}")

            start_time = time.time()

            optimizer.zero_grad()
            losses = model_train([batch_imgs, batch_boxes, batch_classes])
            loss = losses[-1].mean()  # total_loss

            # SPNN cycle and reconstruction losses
            cycle_l = torch.tensor(0.0, device=batch_imgs.device)
            rec_l = torch.tensor(0.0, device=batch_imgs.device)
            if opt.lambda_cycle > 0 or opt.lambda_rec > 0:
                if opt.end_to_end:
                    spnn = model_train.module.spnn
                else:
                    spnn = model_train.module.fcos_body.backbone.spnn

                logits = spnn(batch_imgs)
                x_inv = spnn.pinv(logits)

                if opt.lambda_cycle > 0:
                    y_cycle = spnn(x_inv)
                    cycle_l = (y_cycle - logits).pow(2).mean()
                    loss = loss + opt.lambda_cycle * cycle_l

                if opt.lambda_rec > 0:
                    rec_l = (x_inv - batch_imgs).pow(2).mean()
                    loss = loss + opt.lambda_rec * rec_l

            loss.backward()

            # Gradient clipping
            if opt.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model_train.parameters(), opt.grad_clip)

            optimizer.step()

            end_time = time.time()
            cost_time = int((end_time - start_time) * 1000)

            lr = optimizer.param_groups[0]['lr']

            if GLOBAL_STEPS % 50 == 0 or epoch_step == 0:
                msg = (
                    "global_steps:%d epoch:%d steps:%d/%d cls_loss:%.4f cnt_loss:%.4f "
                    "reg_loss:%.4f cost_time:%dms lr=%.4e total_loss:%.4f" % (
                        GLOBAL_STEPS, epoch + 1, epoch_step + 1, steps_per_epoch,
                        losses[0].mean(), losses[1].mean(), losses[2].mean(),
                        cost_time, lr, loss.item(),
                    )
                )
                if opt.lambda_cycle > 0 or opt.lambda_rec > 0:
                    msg += " cycle:%.4f rec:%.4f" % (cycle_l.item(), rec_l.item())
                print(msg)

            if use_wandb:
                import wandb
                log_dict = {
                    "train/cls_loss": losses[0].mean().item(),
                    "train/cnt_loss": losses[1].mean().item(),
                    "train/reg_loss": losses[2].mean().item(),
                    "train/total_loss": loss.item(),
                    "train/lr": lr,
                    "global_step": GLOBAL_STEPS,
                }
                if opt.lambda_cycle > 0:
                    log_dict["train/cycle_loss"] = cycle_l.item()
                if opt.lambda_rec > 0:
                    log_dict["train/rec_loss"] = rec_l.item()
                wandb.log(log_dict, step=GLOBAL_STEPS)

            GLOBAL_STEPS += 1

        # # Save checkpoint every epoch
        # torch.save(
        #     model_train.state_dict(),
        #     os.path.join(opt.save_dir, f"model_{epoch + 1}.pth"),
        # )

        # Evaluate
        if (epoch + 1) % opt.eval_freq == 0 or epoch == opt.epochs - 1:
            print(f"\n===== Evaluating at epoch {epoch + 1} =====")
            # Build inference model without reloading pretrained backbone
            # (we'll copy trained weights from model_train instead)
            saved_pretrained = config.spnn_pretrained
            config.spnn_pretrained = None
            if opt.end_to_end:
                model_eval = FCOSDetectorE2E(mode="inference", config=config)
            else:
                model_eval = FCOSDetector(mode="inference", config=config)
            config.spnn_pretrained = saved_pretrained
            model_eval = torch.nn.DataParallel(model_eval)

            # Transfer weights from training model to inference model
            # They share the same backbone + FPN + head, just different wrappers
            train_state = model_train.state_dict()
            eval_state = model_eval.state_dict()
            # Copy matching keys (fcos_body.* exists in both)
            for key in eval_state:
                if key in train_state and train_state[key].shape == eval_state[key].shape:
                    eval_state[key] = train_state[key]
            model_eval.load_state_dict(eval_state)
            model_eval = model_eval.cuda().eval()

            mAP, all_ap = evaluate(model_eval, eval_loader, config.class_num)
            print(f"mAP@0.5 = {mAP:.4f}")
            for cls_id, ap in sorted(all_ap.items()):
                cls_name = eval_dataset.id2name.get(int(cls_id), str(cls_id))
                print(f"  {cls_name}: {ap:.4f}")

            if use_wandb:
                import wandb
                log_dict = {"eval/mAP": mAP, "epoch": epoch + 1}
                for cls_id, ap in all_ap.items():
                    cls_name = eval_dataset.id2name.get(int(cls_id), str(cls_id))
                    log_dict[f"eval/AP_{cls_name}"] = ap
                wandb.log(log_dict, step=GLOBAL_STEPS)

            if mAP > best_mAP:
                best_mAP = mAP
                torch.save(
                    model_train.state_dict(),
                    os.path.join(opt.save_dir, "best_model.pth"),
                )
                print(f"New best mAP: {best_mAP:.4f}")

            del model_eval
            model_train.train()
            print("=" * 50 + "\n")

    print(f"\nTraining complete. Best mAP@0.5 = {best_mAP:.4f}")
    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
