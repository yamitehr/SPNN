"""
Detection trainer for SPNN on Pascal VOC.

Combines YOLOv1 detection loss with SPNN cycle losses (right-inverse + image reconstruction).
Follows the same two-stage training as CelebATrainer:
  Stage 1: Forward training (detection + cycle losses)
  Stage 2: r-network optimization

Uses S=8, B=2, C=20 (original YOLOv1 config adapted for 256x256 images).
"""

import os
import torch
import torch.nn as nn
from tqdm import tqdm
from models import SPNN, ConvPINNBlock
from yolov1.loss import YoloLoss
from yolov1.utils import get_bboxes, mean_average_precision
from pytorch_optimization import get_linear_schedule_with_warmup

# Detection config: S=8 (required by PixelUnshuffle), B=2 (original YOLOv1), C=20 (VOC)
S = 8
B = 2
C = 20
OUT_CH = C + B * 5  # 30


def _wandb_log(args, data, step=None):
    if getattr(args, "use_wandb", False):
        import wandb
        wandb.log(data, step=step)


class DetectionTrainer:
    def __init__(self, args, model, train_loader, val_loader, device, logger, n_gpu):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.logger = logger
        self.n_gpu = n_gpu

    def train(self):
        self.logger.info(f'Loss Weights: det={self.args.lambda_det}, '
                         f'cycle={self.args.lambda_right_inverse}, rec={self.args.lambda_img_rec}')

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr,
            betas=(self.args.beta1, self.args.beta2)
        )
        total_steps = int(self.args.epoch * len(self.train_loader) / max(1, self.n_gpu))
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_iters,
            fix_steps=int(self.args.fix_epoch * total_steps),
            num_training_steps=total_steps,
        )

        yolo_loss_fn = YoloLoss(S=S, B=B, C=C)

        current_step = 0
        self.model.train()
        best_val_loss = float("inf")

        self.logger.info(f'Start training for {self.args.epoch} epochs, S={S}, B={B}, C={C}')

        for epoch in range(self.args.epoch):
            show_loss = 0
            show_det_loss = 0
            show_cycle_loss = 0
            show_img_rec_loss = 0
            n_steps_epoch = 0

            for train_img, train_labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}"):
                train_img = train_img.to(device=self.device, non_blocking=True)
                train_labels = train_labels.to(device=self.device, dtype=torch.float32, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                # Forward: SPNN outputs [B, 30, 8, 8]
                output = self.model(train_img)

                # YoloLoss expects [batch, S*S*(C+B*5)] — flatten spatial dims
                # YoloLoss uses reduction="sum", so normalize by batch size
                # to match the mean-reduced cycle losses
                output_flat = output.permute(0, 2, 3, 1).reshape(output.shape[0], -1)
                loss_det = yolo_loss_fn(output_flat, train_labels) / train_img.shape[0]

                # Right-inverse loss: || g(g'(y)) - y ||²
                model_ref = self.model.module if hasattr(self.model, "module") else self.model
                x_inv = model_ref.pinv(output)
                y_cycle = self.model(x_inv)
                cycle_l = (y_cycle - output).pow(2).mean()

                # Image reconstruction loss: || g'(g(x)) - x ||²
                img_rec_l = (x_inv - train_img).pow(2).mean()

                loss = (self.args.lambda_det * loss_det
                        + self.args.lambda_right_inverse * cycle_l
                        + self.args.lambda_img_rec * img_rec_l)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_norm)
                optimizer.step()
                scheduler.step()

                show_loss += loss.detach().item()
                show_det_loss += loss_det.detach().item()
                show_cycle_loss += cycle_l.detach().item()
                show_img_rec_loss += img_rec_l.detach().item()
                n_steps_epoch += 1
                current_step += 1

                if current_step % self.args.print_freq == 0:
                    n = self.args.print_freq
                    avg_loss = show_loss / n
                    avg_det = show_det_loss / n
                    avg_cycle = show_cycle_loss / n
                    avg_rec = show_img_rec_loss / n
                    lr = scheduler.get_lr()[0]

                    self.logger.info(
                        f'[epoch:{epoch+1}/{self.args.epoch}, step:{current_step}, '
                        f'lr:{lr:.3e}] '
                        f'loss:{avg_loss:.4f} det:{avg_det:.4f} '
                        f'cycle:{avg_cycle:.6f} rec:{avg_rec:.6f}'
                    )
                    _wandb_log(self.args, {
                        "train/loss": avg_loss,
                        "train/det_loss": avg_det,
                        "train/cycle_loss": avg_cycle,
                        "train/img_rec_loss": avg_rec,
                        "train/lr": lr,
                    }, step=current_step)

                    show_loss = show_det_loss = show_cycle_loss = show_img_rec_loss = 0

            # End-of-epoch validation
            self.model.eval()
            val_loss_sum = 0.0
            val_det_sum = 0.0
            val_cycle_sum = 0.0
            val_rec_sum = 0.0
            val_batches = 0

            with torch.no_grad():
                for val_img, val_labels in self.val_loader:
                    val_img = val_img.to(self.device, non_blocking=True)
                    val_labels = val_labels.to(self.device, dtype=torch.float32, non_blocking=True)

                    output = self.model(val_img)
                    output_flat = output.permute(0, 2, 3, 1).reshape(output.shape[0], -1)
                    loss_det = yolo_loss_fn(output_flat, val_labels) / val_img.shape[0]

                    model_ref = self.model.module if hasattr(self.model, "module") else self.model
                    x_inv = model_ref.pinv(output)
                    y_cycle = self.model(x_inv)
                    cycle_l = (y_cycle - output).pow(2).mean()
                    img_rec_l = (x_inv - val_img).pow(2).mean()

                    val_loss = (self.args.lambda_det * loss_det
                                + self.args.lambda_right_inverse * cycle_l
                                + self.args.lambda_img_rec * img_rec_l)
                    val_loss_sum += val_loss.item()
                    val_det_sum += loss_det.item()
                    val_cycle_sum += cycle_l.item()
                    val_rec_sum += img_rec_l.item()
                    val_batches += 1

            avg_val_loss = val_loss_sum / max(1, val_batches)
            avg_val_det = val_det_sum / max(1, val_batches)
            avg_val_cycle = val_cycle_sum / max(1, val_batches)
            avg_val_rec = val_rec_sum / max(1, val_batches)
            self.logger.info(f'[epoch:{epoch+1} end] val_loss:{avg_val_loss:.5f}')

            # Compute mAP using repo's get_bboxes + mean_average_precision
            pred_boxes, target_boxes = get_bboxes(
                self.val_loader, self.model,
                iou_threshold=0.5, threshold=0.4,
                device=self.device, S=S,
            )
            map_val = mean_average_precision(
                pred_boxes, target_boxes,
                iou_threshold=0.5, box_format="midpoint",
                num_classes=C,
            )
            self.logger.info(f'[epoch:{epoch+1}] val mAP@0.5: {map_val:.4f}')

            _wandb_log(self.args, {
                "val/loss": avg_val_loss,
                "val/det_loss": avg_val_det,
                "val/cycle_loss": avg_val_cycle,
                "val/img_rec_loss": avg_val_rec,
                "val/mAP_0.5": map_val,
                "epoch": epoch + 1,
            }, step=current_step)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.logger.info(f'[BEST] epoch:{epoch+1} val_loss:{best_val_loss:.5f} mAP:{map_val:.4f}')
                model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                torch.save(model_to_save.state_dict(), os.path.join(self.args.checkpoint_dir, 'best_model.pth'))

            self.model.train()

        self.logger.info('End of forward training.')
        return self.model

    def _train_r_opt(self, model, loader, device, epochs, lr):
        """Train only r-networks in ConvPINNBlocks. Shape-agnostic — works for any output."""
        for p in model.parameters():
            p.requires_grad = False

        r_params = []
        for m in model.modules():
            if isinstance(m, ConvPINNBlock):
                for p in m.r.parameters():
                    p.requires_grad = True
                    r_params.append(p)

        assert len(r_params) > 0, "No r-parameters found for r-opt."

        opt = torch.optim.Adam(r_params, lr=lr)
        model.train()

        for ep in range(epochs):
            total_loss = 0.0
            total_g_pinv = 0.0
            total_rec = 0.0
            total_cycle = 0.0
            steps = 0
            for x_batch, _ in loader:
                x_batch = x_batch.to(device, non_blocking=True)

                with torch.no_grad():
                    y_batch = model(x_batch)

                x_tag = model.pinv(y_batch)

                y, z_list = model(x_tag, return_latents=True)
                y_0, z_list_0 = model(torch.zeros_like(x_tag), return_latents=True)

                B_sz = y.shape[0]

                parts = []
                parts_0 = []
                for z, z_0 in zip(z_list, z_list_0):
                    if z is None:
                        continue
                    parts.append(z.view(B_sz, -1))
                    parts_0.append(z_0.view(B_sz, -1))

                G_pinv = torch.cat(parts, dim=1)
                G_0 = torch.cat(parts_0, dim=1)
                loss_G_pinv = (G_pinv - G_0).pow(2).mean()

                img_rec_l = (x_tag - x_batch).pow(2).mean()

                y_cycle = model(x_tag)
                cycle_l = (y_cycle - y_batch).pow(2).mean()

                loss = (self.args.lambda_r_norm * loss_G_pinv
                        + self.args.lambda_r_rec * img_rec_l
                        + self.args.lambda_r_cycle * cycle_l)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(r_params, max_norm=1.0)
                opt.step()

                total_loss += loss.item()
                total_g_pinv += loss_G_pinv.item()
                total_rec += img_rec_l.item()
                total_cycle += cycle_l.item()
                steps += 1

            avg_loss = total_loss / max(1, steps)
            avg_g_pinv = total_g_pinv / max(1, steps)
            avg_rec = total_rec / max(1, steps)
            avg_cycle = total_cycle / max(1, steps)

            print(f"[r-opt] Epoch {ep+1:3d}/{epochs}: "
                  f"loss={avg_loss:.6f} g_pinv={avg_g_pinv:.6f} "
                  f"rec={avg_rec:.6f} cycle={avg_cycle:.6f}")

            _wandb_log(self.args, {
                "r_opt/loss": avg_loss,
                "r_opt/g_pinv_loss": avg_g_pinv,
                "r_opt/img_rec_loss": avg_rec,
                "r_opt/cycle_loss": avg_cycle,
                "r_opt/epoch": ep + 1,
            })

            if (ep + 1) % 10 == 0:
                checkpoint_path = os.path.join(self.args.checkpoint_dir, f"r_opt_epoch_{ep+1}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"[r-opt] Saved checkpoint: {checkpoint_path}")

        model.eval()

    def train_r_opt(self, checkpoint_path, device, loader, epochs, lr, out_checkpoint_path):
        """Load a forward-trained model and run r-optimization."""
        model = self._build_model().to(device)
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        self._train_r_opt(
            model=model, loader=loader, device=device,
            epochs=epochs, lr=lr,
        )

        torch.save(model.state_dict(), out_checkpoint_path)
        print(f"[r-opt] Saved r-optimized model to {out_checkpoint_path}")
        return out_checkpoint_path

    def _build_model(self):
        """Build a detection SPNN model matching self.args config."""
        from models import PixelUnshuffleBlock
        mix_type = getattr(self.args, "mix_type", "cayley")

        layer_channels = [
            (PixelUnshuffleBlock, {"r": 4}),
            (ConvPINNBlock, {"in_ch": 48, "out_ch": 12, "hidden": 128,
                             "scale_bound": 2.0, "feat_size": 64, "mix_type": mix_type}),
            (PixelUnshuffleBlock, {"r": 4}),
            (ConvPINNBlock, {"in_ch": 192, "out_ch": 48, "hidden": 128,
                             "scale_bound": 2.0, "feat_size": 16, "mix_type": mix_type}),
            (PixelUnshuffleBlock, {"r": 2}),
            (ConvPINNBlock, {"in_ch": 192, "out_ch": OUT_CH, "hidden": 256,
                             "scale_bound": 2.0, "feat_size": S, "mix_type": mix_type}),
        ]

        return SPNN(
            img_ch=3,
            num_classes=OUT_CH,
            img_size=256,
            layer_channels=layer_channels,
            output_spatial_size=(S, S),
        )
