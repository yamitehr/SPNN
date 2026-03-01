import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from models import SPNN, ConvPINNBlock

from utils import torch_show_all_params, torch_init_model
from pytorch_optimization import get_linear_schedule_with_warmup


class CelebATrainer:
    def __init__(self, args, model, train_loader, dev_loader, test_loader, device, logger, n_gpu):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.device = device
        self.logger = logger
        self.n_gpu = n_gpu

    def _adaptive_update_weights(self, model, val_input, val_labels, step, k, lambdas, val_loss, eps=1e-8):
        logits = model(val_input)  # [bs, 40]
        vloss = nn.BCEWithLogitsLoss(reduction='none')(input=logits, target=val_labels)
        vloss = torch.mean(vloss, dim=0)  # [40,]
        val_loss.append(vloss)
        val_loss[:] = val_loss[-(2 * k):]
        if k > 0 and step % k == 0 and len(val_loss) == 2 * k:
            val_loss_list = torch.cat([vl.unsqueeze(0) for vl in val_loss], dim=0)
            pre_mean = torch.mean(val_loss_list[:k, :], dim=0)
            cur_mean = torch.mean(val_loss_list[k:, :], dim=0)
            trend = torch.abs(cur_mean - pre_mean) / (cur_mean + eps)
            norm_trend = trend / (torch.mean(trend) + eps)
            norm_loss = cur_mean / (torch.mean(cur_mean) + eps)
            new_lambdas = norm_trend * norm_loss
            new_lambdas = new_lambdas / (torch.mean(new_lambdas) + eps)
            lambdas.copy_(new_lambdas)

    def train(self):
        self.logger.info('Parameters: ' + str(torch_show_all_params(self.model)))
        self.logger.info(f'Loss Weights: BCE={self.args.lambda_bce}, RightInverse={self.args.lambda_right_inverse}, Rec={self.args.lambda_img_rec}')

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))
        total_steps = int(self.args.epoch * len(self.train_loader) / max(1, self.n_gpu))
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_iters,
                                                    fix_steps=int(self.args.fix_epoch * total_steps),
                                                    num_training_steps=total_steps)

        scaler = torch.amp.GradScaler('cuda', enabled=(self.args.float16 and self.device == "cuda"))

        current_step = 0
        start_epoch = 0
        self.model.train()
        show_loss = 0
        show_loss_bce = 0
        show_cycle_loss = 0
        show_img_rec_loss = 0
        best_val_loss = float("inf")
        lambdas = torch.ones((40,), device=self.device)
        val_loss = []
        dev_iter = iter(self.dev_loader)

        self.logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))

        for epoch in range(start_epoch, self.args.epoch):
            for train_data in self.train_loader:
                train_img, train_labels = train_data
                train_img = train_img.to(device=self.device, non_blocking=True)
                train_labels = train_labels.to(device=self.device, dtype=torch.float32, non_blocking=True)

                if train_labels.min().item() < 0:
                    train_labels = (train_labels + 1) / 2

                optimizer.zero_grad(set_to_none=True)

                if self.args.float16 and self.device == "cuda":
                    with torch.amp.autocast('cuda', enabled=True):
                        logits = self.model(train_img)

                        bce = nn.BCEWithLogitsLoss(reduction='none')(logits, train_labels)
                        loss_bce = torch.sum(bce * lambdas.unsqueeze(0), dim=1).mean()

                        # Right inverse loss: || f(f'(y)) - y ||
                        model_ref = self.model.module if hasattr(self.model, "module") else self.model
                        x_inv = model_ref.pinv(logits)
                        y_cycle = self.model(x_inv)
                        cycle_l = (y_cycle - logits).pow(2).mean()

                        # Image reconstruction loss: || f'(f(x)) - x ||
                        img_rec_l = (x_inv - train_img).pow(2).mean()

                        loss = self.args.lambda_bce * loss_bce + self.args.lambda_right_inverse * cycle_l + self.args.lambda_img_rec * img_rec_l

                    show_loss += loss.detach().item()
                    show_loss_bce += loss_bce.detach().item()
                    show_cycle_loss += cycle_l.detach().item()
                    show_img_rec_loss += img_rec_l.detach().item()
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = self.model(train_img)
                    bce = nn.BCEWithLogitsLoss(reduction='none')(logits, train_labels)
                    loss_bce = torch.sum(bce * lambdas.unsqueeze(0), dim=1).mean()

                    # Right inverse loss: || f(f'(y)) - y ||
                    model_ref = self.model.module if hasattr(self.model, "module") else self.model
                    x_inv = model_ref.pinv(logits)
                    y_cycle = self.model(x_inv)
                    cycle_l = (y_cycle - logits).pow(2).mean()

                    # Image reconstruction loss: || f'(f(x)) - x ||
                    img_rec_l = (x_inv - train_img).pow(2).mean()

                    loss = self.args.lambda_bce * loss_bce + self.args.lambda_right_inverse * cycle_l + self.args.lambda_img_rec * img_rec_l

                    show_loss += loss.detach().item()
                    show_loss_bce += loss_bce.detach().item()
                    show_cycle_loss += cycle_l.detach().item()
                    show_img_rec_loss += img_rec_l.detach().item()
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_norm)
                    optimizer.step()

                scheduler.step()

                if self.args.adaptive_weights and self.args.k > 0 and (current_step % self.args.k == 0):
                    self.model.eval()
                    with torch.no_grad():
                        try:
                            aux_img, aux_labels = next(dev_iter)
                        except StopIteration:
                            dev_iter = iter(self.dev_loader)
                            aux_img, aux_labels = next(dev_iter)

                        aux_img = aux_img.to(device=self.device, non_blocking=True)
                        aux_labels = aux_labels.to(device=self.device, dtype=torch.float32, non_blocking=True)
                        if aux_labels.min().item() < 0:
                            aux_labels = (aux_labels + 1) / 2

                        model_for_eval = self.model.module if hasattr(self.model, "module") else self.model
                        self._adaptive_update_weights(model_for_eval, aux_img, aux_labels, current_step, self.args.k, lambdas, val_loss)
                    self.model.train()

                current_step += 1

                if current_step % self.args.print_freq == 0:
                    avg_loss = show_loss / self.args.print_freq
                    avg_loss_bce = show_loss_bce / self.args.print_freq
                    avg_cycle = show_cycle_loss / self.args.print_freq
                    avg_img_rec = show_img_rec_loss / self.args.print_freq
                    message = '[epoch:{0}/{1}, steps:{2}/{3}, lr:{4:.3e}, loss:{5:.5}, bce:{6:.5}, cycle:{7:.5}, img_rec:{8:.5}] '.format(
                        epoch, self.args.epoch, current_step, total_steps, scheduler.get_lr()[0], avg_loss,
                        avg_loss_bce, avg_cycle, avg_img_rec)
                    show_loss = 0
                    show_loss_bce = 0
                    show_cycle_loss = 0
                    show_img_rec_loss = 0
                    self.logger.info(message)

            self.model.eval()
            val_loss_unweighted_sum = 0.0
            val_loss_weighted_sum = 0.0
            val_batches = 0
            att_wrong = np.zeros(40)

            with torch.no_grad():
                for val_data in tqdm(self.dev_loader):
                    val_img, val_labels = val_data
                    val_img = val_img.to(self.device, non_blocking=True)
                    val_labels_t = val_labels.to(device=self.device, dtype=torch.float32, non_blocking=True)

                    if val_labels_t.min().item() < 0:
                        val_labels_t = (val_labels_t + 1) / 2

                    logits_t = self.model(val_img)
                    bce = nn.BCEWithLogitsLoss(reduction='none')(logits_t, val_labels_t)
                    
                    # Compute losses matching training
                    loss_bce_unweighted = torch.sum(bce, dim=1).mean()
                    loss_bce_weighted = torch.sum(bce * lambdas.unsqueeze(0), dim=1).mean()

                    # Right inverse loss: || f(f'(y)) - y ||
                    model_ref = self.model.module if hasattr(self.model, "module") else self.model
                    x_inv = model_ref.pinv(logits_t)
                    y_cycle = self.model(x_inv)
                    cycle_l = (y_cycle - logits_t).pow(2).mean()

                    # Image reconstruction loss: || f'(f(x)) - x ||
                    img_rec_l = (x_inv - val_img).pow(2).mean()

                    aux_loss = self.args.lambda_right_inverse * cycle_l + self.args.lambda_img_rec * img_rec_l

                    val_loss_unweighted_sum += (self.args.lambda_bce * loss_bce_unweighted + aux_loss).item()
                    val_loss_weighted_sum += (self.args.lambda_bce * loss_bce_weighted + aux_loss).item()
                    val_batches += 1

                    logits = logits_t.detach().cpu().numpy()
                    val_labels_np = val_labels.numpy()
                    if val_labels_np.min() < 0:
                        val_labels_np = (val_labels_np + 1) / 2

                    preds = np.zeros_like(logits)
                    preds[logits > 0] = 1
                    diff = np.abs(preds - val_labels_np)
                    diff = np.sum(diff, axis=0)
                    att_wrong += diff

            att_wrong /= len(self.dev_loader.dataset)
            att_acc = 1 - att_wrong
            val_mean_acc = np.mean(att_acc)
            val_loss_unweighted = val_loss_unweighted_sum / val_batches
            val_loss_weighted = val_loss_weighted_sum / val_batches
            val_loss_for_ckpt = (val_loss_weighted if self.args.adaptive_weights else val_loss_unweighted)

            message = f'[epoch:{epoch} end] mean_acc:{val_mean_acc:.5} '
            if self.args.adaptive_weights:
                message += f'val_loss_w:{val_loss_weighted:.5} val_loss:{val_loss_unweighted:.5}'
            else:
                message += f'val_loss:{val_loss_unweighted:.5}'
            self.logger.info(message)

            if val_loss_for_ckpt < best_val_loss:
                best_val_loss = val_loss_for_ckpt
                self.logger.info(f'[BEST] epoch:{epoch} val_loss:{best_val_loss:.5}' + (' (weighted)' if self.args.adaptive_weights else ' (unweighted)'))
                model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                torch.save(model_to_save.state_dict(), os.path.join(self.args.checkpoint_dir, 'best_model.pth'))
                print("[BEST] model saved")

            self.model.train()

        self.logger.info('End of training.')
        torch_init_model(self.model, os.path.join(self.args.checkpoint_dir, 'best_model.pth'))
        return self.model

    def test(self):
        self.model.eval()
        att_wrong = np.zeros(40)
        with torch.no_grad():
            for test_data in tqdm(self.test_loader):
                test_img, test_labels = test_data
                logits = self.model(test_img.to(self.device, non_blocking=True))
                test_labels = test_labels.numpy()
                if test_labels.min() < 0:
                    test_labels = (test_labels + 1) / 2
                logits = logits.detach().cpu().numpy()
                preds = np.zeros_like(logits)
                preds[logits > 0] = 1
                diff = np.abs(preds - test_labels)
                diff = np.sum(diff, axis=0)
                att_wrong += diff

        att_wrong /= len(self.test_loader.dataset)
        att_acc = 1 - att_wrong
        test_mean_acc = np.mean(att_acc)
        message = '[test mean_acc:{:.5}]'.format(test_mean_acc)
        self.logger.info(message)

    def _train_r_opt_classifier(self, model, loader, device, H, W, epochs, lr):
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
            steps = 0
            for x_batch, _ in loader:
                x_batch = x_batch.to(device, non_blocking=True)

                # Compute logits on the fly (detached, as we don't optimizing forward path)
                with torch.no_grad():
                    y_batch = model(x_batch)

                x_tag = model.pinv(y_batch)

                y, z_list = model(x_tag, return_latents=True)

                y_0, z_list_0 = model(torch.zeros_like(x_tag), return_latents=True)

                B = y.shape[0]

                parts = []
                parts_0 = []

                for z, z_0 in zip(z_list, z_list_0):
                    if z is None:
                        continue
                    parts.append(z.view(B, -1))
                    parts_0.append(z_0.view(B, -1))

                G_pinv = torch.cat(parts, dim=1)
                G_0 = torch.cat(parts_0, dim=1)
                loss_G_pinv = (G_pinv - G_0).pow(2).mean()

                # Image reconstruction loss: || f'(f(x)) - x ||
                img_rec_l = (x_tag - x_batch).pow(2).mean()

                # Right inverse loss
                y_cycle = model(x_tag)
                cycle_l = (y_cycle - y_batch).pow(2).mean()

                loss = self.args.lambda_r_norm * loss_G_pinv + self.args.lambda_r_rec * img_rec_l + self.args.lambda_r_cycle * cycle_l

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(r_params, max_norm=1.0)
                opt.step()

                total_loss += loss.item()
                steps += 1

            print(f"[r-opt-real] Epoch {ep + 1:3d} / {epochs}: avg_loss={total_loss / max(1, steps):.6f}")

            if (ep + 1) % 10 == 0:
                checkpoint_path = os.path.join(self.args.checkpoint_dir, f"r_opt_epoch_{ep + 1}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"[r-opt] Saved checkpoint: {checkpoint_path}")

        model.eval()
        return

    def train_r_opt_on_real_logits(self, checkpoint_path, device, loader, num_classes, H, W, epochs, lr, batch_size, out_checkpoint_path, img_size):
        net = SPNN(img_ch=3, num_classes=num_classes, hidden=128, scale_bound=2.0, img_size=img_size).to(device)
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        net.load_state_dict(state_dict)
        net.eval()

        self._train_r_opt_classifier(model=net, loader=loader, device=device, H=H, W=W, epochs=epochs, lr=lr)

        torch.save(net.state_dict(), out_checkpoint_path)
        print(f"[r-opt-real] Saved r-optimized model to {out_checkpoint_path}")
        return out_checkpoint_path

