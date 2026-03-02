import os
import torch
import torch.nn as nn
from tqdm import tqdm
from models import ConvPINNBlock, SPNN



class PenroseChecker:
    def __init__(self, logger):
        self.logger = logger

    @torch.no_grad()
    def run_penrose_batched(self, checkpoint_path, test_loader, device, img_ch, num_classes, hidden, scale_bound, img_size):
        """Tests the Penrose and specific inverse identities."""
        net = SPNN(img_ch=img_ch, num_classes=num_classes, hidden=hidden, scale_bound=scale_bound, img_size=img_size).to(device)
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        net.load_state_dict(state_dict)
        net.eval()
        mse_fn = nn.MSELoss(reduction="mean")
        err_ggg_sum, err_g_prime_sum, err_gg_prime_sum, n_batches = 0.0, 0.0, 0.0, 0

        for images, _ in tqdm(test_loader, desc="Penrose check"):
            images = images.to(device, non_blocking=True)
            B = images.shape[0]
            y = torch.randn(B, num_classes, device=device)

            # 1. Test gg'g = g
            y_g = net(images)
            x_pinv_g = net.pinv(y_g)
            y_g_cycle = net(x_pinv_g)
            err_ggg_sum += mse_fn(y_g_cycle, y_g).item()

            # 2. Test g'gg' = g'
            x_g_prime = net.pinv(y)
            y_fwd_g_prime = net(x_g_prime)
            x_g_prime_cycle = net.pinv(y_fwd_g_prime)
            err_g_prime_sum += mse_fn(x_g_prime_cycle, x_g_prime).item()

            # 3. Test gg' = I (Right Inverse)
            y_cycle_right = net(net.pinv(y))
            err_gg_prime_sum += mse_fn(y_cycle_right, y).item()
            n_batches += 1

        metrics = {"penrose/ggg_mse": err_ggg_sum / max(1, n_batches), "penrose/g_prime_mse": err_g_prime_sum / max(1, n_batches), "penrose/gg_prime_mse": err_gg_prime_sum / max(1, n_batches)}
        print("\n--- Penrose Identities (Should be ~0) ---")
        print(f"  g(g'(g(x))) == g(x)   | MSE: {metrics['penrose/ggg_mse']:.2e}")
        print(f"  g'(g(g'(y))) == g'(y) | MSE: {metrics['penrose/g_prime_mse']:.2e}")
        print(" (averaged over entire test set)")

        print("\n--- Specific Inverse Identities ---")
        print(f"  g(g'(y)) == y         | MSE: {metrics['penrose/gg_prime_mse']:.2e}")
        print(" (averaged over entire test set)")
        print("----------------------------------------------------")
        return metrics

    @torch.no_grad()
    def run_penrose(self, checkpoint_path, test_loader, device, img_ch, num_classes, hidden, scale_bound, img_size):
        """Tests the Penrose and specific inverse identities."""
        net = SPNN(img_ch=img_ch, num_classes=num_classes, hidden=hidden, scale_bound=scale_bound, img_size=img_size).to(device)
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        net.load_state_dict(state_dict)
        net.eval()
        imgs = []
        for images, _ in test_loader:
            imgs.append(images)
        images_all = torch.cat(imgs, dim=0).to(device)
        N = images_all.shape[0]
        y = torch.randn(N, num_classes, device=device)
        mse_fn = nn.MSELoss(reduction="mean")

        # 1. Test gg'g = g
        y_g = net(images_all)
        x_pinv_g = net.pinv(y_g)
        y_g_cycle = net(x_pinv_g)
        err_ggg = mse_fn(y_g_cycle, y_g)

        # 2. Test g'gg' = g'
        x_g_prime = net.pinv(y)
        y_fwd_g_prime = net(x_g_prime)
        x_g_prime_cycle = net.pinv(y_fwd_g_prime)
        err_g_prime = mse_fn(x_g_prime_cycle, x_g_prime)

        # 3. Test gg' = I (Right Inverse)
        y_cycle_right = net(net.pinv(y))
        err_gg_prime = mse_fn(y_cycle_right, y)

        metrics = {
            "penrose/ggg_mse": err_ggg.item(),
            "penrose/g_prime_mse": err_g_prime.item(),
            "penrose/gg_prime_mse": err_gg_prime.item(),
        }

        print("\n--- Penrose Identities (Should be ~0) ---")
        print(f"  g(g'(g(x))) == g(x)   | MSE: {metrics['penrose/ggg_mse']:.2e}")
        print(f"  g'(g(g'(y))) == g'(y) | MSE: {metrics['penrose/g_prime_mse']:.2e}")

        print("\n--- Specific Inverse Identities ---")
        print(f"  g(g'(y)) == y         | MSE: {metrics['penrose/gg_prime_mse']:.2e}")
        print("---------------------------------------------------------------")
        return metrics


class GinvNormCalculator:
    def __init__(self, logger):
        self.logger = logger

    @torch.no_grad()
    def run(self, checkpoint_path, loader, device, model_cls, model_kwargs):
        model = model_cls(**model_kwargs).to(device)
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        total_norm = 0.0
        total_count = 0
        for i, (x, _) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            B = x.shape[0]
            y = model(x)
            x_hat = model.pinv(y)
            batch_norm = x_hat.pow(2).mean().item()
            total_norm += batch_norm * B
            total_count += B

        mean_norm = total_norm / total_count
        print("\n--- || g'(g(x)) ||^2 on real data ---")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Mean ||g'(g(x))||^2 = {mean_norm:.6e}")
        print("------------------------------------")
        return mean_norm
