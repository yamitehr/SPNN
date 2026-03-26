import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from models import SPNN
from train_resnet_baseline import ResNet50CelebA
from huggingface_hub import hf_hub_download

import json
import time
import numpy as np
import tqdm
import torch
import torch.utils.data as data

from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import download

import torchvision.utils as tvu

from guided_diffusion.models import Model
import random


CELEBA_ATTR_NAMES = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
    'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
    'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
    'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
    'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
    'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young',
]


def _load_spnn_evaluator(device):
    """Load the SPNN classifier for evaluation (used by all methods)."""
    evaluator = SPNN(img_ch=3, num_classes=40, hidden=128, scale_bound=2.0, img_size=256).to(device)
    ckpt_path = hf_hub_download(repo_id="yamitehr/SPNN", filename="spnn_celebahq_256.pth")
    evaluator.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    evaluator.eval()
    return evaluator


def _compute_and_save_metrics(all_y_target, all_y_pred, output_dir, total_time, num_images):
    """Compute per-attribute binary agreement and probability error, save to JSON."""
    y_target = torch.stack(all_y_target)  # [N, 40]
    y_pred = torch.stack(all_y_pred)      # [N, 40]

    # Binary agreement: (pred > 0) == (target > 0)
    binary_agree = ((y_pred > 0) == (y_target > 0)).float()  # [N, 40]
    per_attr_agreement = binary_agree.mean(dim=0) * 100       # percentage

    # Probability error: |sigmoid(pred) - sigmoid(target)|
    prob_error = (y_pred.sigmoid() - y_target.sigmoid()).abs()  # [N, 40]
    per_attr_prob_error = prob_error.mean(dim=0)

    mean_agreement = per_attr_agreement.mean().item()
    mean_prob_error = per_attr_prob_error.mean().item()
    time_per_image = total_time / max(1, num_images)

    metrics = {
        "summary": {
            "mean_agreement_pct": round(mean_agreement, 2),
            "mean_prob_error": round(mean_prob_error, 4),
            "num_images": num_images,
            "total_time_sec": round(total_time, 1),
            "time_per_image_sec": round(time_per_image, 1),
        },
        "per_attribute": {}
    }
    for i, name in enumerate(CELEBA_ATTR_NAMES):
        metrics["per_attribute"][name] = {
            "agreement_pct": round(per_attr_agreement[i].item(), 1),
            "prob_error": round(per_attr_prob_error[i].item(), 4),
        }

    out_path = os.path.join(output_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Mean Binary Agreement: {mean_agreement:.1f}%")
    print(f"  Mean Probability Error: {mean_prob_error:.4f}")
    print(f"  Time per image: {time_per_image:.1f}s")
    print(f"  Metrics saved to: {out_path}")
    print(f"{'='*60}")
    return metrics



def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device


        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self):
        model = Model(self.config)
        ckpt = os.path.join(self.args.exp, "logs/celeba/celeba_hq.ckpt")
        if not os.path.exists(ckpt):
            download('https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt', ckpt)
        model.load_state_dict(torch.load(ckpt, map_location=self.device, weights_only=False))
        model.to(self.device)
        model = torch.nn.DataParallel(model)
        guidance = getattr(self.args, 'guidance_method', 'nlbp')
        print(f'Run {guidance.upper()} guidance.',
              f'{self.config.time_travel.T_sampling} sampling steps.',
              f'travel_length = {self.config.time_travel.travel_length},',
              f'travel_repeat = {self.config.time_travel.travel_repeat}.'
             )
        if guidance == 'dps':
            self.dps_guidance(model)
        else:
            self.simplified_ddnm_plus(model)


    def simplified_ddnm_plus(self, model):
        args, config = self.args, self.config
        _, test_dataset = get_dataset(args, config)

        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)

        print(f'Dataset has size {len(test_dataset)}')

        def seed_worker(worker_id):
            worker_seed = args.seed % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(args.seed)
        val_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )

        classifier = SPNN(img_ch=3, num_classes=40, hidden=128, scale_bound=2.0, img_size=256).to(
            self.device)

        ckpt_path = hf_hub_download(repo_id="yamitehr/SPNN", filename="spnn_celebahq_256.pth")

        print(f"Loading classifier from {ckpt_path}")
        classifier.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=False))
        classifier.eval()

        A = lambda z, **kwargs: classifier(z, **kwargs)
        Ap = lambda logits, **kwargs: (classifier.pinv(logits, **kwargs))

        # Evaluate with the same SPNN classifier
        evaluator = classifier
        all_y_target = []
        all_y_pred = []

        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        t_start = time.time()
        pbar = tqdm.tqdm(val_loader)
        for x_orig, classes in pbar:
            x_orig = x_orig.to(self.device)
            x_orig = data_transform(self.config, x_orig)

            y = A(x_orig)

            if config.sampling.batch_size != 1:
                raise ValueError("please change the config file to set batch size as 1")
                
            # init x_T
            torch.manual_seed(args.seed)
            x = torch.randn(
                y.shape[0],
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )

            with torch.no_grad():
                skip = config.diffusion.num_diffusion_timesteps // config.time_travel.T_sampling
                n = x.size(0)
                x0_preds = []
                xs = [x]

                times = get_schedule_jump(config.time_travel.T_sampling,
                                          config.time_travel.travel_length,
                                          config.time_travel.travel_repeat,
                                          )
                time_pairs = list(zip(times[:-1], times[1:]))
                
                for step_idx, (i, j) in enumerate(tqdm.tqdm(time_pairs)):
                    i, j = i * skip, j * skip
                    if j < 0: j = -1

                    if j < i:  # normal sampling
                        t = (torch.ones(n) * i).to(x.device)
                        next_t = (torch.ones(n) * j).to(x.device)
                        at = compute_alpha(self.betas, t.long())
                        at_next = compute_alpha(self.betas, next_t.long())
                        sigma_t = (1 - at / at_next).sqrt()
                        xt = xs[-1].to('cuda')

                        et = model(xt, t)

                        if et.size(1) == 6:
                            et = et[:, :3]

                        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                        x0_t_hat = x0_t
                        min_bp_step = args.min_bp_step

                        lambda1 = args.lambda1
                        lambda2 = args.lambda2

                        if i < min_bp_step:
                            lambda_t = lambda1
                        else: # i > min
                            lambda_t = lambda2

                        # None-Linear Back Projection
                        y_cur, z_cur = A(x0_t_hat, return_latents=True)

                        if (y_cur.sigmoid() - y.sigmoid()).abs().mean() > args.nlbp_stop_cond:

                            y_tar, z_tar = A(Ap(y), return_latents=True)
                            y_proj, z_proj = A(Ap(A(x0_t_hat)), return_latents=True)
                            z_final = []
                            for z0, z1, z2 in zip(z_cur, z_tar, z_proj):
                                if z0 is not None:
                                    z_final.append(z0 + lambda_t * (z1 - z2))
                                else:
                                    z_final.append(None)

                            y_final = y_cur + lambda_t * (y_tar - y_proj)

                            x0_t_hat = Ap(y_final, latents=z_final)

                        c2 = (1 - at_next - sigma_t ** 2).clamp(min=0).sqrt()
                        xt_next = at_next.sqrt() * x0_t_hat + c2 * et + sigma_t * torch.randn_like(x0_t)

                        x0_preds.append(x0_t_hat.to('cpu'))
                        xs.append(xt_next.to('cpu'))
                    else: # time-travel back
                        next_t = (torch.ones(n) * j).to(x.device)
                        at_next = compute_alpha(self.betas, next_t.long())
                        x0_t = x0_preds[-1].to('cuda')

                        xt_next = at_next.sqrt() * x0_t_hat + torch.randn_like(x0_t) * (1 - at_next).sqrt()

                        xs.append(xt_next.to('cpu'))

            final_x0 = inverse_data_transform(config, x0_preds[-1])

            tvu.save_image(
                final_x0[0], os.path.join(self.args.image_folder, f"{idx_so_far}_{0}.png")
            )
            orig = inverse_data_transform(config, x_orig[0])
            # Save result grid
            results_dir = self.args.image_folder
            os.makedirs(results_dir, exist_ok=True)
            res_grid = torch.cat([orig.cpu(), final_x0[0].cpu()], dim=-1)
            grid_path = os.path.join(results_dir, f"grid_{idx_so_far}.png")
            tvu.save_image(res_grid, grid_path)

            mse = torch.mean((final_x0[0].to(self.device) - orig) ** 2)
            psnr = 10 * torch.log10(1 / mse)
            avg_psnr += psnr

            # Evaluate reconstruction: re-normalize [0,1] -> [-1,1] for classifier
            with torch.no_grad():
                final_x0_normalized = final_x0[0].unsqueeze(0).to(self.device) * 2.0 - 1.0
                y_pred = evaluator(final_x0_normalized)
                all_y_target.append(y.cpu().squeeze(0))
                all_y_pred.append(y_pred.cpu().squeeze(0))

            idx_so_far += y.shape[0]

            pbar.set_description("PSNR: %.2f" % (avg_psnr / (idx_so_far - idx_init)))

        total_time = time.time() - t_start
        avg_psnr = avg_psnr / (idx_so_far - idx_init)
        print("Total Average PSNR: %.2f" % avg_psnr)
        print("Number of samples: %d" % (idx_so_far - idx_init))
        print(f"Results saved to: {os.path.abspath(self.args.image_folder)}")
        _compute_and_save_metrics(all_y_target, all_y_pred, self.args.image_folder, total_time, idx_so_far - idx_init)


    def dps_guidance(self, model):
        """
        DPS (Diffusion Posterior Sampling) baseline.

        Instead of NLBP's closed-form back-projection, this uses gradient-based
        guidance: at each step, backprop through the classifier to compute
        nabla_{x_t} ||y - g(x0_hat(x_t))||^2 and subtract from x_t.

        Reference: Chung et al., "Diffusion Posterior Sampling for General Noisy
        Inverse Problems", ICLR 2023.
        """
        args, config = self.args, self.config
        _, test_dataset = get_dataset(args, config)

        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)

        print(f'Dataset has size {len(test_dataset)}')

        def seed_worker(worker_id):
            worker_seed = args.seed % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(args.seed)
        val_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )

        # Load classifier
        classifier_type = getattr(args, 'classifier', 'spnn')
        if classifier_type == 'resnet':
            assert args.resnet_ckpt is not None, "Must provide --resnet_ckpt when using --classifier resnet"
            classifier = ResNet50CelebA(num_classes=40, pretrained=False).to(self.device)
            classifier.load_state_dict(torch.load(args.resnet_ckpt, map_location=self.device, weights_only=True))
            print(f"Loaded ResNet-50 baseline classifier from {args.resnet_ckpt}")
        else:
            classifier = SPNN(img_ch=3, num_classes=40, hidden=128, scale_bound=2.0, img_size=256).to(
                self.device)
            ckpt_path = hf_hub_download(repo_id="yamitehr/SPNN", filename="spnn_celebahq_256.pth")
            classifier.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=False))
            print(f"Loaded SPNN classifier from {ckpt_path}")
        classifier.eval()

        zeta = args.dps_step_size

        # Evaluate with the SAME classifier used for guidance (fair: each method evaluated on its own terms)
        evaluator = classifier
        all_y_target = []
        all_y_pred = []

        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        t_start = time.time()
        pbar = tqdm.tqdm(val_loader)
        for x_orig, classes in pbar:
            x_orig = x_orig.to(self.device)
            x_orig = data_transform(self.config, x_orig)

            # Target measurement for guidance
            with torch.no_grad():
                y = classifier(x_orig)

            if config.sampling.batch_size != 1:
                raise ValueError("please change the config file to set batch size as 1")

            # init x_T (same seed as NLBP for fair comparison)
            torch.manual_seed(args.seed)
            x = torch.randn(
                y.shape[0],
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )

            skip = config.diffusion.num_diffusion_timesteps // config.time_travel.T_sampling
            n = x.size(0)
            x0_preds = []
            xs = [x]

            times = get_schedule_jump(config.time_travel.T_sampling,
                                      config.time_travel.travel_length,
                                      config.time_travel.travel_repeat,
                                      )
            time_pairs = list(zip(times[:-1], times[1:]))

            for step_idx, (i, j) in enumerate(tqdm.tqdm(time_pairs)):
                i, j = i * skip, j * skip
                if j < 0: j = -1

                if j < i:  # normal sampling
                    t = (torch.ones(n) * i).to(x.device)
                    next_t = (torch.ones(n) * j).to(x.device)
                    at = compute_alpha(self.betas, t.long())
                    at_next = compute_alpha(self.betas, next_t.long())
                    sigma_t = (1 - at / at_next).sqrt()
                    xt = xs[-1].to('cuda').requires_grad_(True)

                    # Diffusion model predicts noise (no grad through U-Net)
                    with torch.no_grad():
                        et = model(xt, t)
                        if et.size(1) == 6:
                            et = et[:, :3]

                    # Tweedie estimate of x0 — MUST be in grad graph w.r.t. xt
                    x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                    # DPS guidance: nabla_{x_t} ||y - g(x0_hat)||^2
                    y_pred = classifier(x0_t)
                    loss = (y_pred - y).pow(2).sum()
                    loss.backward()

                    with torch.no_grad():
                        # Gradient step on x0_t
                        x0_t_hat = x0_t.detach() - zeta * xt.grad.detach()

                        # DDPM sampling step
                        c2 = (1 - at_next - sigma_t ** 2).clamp(min=0).sqrt()
                        xt_next = at_next.sqrt() * x0_t_hat + c2 * et + sigma_t * torch.randn_like(x0_t_hat)

                    x0_preds.append(x0_t_hat.to('cpu'))
                    xs.append(xt_next.to('cpu'))
                else:  # time-travel back
                    next_t = (torch.ones(n) * j).to(x.device)
                    at_next = compute_alpha(self.betas, next_t.long())
                    x0_t = x0_preds[-1].to('cuda')

                    with torch.no_grad():
                        xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

                    xs.append(xt_next.to('cpu'))

            final_x0 = inverse_data_transform(config, x0_preds[-1])

            tvu.save_image(
                final_x0[0], os.path.join(self.args.image_folder, f"{idx_so_far}_{0}.png")
            )
            orig = inverse_data_transform(config, x_orig[0])
            # Save result grid
            results_dir = self.args.image_folder
            os.makedirs(results_dir, exist_ok=True)
            res_grid = torch.cat([orig.cpu(), final_x0[0].cpu()], dim=-1)
            grid_path = os.path.join(results_dir, f"grid_{idx_so_far}.png")
            tvu.save_image(res_grid, grid_path)

            mse = torch.mean((final_x0[0].to(self.device) - orig) ** 2)
            psnr = 10 * torch.log10(1 / mse)
            avg_psnr += psnr

            # Evaluate reconstruction: re-normalize [0,1] -> [-1,1] for classifier
            with torch.no_grad():
                final_x0_normalized = final_x0[0].unsqueeze(0).to(self.device) * 2.0 - 1.0
                y_pred = evaluator(final_x0_normalized)
                all_y_target.append(y.detach().cpu().squeeze(0))
                all_y_pred.append(y_pred.cpu().squeeze(0))

            idx_so_far += y.shape[0]

            pbar.set_description("PSNR: %.2f" % (avg_psnr / (idx_so_far - idx_init)))

        total_time = time.time() - t_start
        avg_psnr = avg_psnr / (idx_so_far - idx_init)
        print("Total Average PSNR: %.2f" % avg_psnr)
        print("Number of samples: %d" % (idx_so_far - idx_init))
        print(f"Results saved to: {os.path.abspath(self.args.image_folder)}")
        _compute_and_save_metrics(all_y_target, all_y_pred, self.args.image_folder, total_time, idx_so_far - idx_init)


# Code form RePaint
def get_schedule_jump(T_sampling, travel_length, travel_repeat):
    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = T_sampling
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, T_sampling)
    return ts

def _check_times(times, t_0, T_sampling):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)
        
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a
