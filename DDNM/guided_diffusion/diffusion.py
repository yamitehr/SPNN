import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from models import SPNN
from train_resnet_baseline import ResNet50CelebA
from train_noisy_classifier import NoisyResNet50CelebA
from huggingface_hub import hf_hub_download

import json
import subprocess
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
    print(f"  Total runtime: {total_time:.1f}s")
    print(f"  Time per image: {time_per_image:.1f}s")
    print(f"  Metrics saved to: {out_path}")
    print(f"{'='*60}")
    return metrics


def _compute_fid_and_lpips(originals_dir, reconstructions_dir, output_dir, device="cuda"):
    """
    Compute FID and LPIPS between originals and reconstructions.

    FID: uses pytorch-fid (same as DPS paper, Chung et al. ICLR 2023).
         Compares Inception-v3 feature distributions of two image directories.
         Images are 256x256 PNG in [0,1] range; Inception resizes to 299x299 internally.

    LPIPS: per-image perceptual distance (AlexNet), averaged over all pairs.
    """
    metrics = {}

    # --- FID via pytorch-fid CLI (most reliable, matches DPS paper) ---
    try:
        result = subprocess.run(
            ["python", "-m", "pytorch_fid", originals_dir, reconstructions_dir,
             "--device", device],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode == 0:
            # Output format: "FID:  <value>"
            for line in result.stdout.strip().split("\n"):
                if "FID" in line:
                    fid_value = float(line.split()[-1])
                    metrics["fid"] = round(fid_value, 2)
                    print(f"  FID: {fid_value:.2f}")
                    break
        else:
            print(f"  pytorch-fid failed: {result.stderr.strip()}")
    except FileNotFoundError:
        print("  pytorch-fid not installed. Install with: pip install pytorch-fid")
    except Exception as e:
        print(f"  FID computation error: {e}")

    # --- LPIPS ---
    try:
        import lpips
        from PIL import Image
        from torchvision import transforms

        loss_fn = lpips.LPIPS(net='alex').to(device)
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        lpips_values = []
        recon_files = sorted(os.listdir(reconstructions_dir))
        orig_files = sorted(os.listdir(originals_dir))

        # Match by filename
        recon_set = {f for f in recon_files if f.lower().endswith(('.png', '.jpg'))}
        orig_set = {f for f in orig_files if f.lower().endswith(('.png', '.jpg'))}
        common = sorted(recon_set & orig_set)

        if not common:
            # Fall back to pairing by sorted order
            recon_list = sorted(f for f in recon_files if f.lower().endswith(('.png', '.jpg')))
            orig_list = sorted(f for f in orig_files if f.lower().endswith(('.png', '.jpg')))
            common_count = min(len(recon_list), len(orig_list))
            pairs = list(zip(orig_list[:common_count], recon_list[:common_count]))
        else:
            pairs = [(f, f) for f in common]

        with torch.no_grad():
            for orig_name, recon_name in pairs:
                img_orig = transform(Image.open(os.path.join(originals_dir, orig_name)).convert("RGB"))
                img_recon = transform(Image.open(os.path.join(reconstructions_dir, recon_name)).convert("RGB"))
                # LPIPS expects [-1, 1]
                img_orig = img_orig.unsqueeze(0).to(device) * 2.0 - 1.0
                img_recon = img_recon.unsqueeze(0).to(device) * 2.0 - 1.0
                d = loss_fn(img_orig, img_recon)
                lpips_values.append(d.item())

        if lpips_values:
            mean_lpips = sum(lpips_values) / len(lpips_values)
            metrics["lpips"] = round(mean_lpips, 4)
            print(f"  LPIPS: {mean_lpips:.4f}")
    except ImportError:
        print("  lpips not installed. Install with: pip install lpips")
    except Exception as e:
        print(f"  LPIPS computation error: {e}")

    # Save to JSON
    if metrics:
        out_path = os.path.join(output_dir, "fid_lpips.json")
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Saved to: {out_path}")

    return metrics


def reverse_step(x0_hat, et, at, at_next, betas, t, sampler="ddim"):
    """
    Compute x_{t-1} from x0_hat and noise prediction.

    ddim:  DDIM-style (Song et al., 2021). sigma_t = sqrt(1 - at/at_next).
    ddpm:  Standard DDPM posterior (Ho et al., 2020; used in DPS paper).
           mu = sqrt(alpha_bar_{t-1}) * beta_t / (1-alpha_bar_t) * x0_hat
              + sqrt(alpha_t) * (1-alpha_bar_{t-1}) / (1-alpha_bar_t) * x_t
           But we don't have x_t here, so use the equivalent DDIM form with
           sigma = sqrt(beta_tilde_t) where beta_tilde = beta_t * (1-alpha_bar_{t-1}) / (1-alpha_bar_t).
    """
    if sampler == "ddpm":
        # DDPM posterior variance: beta_tilde = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        beta_t = betas[t.long().clamp(min=0)]
        if beta_t.dim() == 0:
            beta_t = beta_t.view(1, 1, 1, 1)
        elif beta_t.dim() == 1:
            beta_t = beta_t.view(-1, 1, 1, 1)
        beta_tilde = beta_t * (1 - at_next) / (1 - at).clamp(min=1e-20)
        sigma_t = beta_tilde.clamp(min=1e-20).sqrt()
    else:  # ddim
        sigma_t = (1 - at / at_next).sqrt()

    c2 = (1 - at_next - sigma_t ** 2).clamp(min=0).sqrt()
    xt_next = at_next.sqrt() * x0_hat + c2 * et + sigma_t * torch.randn_like(x0_hat)
    return xt_next, sigma_t


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
        elif guidance == 'cg':
            self.classifier_guidance(model)
        elif guidance == 'true_cg':
            self.true_classifier_guidance(model)
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

        if getattr(args, 'spnn_ckpt', None) is not None:
            ckpt_path = args.spnn_ckpt
        else:
            ckpt_path = hf_hub_download(repo_id="yamitehr/SPNN", filename="spnn_celebahq_256.pth")

        print(f"Loading classifier from {ckpt_path}")
        classifier.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=False))
        classifier.eval()

        A = lambda z, **kwargs: classifier(z, **kwargs)
        Ap = lambda logits, **kwargs: (classifier.pinv(logits, **kwargs))

        # Evaluate with SPNN (independent evaluator, consistent across all methods)
        evaluator = _load_spnn_evaluator(self.device)
        all_y_target = []
        all_y_pred = []

        # Create output dirs for FID computation before the loop
        orig_dir = os.path.join(self.args.image_folder, "originals")
        recon_dir = os.path.join(self.args.image_folder, "reconstructions")
        os.makedirs(orig_dir, exist_ok=True)
        os.makedirs(recon_dir, exist_ok=True)

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
            orig = inverse_data_transform(config, x_orig[0])

            img_name = f"{idx_so_far}.png"
            tvu.save_image(orig.cpu(), os.path.join(orig_dir, img_name))
            tvu.save_image(final_x0[0].cpu(), os.path.join(recon_dir, img_name))

            # Save result grid
            res_grid = torch.cat([orig.cpu(), final_x0[0].cpu()], dim=-1)
            grid_path = os.path.join(self.args.image_folder, f"grid_{idx_so_far}.png")
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

        # FID & LPIPS (matching DPS paper evaluation protocol)
        print(f"\n{'='*60}")
        print("  Computing FID & LPIPS...")
        _compute_fid_and_lpips(orig_dir, recon_dir, self.args.image_folder, device=str(self.device))
        print(f"{'='*60}")


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

        # Evaluate with SPNN (independent of guidance classifier for fair comparison)
        evaluator = _load_spnn_evaluator(self.device)
        all_y_target = []
        all_y_pred = []

        # Create output dirs for FID computation before the loop
        orig_dir = os.path.join(self.args.image_folder, "originals")
        recon_dir = os.path.join(self.args.image_folder, "reconstructions")
        os.makedirs(orig_dir, exist_ok=True)
        os.makedirs(recon_dir, exist_ok=True)

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

                sampler = getattr(args, 'sampler', 'ddim')

                if j < i:  # normal sampling
                    t = (torch.ones(n) * i).to(x.device)
                    next_t = (torch.ones(n) * j).to(x.device)
                    at = compute_alpha(self.betas, t.long())
                    at_next = compute_alpha(self.betas, next_t.long())
                    xt = xs[-1].to('cuda').requires_grad_(True)

                    # Diffusion model predicts noise (no grad through U-Net)
                    with torch.no_grad():
                        et = model(xt, t)
                        if et.size(1) == 6:
                            et = et[:, :3]

                    # Tweedie estimate of x0 — MUST be in grad graph w.r.t. xt
                    x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                    # DPS guidance (Alg. 1, Chung et al.): use unsquared norm
                    # as in official impl — equivalent to squared + zeta/||r|| normalization
                    y_pred = classifier(x0_t)
                    difference = y - y_pred
                    norm = torch.linalg.norm(difference)
                    norm.backward()

                    with torch.no_grad():
                        x0_t_hat = x0_t.detach()

                        # Unconditional reverse step
                        xt_next, _ = reverse_step(x0_t_hat, et, at, at_next, self.betas, t, sampler=sampler)

                        # DPS correction applied to x_{t-1}
                        xt_next = xt_next - zeta * xt.grad.detach()

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
            orig = inverse_data_transform(config, x_orig[0])

            img_name = f"{idx_so_far}.png"
            tvu.save_image(orig.cpu(), os.path.join(orig_dir, img_name))
            tvu.save_image(final_x0[0].cpu(), os.path.join(recon_dir, img_name))

            # Save result grid
            res_grid = torch.cat([orig.cpu(), final_x0[0].cpu()], dim=-1)
            grid_path = os.path.join(self.args.image_folder, f"grid_{idx_so_far}.png")
            tvu.save_image(res_grid, grid_path)

            mse = torch.mean((final_x0[0].to(self.device) - orig) ** 2)
            psnr = 10 * torch.log10(1 / mse)
            avg_psnr += psnr

            # Evaluate reconstruction with SPNN evaluator on both original and reconstruction
            with torch.no_grad():
                orig_normalized = orig.unsqueeze(0).to(self.device) * 2.0 - 1.0 if orig.dim() == 3 else orig.to(self.device) * 2.0 - 1.0
                final_x0_normalized = final_x0[0].unsqueeze(0).to(self.device) * 2.0 - 1.0
                y_target_eval = evaluator(orig_normalized)
                y_pred_eval = evaluator(final_x0_normalized)
                all_y_target.append(y_target_eval.cpu().squeeze(0))
                all_y_pred.append(y_pred_eval.cpu().squeeze(0))

            idx_so_far += y.shape[0]

            pbar.set_description("PSNR: %.2f" % (avg_psnr / (idx_so_far - idx_init)))

        total_time = time.time() - t_start
        avg_psnr = avg_psnr / (idx_so_far - idx_init)
        print("Total Average PSNR: %.2f" % avg_psnr)
        print("Number of samples: %d" % (idx_so_far - idx_init))
        print(f"Results saved to: {os.path.abspath(self.args.image_folder)}")
        _compute_and_save_metrics(all_y_target, all_y_pred, self.args.image_folder, total_time, idx_so_far - idx_init)

        # FID & LPIPS (matching DPS paper evaluation protocol)
        print(f"\n{'='*60}")
        print("  Computing FID & LPIPS...")
        _compute_fid_and_lpips(orig_dir, recon_dir, self.args.image_folder, device=str(self.device))
        print(f"{'='*60}")


    def classifier_guidance(self, model):
        """
        Classifier Guidance baseline (Dhariwal & Nichol, NeurIPS 2021).

        Modifies the noise prediction by adding the classifier gradient:
          ε̃ = ε_θ(x_t, t) - √(1 - ᾱ_t) · s · ∇_{x_t} log p(y|x̂_0)
        then uses ε̃ for both the Tweedie estimate and the DDPM step.

        Key difference from DPS:
          - DPS: unconditional DDPM step, then subtract gradient from x_{t-1}
          - CG:  gradient modifies ε, so it affects x̂_0 and the entire DDPM step

        Since our classifier is trained on clean images (not noisy x_t), we
        approximate ∇_{x_t} log p(y|x_t) ≈ ∇_{x_t} log p(y|x̂_0(x_t)) via
        the Tweedie estimate, same approximation as DPS.

        Reference: Dhariwal & Nichol, "Diffusion Models Beat GANs on Image
        Synthesis", NeurIPS 2021.
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
            print(f"Loaded ResNet-50 classifier from {args.resnet_ckpt}")
        else:
            classifier = SPNN(img_ch=3, num_classes=40, hidden=128, scale_bound=2.0, img_size=256).to(
                self.device)
            ckpt_path = hf_hub_download(repo_id="yamitehr/SPNN", filename="spnn_celebahq_256.pth")
            classifier.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=False))
            print(f"Loaded SPNN classifier from {ckpt_path}")
        classifier.eval()

        scale = args.cg_scale

        # Evaluate with SPNN (independent evaluator)
        evaluator = _load_spnn_evaluator(self.device)
        all_y_target = []
        all_y_pred = []

        # Create output dirs for FID computation before the loop
        orig_dir = os.path.join(self.args.image_folder, "originals")
        recon_dir = os.path.join(self.args.image_folder, "reconstructions")
        os.makedirs(orig_dir, exist_ok=True)
        os.makedirs(recon_dir, exist_ok=True)

        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        t_start = time.time()
        pbar = tqdm.tqdm(val_loader)
        for x_orig, classes in pbar:
            x_orig = x_orig.to(self.device)
            x_orig = data_transform(self.config, x_orig)

            # Target measurement
            with torch.no_grad():
                y = classifier(x_orig)

            if config.sampling.batch_size != 1:
                raise ValueError("please change the config file to set batch size as 1")

            # init x_T (same seed for fair comparison)
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

                sampler = getattr(args, 'sampler', 'ddim')

                if j < i:  # normal sampling
                    t = (torch.ones(n) * i).to(x.device)
                    next_t = (torch.ones(n) * j).to(x.device)
                    at = compute_alpha(self.betas, t.long())
                    at_next = compute_alpha(self.betas, next_t.long())
                    xt = xs[-1].to('cuda').requires_grad_(True)

                    # Diffusion model predicts noise (no grad through U-Net)
                    with torch.no_grad():
                        et = model(xt, t)
                        if et.size(1) == 6:
                            et = et[:, :3]

                    # Tweedie estimate — in grad graph w.r.t. xt
                    x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                    # Classifier guidance: ∇_{x_t} log p(y|x̂_0)
                    # Use unsquared norm (same as DPS for consistency)
                    y_pred = classifier(x0_t)
                    difference = y - y_pred
                    norm = torch.linalg.norm(difference)
                    norm.backward()

                    with torch.no_grad():
                        # Modify noise prediction: ε̃ = ε - √(1-ᾱ_t) · s · ∇_{x_t}
                        et_guided = et - (1 - at).sqrt() * scale * xt.grad.detach()

                        # Recompute x̂_0 with guided noise
                        x0_t_hat = (xt - et_guided * (1 - at).sqrt()) / at.sqrt()

                        # Reverse step with guided estimates
                        xt_next, _ = reverse_step(x0_t_hat, et_guided, at, at_next, self.betas, t, sampler=sampler)

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
            orig = inverse_data_transform(config, x_orig[0])

            img_name = f"{idx_so_far}.png"
            tvu.save_image(orig.cpu(), os.path.join(orig_dir, img_name))
            tvu.save_image(final_x0[0].cpu(), os.path.join(recon_dir, img_name))

            # Save result grid
            res_grid = torch.cat([orig.cpu(), final_x0[0].cpu()], dim=-1)
            grid_path = os.path.join(self.args.image_folder, f"grid_{idx_so_far}.png")
            tvu.save_image(res_grid, grid_path)

            mse = torch.mean((final_x0[0].to(self.device) - orig) ** 2)
            psnr = 10 * torch.log10(1 / mse)
            avg_psnr += psnr

            # Evaluate reconstruction with SPNN evaluator on both original and reconstruction
            with torch.no_grad():
                orig_normalized = orig.unsqueeze(0).to(self.device) * 2.0 - 1.0 if orig.dim() == 3 else orig.to(self.device) * 2.0 - 1.0
                final_x0_normalized = final_x0[0].unsqueeze(0).to(self.device) * 2.0 - 1.0
                y_target_eval = evaluator(orig_normalized)
                y_pred_eval = evaluator(final_x0_normalized)
                all_y_target.append(y_target_eval.cpu().squeeze(0))
                all_y_pred.append(y_pred_eval.cpu().squeeze(0))

            idx_so_far += y.shape[0]

            pbar.set_description("PSNR: %.2f" % (avg_psnr / (idx_so_far - idx_init)))

        total_time = time.time() - t_start
        avg_psnr = avg_psnr / (idx_so_far - idx_init)
        print("Total Average PSNR: %.2f" % avg_psnr)
        print("Number of samples: %d" % (idx_so_far - idx_init))
        print(f"Results saved to: {os.path.abspath(self.args.image_folder)}")
        _compute_and_save_metrics(all_y_target, all_y_pred, self.args.image_folder, total_time, idx_so_far - idx_init)

        # FID & LPIPS
        print(f"\n{'='*60}")
        print("  Computing FID & LPIPS...")
        _compute_fid_and_lpips(orig_dir, recon_dir, self.args.image_folder, device=str(self.device))
        print(f"{'='*60}")


    def true_classifier_guidance(self, model):
        """
        True Classifier Guidance (Dhariwal & Nichol, NeurIPS 2021).

        Uses a noise-aware classifier trained on (x_t, t) at all noise levels.
        No Tweedie approximation needed — the classifier directly evaluates
        noisy x_t and provides ∇_{x_t} log p(y|x_t, t).

        The score is modified as:
          ε̃ = ε_θ(x_t, t) − √(1−ᾱ_t) · s · ∇_{x_t} log p(y|x_t, t)
        where log p(y|x_t, t) = −BCE(classifier(x_t, t), y).

        Reference: Dhariwal & Nichol, "Diffusion Models Beat GANs on Image
        Synthesis", NeurIPS 2021.
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

        # Load noise-aware classifier
        assert args.noisy_resnet_ckpt is not None, \
            "Must provide --noisy_resnet_ckpt for --guidance_method true_cg"
        classifier = NoisyResNet50CelebA(num_classes=40, t_emb_dim=256).to(self.device)
        classifier.load_state_dict(torch.load(args.noisy_resnet_ckpt, map_location=self.device, weights_only=True))
        classifier.eval()
        print(f"Loaded noise-aware classifier from {args.noisy_resnet_ckpt}")

        # We also need the clean classifier to compute target y from x_orig
        # Use a standard ResNet or SPNN for this
        classifier_type = getattr(args, 'classifier', 'spnn')
        if classifier_type == 'resnet':
            assert args.resnet_ckpt is not None
            target_classifier = ResNet50CelebA(num_classes=40, pretrained=False).to(self.device)
            target_classifier.load_state_dict(torch.load(args.resnet_ckpt, map_location=self.device, weights_only=True))
            print(f"Target classifier: ResNet-50 from {args.resnet_ckpt}")
        else:
            target_classifier = SPNN(img_ch=3, num_classes=40, hidden=128, scale_bound=2.0, img_size=256).to(self.device)
            ckpt_path = hf_hub_download(repo_id="yamitehr/SPNN", filename="spnn_celebahq_256.pth")
            target_classifier.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=False))
            print(f"Target classifier: SPNN from {ckpt_path}")
        target_classifier.eval()

        scale = args.cg_scale

        # Evaluate with SPNN (independent evaluator)
        evaluator = _load_spnn_evaluator(self.device)
        all_y_target = []
        all_y_pred = []

        # Create output dirs
        orig_dir = os.path.join(self.args.image_folder, "originals")
        recon_dir = os.path.join(self.args.image_folder, "reconstructions")
        os.makedirs(orig_dir, exist_ok=True)
        os.makedirs(recon_dir, exist_ok=True)

        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        t_start = time.time()
        pbar = tqdm.tqdm(val_loader)
        for x_orig, classes in pbar:
            x_orig = x_orig.to(self.device)
            x_orig = data_transform(self.config, x_orig)

            # Target: binary attributes from clean image (as probabilities for BCE)
            with torch.no_grad():
                y_logits = target_classifier(x_orig)
                y_binary = (y_logits > 0).float()  # binary targets for BCE

            if config.sampling.batch_size != 1:
                raise ValueError("please change the config file to set batch size as 1")

            # init x_T
            torch.manual_seed(args.seed)
            x = torch.randn(
                y_binary.shape[0],
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

            sampler = getattr(args, 'sampler', 'ddim')

            for step_idx, (i, j) in enumerate(tqdm.tqdm(time_pairs)):
                i, j = i * skip, j * skip
                if j < 0: j = -1

                if j < i:  # normal sampling
                    t = (torch.ones(n) * i).to(x.device)
                    next_t = (torch.ones(n) * j).to(x.device)
                    at = compute_alpha(self.betas, t.long())
                    at_next = compute_alpha(self.betas, next_t.long())
                    xt = xs[-1].to('cuda').requires_grad_(True)

                    # Diffusion model predicts noise (no grad through U-Net)
                    with torch.no_grad():
                        et = model(xt, t)
                        if et.size(1) == 6:
                            et = et[:, :3]

                    # True classifier guidance: ∇_{x_t} log p(y|x_t, t)
                    # log p(y|x_t, t) = −BCE(classifier(x_t, t), y_binary)
                    t_int = t.long()
                    y_pred_noisy = classifier(xt, t_int)
                    log_likelihood = -torch.nn.functional.binary_cross_entropy_with_logits(
                        y_pred_noisy, y_binary, reduction='sum'
                    )
                    log_likelihood.backward()

                    with torch.no_grad():
                        # Modify noise prediction: ε̃ = ε − √(1−ᾱ_t) · s · ∇_{x_t} log p(y|x_t)
                        et_guided = et - (1 - at).sqrt() * scale * xt.grad.detach()

                        # Recompute x̂_0 with guided noise
                        x0_t_hat = (xt - et_guided * (1 - at).sqrt()) / at.sqrt()

                        # Reverse step
                        xt_next, _ = reverse_step(x0_t_hat, et_guided, at, at_next, self.betas, t, sampler=sampler)

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
            orig = inverse_data_transform(config, x_orig[0])

            img_name = f"{idx_so_far}.png"
            tvu.save_image(orig.cpu(), os.path.join(orig_dir, img_name))
            tvu.save_image(final_x0[0].cpu(), os.path.join(recon_dir, img_name))

            # Save result grid
            res_grid = torch.cat([orig.cpu(), final_x0[0].cpu()], dim=-1)
            grid_path = os.path.join(self.args.image_folder, f"grid_{idx_so_far}.png")
            tvu.save_image(res_grid, grid_path)

            mse = torch.mean((final_x0[0].to(self.device) - orig) ** 2)
            psnr = 10 * torch.log10(1 / mse)
            avg_psnr += psnr

            # Evaluate reconstruction with SPNN evaluator on both original and reconstruction
            with torch.no_grad():
                orig_normalized = orig.unsqueeze(0).to(self.device) * 2.0 - 1.0 if orig.dim() == 3 else orig.to(self.device) * 2.0 - 1.0
                final_x0_normalized = final_x0[0].unsqueeze(0).to(self.device) * 2.0 - 1.0
                y_target_eval = evaluator(orig_normalized)
                y_pred_eval = evaluator(final_x0_normalized)
                all_y_target.append(y_target_eval.cpu().squeeze(0))
                all_y_pred.append(y_pred_eval.cpu().squeeze(0))

            idx_so_far += y_binary.shape[0]

            pbar.set_description("PSNR: %.2f" % (avg_psnr / (idx_so_far - idx_init)))

        total_time = time.time() - t_start
        avg_psnr = avg_psnr / (idx_so_far - idx_init)
        print("Total Average PSNR: %.2f" % avg_psnr)
        print("Number of samples: %d" % (idx_so_far - idx_init))
        print(f"Results saved to: {os.path.abspath(self.args.image_folder)}")
        _compute_and_save_metrics(all_y_target, all_y_pred, self.args.image_folder, total_time, idx_so_far - idx_init)

        # FID & LPIPS
        print(f"\n{'='*60}")
        print("  Computing FID & LPIPS...")
        _compute_fid_and_lpips(orig_dir, recon_dir, self.args.image_folder, device=str(self.device))
        print(f"{'='*60}")


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
