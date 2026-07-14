import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from models import SPNN, ConvPINNBlock, PixelUnshuffleBlock
from huggingface_hub import hf_hub_download

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import download

import torchvision.utils as tvu

from guided_diffusion.models import Model
from guided_diffusion.script_util import create_model, create_classifier, classifier_defaults, args_to_dict
import random



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
        cls_fn = None

        if self.config.model.type == 'simple':
            model = Model(self.config)
            if self.config.data.dataset == 'CelebA_HQ':
                ckpt = os.path.join(self.args.exp, "logs/celeba/celeba_hq.ckpt")
                if not os.path.exists(ckpt):
                    download('https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt', ckpt)
            else:
                raise ValueError(f"Unsupported dataset for 'simple' model type: {self.config.data.dataset}")
            model.load_state_dict(torch.load(ckpt, map_location=self.device, weights_only=False))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        elif self.config.model.type == 'openai':
            config_dict = vars(self.config.model)
            model = create_model(**config_dict)
            if self.config.model.use_fp16:
                model.convert_to_fp16()
            if getattr(self.config.model, 'class_cond', False):
                ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_diffusion.pt' % (
                    self.config.data.image_size, self.config.data.image_size))
                if not os.path.exists(ckpt):
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_diffusion_uncond.pt' % (
                            self.config.data.image_size, self.config.data.image_size), ckpt)
            else:
                ckpt = os.path.join(self.args.exp, "logs/imagenet/256x256_diffusion_uncond.pt")
                if not os.path.exists(ckpt):
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt',
                        ckpt)

            model.load_state_dict(torch.load(ckpt, map_location=self.device, weights_only=False))
            model.to(self.device)
            model.eval()
            model = torch.nn.DataParallel(model)

        else:
            raise ValueError(f"Unknown model type: {self.config.model.type}")

        print('Run Simplified DDNM.',
              f'{self.config.time_travel.T_sampling} sampling steps.',
              f'travel_length = {self.config.time_travel.travel_length},',
              f'travel_repeat = {self.config.time_travel.travel_repeat}.'
             )
        self.simplified_ddnm_plus(model)


    # ===================================================================
    # A / A† builders for the three supported tasks
    # ===================================================================

    def _build_imagenet_classification_A_Ap(self, args):
        """Original SPNN-classifier path: A: image → class logits."""
        spnn_ckpt = getattr(args, 'spnn_ckpt', None)
        num_classes = getattr(args, 'spnn_num_classes', 10)
        mix_type = getattr(args, 'spnn_mix_type', 'householder')
        scale_bound = getattr(args, 'spnn_scale_bound', 1.0)
        hidden = 256

        layer_channels = [
            (PixelUnshuffleBlock, {"r": 4}),
            (ConvPINNBlock, {"in_ch": 48, "out_ch": 24, "hidden": hidden,
                             "scale_bound": scale_bound, "feat_size": 64, "mix_type": mix_type}),
            (ConvPINNBlock, {"in_ch": 24, "out_ch": 12, "hidden": hidden,
                             "scale_bound": scale_bound, "feat_size": 64, "mix_type": mix_type}),
            (PixelUnshuffleBlock, {"r": 4}),
            (ConvPINNBlock, {"in_ch": 192, "out_ch": 96, "hidden": hidden,
                             "scale_bound": scale_bound, "feat_size": 16, "mix_type": mix_type}),
            (ConvPINNBlock, {"in_ch": 96, "out_ch": 48, "hidden": hidden,
                             "scale_bound": scale_bound, "feat_size": 16, "mix_type": mix_type}),
            (PixelUnshuffleBlock, {"r": 4}),
            (ConvPINNBlock, {"in_ch": 768, "out_ch": 192, "hidden": hidden,
                             "scale_bound": scale_bound, "feat_size": 4, "mix_type": mix_type}),
            (PixelUnshuffleBlock, {"r": 4}),
            (ConvPINNBlock, {"in_ch": 3072, "out_ch": 1024, "hidden": hidden,
                             "scale_bound": scale_bound, "feat_size": 1, "mix_type": mix_type}),
            (ConvPINNBlock, {"in_ch": 1024, "out_ch": num_classes, "hidden": hidden,
                             "scale_bound": scale_bound, "feat_size": 1, "mix_type": mix_type}),
        ]
        classifier = SPNN(img_ch=3, num_classes=num_classes, img_size=256,
                          layer_channels=layer_channels).to(self.device)

        assert spnn_ckpt is not None, "Must provide --spnn_ckpt for ImageNet DDNM"
        raw = torch.load(spnn_ckpt, map_location=self.device, weights_only=False)
        state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) and "state_dict" in raw else raw
        classifier.load_state_dict(state_dict)
        print(f"Loaded SPNN classifier ({num_classes} classes, {mix_type}) from {spnn_ckpt}")

        img_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        img_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        def diffusion_to_spnn(x):
            x01 = (x + 1.0) / 2.0
            return (x01 - img_mean) / img_std

        def spnn_to_diffusion(x):
            x01 = x * img_std + img_mean
            return x01 * 2.0 - 1.0

        A = lambda z, **kw: classifier(diffusion_to_spnn(z), **kw)
        Ap = lambda logits, **kw: spnn_to_diffusion(classifier.pinv(logits, **kw))
        return classifier, A, Ap

    def _build_celeba_classification_A_Ap(self):
        """CelebA SPNN classifier path."""
        classifier = SPNN(img_ch=3, num_classes=40, hidden=128,
                          scale_bound=2.0, img_size=256).to(self.device)
        ckpt_path = hf_hub_download(repo_id="yamitehr/SPNN",
                                    filename="spnn_celebahq_256.pth")
        print(f"Loading classifier from {ckpt_path}")
        classifier.load_state_dict(torch.load(ckpt_path, map_location=self.device,
                                              weights_only=False))
        A = lambda z, **kw: classifier(z, **kw)
        Ap = lambda logits, **kw: classifier.pinv(logits, **kw)
        return classifier, A, Ap

    def _build_detection_A_Ap(self, args):
        """SPNN-CenterNet path: A: image → full detector output (post-everything),
        a single tensor [B, num_classes+4, 64, 64].

        Bijective chain through ALL layers:
          diffusion (RGB, [-1,1])
            -> diffusion_to_detector: BGR + ImageNet-normalized
            -> detector.spnn  -> raw [B, 24, 64, 64]
            -> hmap channels: * scale, then orthogonal mix (if used), then + bias
            -> regs / w_h_ pass through unchanged
            -> y = cat([hmap, regs, w_h_], dim=1)

        A† reverses every step. The wrapper's affine + (optional) orthogonal
        mixer are both bijective, so they don't introduce extra latents — the
        latent list comes entirely from `detector.spnn`.
        """
        # Lazy import so the classification path doesn't pay the import cost
        # if centernet_ref isn't on sys.path.
        import sys as _sys
        _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        _centernet_ref = os.path.join(_project_root, "centernet_ref")
        for p in (_project_root, _centernet_ref):
            if p not in _sys.path:
                _sys.path.insert(0, p)
        from nets.spnn_centernet import get_spnn_centernet  # noqa: E402

        assert args.detector_ckpt is not None, \
            "Must provide --detector_ckpt for detection DDNM"

        detector = get_spnn_centernet(
            num_classes=args.detector_num_classes,
            pretrained_backbone=None,  # we're loading the full ckpt below
            hmap_init_scale=args.detector_hmap_init_scale,
            hmap_init_bias=args.detector_hmap_init_bias,
            deep_det_head=args.detector_deep_det_head,
            deep_head_hidden=args.detector_deep_head_hidden,
            two_block_head=args.detector_two_block_head,
            freeze_backbone=False,
        ).to(self.device)

        raw = torch.load(args.detector_ckpt, map_location=self.device,
                         weights_only=False)
        if isinstance(raw, dict) and 'state_dict' in raw:
            state = raw['state_dict']
        elif isinstance(raw, dict) and 'model' in raw:
            state = raw['model']
        else:
            state = raw
        # Strip 'module.' prefix from DataParallel / DDP saves
        state = {k[7:] if k.startswith('module.') else k: v
                 for k, v in state.items()}
        missing, unexpected = detector.load_state_dict(state, strict=False)
        print(f"[detector] Loaded {args.detector_ckpt} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
        if missing:
            print(f"[detector] first missing keys: {missing[:6]}")
        if unexpected:
            print(f"[detector] first unexpected keys: {unexpected[:6]}")
        detector.eval()
        for p in detector.parameters():
            p.requires_grad_(False)

        nc = args.detector_num_classes  # 20 for VOC
        img_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        img_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        def diffusion_to_detector(x):
            """[-1,1] RGB  →  BGR + ImageNet-normalized.

            VOC pascal.py loaded images via cv2.imread (BGR) and never swapped
            to RGB before applying RGB-named ImageNet stats — that's the
            CenterNet convention. To match the detector's training distribution
            we replicate it: RGB→BGR swap, then (x/255 − μ)/σ-equivalent step.
            (We start in [0,1] post-denorm, so it's just (x − μ)/σ.)
            """
            x01 = (x + 1.0) / 2.0                # [-1,1] RGB → [0,1] RGB
            x_bgr = x01[:, [2, 1, 0]]            # RGB → BGR
            return (x_bgr - img_mean) / img_std

        def detector_to_diffusion(x):
            """BGR + ImageNet-normalized  →  [-1,1] RGB."""
            x01_bgr = x * img_std + img_mean     # → [0,1] BGR
            x01_rgb = x01_bgr[:, [2, 1, 0]]      # BGR → RGB
            return x01_rgb * 2.0 - 1.0           # → [-1,1] RGB

        def A(z, return_latents=False):
            # The SPNN raw output IS the detection output (the per-class
            # affine lives inside the last head block's s/t, applied
            # automatically by spnn forward). No external head adapter to
            # replay, so we can call spnn directly even on the latent path.
            x_det = diffusion_to_detector(z)
            if return_latents:
                y, latents = detector.spnn(x_det, return_latents=True)
                return y, latents
            return detector.spnn(x_det)

        def Ap(y, latents=None):
            # No external head adapter to invert; spnn.pinv handles the
            # internal affine via its sign-flipped log_scale in s.
            x_det = detector.spnn.pinv(y, latents=latents)
            return detector_to_diffusion(x_det)

        return detector, A, Ap

    def simplified_ddnm_plus(self, model):
        args, config = self.args, self.config

        # ---- dataset selection ---------------------------------------------
        # For the detection task we use VOC2007 test split. The default
        # get_dataset(...) doesn't know about VOC; bypass it.
        if getattr(args, "task", "classification") == "detection":
            from datasets.voc import VOCValForDDNM
            test_dataset = VOCValForDDNM(args.voc_data_dir,
                                         image_size=config.data.image_size)
        else:
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

        # ---- A / A† selection ---------------------------------------------
        if getattr(args, "task", "classification") == "detection":
            classifier, A, Ap = self._build_detection_A_Ap(args)
        elif config.data.dataset == 'ImageNet':
            classifier, A, Ap = self._build_imagenet_classification_A_Ap(args)
        else:
            classifier, A, Ap = self._build_celeba_classification_A_Ap()

        classifier.eval()

        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
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

                        # Stopping condition: task-dependent error metric
                        if y_cur.dim() == 2:
                            # Classification: sigmoid-based attribute error.
                            # Bounded in [0, 1]; threshold ~0.1 is meaningful.
                            nlbp_error = (y_cur.sigmoid() - y.sigmoid()).abs().mean()
                        elif (getattr(args, "task", "classification") == "detection"
                              and y_cur.dim() == 4
                              and y_cur.shape[1] == args.detector_num_classes + 4):
                            # Detection: match the supervised training loss space.
                            #   hmap channels in probability space (focal-loss-like)
                            #   regs / w_h_ in raw L1, weighted as in training (1.0 / 0.1)
                            nc = args.detector_num_classes
                            e_hmap = (y_cur[:, :nc].sigmoid()
                                      - y[:, :nc].sigmoid()).abs().mean()
                            e_regs = (y_cur[:, nc:nc + 2]
                                      - y[:, nc:nc + 2]).abs().mean()
                            e_wh = (y_cur[:, nc + 2:nc + 4]
                                    - y[:, nc + 2:nc + 4]).abs().mean()
                            nlbp_error = e_hmap + 1.0 * e_regs + 0.1 * e_wh
                        else:
                            # Generic spatial output: raw tensor distance.
                            nlbp_error = (y_cur - y).abs().mean()

                        if nlbp_error > args.nlbp_stop_cond:

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
                            x0_t_hat = x0_t_hat.clamp(-1, 1)

                        if step_idx % 10 == 0 or step_idx < 5:
                            print(f"  step {step_idx}: t={i} | x0_t range=[{x0_t.min():.3f}, {x0_t.max():.3f}] mean={x0_t.mean():.3f} | "
                                  f"x0_t_hat range=[{x0_t_hat.min():.3f}, {x0_t_hat.max():.3f}] mean={x0_t_hat.mean():.3f} | "
                                  f"nlbp_error={nlbp_error:.4f} lambda_t={lambda_t:.2f}")

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

            # Per-image diagnostic: how well does A(generated) match A(original)?
            with torch.no_grad():
                y_orig = A(x_orig)
                y_gen = A(data_transform(config, final_x0.to(self.device)))
                if getattr(args, "task", "classification") == "detection":
                    # Per-component error in the same space as the NLBP stopping
                    # condition above, so the numbers are comparable across runs.
                    nc = args.detector_num_classes
                    e_h = (y_orig[:, :nc].sigmoid()
                           - y_gen[:, :nc].sigmoid()).abs().mean().item()
                    e_r = (y_orig[:, nc:nc + 2]
                           - y_gen[:, nc:nc + 2]).abs().mean().item()
                    e_w = (y_orig[:, nc + 2:nc + 4]
                           - y_gen[:, nc + 2:nc + 4]).abs().mean().item()
                    voc_id = classes[0].item() if classes.dim() > 0 else classes.item()
                    print(f"  [img {idx_so_far}] voc_id={voc_id} | "
                          f"hmap_err(prob)={e_h:.4f} regs_err={e_r:.4f} wh_err={e_w:.4f} | "
                          f"weighted={(e_h + e_r + 0.1*e_w):.4f}")
                else:
                    pred_orig = y_orig.argmax(dim=1).item()
                    pred_gen = y_gen.argmax(dim=1).item()
                    true_cls = classes[0].item() if classes.dim() > 0 else classes.item()
                    print(f"  [img {idx_so_far}] true_class={true_cls} | "
                          f"spnn_on_orig={pred_orig} | spnn_on_generated={pred_gen}")

            mse = torch.mean((final_x0[0].to(self.device) - orig) ** 2)
            psnr = 10 * torch.log10(1 / mse)
            avg_psnr += psnr

            idx_so_far += y.shape[0]

            pbar.set_description("PSNR: %.2f" % (avg_psnr / (idx_so_far - idx_init)))

        avg_psnr = avg_psnr / (idx_so_far - idx_init)
        print("Total Average PSNR: %.2f" % avg_psnr)
        print("Number of samples: %d" % (idx_so_far - idx_init))
        print(f"Results saved to: {os.path.abspath(self.args.image_folder)}")



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
