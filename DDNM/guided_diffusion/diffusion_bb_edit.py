"""DDNM diffusion driver — "designed bbox" mode (source-paste flavor).

Differs from diffusion_copy.py only in the conditioning target: instead of
running the detector on a real image and using the *full* y as the target,
we pick a real source image with the desired class, run A on it to get
y_src, and pin only the cells corresponding to that object's GT box. The
rest of y is left free (taken from y_cur each step). The diffusion then
hallucinates an image whose detector output, *in the pinned region*,
matches what the source's bird/dog/etc. would produce.

Concrete y-space mask:
  - hmap[cls]: rectangular footprint = the GT box's extent at stride-4
              (every grid cell the box touches). Values copied from y_src.
  - regs/wh:  single pixel at the box's integer center — matches CenterNet's
              regs/wh supervision (only the GT-peak cell is supervised at
              training, and ctdet_decode reads only that cell).
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from models import SPNN, ConvPINNBlock, PixelUnshuffleBlock
from huggingface_hub import hf_hub_download

import json
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


# Pascal VOC class names (alphabetical, matches the channel order of the
# trained SPNN-CenterNet head).
_VOC_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


# Hard-coded prototype: pick a VOC train image whose annotation contains
# this class, run A on it, and use that real detector output as the pinned
# target. The mask covers exactly the GT box's hmap-disk + regs/wh-peak.
# Position is determined by the source image (no translation).
PROTOTYPE_TARGET_CLASS = "bird"


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


# ---------------------------------------------------------------------------
# y_pinned + mask construction helpers
# ---------------------------------------------------------------------------

def pick_source_image_for_class(target_class, voc_data_dir,
                                image_size=256,
                                min_box_area_after_crop=400.0):
    """Auto-pick a VOC trainval0712 image where target_class has a GT box
    that survives the same center-crop + resize pipeline used by
    VOCValForDDNM. Returns:
        source_chw_01 — torch tensor [3, image_size, image_size] in [0,1] RGB
        gt_box        — (x1, y1, x2, y2) in cropped+resized image coords
        cls_idx       — int (0..19), index into _VOC_NAMES
    """
    import cv2 as _cv2

    ann_path = os.path.join(voc_data_dir, "voc", "annotations",
                            "pascal_trainval0712.json")
    img_dir = os.path.join(voc_data_dir, "voc", "images")
    with open(ann_path) as f:
        ann = json.load(f)
    cat_name_to_id = {c["name"]: c["id"] for c in ann.get("categories", [])}
    image_by_id = {im["id"]: im for im in ann.get("images", [])}
    if target_class not in cat_name_to_id:
        raise RuntimeError(f"VOC has no category named {target_class!r}")
    target_cat_id = cat_name_to_id[target_class]

    for a in ann.get("annotations", []):
        if a["category_id"] != target_cat_id:
            continue
        info = image_by_id.get(a["image_id"])
        if info is None:
            continue
        h, w = info["height"], info["width"]
        s = min(h, w)
        y0 = (h - s) // 2
        x0 = (w - s) // 2
        scale = image_size / s
        bx, by, bw, bh = a["bbox"]
        x1 = max(0.0, min(float(s), bx - x0))
        y1 = max(0.0, min(float(s), by - y0))
        x2 = max(0.0, min(float(s), bx + bw - x0))
        y2 = max(0.0, min(float(s), by + bh - y0))
        if x2 <= x1 or y2 <= y1:
            continue
        x1 *= scale; y1 *= scale; x2 *= scale; y2 *= scale
        if (x2 - x1) * (y2 - y1) < min_box_area_after_crop:
            continue
        img_path = os.path.join(img_dir, info["file_name"])
        img = _cv2.imread(img_path)
        if img is None:
            continue
        img = img[y0:y0 + s, x0:x0 + s]
        img = _cv2.resize(img, (image_size, image_size),
                          interpolation=_cv2.INTER_AREA)
        img = _cv2.cvtColor(img, _cv2.COLOR_BGR2RGB)
        chw_01 = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)
        return (torch.from_numpy(chw_01).contiguous(),
                (float(x1), float(y1), float(x2), float(y2)),
                _VOC_NAMES.index(target_class),
                info["file_name"])

    raise RuntimeError(
        f"No VOC trainval image found with class {target_class!r}")


def build_source_y_and_mask(target_class, A_fn, voc_data_dir, config,
                            *, num_classes=20, image_size=256, stride=4,
                            device='cuda'):
    """Run A on an auto-picked VOC train image containing target_class.
    Build a binary mask covering only that class's GT bbox at stride-4
    (Gaussian-disk for hmap, single-pixel for regs/wh — same shape as
    before, but values come from the real detector output).

    Returns:
        y_src         — A(source_image), [1, nc+4, H/4, W/4]
        mask          — binary mask, 1 on pinned cells
        source_chw_01 — source image in [0,1] CHW (for saving)
        gt_box_256    — source GT box in image-256 coords (for grid drawing)
        source_file   — VOC filename (for logging)
    """
    source_chw_01, gt_box, cls_idx, source_file = pick_source_image_for_class(
        target_class, voc_data_dir, image_size=image_size)

    # [0, 1] → diffusion space ([-1, 1] when config.data.rescaled=true) → A.
    # Match diffusion_copy.py exactly: route the conversion through
    # data_transform(config, ...) instead of an inline *2-1 so that any
    # config change (e.g. rescaled=false, logit, etc.) is honored here too.
    x_diff = data_transform(config, source_chw_01.unsqueeze(0).to(device))
    with torch.no_grad():
        y_src = A_fn(x_diff)  # [1, nc+4, H/4, W/4]

    H = image_size // stride
    nc = num_classes
    mask = torch.zeros_like(y_src)

    x1, y1, x2, y2 = gt_box
    # Map the GT box from 256-coords into the stride-4 grid. floor for the
    # top-left corner and ceil for the bottom-right so every grid cell the
    # box touches is included; clip to grid bounds.
    gx1 = max(0,     int(x1 // stride))
    gy1 = max(0,     int(y1 // stride))
    gx2 = min(H,     int(-(-x2 // stride)))   # ceil
    gy2 = min(H,     int(-(-y2 // stride)))
    if gx2 > gx1 and gy2 > gy1:
        # Mask the GT-box's stride-4 footprint in the target class's hmap
        # channel — this is "extract the object's tensor region from the
        # source y". Other class channels stay free everywhere.
        mask[0, cls_idx, gy1:gy2, gx1:gx2] = 1.0

    # regs/wh peak cell: still single-pixel (matches CenterNet training-time
    # supervision and ctdet_decode's read pattern). Take the box's integer
    # center as the peak.
    cx_int = int(((x1 + x2) / 2.0) // stride)
    cy_int = int(((y1 + y2) / 2.0) // stride)
    if 0 <= cy_int < H and 0 <= cx_int < H:
        mask[0, nc + 0:nc + 4, cy_int, cx_int] = 1.0

    return y_src, mask, source_chw_01, gt_box, source_file


def compute_voc_cooccurrence(annotations_path, top_k=10):
    """One-time descriptive stat: count class-pair co-occurrences on
    VOC train+val 0712, return the top-K most frequent unordered pairs.

    Pairs are computed over distinct class sets per image (so "two dogs in
    one image" doesn't inflate dog-dog).
    """
    from collections import Counter
    with open(annotations_path) as f:
        ann = json.load(f)
    cat_name = {c["id"]: c["name"] for c in ann.get("categories", [])}
    by_image = {}
    for a in ann.get("annotations", []):
        by_image.setdefault(a["image_id"], set()).add(cat_name.get(a["category_id"]))
    pair_counts = Counter()
    for classes in by_image.values():
        clist = sorted(c for c in classes if c is not None)
        for i in range(len(clist)):
            for j in range(i + 1, len(clist)):
                pair_counts[(clist[i], clist[j])] += 1
    return pair_counts.most_common(top_k)


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

        print('Run BB-edit DDNM.',
              f'{self.config.time_travel.T_sampling} sampling steps.',
              f'travel_length = {self.config.time_travel.travel_length},',
              f'travel_repeat = {self.config.time_travel.travel_repeat}.'
             )
        self.simplified_ddnm_plus(model)


    def _build_detection_A_Ap(self, args):
        """Same SPNN-CenterNet path as diffusion_copy.py — kept verbatim."""
        import sys as _sys
        _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        _centernet_ref = os.path.join(_project_root, "centernet_ref")
        for p in (_project_root, _centernet_ref):
            if p not in _sys.path:
                _sys.path.insert(0, p)
        from nets.spnn_centernet_copy import get_spnn_centernet  # noqa: E402

        assert args.detector_ckpt is not None, \
            "Must provide --detector_ckpt for detection DDNM"

        head_mix_reflections = (args.detector_head_mix_reflections
                                if args.detector_head_mix_reflections > 0 else None)
        detector = get_spnn_centernet(
            num_classes=args.detector_num_classes,
            pretrained_backbone=None,
            hmap_init_scale=args.detector_hmap_init_scale,
            hmap_init_bias=args.detector_hmap_init_bias,
            head_mode=args.detector_head_mode,
            head_mix_type=args.detector_head_mix_type,
            head_mix_reflections=head_mix_reflections,
            deep_det_head=args.detector_deep_det_head,
            deep_head_hidden=args.detector_deep_head_hidden,
            two_block_head=args.detector_two_block_head,
            freeze_backbone=False,
            no_hmap_scale=getattr(args, 'no_hmap_scale', False),
            no_hmap_bias=getattr(args, 'no_hmap_bias', False),
            internal_head_affine=getattr(args, 'internal_head_affine', False),
        ).to(self.device)

        raw = torch.load(args.detector_ckpt, map_location=self.device,
                         weights_only=False)
        if isinstance(raw, dict) and 'state_dict' in raw:
            state = raw['state_dict']
        elif isinstance(raw, dict) and 'model' in raw:
            state = raw['model']
        else:
            state = raw
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

        nc = args.detector_num_classes
        img_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        img_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        def diffusion_to_detector(x):
            x01 = (x + 1.0) / 2.0
            x_bgr = x01[:, [2, 1, 0]]
            return (x_bgr - img_mean) / img_std

        def detector_to_diffusion(x):
            x01_bgr = x * img_std + img_mean
            x01_rgb = x01_bgr[:, [2, 1, 0]]
            return x01_rgb * 2.0 - 1.0

        def _wrapper_post_spnn(raw_24ch):
            hmap = raw_24ch[:, :nc]
            if getattr(detector, 'use_hmap_scale', True):
                hmap = hmap * detector.hmap_scale
            if detector.head_mode == 'orthogonal_mix':
                hmap = detector.hmap_mix(hmap)
            if getattr(detector, 'use_hmap_bias', True):
                hmap = hmap + detector.hmap_bias
            regs = raw_24ch[:, nc:nc + 2]
            w_h_ = raw_24ch[:, nc + 2:]
            return torch.cat([hmap, regs, w_h_], dim=1)

        def A(z, return_latents=False):
            x_det = diffusion_to_detector(z)
            if return_latents:
                raw, latents = detector.spnn(x_det, return_latents=True)
                y = _wrapper_post_spnn(raw)
                return y, latents
            out = detector(x_det)
            hmap, regs, w_h_ = out[0]
            return torch.cat([hmap, regs, w_h_], dim=1)

        def Ap(y, latents=None):
            hmap = y[:, :nc]
            regs = y[:, nc:nc + 2]
            w_h_ = y[:, nc + 2:]
            hmap_raw = detector.hmap_to_raw(hmap)
            raw = torch.cat([hmap_raw, regs, w_h_], dim=1)
            x_det = detector.spnn.pinv(raw, latents=latents)
            return detector_to_diffusion(x_det)

        return detector, A, Ap

    _VOC_NAMES = _VOC_NAMES  # class-level alias

    def _save_pinned_grid(self, idx, results_dir, generated_chw_01,
                         pins, *, image_size=256):
        """Visualize the result: generated image with the pinned bboxes drawn
        in red. No SPNN run needed — we just draw the requested pins."""
        from PIL import Image, ImageDraw
        arr = (generated_chw_01.cpu().clamp(0, 1)
               .permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil = Image.fromarray(arr)
        draw = ImageDraw.Draw(pil)
        for pin in pins:
            x1, y1, x2, y2 = pin["box"]
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
            draw.text((x1 + 2, y1 + 2), pin["cls"], fill=(255, 255, 0))
        out_path = os.path.join(results_dir, f"grid_pinned_{idx}.png")
        pil.save(out_path)

    def _save_decoded_grid(self, idx, results_dir, generated_chw_01,
                           A_fn, *, nc=20, score_thresh=0.2,
                           max_boxes=15, K=100):
        """Run the detector on the generated image, draw top-K predictions.
        Lets the user check whether the detector actually 'sees' the pinned
        objects in the synthesized image."""
        import sys as _sys
        _project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", ".."))
        _centernet_ref = os.path.join(_project_root, "centernet_ref")
        for _p in (_project_root, _centernet_ref):
            if _p not in _sys.path:
                _sys.path.insert(0, _p)
        from utils.post_process import ctdet_decode  # noqa: E402
        from PIL import Image, ImageDraw

        # Run A on the generated image (in diffusion space). Use
        # data_transform so this stays consistent with build_source_y_and_mask
        # and diffusion_copy.py if the config's normalization changes.
        x_diff = data_transform(
            self.config, generated_chw_01.unsqueeze(0).to(self.device))
        with torch.no_grad():
            y = A_fn(x_diff)
            hmap = y[:, :nc]
            regs = y[:, nc:nc + 2]
            w_h_ = y[:, nc + 2:nc + 4]
            dets = ctdet_decode(hmap, regs, w_h_, K=K)[0].cpu().numpy()
        dets = dets[np.argsort(-dets[:, 4])]

        stride = 4
        arr = (generated_chw_01.cpu().clamp(0, 1)
               .permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil = Image.fromarray(arr)
        draw = ImageDraw.Draw(pil)
        shown = 0
        for d in dets:
            x1, y1, x2, y2, sc, cls = d
            if sc < score_thresh:
                continue
            x1p, y1p = x1 * stride, y1 * stride
            x2p, y2p = x2 * stride, y2 * stride
            cls = int(cls)
            cname = (self._VOC_NAMES[cls] if 0 <= cls < len(self._VOC_NAMES)
                     else str(cls))
            draw.rectangle([x1p, y1p, x2p, y2p], outline=(0, 255, 255), width=2)
            draw.text((x1p + 2, y1p + 2), f"{cname}:{sc:.2f}",
                      fill=(255, 255, 0))
            shown += 1
            if shown >= max_boxes:
                break
        out_path = os.path.join(results_dir, f"grid_decoded_{idx}.png")
        pil.save(out_path)

    def simplified_ddnm_plus(self, model):
        args, config = self.args, self.config
        nc = args.detector_num_classes

        # ---- Build A / A† for the detection task -------------------------
        classifier, A, Ap = self._build_detection_A_Ap(args)
        classifier.eval()

        # ---- Build pinned y + mask once. Source-paste flavor: pick a real
        # VOC train image containing PROTOTYPE_TARGET_CLASS, run A on it,
        # and use those values in the mask region. -------------------------
        target_class = PROTOTYPE_TARGET_CLASS
        results_dir = self.args.image_folder
        os.makedirs(results_dir, exist_ok=True)
        y_pinned, mask_y, source_chw_01, gt_box, source_file = \
            build_source_y_and_mask(
                target_class, A, args.voc_data_dir, config,
                num_classes=nc, image_size=config.data.image_size,
                stride=4, device=self.device)
        # Save the source image so the user can see what was sampled.
        tvu.save_image(
            source_chw_01,
            os.path.join(results_dir, f"source_{target_class}.png"))
        # List of (cls, xyxy_256) used by _save_pinned_grid below.
        pinned_boxes = [{"cls": target_class, "box": gt_box}]
        print(f"[bb-edit] target_class={target_class}  "
              f"source={source_file}  gt_box(256)={gt_box}")
        print(f"[bb-edit] y_pinned shape={tuple(y_pinned.shape)} "
              f"mask cells pinned={int(mask_y.sum().item())} "
              f"({100*mask_y.mean().item():.3f}% of total)")

        # ---- One-time co-occurrence dump (informational) -----------------
        train_ann = os.path.join(args.voc_data_dir, "voc", "annotations",
                                 "pascal_trainval0712.json")
        if os.path.exists(train_ann):
            try:
                top = compute_voc_cooccurrence(train_ann, top_k=10)
                print("[bb-edit] Top-10 VOC train+val class-pair co-occurrences:")
                for (c1, c2), n in top:
                    print(f"  {n:5d}  {c1:>12s}  +  {c2}")
            except Exception as e:
                print(f"[bb-edit] couldn't compute co-occurrence: {e}")

        # ---- Number of samples to draw from this single pin spec ---------
        # Each "image" idx is just a different noise seed for the same target.
        n_samples = max(1, args.subset_end - max(0, args.subset_start))

        idx_init = max(0, args.subset_start)
        for offset in range(n_samples):
            idx_so_far = idx_init + offset

            # Init x_T from a per-sample seed so each idx differs.
            torch.manual_seed(args.seed + offset)
            x = torch.randn(
                1,
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

                for step_idx, (i, j) in enumerate(tqdm.tqdm(time_pairs,
                                                            desc=f"img {idx_so_far}")):
                    i, j = i * skip, j * skip
                    if j < 0:
                        j = -1

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
                        lambda_t = (args.lambda1 if i < min_bp_step
                                    else args.lambda2)

                        # Current detector output for the unconditional
                        # one-step prediction.
                        y_cur, z_cur = A(x0_t_hat, return_latents=True)

                        # Fuse: free pixels = y_cur (so BP gradient is zero
                        # there); pinned pixels = y_pinned (force the bbox).
                        y_target = y_pinned * mask_y + y_cur * (1.0 - mask_y)

                        # Mask-weighted error: average sigmoid-diff over the
                        # pinned cells only. Outside the mask y_target ≡ y_cur,
                        # so unweighted .mean() would dilute the metric to ~0.
                        diff = (y_target.sigmoid() - y_cur.sigmoid()).abs()
                        denom = mask_y.sum().clamp(min=1)
                        nlbp_error = (diff * mask_y).sum() / denom

                        # lambda_t == 0 means "skip BP for this regime entirely":
                        # x0_t_hat stays as x0_t (the assignment a few lines up).
                        if lambda_t != 0.0 and nlbp_error > args.nlbp_stop_cond:

                            y_tar, z_tar = A(Ap(y_target), return_latents=True)
                            y_proj, z_proj = A(Ap(A(x0_t_hat)), return_latents=True)
                            z_final = []
                            for z0, z1, z2 in zip(z_cur, z_tar, z_proj):
                                if z0 is not None:
                                    z_final.append(z0 + lambda_t * (z1 - z2))
                                else:
                                    z_final.append(None)

                            y_final = y_cur + lambda_t * (y_tar - y_proj)

                            x0_t_hat = Ap(y_final, latents=z_final)
                            # Clamp to the valid diffusion-image range. Without
                            # this, large lambda_t (e.g. 1.0) can drive
                            # x0_t_hat to ±40+, which pushes the diffusion
                            # model out of distribution on the next step.
                            # Only inside the lambda_t != 0 branch — the
                            # short-circuit above leaves x0_t_hat = x0_t when
                            # lambda_t == 0, no clamp needed there.
                            x0_t_hat = x0_t_hat.clamp(-1, 1)

                        if step_idx % 10 == 0 or step_idx < 5:
                            print(f"  step {step_idx}: t={i} | x0_t range=[{x0_t.min():.3f}, {x0_t.max():.3f}] mean={x0_t.mean():.3f} | "
                                  f"x0_t_hat range=[{x0_t_hat.min():.3f}, {x0_t_hat.max():.3f}] mean={x0_t_hat.mean():.3f} | "
                                  f"nlbp_error={nlbp_error:.4f} lambda_t={lambda_t:.2f}")

                        # Per-step debug dump.
                        debug_dir = os.path.join(
                            self.args.image_folder, "debug_x0",
                            f"img_{idx_so_far}")
                        os.makedirs(debug_dir, exist_ok=True)
                        tvu.save_image(
                            inverse_data_transform(config, x0_t[0].cpu()),
                            os.path.join(debug_dir,
                                         f"step{step_idx:03d}_x0_t.png"))
                        tvu.save_image(
                            inverse_data_transform(config, x0_t_hat[0].cpu()),
                            os.path.join(debug_dir,
                                         f"step{step_idx:03d}_x0_t_hat.png"))

                        c2 = (1 - at_next - sigma_t ** 2).clamp(min=0).sqrt()
                        xt_next = at_next.sqrt() * x0_t_hat + c2 * et + sigma_t * torch.randn_like(x0_t)

                        x0_preds.append(x0_t_hat.to('cpu'))
                        xs.append(xt_next.to('cpu'))
                    else:  # time-travel back
                        next_t = (torch.ones(n) * j).to(x.device)
                        at_next = compute_alpha(self.betas, next_t.long())
                        x0_t = x0_preds[-1].to('cuda')
                        xt_next = at_next.sqrt() * x0_t_hat + torch.randn_like(x0_t) * (1 - at_next).sqrt()
                        xs.append(xt_next.to('cpu'))

            final_x0 = inverse_data_transform(config, x0_preds[-1])

            tvu.save_image(
                final_x0[0],
                os.path.join(self.args.image_folder, f"{idx_so_far}_0.png"),
            )

            # Two visual grids per sample:
            #  - grid_pinned_<i>.png: generated image + the requested pin boxes.
            #  - grid_decoded_<i>.png: generated image + detector's actual top-K.
            self._save_pinned_grid(idx_so_far, results_dir, final_x0[0],
                                  pinned_boxes,
                                  image_size=config.data.image_size)
            self._save_decoded_grid(idx_so_far, results_dir, final_x0[0], A,
                                    nc=nc, score_thresh=0.2)

        print(f"[bb-edit] {n_samples} sample(s) saved to: "
              f"{os.path.abspath(self.args.image_folder)}")



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
    assert times[0] > times[1], (times[0], times[1])
    assert times[-1] == -1, times[-1]
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a
