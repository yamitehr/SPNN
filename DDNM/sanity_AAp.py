"""Sanity-check A(Ap(y)) == y for the SPNN-CenterNet pipeline used by NLBP.

Tests:
  1. Real natural y from A(x_orig).
  2. Masked-outside-BB y (the 'crop_pinv' input).
  3. Random y in roughly the natural range.
"""
import argparse
import os
import sys

import numpy as np
import torch

THIS = os.path.abspath(os.path.dirname(__file__))
PROJ = os.path.abspath(os.path.join(THIS, ".."))
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "centernet_ref"))
sys.path.insert(0, THIS)

from datasets.voc import VOCValForDDNM  # noqa: E402
from nets.spnn_centernet_copy import get_spnn_centernet  # noqa: E402

VOC_NAMES = ("aeroplane bicycle bird boat bottle bus car cat chair cow "
             "diningtable dog horse motorbike person pottedplant sheep sofa "
             "train tvmonitor").split()
VOC_LABEL = {n: i for i, n in enumerate(VOC_NAMES)}


def build(args, device):
    detector = get_spnn_centernet(
        num_classes=20, pretrained_backbone=None,
        hmap_init_scale=0.01, hmap_init_bias=-2.19,
        head_mode="affine", head_mix_type="householder",
        head_mix_reflections=None,
        deep_det_head=True, deep_head_hidden=128,
        two_block_head=False, freeze_backbone=False,
        no_hmap_scale=True, no_hmap_bias=True,
        internal_head_affine=True,
    ).to(device)
    raw = torch.load(args.detector_ckpt, map_location=device, weights_only=False)
    state = raw['state_dict'] if isinstance(raw, dict) and 'state_dict' in raw \
        else raw['model'] if isinstance(raw, dict) and 'model' in raw else raw
    state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
    detector.load_state_dict(state, strict=False)
    detector.eval()
    for p in detector.parameters():
        p.requires_grad_(False)

    img_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    img_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    nc = 20

    def A(z):
        x01 = (z + 1.0) / 2.0
        x_bgr = x01[:, [2, 1, 0]]
        x_det = (x_bgr - img_mean) / img_std
        out = detector(x_det)
        hmap, regs, wh = out[0]
        return torch.cat([hmap, regs, wh], dim=1)

    def Ap(y, latents=None):
        hmap = y[:, :nc]
        regs = y[:, nc:nc + 2]
        w_h_ = y[:, nc + 2:]
        hmap_raw = detector.hmap_to_raw(hmap)
        raw_y = torch.cat([hmap_raw, regs, w_h_], dim=1)
        x_det = detector.spnn.pinv(raw_y, latents=latents)
        x01_bgr = x_det * img_std + img_mean
        x01_rgb = x01_bgr[:, [2, 1, 0]]
        return x01_rgb * 2.0 - 1.0

    return detector, A, Ap


def report(name, y_in, y_out):
    diff = (y_out - y_in).abs()
    print(f"  {name:40s} | "
          f"max_err={diff.max().item():.3e} | "
          f"mean_err={diff.mean().item():.3e} | "
          f"||y_in||={y_in.norm().item():.2f} | "
          f"||y_out||={y_out.norm().item():.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--detector_ckpt", required=True)
    p.add_argument("--voc_data_dir", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector, A, Ap = build(args, device)
    dataset = VOCValForDDNM(args.voc_data_dir, image_size=256)

    # Find first image with a GT box
    img_idx = None
    for i in range(len(dataset)):
        gts = dataset.get_gt_in_image_coords(dataset[i][1])
        if gts:
            img_idx = i
            break
    assert img_idx is not None
    x01, img_id = dataset[img_idx]
    gts = dataset.get_gt_in_image_coords(img_id)
    cls_name, box = gts[0]
    print(f"Using image idx={img_idx}, GT class={cls_name}, box={box}\n")

    x = (x01.unsqueeze(0).to(device) * 2.0 - 1.0)
    with torch.no_grad():
        y = A(x)

    print("=== A(Ap(y)) vs y for various y inputs ===")

    # 1. Real natural y
    with torch.no_grad():
        x_recon = Ap(y)
        y_re = A(x_recon)
    report("(1) y = A(x_orig)  [natural]", y, y_re)

    # Build BB mask in fmap coords
    fmap, stride = 64, 4
    bx1, by1, bx2, by2 = box
    px1 = max(0, int(bx1 // stride)); px2 = min(fmap, int(np.ceil(bx2 / stride)))
    py1 = max(0, int(by1 // stride)); py2 = min(fmap, int(np.ceil(by2 / stride)))
    M = torch.zeros(1, 1, fmap, fmap, device=device, dtype=y.dtype)
    M[..., py1:py2, px1:px2] = 1.0

    # 2. Masked y
    masked_y = M * y
    with torch.no_grad():
        x_recon = Ap(masked_y)
        y_re = A(x_recon)
    report("(2) y = M*y_orig  [zero outside BB]", masked_y, y_re)

    # 3. Random y in natural range (per analysis: hmap mean -6, regs 0.5, wh 18)
    nc = 20
    rand_y = torch.empty_like(y)
    rand_y[:, :nc] = -6 + torch.randn(1, nc, fmap, fmap, device=device) * 1.0
    rand_y[:, nc:nc+2] = 0.5 + torch.randn(1, 2, fmap, fmap, device=device) * 0.07
    rand_y[:, nc+2:nc+4] = 18 + torch.randn(1, 2, fmap, fmap, device=device) * 11
    with torch.no_grad():
        x_recon = Ap(rand_y)
        y_re = A(x_recon)
    report("(3) y = random natural-stats", rand_y, y_re)

    # 4. Masked diff (our BP residual)
    masked_diff = M * (y - 0)  # y_target - y_cur with y_cur=0 just for test
    with torch.no_grad():
        x_recon = Ap(masked_diff)
        y_re = A(x_recon)
    report("(4) y = M*(y_orig - 0)  [diff]", masked_diff, y_re)

    # Per-channel max error breakdown for (2)
    print("\n=== Per-channel A(Ap(M*y)) - M*y max err ===")
    masked_y = M * y
    with torch.no_grad():
        y_re = A(Ap(masked_y))
    diff = (y_re - masked_y).abs()
    print(f"  hmap (ch 0..{nc-1}): max={diff[:, :nc].max().item():.3e}")
    print(f"  regs (ch {nc}..{nc+1}): max={diff[:, nc:nc+2].max().item():.3e}")
    print(f"  wh   (ch {nc+2}..{nc+3}): max={diff[:, nc+2:nc+4].max().item():.3e}")


if __name__ == "__main__":
    main()
