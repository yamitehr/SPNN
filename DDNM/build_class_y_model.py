"""Build a per-class y-window statistics model.

For each instance of the target class in VOC, extract a fixed-size window
of the detector's y output around the peak cell. Compute per-cell,
per-channel mean and std. Save to disk + visualize.

Usage:
  python build_class_y_model.py --class horse --window_radius 10 \\
      --detector_ckpt ... --voc_data_dir ... --out_dir exp/horse_model
"""
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def build_A(args, device):
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

    return A


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--detector_ckpt", required=True)
    p.add_argument("--voc_data_dir", required=True)
    p.add_argument("--cls", default="horse",
                   help="VOC class to model")
    p.add_argument("--window_radius", type=int, default=10,
                   help="Half-size of the y-window in fmap cells (window "
                        "is (2r+1) x (2r+1)). Default 10 -> 21x21 cells.")
    p.add_argument("--out_dir", default="exp/class_y_models")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = build_A(args, device)
    dataset = VOCValForDDNM(args.voc_data_dir, image_size=256)
    label = VOC_LABEL[args.cls]
    fmap = 64
    stride = 4
    r = args.window_radius
    win_sz = 2 * r + 1

    windows, box_dims, subpx, peak_xy = [], [], [], []
    n_imgs_with_class = 0
    for idx in range(len(dataset)):
        x01, img_id = dataset[idx]
        gts = dataset.get_gt_in_image_coords(img_id)
        if not gts:
            continue
        gts_cls = [(c, b) for (c, b) in gts if c == args.cls]
        if not gts_cls:
            continue
        x = (x01.unsqueeze(0).to(device) * 2.0 - 1.0)
        with torch.no_grad():
            y = A(x)[0].cpu()  # [24, 64, 64]
        n_imgs_with_class += 1
        for _cls, box in gts_cls:
            bx1, by1, bx2, by2 = box
            cx = (bx1 + bx2) / 2.0; cy = (by1 + by2) / 2.0
            w_f = (bx2 - bx1) / stride; h_f = (by2 - by1) / stride
            px = int(cx / stride); py = int(cy / stride)
            if not (0 <= px < fmap and 0 <= py < fmap):
                continue
            window = torch.zeros(y.shape[0], win_sz, win_sz, dtype=y.dtype)
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    yy, xx = py + dy, px + dx
                    if 0 <= yy < fmap and 0 <= xx < fmap:
                        window[:, dy + r, dx + r] = y[:, yy, xx]
                    # else: leave at 0 (zero-pad outside image)
            windows.append(window)
            box_dims.append((h_f, w_f))
            subpx.append((cx / stride - px, cy / stride - py))
            peak_xy.append((px, py))

    if not windows:
        raise SystemExit(f"No '{args.cls}' instances found in dataset.")
    W = torch.stack(windows)  # [N, 24, win_sz, win_sz]
    print(f"Found {len(W)} {args.cls} instances across {n_imgs_with_class} images.")
    mean = W.mean(0)  # [24, win_sz, win_sz]
    std = W.std(0)
    box_dims = torch.tensor(box_dims)
    subpx = torch.tensor(subpx)
    print(f"box_dims (h_f, w_f) mean: {box_dims.mean(0).tolist()}")
    print(f"box_dims (h_f, w_f) std:  {box_dims.std(0).tolist()}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_pt = os.path.join(args.out_dir, f"{args.cls}_y_model.pt")
    torch.save({
        "cls": args.cls, "label": label,
        "window_radius": r,
        "mean": mean, "std": std,
        "box_dims_mean": box_dims.mean(0),
        "box_dims_std": box_dims.std(0),
        "subpx_mean": subpx.mean(0),
        "subpx_std": subpx.std(0),
        "n_instances": len(W),
    }, out_pt)
    print(f"Saved {out_pt}")

    # Visualize: target-class hmap mean, std + a few neighbors
    nc = 20
    fig, axs = plt.subplots(3, 4, figsize=(13, 9))
    # Row 1: target class hmap + wh + regs
    im = axs[0, 0].imshow(mean[label], cmap="viridis")
    axs[0, 0].set_title(f"hmap[{args.cls}]  mean")
    plt.colorbar(im, ax=axs[0, 0], fraction=0.046)
    im = axs[0, 1].imshow(std[label], cmap="magma")
    axs[0, 1].set_title(f"hmap[{args.cls}]  std")
    plt.colorbar(im, ax=axs[0, 1], fraction=0.046)
    # Row 1 cont: regs
    im = axs[0, 2].imshow(mean[nc], cmap="viridis")
    axs[0, 2].set_title("regs[sub_x] mean"); plt.colorbar(im, ax=axs[0, 2], fraction=0.046)
    im = axs[0, 3].imshow(mean[nc + 1], cmap="viridis")
    axs[0, 3].set_title("regs[sub_y] mean"); plt.colorbar(im, ax=axs[0, 3], fraction=0.046)
    # Row 2: wh + a couple of confused classes
    im = axs[1, 0].imshow(mean[nc + 2], cmap="viridis")
    axs[1, 0].set_title("wh[w] mean"); plt.colorbar(im, ax=axs[1, 0], fraction=0.046)
    im = axs[1, 1].imshow(mean[nc + 3], cmap="viridis")
    axs[1, 1].set_title("wh[h] mean"); plt.colorbar(im, ax=axs[1, 1], fraction=0.046)
    # Row 2 cont: top confused classes (highest mean at center)
    center = win_sz // 2
    other_means = [(c, mean[c, center, center].item())
                   for c in range(nc) if c != label]
    other_means.sort(key=lambda v: v[1], reverse=True)
    for k, ax in enumerate(axs[1, 2:].tolist() + axs[2, :].tolist()):
        if k >= len(other_means): break
        c, m = other_means[k]
        im = ax.imshow(mean[c], cmap="viridis")
        ax.set_title(f"hmap[{VOC_NAMES[c]}] mean "
                     f"(center={mean[c, center, center]:.2f})")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(f"{args.cls} y-window model (n={len(W)})", fontsize=14)
    plt.tight_layout()
    out_png = os.path.join(args.out_dir, f"{args.cls}_y_model.png")
    plt.savefig(out_png, dpi=110)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
