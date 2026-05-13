"""Analyze the structure of y = A(x_orig) on VOC, around GT peak cells.

Run from DDNM/. Uses the SAME detector + RGB→BGR + ImageNet-norm pipeline
as the diffusion code's _build_detection_A_Ap.
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

VOC_NAMES_NO_BG = (
    "aeroplane bicycle bird boat bottle bus car cat chair cow "
    "diningtable dog horse motorbike person pottedplant sheep sofa "
    "train tvmonitor"
).split()
VOC_LABEL = {n: i for i, n in enumerate(VOC_NAMES_NO_BG)}


def build_detector(args, device):
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
    return detector


def make_A(detector, device):
    img_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    img_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def A(x):  # x in [-1, 1] RGB
        x01 = (x + 1.0) / 2.0
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
    p.add_argument("--n_images", type=int, default=100)
    p.add_argument("--out_dir", default="exp/y_analysis")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    detector = build_detector(args, device)
    A = make_A(detector, device)
    dataset = VOCValForDDNM(args.voc_data_dir, image_size=256)

    nc, stride, fmap = 20, 4, 64

    # Per-object collected values
    peak_hmap_gt = []          # value at peak cell, GT class
    peak_hmap_other = []       # value at peak cell, NON-GT classes (avg)
    bg_hmap_all = []           # all hmap values from far-away cells, all classes
    radial_profiles = []       # per-object: array of mean(|h(d)|) over d=0..15
    peak_regs_subx = []
    peak_regs_suby = []
    gt_subx = []
    gt_suby = []
    peak_wh_w = []
    peak_wh_h = []
    gt_wf = []
    gt_hf = []
    nonpeak_regs = []
    nonpeak_wh = []
    n_objs = 0
    n_imgs = 0

    for idx in range(min(args.n_images, len(dataset))):
        x01, img_id = dataset[idx]
        gts = dataset.get_gt_in_image_coords(img_id)
        if not gts:
            continue
        x = (x01.unsqueeze(0).to(device) * 2.0 - 1.0)
        with torch.no_grad():
            y = A(x)  # [1, 24, 64, 64]
        hmap = y[0, :nc]            # [20, 64, 64]
        regs = y[0, nc:nc + 2]      # [2, 64, 64]
        wh = y[0, nc + 2:nc + 4]    # [2, 64, 64]

        peak_mask_all = torch.zeros(fmap, fmap, dtype=torch.bool, device=device)
        for cls_name, box in gts:
            if cls_name not in VOC_LABEL:
                continue
            label = VOC_LABEL[cls_name]
            bx1, by1, bx2, by2 = box
            cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
            px, py = int(cx / stride), int(cy / stride)
            if not (0 <= px < fmap and 0 <= py < fmap):
                continue
            n_objs += 1

            peak_hmap_gt.append(hmap[label, py, px].item())
            other_vals = torch.cat([hmap[:label, py, px], hmap[label + 1:, py, px]])
            peak_hmap_other.append(other_vals.mean().item())

            # Radial profile of GT-class hmap
            yy, xx = torch.meshgrid(
                torch.arange(fmap, device=device, dtype=torch.float32),
                torch.arange(fmap, device=device, dtype=torch.float32),
                indexing="ij",
            )
            d = torch.round(torch.sqrt((yy - py) ** 2 + (xx - px) ** 2)).long()
            prof = []
            for r in range(16):
                mask = (d == r)
                if mask.any():
                    prof.append(hmap[label][mask].mean().item())
                else:
                    prof.append(np.nan)
            radial_profiles.append(prof)

            peak_regs_subx.append(regs[0, py, px].item())
            peak_regs_suby.append(regs[1, py, px].item())
            gt_subx.append(cx / stride - px)
            gt_suby.append(cy / stride - py)
            peak_wh_w.append(wh[0, py, px].item())
            peak_wh_h.append(wh[1, py, px].item())
            gt_wf.append((bx2 - bx1) / stride)
            gt_hf.append((by2 - by1) / stride)

            r = 6
            peak_mask_all[max(0, py - r):min(fmap, py + r + 1),
                          max(0, px - r):min(fmap, px + r + 1)] = True

        # BG = cells far from any peak
        bg_mask = ~peak_mask_all
        bg_hmap_all.extend(hmap[:, bg_mask].cpu().numpy().flatten().tolist())
        nonpeak_regs.extend(regs[:, bg_mask].cpu().numpy().flatten().tolist())
        nonpeak_wh.extend(wh[:, bg_mask].cpu().numpy().flatten().tolist())

        n_imgs += 1

    def stats(name, arr):
        a = np.asarray(arr)
        if a.size == 0:
            print(f"{name}: empty"); return
        print(f"{name}: n={len(a)} mean={a.mean():.3f} std={a.std():.3f} "
              f"min={a.min():.3f} q10={np.quantile(a, .1):.3f} med={np.median(a):.3f} "
              f"q90={np.quantile(a, .9):.3f} max={a.max():.3f}")

    print(f"\nProcessed {n_imgs} images, {n_objs} GT objects.\n")
    print("==== HMAP (pre-sigmoid logits) ====")
    stats("  peak cell, GT class      ", peak_hmap_gt)
    stats("  peak cell, OTHER classes ", peak_hmap_other)
    stats("  BG cells (>6 from peak)  ", bg_hmap_all)
    print("\n==== REGS (sub-pixel offsets, peak cell) ====")
    stats("  peak regs[0] (sub_x)     ", peak_regs_subx)
    stats("  GT     regs[0] (sub_x)   ", gt_subx)
    stats("  peak regs[1] (sub_y)     ", peak_regs_suby)
    stats("  GT     regs[1] (sub_y)   ", gt_suby)
    stats("  non-peak regs (any axis) ", nonpeak_regs)
    print("\n==== WH (peak cell, fmap units) ====")
    stats("  peak wh[0] (w_f)         ", peak_wh_w)
    stats("  GT   w_f                 ", gt_wf)
    stats("  peak wh[1] (h_f)         ", peak_wh_h)
    stats("  GT   h_f                 ", gt_hf)
    stats("  non-peak wh (any axis)   ", nonpeak_wh)

    # Plots
    profs = np.asarray(radial_profiles)
    mean_prof = np.nanmean(profs, axis=0)
    p25 = np.nanpercentile(profs, 25, axis=0)
    p75 = np.nanpercentile(profs, 75, axis=0)

    fig, axs = plt.subplots(2, 2, figsize=(11, 8))

    ax = axs[0, 0]
    rs = np.arange(len(mean_prof))
    ax.plot(rs, mean_prof, label="mean", lw=2)
    ax.fill_between(rs, p25, p75, alpha=0.3, label="IQR")
    ax.axhline(np.mean(bg_hmap_all), color="red", ls="--", label="BG mean (all classes)")
    ax.set(xlabel="distance from peak (cells)",
           ylabel="hmap (pre-sigmoid) value, GT class",
           title=f"Radial profile of GT-class hmap (n={len(profs)})")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axs[0, 1]
    ax.hist(peak_hmap_gt, bins=40, alpha=0.7, label="peak (GT cls)", color="C0")
    ax.hist(peak_hmap_other, bins=40, alpha=0.7, label="peak (other cls)", color="C1")
    ax.hist(np.random.choice(bg_hmap_all, size=min(20000, len(bg_hmap_all))),
            bins=40, alpha=0.5, label="BG cells", color="C3")
    ax.set(xlabel="hmap value (pre-sigmoid)", ylabel="count",
           title="Distributions of hmap values")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axs[1, 0]
    ax.scatter(gt_wf, peak_wh_w, s=8, alpha=0.5, label="w_f")
    ax.scatter(gt_hf, peak_wh_h, s=8, alpha=0.5, label="h_f")
    lim = max(max(gt_wf, default=1), max(gt_hf, default=1),
              max(peak_wh_w, default=1), max(peak_wh_h, default=1))
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, label="y=x")
    ax.set(xlabel="GT wh (fmap units)", ylabel="predicted peak wh",
           title="wh: GT vs detector at peak cell")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axs[1, 1]
    ax.scatter(gt_subx, peak_regs_subx, s=8, alpha=0.5, label="sub_x")
    ax.scatter(gt_suby, peak_regs_suby, s=8, alpha=0.5, label="sub_y")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="y=x")
    ax.set(xlabel="GT subpixel offset", ylabel="predicted peak regs",
           title="regs: GT vs detector at peak cell")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(args.out_dir, "y_analysis.png")
    plt.savefig(out_png, dpi=110)
    print(f"\nSaved figure: {out_png}")

    # Also dump a few representative hmap heatmaps
    fig, axs = plt.subplots(2, 4, figsize=(14, 7))
    shown = 0
    for idx in range(len(dataset)):
        if shown >= 8: break
        x01, img_id = dataset[idx]
        gts = dataset.get_gt_in_image_coords(img_id)
        if not gts: continue
        cls_name, box = gts[0]
        if cls_name not in VOC_LABEL: continue
        label = VOC_LABEL[cls_name]
        x = (x01.unsqueeze(0).to(device) * 2.0 - 1.0)
        with torch.no_grad():
            y = A(x)
        hm = y[0, label].cpu().numpy()
        ax = axs[shown // 4, shown % 4]
        im = ax.imshow(hm, cmap="viridis")
        bx1, by1, bx2, by2 = box
        cx, cy = (bx1 + bx2) / 2 / stride, (by1 + by2) / 2 / stride
        ax.scatter([cx], [cy], color="red", s=40, marker="x")
        ax.set_title(f"{cls_name}  range=[{hm.min():.1f},{hm.max():.1f}]")
        plt.colorbar(im, ax=ax, fraction=0.046)
        shown += 1
    plt.tight_layout()
    out_png2 = os.path.join(args.out_dir, "hmap_examples.png")
    plt.savefig(out_png2, dpi=110)
    print(f"Saved figure: {out_png2}")


if __name__ == "__main__":
    main()
