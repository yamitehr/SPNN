"""Standalone smoke-test for the SPNN-CenterNet preprocessing chain used by
the DDNM detection branch.

Run BEFORE submitting any DDNM job. It loads the trained detector, takes a
single VOC test image, runs the detector with two channel orderings (BGR
vs RGB), and prints which one produces sensible peaks. The "winner" should
match what's hard-coded in `_build_detection_A_Ap` (currently: BGR).

Usage:
    cd /shared/cycle1_iit_shocher_prj/SPNN
    source .venv-cluster/bin/activate
    python DDNM/verify_detector_preprocessing.py \
        --detector_ckpt /shared/cycle1_iit_shocher_prj/SPNN/centernet_ref/ckpt/spnn_centernet_deephead_NOdistill_hidden128_GN_UnetLike/checkpoint.t7 \
        --voc_data_dir /shared/cycle1_iit_shocher_prj/SPNN/centernet_ref/data \
        --img_idx 0 \
        --detector_deep_det_head --detector_deep_head_hidden 128

Pass any extra detector-arch flags exactly as you would to DDNM/main.py.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CR = os.path.join(_ROOT, "centernet_ref")
for p in (_ROOT, _CR):
    if p not in sys.path:
        sys.path.insert(0, p)

from nets.spnn_centernet import get_spnn_centernet  # noqa: E402

VOC_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


def load_one_voc_image(voc_data_dir, idx, image_size=256):
    """Returns image as a single torch tensor [1, 3, image_size, image_size]
    in BGR uint8 (the raw CenterNet-training format), already resized."""
    import json
    annot = json.load(open(os.path.join(voc_data_dir, "voc", "annotations",
                                         "pascal_test2007.json")))
    info = annot["images"][idx]
    img_path = os.path.join(voc_data_dir, "voc", "images", info["file_name"])
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    s = min(h, w)
    img = img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return img, info["file_name"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector_ckpt", required=True)
    ap.add_argument("--voc_data_dir", default="DDNM/data")
    ap.add_argument("--img_idx", type=int, default=0)
    ap.add_argument("--detector_num_classes", type=int, default=20)
    ap.add_argument("--detector_deep_det_head", action="store_true")
    ap.add_argument("--detector_deep_head_hidden", type=int, default=128)
    ap.add_argument("--detector_two_block_head", action="store_true")
    ap.add_argument("--detector_head_mode", default="affine")
    ap.add_argument("--detector_head_mix_type", default="householder")
    ap.add_argument("--detector_head_mix_reflections", type=int, default=0)
    ap.add_argument("--detector_hmap_init_scale", type=float, default=0.01)
    ap.add_argument("--detector_hmap_init_bias", type=float, default=-2.19)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

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
    ).to(device)

    raw = torch.load(args.detector_ckpt, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        state = raw["state_dict"]
    elif isinstance(raw, dict) and "model" in raw:
        state = raw["model"]
    else:
        state = raw
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    miss, unex = detector.load_state_dict(state, strict=False)
    print(f"loaded ckpt | missing={len(miss)} unexpected={len(unex)}")
    if miss:
        print(f"  first missing: {miss[:6]}")
    if unex:
        print(f"  first unexpected: {unex[:6]}")
    detector.eval()

    img_bgr_uint8, fname = load_one_voc_image(args.voc_data_dir, args.img_idx)
    print(f"\nImage: {fname}  shape={img_bgr_uint8.shape}\n")

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def preprocess(arr_uint8):
        arr = arr_uint8.astype(np.float32) / 255.0
        arr = (arr - mean) / std
        arr = arr.transpose(2, 0, 1)[None]  # NCHW
        return torch.from_numpy(arr.copy()).to(device)

    # Variant A: BGR (matches pascal.py training preprocessing exactly)
    x_bgr = preprocess(img_bgr_uint8)
    # Variant B: RGB (cv2 → cvtColor → same normalization)
    img_rgb_uint8 = cv2.cvtColor(img_bgr_uint8, cv2.COLOR_BGR2RGB)
    x_rgb = preprocess(img_rgb_uint8)

    with torch.no_grad():
        out_bgr = detector(x_bgr)
        out_rgb = detector(x_rgb)
        hmap_bgr, regs_bgr, wh_bgr = out_bgr[0]
        hmap_rgb, regs_rgb, wh_rgb = out_rgb[0]
        prob_bgr = hmap_bgr.sigmoid()[0].cpu().numpy()  # [20, 64, 64]
        prob_rgb = hmap_rgb.sigmoid()[0].cpu().numpy()

    def _print_top_per_class(prob, label):
        max_per_cls = prob.reshape(20, -1).max(axis=1)
        order = max_per_cls.argsort()[::-1][:5]
        print(f"  [{label}] top-5 classes by max sigmoid:")
        for c in order:
            n_above_05 = int((prob[c] > 0.5).sum())
            n_above_03 = int((prob[c] > 0.3).sum())
            print(f"    {VOC_NAMES[c]:14s}: max={max_per_cls[c]:.3f} "
                  f"#>0.5={n_above_05:3d} #>0.3={n_above_03:3d}")
        print(f"  [{label}] global max sigmoid: {prob.max():.3f}, "
              f"#cells>0.5: {(prob > 0.5).sum()}, "
              f"#cells>0.3: {(prob > 0.3).sum()}")

    print("=" * 60)
    print("Variant A: BGR input (cv2.imread directly, no swap) — matches pascal.py")
    print("=" * 60)
    _print_top_per_class(prob_bgr, "BGR")

    print()
    print("=" * 60)
    print("Variant B: RGB input (cv2 + cvtColor BGR→RGB)")
    print("=" * 60)
    _print_top_per_class(prob_rgb, "RGB")

    print()
    print("=" * 60)
    print("INTERPRETATION:")
    print("  Pick whichever variant produces higher max sigmoid AND")
    print("  more cells above 0.3/0.5. That's the channel order the")
    print("  detector was trained on. The DDNM code uses BGR by default")
    print("  (matches pascal.py); if RGB wins here, you need to flip the")
    print("  swap in `diffusion_to_detector` / `detector_to_diffusion`.")
    print("=" * 60)


if __name__ == "__main__":
    main()
