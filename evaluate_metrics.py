"""
Evaluate LPIPS, ArcFace identity similarity, FID, and KID between original and diffusion-generated images.
Downloads images from Google Drive, then computes metrics over all image pairs.
"""

import argparse
import glob
import os
import sys

import shutil

# import cv2
# import gdown
# import lpips
import numpy as np
import torch
from cleanfid import fid
# from insightface.app import FaceAnalysis
from PIL import Image
# from torchvision import transforms


def download_from_gdrive(folder_url: str, output_dir: str):
    """Download a Google Drive folder using gdown."""
    if os.path.exists(output_dir) and os.listdir(output_dir):
        print(f"Data already exists at {output_dir}, skipping download.")
        return
    print(f"Downloading from Google Drive to {output_dir}...")
    gdown.download_folder(folder_url, output=output_dir, quiet=False)
    print("Download complete.")


def find_image_pairs(data_dir: str):
    """Find (original, x0_t) image pairs across all img_X folders."""
    pairs = []
    for img_idx in range(100):
        folder_name = f"img_{img_idx}"
        # Try multiple possible directory structures
        candidates = [
            os.path.join(data_dir, folder_name, "media", "images", folder_name, "Images"),
            os.path.join(data_dir, folder_name, "media", "images", folder_name, "images"),
            os.path.join(data_dir, folder_name),
        ]
        # Also check for an extra subfolder from zip extraction
        for sub in os.listdir(data_dir) if os.path.isdir(data_dir) else []:
            sub_path = os.path.join(data_dir, sub)
            if os.path.isdir(sub_path) and sub != folder_name:
                candidates.append(os.path.join(sub_path, folder_name, "media", "images", folder_name, "Images"))
                candidates.append(os.path.join(sub_path, folder_name, "media", "images", folder_name, "images"))

        images_dir = None
        for c in candidates:
            if os.path.isdir(c):
                images_dir = c
                break

        if images_dir is None:
            print(f"Warning: folder not found for img_{img_idx}, skipping.")
            continue

        # Find original image (starts with "original")
        originals = glob.glob(os.path.join(images_dir, "original*.png"))

        # Find x0_t_final image
        x0t_candidates = glob.glob(os.path.join(images_dir, "x0_t_final*.png"))

        if not originals:
            print(f"Warning: no original image found in {images_dir}")
            continue
        if not x0t_candidates:
            print(f"Warning: no x0_t_final image found in {images_dir}")
            continue

        pairs.append((originals[0], x0t_candidates[0], img_idx))

    print(f"Found {len(pairs)} image pairs.")
    return pairs


def compute_lpips(pairs, device):
    """Compute LPIPS perceptual distance for all pairs."""
    loss_fn = lpips.LPIPS(net="alex").to(device)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # map to [-1, 1]
    ])

    scores = []
    for orig_path, x0t_path, idx in pairs:
        img_orig = transform(Image.open(orig_path).convert("RGB")).unsqueeze(0).to(device)
        img_x0t = transform(Image.open(x0t_path).convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad():
            score = loss_fn(img_orig, img_x0t).item()
        scores.append(score)
        print(f"  img_{idx}: LPIPS = {score:.4f}")

    return scores


def compute_arcface_similarity(pairs, device):
    """Compute ArcFace cosine similarity for all pairs.

    Uses face detection + landmark alignment when possible (proper pipeline).
    Falls back to direct center-crop + resize for images where detection fails.
    """
    import onnxruntime as ort
    from skimage import transform as skimage_transform

    # Standard ArcFace alignment template (5 landmarks for 112x112)
    ARCFACE_DST = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ], dtype=np.float32)

    def align_face(img, landmarks):
        """Align face using 5 landmarks via similarity transform to 112x112."""
        tform = skimage_transform.SimilarityTransform()
        tform.estimate(landmarks, ARCFACE_DST)
        aligned = cv2.warpAffine(img, tform.params[0:2, :], (112, 112), borderValue=0.0)
        return aligned

    # Initialize face detector + landmark finder
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0 if device.type == "cuda" else -1, det_size=(640, 640), det_thresh=0.2)

    # Load recognition model directly (for fallback)
    model_dir = os.path.expanduser("~/.insightface/models/buffalo_l")
    model_path = os.path.join(model_dir, "w600k_r50.onnx")
    sess = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    def get_embedding_aligned(img_bgr):
        """Try detection+alignment first, fall back to center resize."""
        # Upscale small images for better detection
        h, w = img_bgr.shape[:2]
        scale = max(1, 512 / min(h, w))
        if scale > 1:
            img_up = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        else:
            img_up = img_bgr

        faces = app.get(img_up)
        if faces:
            # Use detected face with highest score
            face = max(faces, key=lambda f: f.det_score)
            # Get landmarks and scale back
            lmk = face.kps / scale
            aligned = align_face(img_bgr, lmk)
            method = "aligned"
        else:
            # Fallback: direct resize (assuming image is already a face crop)
            aligned = cv2.resize(img_bgr, (112, 112))
            method = "fallback"

        blob = (aligned.astype(np.float32) - 127.5) / 127.5
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
        emb = sess.run(None, {input_name: blob})[0][0]
        return emb, method

    scores = []
    failed = []
    aligned_count = 0
    fallback_count = 0

    for orig_path, x0t_path, idx in pairs:
        try:
            emb_orig, method_orig = get_embedding_aligned(cv2.imread(orig_path))
            emb_x0t, method_x0t = get_embedding_aligned(cv2.imread(x0t_path))

            cos_sim = np.dot(emb_orig, emb_x0t) / (
                np.linalg.norm(emb_orig) * np.linalg.norm(emb_x0t)
            )
            scores.append(cos_sim)

            methods = f"({method_orig}/{method_x0t})"
            if method_orig == "aligned" and method_x0t == "aligned":
                aligned_count += 1
            else:
                fallback_count += 1
            print(f"  img_{idx}: ArcFace cosine similarity = {cos_sim:.4f} {methods}")
        except Exception as e:
            print(f"  img_{idx}: Failed - {e}")
            failed.append(idx)

    print(f"\n  Alignment stats: {aligned_count} fully aligned, {fallback_count} used fallback")
    if failed:
        print(f"  Failed for {len(failed)} images: {failed}")

    return scores, failed


def prepare_fid_folders(pairs, output_base):
    """Copy originals and x0_t images into separate flat folders for FID/KID computation."""
    orig_dir = os.path.join(output_base, "originals")
    x0t_dir = os.path.join(output_base, "x0t")
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(x0t_dir, exist_ok=True)

    for orig_path, x0t_path, idx in pairs:
        shutil.copy2(orig_path, os.path.join(orig_dir, f"img_{idx}.png"))
        shutil.copy2(x0t_path, os.path.join(x0t_dir, f"img_{idx}.png"))

    return orig_dir, x0t_dir


def compute_fid_kid(pairs, device):
    """Compute FID and KID between original and x0_t image sets."""
    tmp_dir = os.path.join(".", "fid_tmp")
    orig_dir, x0t_dir = prepare_fid_folders(pairs, tmp_dir)

    # Verification: print sample file paths and check images are valid
    print("\n  === Verification ===")
    print(f"  Originals folder: {orig_dir} ({len(os.listdir(orig_dir))} files)")
    print(f"  x0_t folder:      {x0t_dir} ({len(os.listdir(x0t_dir))} files)")

    # Verify all originals are unique (not duplicated files)
    import hashlib
    orig_hashes = {}
    x0t_hashes = {}
    for orig_path, x0t_path, idx in pairs:
        with open(orig_path, "rb") as f:
            h = hashlib.md5(f.read()).hexdigest()
            orig_hashes[idx] = h
        with open(x0t_path, "rb") as f:
            h = hashlib.md5(f.read()).hexdigest()
            x0t_hashes[idx] = h

    unique_originals = len(set(orig_hashes.values()))
    unique_x0t = len(set(x0t_hashes.values()))
    print(f"  Unique original images: {unique_originals}/{len(pairs)}")
    print(f"  Unique x0_t images:     {unique_x0t}/{len(pairs)}")

    if unique_originals < len(pairs):
        # Find duplicates
        from collections import Counter
        dup_hashes = [h for h, cnt in Counter(orig_hashes.values()).items() if cnt > 1]
        for dh in dup_hashes:
            dup_ids = [idx for idx, h in orig_hashes.items() if h == dh]
            print(f"  WARNING: Duplicate original images at img indices: {dup_ids}")

    if unique_x0t < len(pairs):
        from collections import Counter
        dup_hashes = [h for h, cnt in Counter(x0t_hashes.values()).items() if cnt > 1]
        for dh in dup_hashes:
            dup_ids = [idx for idx, h in x0t_hashes.items() if h == dh]
            print(f"  WARNING: Duplicate x0_t images at img indices: {dup_ids}")

    # Check no cross-contamination (original accidentally in x0t or vice versa)
    cross = set(orig_hashes.values()) & set(x0t_hashes.values())
    if cross:
        print(f"  WARNING: {len(cross)} images appear in BOTH original and x0_t sets!")
    else:
        print(f"  No cross-contamination between sets.")

    # Exclude duplicate x0_t images
    if unique_x0t < len(pairs):
        from collections import Counter
        hash_counts = Counter(x0t_hashes.values())
        dup_hash_set = {h for h, cnt in hash_counts.items() if cnt > 1}
        exclude_ids = {idx for idx, h in x0t_hashes.items() if h in dup_hash_set}
        print(f"  Excluding {len(exclude_ids)} images with duplicate x0_t: {sorted(exclude_ids)}")
        pairs = [(o, x, i) for o, x, i in pairs if i not in exclude_ids]
        print(f"  Remaining pairs: {len(pairs)}")

        # Rebuild folders
        shutil.rmtree(tmp_dir, ignore_errors=True)
        orig_dir, x0t_dir = prepare_fid_folders(pairs, tmp_dir)
        print(f"  Rebuilt folders: {len(os.listdir(orig_dir))} originals, {len(os.listdir(x0t_dir))} x0_t")

    print("\n  Computing FID...")
    fid_score = fid.compute_fid(orig_dir, x0t_dir, device=device, num_workers=4)

    print("  Computing KID...")
    kid_score = fid.compute_kid(orig_dir, x0t_dir, device=device, num_workers=4)

    # Cleanup temp folders
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return fid_score, kid_score, len(pairs)


def main():
    parser = argparse.ArgumentParser(description="Evaluate LPIPS and ArcFace metrics")
    parser.add_argument(
        "--gdrive_url",
        type=str,
        default=None,
        help="Google Drive folder URL (optional, skip if data already downloaded)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Local directory containing the image data (or where to download it)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Step 1: Download data (only if URL provided)
    if args.gdrive_url:
        download_from_gdrive(args.gdrive_url, args.data_dir)

    # Step 2: Find image pairs
    pairs = find_image_pairs(args.data_dir)
    if not pairs:
        print("No image pairs found. Check folder structure.")
        sys.exit(1)

    # # Step 3: Compute LPIPS
    # print("\n=== LPIPS (lower = more similar) ===")
    # lpips_scores = compute_lpips(pairs, device)

    # # Step 4: Compute ArcFace similarity
    # print("\n=== ArcFace Identity Similarity (higher = more similar) ===")
    # arcface_scores, arcface_failed = compute_arcface_similarity(pairs, device)

    # Step 5: Compute FID and KID
    print("\n=== FID / KID (lower = more similar distributions) ===")
    fid_score, kid_score, num_evaluated = compute_fid_kid(pairs, device)

    # Step 6: Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total image pairs found: {len(pairs)}")
    print(f"Total image pairs evaluated: {num_evaluated}")
    # print(f"\nLPIPS (perceptual distance):")
    # print(f"  Mean:   {np.mean(lpips_scores):.4f}")
    # print(f"  Std:    {np.std(lpips_scores):.4f}")
    # print(f"  Min:    {np.min(lpips_scores):.4f}")
    # print(f"  Max:    {np.max(lpips_scores):.4f}")
    # print(f"\nArcFace Identity Similarity (cosine):")
    # print(f"  Evaluated: {len(arcface_scores)}/{len(pairs)} (face detection failed on {len(arcface_failed)})")
    # if arcface_scores:
    #     print(f"  Mean:   {np.mean(arcface_scores):.4f}")
    #     print(f"  Std:    {np.std(arcface_scores):.4f}")
    #     print(f"  Min:    {np.min(arcface_scores):.4f}")
    #     print(f"  Max:    {np.max(arcface_scores):.4f}")
    print(f"\nFID (Frechet Inception Distance):")
    print(f"  Score:  {fid_score:.4f}")
    print(f"\nKID (Kernel Inception Distance):")
    print(f"  Score:  {kid_score:.4f}")


if __name__ == "__main__":
    main()
