"""Extract all 100 original images into a single flat folder."""

import argparse
import glob
import os
import shutil


def find_originals(data_dir):
    """Find all original images across the folder structure."""
    images = []
    for img_idx in range(100):
        folder_name = f"img_{img_idx}"
        candidates = [
            os.path.join(data_dir, folder_name, "media", "images", folder_name, "Images"),
            os.path.join(data_dir, folder_name, "media", "images", folder_name, "images"),
            os.path.join(data_dir, folder_name),
        ]
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

        originals = glob.glob(os.path.join(images_dir, "original*.png"))
        if not originals:
            print(f"Warning: no original image found in {images_dir}")
            continue

        images.append((originals[0], img_idx))

    return images


def main():
    parser = argparse.ArgumentParser(description="Extract original images into a flat folder")
    parser.add_argument("--data_dir", type=str, required=True, help="Root data directory")
    parser.add_argument("--output_dir", type=str, default="./originals_only", help="Output folder")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    images = find_originals(args.data_dir)
    print(f"Found {len(images)} original images.")

    for orig_path, idx in images:
        dst = os.path.join(args.output_dir, f"img_{idx}.png")
        shutil.copy2(orig_path, dst)

    print(f"Copied {len(images)} images to {args.output_dir}")


if __name__ == "__main__":
    main()
