"""
Convert ImageNet-1K from HuggingFace format to ImageFolder format.

Prerequisites:
  Download raw files first (fast, resumable):
    huggingface-cli download --repo-type dataset ILSVRC/imagenet-1k \
        --local-dir datasets/imagenet-hf

Then run this script:
    python download_imagenet.py --output-dir datasets/imagenet

This creates:
    datasets/imagenet/train/<class_id>/img_XXXXXX.JPEG  (1,281,167 images)
    datasets/imagenet/val/<class_id>/img_XXXXXX.JPEG    (50,000 images)
"""

import os
import argparse
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Convert ImageNet-1K to ImageFolder format")
    parser.add_argument("--output-dir", type=str, default="datasets/imagenet",
                        help="Output directory for ImageFolder format")
    parser.add_argument("--hf-dir", type=str, default="datasets/imagenet-hf",
                        help="Path to HuggingFace downloaded data (from huggingface-cli)")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of workers for dataset loading")
    args = parser.parse_args()

    from datasets import load_dataset

    output_dir = args.output_dir
    hf_dir = args.hf_dir

    # Load from local HuggingFace download if available
    if os.path.exists(hf_dir):
        print(f"Loading ImageNet-1K from local HuggingFace cache: {hf_dir}")
        ds = load_dataset(hf_dir, num_proc=args.num_workers)
    else:
        print(f"No local cache found at {hf_dir}, downloading from HuggingFace...")
        print("This will take a while (~150GB)")
        ds = load_dataset("ILSVRC/imagenet-1k", num_proc=args.num_workers)

    # Pre-create all 1000 class directories
    for split_name in ["train", "val"]:
        split_dir = os.path.join(output_dir, split_name)
        for label in range(1000):
            os.makedirs(os.path.join(split_dir, f"{label:04d}"), exist_ok=True)

    for split_name, hf_split in [("train", "train"), ("val", "validation")]:
        split_dir = os.path.join(output_dir, split_name)
        split_data = ds[hf_split]
        print(f"\nConverting {split_name} split: {len(split_data)} images")

        for idx in tqdm(range(len(split_data)), desc=split_name):
            example = split_data[idx]
            image = example["image"]
            label = example["label"]

            img_path = os.path.join(split_dir, f"{label:04d}", f"img_{idx:07d}.JPEG")
            if not os.path.exists(img_path):
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image.save(img_path)

    print(f"\nDone! ImageNet saved to {output_dir}")
    print(f"  train/ : {len(os.listdir(os.path.join(output_dir, 'train')))} classes")
    print(f"  val/   : {len(os.listdir(os.path.join(output_dir, 'val')))} classes")


if __name__ == "__main__":
    main()
