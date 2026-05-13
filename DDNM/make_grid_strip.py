"""Tile every grid_<idx>.png from a run dir into one big image. Each grid
is already orig+result side-by-side; this just stacks them in rows."""
import argparse
import glob
import os
import re

from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_cols", type=int, default=2,
                   help="How many grid_*.png panels per row.")
    args = p.parse_args()

    files = sorted(
        [f for f in glob.glob(os.path.join(args.run_dir, "grid_*.png"))
         if re.search(r"grid_(\d+)\.png$", f)],
        key=lambda f: int(re.search(r"grid_(\d+)\.png$", f).group(1)))
    if not files:
        raise SystemExit(f"no grid_*.png in {args.run_dir}")

    sample = Image.open(files[0]).convert("RGB")
    cw, ch = sample.size
    n = len(files)
    n_rows = (n + args.n_cols - 1) // args.n_cols
    canvas = Image.new("RGB", (args.n_cols * cw, n_rows * ch), (20, 20, 20))
    for k, f in enumerate(files):
        r, c = k // args.n_cols, k % args.n_cols
        canvas.paste(Image.open(f).convert("RGB"), (c * cw, r * ch))
    canvas.save(args.out)
    print(f"wrote {args.out}  ({canvas.size}, {n} panels)")


if __name__ == "__main__":
    main()
