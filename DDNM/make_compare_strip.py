"""Build an N-row, M-column comparison panel. Row = image. Cols =
orig (with red GT box) + one column per run dir."""
import argparse
import os
import re

from PIL import Image, ImageDraw, ImageFont


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref_dir", required=True,
                   help="Source of orig (uses grid_<idx>.png left half).")
    p.add_argument("--run_dirs", nargs="+", required=True,
                   help="Each dir holds <idx>_0.png outputs.")
    p.add_argument("--col_labels", nargs="+", required=True,
                   help="One label per col after 'orig'. Same len as run_dirs.")
    p.add_argument("--n_imgs", type=int, default=8)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    assert len(args.col_labels) == len(args.run_dirs)
    n_cols = 1 + len(args.run_dirs)

    # Probe size from first orig
    grid0 = Image.open(os.path.join(args.ref_dir, "grid_0.png")).convert("RGB")
    cell_w = grid0.size[0] // 2
    cell_h = grid0.size[1]

    label_h = 22
    W = n_cols * cell_w
    H = label_h + args.n_imgs * cell_h
    canvas = Image.new("RGB", (W, H), (20, 20, 20))
    d = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    # Header row
    headers = ["orig"] + args.col_labels
    for c, h in enumerate(headers):
        d.text((c * cell_w + 6, 4), h, fill=(230, 230, 230), font=font)

    for r in range(args.n_imgs):
        # orig from grid (left half)
        gp = os.path.join(args.ref_dir, f"grid_{r}.png")
        if not os.path.exists(gp):
            continue
        g = Image.open(gp).convert("RGB")
        orig = g.crop((0, 0, cell_w, cell_h))
        canvas.paste(orig, (0, label_h + r * cell_h))
        for c, rd in enumerate(args.run_dirs):
            ip = os.path.join(rd, f"{r}_0.png")
            if not os.path.exists(ip):
                continue
            im = Image.open(ip).convert("RGB").resize((cell_w, cell_h))
            canvas.paste(im, ((1 + c) * cell_w, label_h + r * cell_h))

    canvas.save(args.out)
    print(f"wrote {args.out}  ({canvas.size})")


if __name__ == "__main__":
    main()
