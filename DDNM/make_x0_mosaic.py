"""Build a single-image mosaic showing x0_t (top row) and x0_t_hat (bottom row)
at sampled BP steps. Quick at-a-glance summary of the evolution."""
import argparse
import os
import re

from PIL import Image, ImageDraw, ImageFont


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--debug_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_cols", type=int, default=8)
    args = p.parse_args()

    files = os.listdir(args.debug_dir)
    pattern = re.compile(r"step(\d+)_(x0_t|x0_t_hat)\.png")
    steps = {}
    for f in files:
        m = pattern.match(f)
        if not m: continue
        s, kind = int(m.group(1)), m.group(2)
        steps.setdefault(s, {})[kind] = os.path.join(args.debug_dir, f)
    sorted_steps = sorted(s for s, d in steps.items()
                          if "x0_t" in d and "x0_t_hat" in d)
    if not sorted_steps:
        raise SystemExit("no steps found")

    n_cols = min(args.n_cols, len(sorted_steps))
    idxs = [round(i * (len(sorted_steps) - 1) / (n_cols - 1))
            for i in range(n_cols)] if n_cols > 1 else [0]
    picked = [sorted_steps[i] for i in idxs]

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except Exception:
        font = ImageFont.load_default()

    # Probe a frame for dims
    sample = Image.open(steps[picked[0]]["x0_t"]).convert("RGB")
    cell_w, cell_h = sample.size
    label_h = 18
    row_label = 80
    W = row_label + n_cols * cell_w
    H = label_h * 2 + cell_h * 2
    canvas = Image.new("RGB", (W, H), (20, 20, 20))
    d = ImageDraw.Draw(canvas)

    d.text((4, label_h + cell_h // 2 - 6), "pre-BP\n(x0_t)",
           fill=(230, 230, 230), font=font)
    d.text((4, label_h * 2 + cell_h + cell_h // 2 - 6),
           "post-BP\n(x0_t_hat)", fill=(230, 230, 230), font=font)

    for col, s in enumerate(picked):
        x_off = row_label + col * cell_w
        d.text((x_off + 4, 2), f"step {s:03d}",
               fill=(230, 230, 230), font=font)
        a = Image.open(steps[s]["x0_t"]).convert("RGB")
        b = Image.open(steps[s]["x0_t_hat"]).convert("RGB")
        canvas.paste(a, (x_off, label_h))
        canvas.paste(b, (x_off, label_h * 2 + cell_h))
    canvas.save(args.out)
    print(f"wrote {args.out}  ({canvas.size})")


if __name__ == "__main__":
    main()
