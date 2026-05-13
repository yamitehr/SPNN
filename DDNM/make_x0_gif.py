"""Compose a GIF of (x0_t before BP | x0_t_hat after BP) per BP step.

Reads debug_x0/img_<idx>/step{N}_x0_t.png and step{N}_x0_t_hat.png from an
image_folder, side-by-sides them with a step label, and writes a GIF.
"""
import argparse
import os
import re

from PIL import Image, ImageDraw, ImageFont


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--debug_dir", required=True,
                   help="Path to debug_x0/img_<idx>/")
    p.add_argument("--out", required=True, help="Output GIF path")
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--scale", type=int, default=2,
                   help="Upsample factor for visibility")
    args = p.parse_args()

    files = os.listdir(args.debug_dir)
    pattern = re.compile(r"step(\d+)_(x0_t|x0_t_hat)\.png")
    steps = {}
    for f in files:
        m = pattern.match(f)
        if not m:
            continue
        s, kind = int(m.group(1)), m.group(2)
        steps.setdefault(s, {})[kind] = os.path.join(args.debug_dir, f)
    sorted_steps = sorted(steps.keys())
    print(f"Found {len(sorted_steps)} steps in {args.debug_dir}")

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    frames = []
    for s in sorted_steps:
        if "x0_t" not in steps[s] or "x0_t_hat" not in steps[s]:
            continue
        a = Image.open(steps[s]["x0_t"]).convert("RGB")
        b = Image.open(steps[s]["x0_t_hat"]).convert("RGB")
        if args.scale != 1:
            a = a.resize((a.width * args.scale, a.height * args.scale),
                         Image.NEAREST)
            b = b.resize((b.width * args.scale, b.height * args.scale),
                         Image.NEAREST)
        gap = 8
        label_h = 24
        W = a.width + gap + b.width
        H = a.height + label_h
        frame = Image.new("RGB", (W, H), (20, 20, 20))
        frame.paste(a, (0, label_h))
        frame.paste(b, (a.width + gap, label_h))
        d = ImageDraw.Draw(frame)
        d.text((4, 2), f"step {s:03d}  |  L: x0_t (pre-BP)   "
                       f"R: x0_t_hat (post-BP)",
               fill=(230, 230, 230), font=font)
        frames.append(frame)

    if not frames:
        raise SystemExit("no frames")
    duration_ms = int(1000.0 / args.fps)
    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0, optimize=False, disposal=2)
    print(f"Wrote {len(frames)} frames at {args.fps} fps -> {args.out} "
          f"({os.path.getsize(args.out) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
