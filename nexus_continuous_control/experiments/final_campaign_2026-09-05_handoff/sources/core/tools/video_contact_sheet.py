"""Turn a rollout .mp4 into a single annotated contact sheet PNG.

Why this exists: the campaign's conclusions are drawn from scalar success metrics, and a scalar
cannot tell you *how* a policy succeeded or failed. `panda flat` scoring 0.000 on a third of its
seeds and `hopper flat` scoring 0.12 are both compatible with several very different behaviours —
the arm flinging the cube, the hopper dragging itself along the floor, the robot standing still
and collecting a survival bonus. Those are distinguishable by eye in two seconds and not at all
from the number.

A contact sheet (one image, N frames evenly spaced across the episode, each stamped with its
frame index and time) is the form that survives being handed to a reviewer: it shows the whole
episode at once, so motion, drift and failure mode are visible in a single look.

ffmpeg is not installed on this box, but `imageio_ffmpeg` ships its own binary and `imageio`
reads mp4 through it — no system dependency, no apt install.

    python tools/video_contact_sheet.py runs/videos/panda_flat_s0.mp4 \
        --frames 12 --cols 4 --out runs/frames/panda_flat_s0.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw


def contact_sheet(video: Path, n_frames: int, cols: int, width: int,
                  drop_last: int = 1) -> Image.Image:
    frames = np.stack(list(iio.imiter(video)))
    total = len(frames)
    if total == 0:
        raise SystemExit(f"no frames decoded from {video}")
    # `render_rollout.py` breaks on `done` and renders the state AFTER env.step, and the env
    # auto-resets — so the final frame of every clip is the post-reset home pose, not policy
    # behaviour. On a contact sheet that is one tile in twelve showing the robot upright and the
    # object back at its start, which reads as "the policy dropped it at the end". A reviewer
    # would draw exactly the wrong conclusion from a tile that contains no policy at all.
    # Dropped here rather than re-rendering 33 clips; the underlying render bug is recorded in
    # docs/ROUGH_TERRAIN_FINDINGS.md.
    if drop_last and total > drop_last + 1:
        frames = frames[:-drop_last]
        total = len(frames)
    # Evenly spaced INCLUDING the last frame: the end of an episode is where the failure mode
    # usually is (fallen, cube dropped, drifted off), so sampling that misses it is worse than
    # useless — it makes a failed rollout look fine.
    idx = np.linspace(0, total - 1, num=min(n_frames, total)).round().astype(int)
    picked = [frames[i] for i in idx]

    h, w = picked[0].shape[:2]
    scale = width / (cols * w)
    tw, th = int(w * scale), int(h * scale)
    rows = (len(picked) + cols - 1) // cols
    band = 16  # caption strip under each tile

    sheet = Image.new("RGB", (cols * tw, rows * (th + band)), (18, 20, 26))
    draw = ImageDraw.Draw(sheet)
    for k, (i, arr) in enumerate(zip(idx, picked)):
        tile = Image.fromarray(arr).resize((tw, th), Image.LANCZOS)
        x, y = (k % cols) * tw, (k // cols) * (th + band)
        sheet.paste(tile, (x, y))
        draw.text((x + 4, y + th + 3), f"frame {int(i)}/{total - 1}", fill=(210, 215, 225))
    return sheet


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("--frames", type=int, default=12)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--width", type=int, default=1400, help="total sheet width in px")
    p.add_argument("--drop-last", type=int, default=1,
                   help="Drop N trailing frames. Default 1: the last rendered frame is the "
                        "post-reset home pose, not policy behaviour (see contact_sheet()).")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    src = Path(a.video)
    out = Path(a.out) if a.out else Path("runs/frames") / (src.stem + ".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet = contact_sheet(src, a.frames, a.cols, a.width, a.drop_last)
    sheet.save(out)
    print(f"wrote {out}  ({sheet.size[0]}x{sheet.size[1]})")


if __name__ == "__main__":
    main()
