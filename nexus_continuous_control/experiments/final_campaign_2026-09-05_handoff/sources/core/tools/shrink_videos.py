#!/usr/bin/env python3
"""Shrink the V3.4 clips so the dashboard fits the 16 MB Artifact limit.

The clips render one frame per env step (1000 frames, ~33 s at 30 fps) and together came to
10.3 MB, pushing the board to 16 MB. The plan's own media budget is "<= 6 s at 480p".

**Subsample rather than truncate.** Keeping the first 200 frames would show only the first fifth
of the episode and silently misrepresent the rollout — a walker that falls at step 800 would
look fine. Taking every Nth frame keeps the whole episode and plays it at N x speed, which is
what a reviewer wants from a behaviour clip anyway.

The skill-activation strip is untouched: it already summarises the full episode at per-step
resolution, so the pair stays honest — strip for detail, video for gist.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as iio


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="runs/videos")
    ap.add_argument("--every", type=int, default=5, help="keep every Nth frame")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max-frames", type=int, default=210,
                    help="clips already at or below this many frames are left alone")
    args = ap.parse_args()

    total_before = total_after = 0.0
    for mp4 in sorted(Path(args.dir).glob("*.mp4")):
        mb = mp4.stat().st_size / 1048576
        total_before += mb
        # Decide on FRAME COUNT, not file size. Selecting on size left `hopper_flat_s0` at its
        # full 1000 frames because it happened to compress well, so it played at 1x while every
        # other clip played at 5x -- a reviewer comparing arms would have been comparing
        # different playback speeds without being told.
        try:
            probe = iio.get_reader(str(mp4))
            n_frames = probe.count_frames()
            probe.close()
        except Exception:
            n_frames = None
        if n_frames is not None and n_frames <= args.max_frames:
            total_after += mb
            print(f"  skip  {mp4.name:<36} {mb:5.2f} MB ({n_frames} frames, already short)")
            continue
        # Stream frame-by-frame. Materialising the whole clip first blew up with MemoryError:
        # 1000 frames x 368x480x3 is ~530 MB decoded, for a 1.6 MB file on disk.
        tmp = mp4.with_suffix(".shrink.mp4")
        n_in = n_out = 0
        reader = iio.get_reader(str(mp4))
        writer = iio.get_writer(str(tmp), fps=args.fps, macro_block_size=None)
        try:
            for i, frame in enumerate(reader):
                n_in += 1
                if i % args.every == 0:
                    writer.append_data(frame)
                    n_out += 1
        finally:
            writer.close()
            reader.close()
        tmp.replace(mp4)
        nb = mp4.stat().st_size / 1048576
        total_after += nb
        print(f"  shrink {mp4.name:<36} {mb:5.2f} -> {nb:5.2f} MB "
              f"({n_in} -> {n_out} frames, {args.every}x speed)")
    print(f"\ntotal {total_before:.2f} MB -> {total_after:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
