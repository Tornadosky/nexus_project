"""Rollout videos for the state-only vs state+RGB campaign, plus side-by-side pairs.

Drives `nexus_continuous.scripts.rgb_inloop_visualize` once per arm from the
SAVED policies (no retraining), then stitches each env's two arms into one
side-by-side file so the baseline and the extension can be watched together.

Every frame is captioned with the env, arm, seed, meta, budget and the arm's
measured intact score. For pixel arms the caption also states how strongly THAT
SEED depended on its camera, quoting the LARGEST single pixel corruption and
naming it. It deliberately does NOT quote the stored `actor_uses_pixels`
boolean: that thresholds the median of three corruptions at 30%, a bar
calibrated on `RGB_PROPRIO: none` actors, and on WalkerWalk seed 0 it labels
"ignores" a run that loses 94.9% of its performance when the image is held at
the t=0 frame. All three walker seeds use their camera. For state-only arms the
frames are marked as a scene view that is NOT an actor input -- those actors
have no camera pathway at all.

    python tools/make_state_plus_rgb_videos.py
    python tools/make_state_plus_rgb_videos.py --only walker --eval-steps 120
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

CFG = {"cartpole": "cartpole_balance_nesy", "walker": "walker_walk_nesy",
       "cheetah": "cheetah_run_nesy"}
POLDIR = Path.home() / "runs_spr"
RESROOT = Path("results/rgb/state_plus_rgb")

# (env, arm, seed). Seed 0 for every env so the two arms are matched, plus
# walker seed 1. Walker seed 1 was originally included as "the one arm whose
# actor demonstrably USED its camera"; on the corrected reading all three
# walker seeds use theirs (frozen_first costs 94.9 / 74.2 / 80.2 %), so it is
# now simply a second walker pairing.
ARMS = [("cartpole", "state_matched", 0), ("cartpole", "state_plus_rgb", 0),
        ("walker", "state_matched", 0), ("walker", "state_plus_rgb", 0),
        ("walker", "state_plus_rgb", 1),
        ("cheetah", "state_matched", 0), ("cheetah", "state_plus_rgb", 0)]

# Arm-specific caveats that must travel with the video.
#
# The old note on walker/state_matched/0 ("mean dragged down by 1 collapsed
# episode of 5") has been REMOVED. It framed a legitimate result as
# contamination. Eval reset keys are PRNGKey(9000 + 97*ep + seed) and do not
# depend on the arm, so that collapsed episode ran from the SAME initial state
# as episode 4 of walker/state_plus_rgb/0, which scored 0.652: the baseline
# fell over from a start the extension handled. Excluding it moved walker's
# effect from +0.072 to +0.025, i.e. it erased the effect under test.
#
# Keep each note under ~60 characters: the caption is rendered into a 320px
# band with PIL's default bitmap font, which does not wrap, so a longer line
# is silently truncated mid-word.
NOTES = {
    ("walker", "state_matched", 0):
        "ep4 collapsed 0.055; paired state+RGB ep4 = 0.652",
}

PAIRS = [("cartpole", 0, 0), ("walker", 0, 0), ("walker", 0, 1), ("cheetah", 0, 0)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="results/rgb/state_plus_rgb/video")
    ap.add_argument("--eval-steps", type=int, default=250)
    ap.add_argument("--upscale", type=int, default=320)
    ap.add_argument("--updates", type=int, default=250,
                    help="caption only (the policy is loaded, not trained)")
    ap.add_argument("--num-envs", type=int, default=128, help="caption only")
    ap.add_argument("--only", default=None, help="restrict to one env")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args(argv)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    made = {}

    for env, arm, seed in ARMS:
        if args.only and env != args.only:
            continue
        pol = POLDIR / f"{env}_{arm}_s{seed}.pkl"
        res = RESROOT / env / f"{arm}_seed{seed}" / "pixel_ablation.json"
        if not pol.exists():
            print(f"[skip] no saved policy {pol}")
            continue
        tag = f"{env}_{arm}_seed{seed}"
        dest = out / tag
        if args.skip_existing and (dest / "rollout_inloop.mp4").exists():
            print(f"[have] {tag}")
            made[(env, arm, seed)] = dest
            continue
        cmd = [sys.executable, "-m", "nexus_continuous.scripts.rgb_inloop_visualize",
               "--config", f"configs/{CFG[env]}_{arm}.yaml", "--meta", "nesy",
               "--seed", str(seed), "--load-policy", str(pol),
               "--eval-steps", str(args.eval_steps), "--upscale", str(args.upscale),
               "--updates", str(args.updates), "--num-envs", str(args.num_envs),
               "--out", str(dest)]
        if arm.startswith("state_matched"):
            cmd.append("--no-rgb")
        if res.exists():
            cmd += ["--result-json", str(res)]
        note = NOTES.get((env, arm, seed))
        if note:
            cmd += ["--note", note]
        print("\n>>>", " ".join(cmd))
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"[FAIL] {tag} rc={rc}")
            continue
        made[(env, arm, seed)] = dest

    # ---- side-by-side: baseline (left) next to extension (right) -----------
    import numpy as np
    import imageio.v2 as imageio

    for env, s_state, s_rgb in PAIRS:
        if args.only and env != args.only:
            continue
        a = made.get((env, "state_matched", s_state))
        b = made.get((env, "state_plus_rgb", s_rgb))
        if not (a and b):
            continue
        va = imageio.mimread(a / "rollout_inloop.mp4", memtest=False)
        vb = imageio.mimread(b / "rollout_inloop.mp4", memtest=False)
        t = min(len(va), len(vb))
        h = max(va[0].shape[0], vb[0].shape[0])

        def pad(f):
            if f.shape[0] == h:
                return f
            return np.vstack([f, np.zeros((h - f.shape[0],) + f.shape[1:], f.dtype)])

        gap = np.zeros((h, 6, 3), np.uint8)
        gap[:, :, :] = 90
        frames = [np.hstack([pad(va[i]), gap, pad(vb[i])]) for i in range(t)]
        name = (f"{env}_side_by_side_state_s{s_state}_vs_state_plus_rgb_s{s_rgb}.mp4")
        imageio.mimsave(out / name, frames, fps=25, quality=8)
        # The GIF is a convenience preview, not the artifact of record: keep it
        # half-size and every 4th frame, or a 250-frame side-by-side lands at
        # ~5 MB in git for no extra information over the mp4.
        small = [f[::2, ::2] for f in frames[::4]]
        imageio.mimsave(out / name.replace(".mp4", ".gif"), small, fps=12, loop=0)
        print("wrote", out / name)

    write_readme(out, made)


def write_readme(out: Path, made):
    lines = ["# Rollout videos: state-only vs state+RGB\n",
             "Generated by `tools/make_state_plus_rgb_videos.py` from the saved "
             "policies in `~/runs_spr/` (no retraining). Every arm is a 250-step "
             "greedy rollout in a 1-env vision environment.\n",
             "## What the frames show\n",
             "The image is a 64x64 **GRAYSCALE** 3-frame stack from the "
             "MuJoCo-Warp camera render. It is not colour; \"RGB\" throughout "
             "this project is a misnomer inherited from the flag name.\n",
             "* For **state+RGB** arms that image is the actor's input, and the "
             "caption says how strongly that seed's actor depended on it -- the "
             "LARGEST single pixel corruption, named. It does NOT quote the "
             "stored `actor_uses_pixels` boolean; see below.",
             "* For **state-only** arms the actor has no camera pathway at all. "
             "The image is a scene view only, and every frame says so. Do not "
             "read it as something the policy is looking at.\n",
             "## Corrected camera verdicts\n",
             "The captions were rewritten in the correction pass. The stored "
             "`actor_uses_pixels` boolean thresholds the MEDIAN of "
             "{frozen_first, random_replay, zeros} at 30%, a bar calibrated on "
             "`RGB_PROPRIO: none` actors whose only input is the camera. An "
             "actor that also holds the privileged state can ride it through a "
             "corrupted frame, so the median under-reads, and it hides "
             "disagreement between the conditions.\n",
             "**All three WalkerWalk seeds use their camera**: holding the "
             "image at the t=0 frame costs 94.9% / 74.2% / 80.2%, and the "
             "independent in-training probe reads 0.098 / 0.099 / 0.117 across "
             "the three. The earlier \"seed 0 ignores its camera\" reading was "
             "an artifact of the median rule. Cartpole and cheetah show no "
             "camera use on any seed. Full per-condition tables: "
             "`../figures/README.md`.\n",
             "**If a video's baked-in caption still says a walker seed "
             "\"IGNORED its camera\", that file predates the correction and "
             "should be regenerated** -- delete it and re-run the driver.\n",
             "## What these videos do NOT claim\n",
             "* A single 250-step rollout is an illustration, not a "
             "measurement. The numbers to cite are in `../figures/README.md`, "
             "whose PRIMARY metric is the training return over 128 envs, not "
             "the 5-episode eval.",
             "* Side-by-side files pair one seed of each arm; two walker "
             "pairings are provided. The walker state-only seed-0 video "
             "carries a note that its evaluation contains one collapsed "
             "episode. That episode is NOT excluded from any analysis: eval "
             "reset keys depend only on (episode, seed), so it started from "
             "the same state as the state+RGB arm's episode 4, which scored "
             "0.652. It is a paired outcome, not contamination.\n",
             "## Files\n",
             "| file | env | arm | seed |", "|---|---|---|---|"]
    # Scan the directory rather than trusting `made`: the driver is often run one
    # env at a time, and a README listing only the last env would look complete
    # while silently omitting the others.
    for d in sorted(out.iterdir()):
        if not (d.is_dir() and (d / "rollout_inloop.mp4").exists()):
            continue
        env = d.name.split("_", 1)[0]
        arm = "state_plus_rgb" if "state_plus_rgb" in d.name else "state_matched"
        seed = d.name.rsplit("seed", 1)[-1]
        lines.append(f"| `{d.name}/rollout_inloop.mp4` | {env} | {arm} | {seed} |")
    for f in sorted(out.glob("*side_by_side*.mp4")):
        lines.append(f"| `{f.name}` | {f.name.split('_')[0]} | side-by-side | see name |")
    lines += ["", "Each per-arm directory also holds `observation_filmstrip.png` "
                  "(8 sampled frames) and `skill_timeline.png` (which skill the "
                  "meta-policy selected, against reward).", ""]
    (out / "README.md").write_text("\n".join(lines))
    print("wrote", out / "README.md")


if __name__ == "__main__":
    main()
