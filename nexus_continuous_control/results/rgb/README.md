# RGB skill-actor extension — results

**The report to read is
[`docs/reports/RGB_EXTENSION_FINDINGS.md`](../../docs/reports/RGB_EXTENSION_FINDINGS.md)** (short,
with figures). The long version with every per-seed number is
[`docs/reports/rgb_extension_team_briefing.txt`](../../docs/reports/rgb_extension_team_briefing.txt).

## The question

**This is the supervisor's brief, not a NEXUS-paper result** — an *optional* extension. It asks
whether giving the skill actors a camera **in addition to** the state vector they already read
improves performance. It is answered by a matched-budget comparison: state-only control vs
state + RGB, 3 envs × 2 arms × 3 seeds, 2.05M env steps each, with the two arms' configs differing in
**exactly one key** (`RGB_ACTOR`).

Both arms keep `USE_RGB: true`, because in MuJoCo Playground `vision` is a *task* switch (it changes
the reward, `ctrl_dt`, episode length, termination rule and state representation), not a rendering
switch. `RGB_ACTOR` separates "the environment renders" from "the actor consumes".

## The answer

Primary metric is training return (last-20-update mean over 250 updates × 128 envs); ± is the sample
s.d. over 3 seeds. Eval is a 30-episode deterministic score of the final weights — upright fraction
for cartpole, reward/step for walker and cheetah, so eval values are not comparable across rows.

| env | training return (state only → state+RGB) | Δ | 30-ep eval | camera used by actor |
|---|---|---|---|---|
| CartpoleBalance | 24.10 ± 1.18 → 16.29 ± 13.34 | −32% | 1.000 → 0.658 | ignored 3/3 |
| WalkerWalk | 189.51 ± 1.80 → **203.20 ± 6.25** | **+7.2%** | 0.713 → 0.816 | **used 3/3** |
| CheetahRun | 121.15 ± 1.38 → 118.35 ± 1.72 | −2.3% | 0.481 → 0.480 | ignored 3/3 |

Walker helped (3/3 seeds, non-overlapping ranges, Welch p = 0.054, d = +2.98, paired p = 0.047, and
+14.4% at the 30-episode eval). Cheetah cost 1–3%. Cartpole is a **stability** finding, not a lower
ceiling: one seed finished at 0.9999 upright fraction, two collapsed late and were scored after the
collapse. The split tracks whether the actor learned to use its camera at all.

**Two limits bind all of it:** the runs are undertrained (4–21% of this project's own configured
budgets, nothing plateaued) and n = 3 seeds. The caveats section of the report is not optional
reading.

## What is in here

- **`state_plus_rgb/`** — the campaign, at 2.05M env steps.
  - `<env>/<arm>_seed<n>/` — raw per-run artifacts exactly as produced, never rewritten:
    `training_curves.json` (250 updates × 128 envs: episode return, in-training pixel sensitivity),
    `pixel_ablation.json` (**5-episode** deterministic eval, six camera-corruption conditions),
    plus `run.log` and the two per-run PNGs. `<env>` ∈ {`cartpole`, `walker`, `cheetah`},
    `<arm>` ∈ {`state_matched`, `state_plus_rgb`}.
  - `corrected_analysis.json` — every statistic in the report, machine-readable, with the source path
    of every input. Regenerate with `python tools/analyze_state_plus_rgb.py`.
  - `reference_baselines.json` — the few pixels-only scalars used as reference *lines* in the figures,
    with their provenance, budget, and an explicit flag for which are one-key contrasts and which are
    not. No conclusion rests on them.
  - `figures/` — eight figures plus a per-figure README that states what each one does **and does
    not** claim. Regenerate with `python tools/plot_state_plus_rgb_figures.py --figure all`.
  - `video/` — 11 mp4 rollouts (seven per-arm, four side-by-side pairings) generated from the saved
    policies, no retraining; each per-arm directory also holds an observation filmstrip and a skill
    timeline. A single rollout illustrates, it does not measure.
- **`state_plus_rgb_eval30/`** — the **same 18 frozen policies** re-scored at 30 evaluation episodes
  instead of 5, no retraining (`tools/run_state_plus_rgb_eval30.sh`). Same layout, `pixel_ablation.json`
  only. Kept alongside the 5-episode tree rather than replacing it, because the difference between
  them is itself a finding: the 5-episode metric had an achieved power of 0.16 and reported walker as
  a null.

Earlier campaigns in this line — offline distillation of state skills into pixel policies, and in-loop
pixel RL with `RGB_PROPRIO: none` — tested pixels **replacing** the state, which is a different
question and had no matched-budget control. Their artifact trees (`distill/`, `inloop/`, `ablation/`)
were deleted from the working tree and remain in git history at `8333c39` and earlier.

## Reproduce

```bash
python tools/gen_state_plus_rgb_configs.py    # regenerates the 6 configs, asserts the one-key property
tools/run_state_plus_rgb_campaign.sh cartpole:state_matched:0 ...   # training, 18 arms, GPU
tools/run_state_plus_rgb_eval30.sh walker:state_plus_rgb:0 ...      # 30-episode re-scoring, GPU
python tools/analyze_state_plus_rgb.py                              # statistics, CPU
python tools/analyze_state_plus_rgb.py --readme                     # figures/README.md, CPU
python tools/plot_state_plus_rgb_figures.py --figure all            # figures, CPU
```

Run environment: TU student pool (RTX 2080 Ti), jax 0.11 CUDA, mujoco 3.11 + mujoco-warp 3.11,
in-loop rendering via MJWarp (CUDA). Headless-rendering setup is in the top-level
[`README.md`](../../README.md).
