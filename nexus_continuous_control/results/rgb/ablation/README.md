# Pixel-dependence ablation campaign — directory legend

Each leaf directory is one trained in-loop pixel-RL run, corrupted six ways to
test whether the actor actually uses its camera (see
[`../../../docs/reports/rgb_extension_team_briefing.txt`](../../../docs/reports/rgb_extension_team_briefing.txt)
section 7-8 for the full story). Layout: `<env>/<meta>_<status>[_seedN]/`.

- **`<env>`** — `cartpole`, `cheetah`, `walker`, `hopper`.
- **`<meta>`** — `nesy`, the project's own flagship neuro-symbolic meta; these
  are the headline numbers. (An earlier pass accidentally ran this whole
  campaign with `--meta neural` instead; once the `nesy` re-run independently
  reproduced every conclusion, the `neural` in-loop data was deleted as
  redundant. `neural` still exists separately for the distillation comparison
  in `results/rgb/distill/`, which is unaffected.)
- **`<status>`**
  - `blind` — baseline config, the actor was found to ignore its pixels
    (privileged meta solved the task alone).
  - `fixed` — trained with the repair (`RGB_AUX_STATE_COEF 1.0` +
    `META_DECISION_INTERVAL 4`); verified genuinely pixel-driven.
  - no status (`cheetah/nesy_seed0`, `hopper/nesy_seed0`) — only one variant
    was ever run for that env (cheetah was never blind; hopper's score is
    inconclusive so there was nothing to fix).
- **`_seedN`** — multi-seed replicate (seed 0 is the primary/first run). Only
  `cartpole` and `walker`'s `fixed` runs and `cheetah` have 3-seed replicates
  (`_seed0/_seed1/_seed2`); everything else is single-seed.

Each leaf directory contains: `pixel_ablation.json`/`.png` (six-condition
camera ablation + verdict), `pixel_sensitivity.json` (open-loop responsiveness
probe), `training_curves.json`/`.png`, and `viz/` (rollout video, 64×64
filmstrip, skill/reward timeline).

- **`summary/`** — cross-run figures aggregating multiple leaf directories:
  `comparison_nesy.png`, `method_comparison_nesy.png`,
  `pixel_responsiveness_nesy.png`. Regenerate with
  `tools/plot_rgb_ablation_comparison.py`, `tools/plot_rgb_summary_figures.py`,
  `tools/plot_rgb_sensitivity_figure.py` (no GPU needed — they read the
  committed JSON).
- **`seed_variance.json`** — mean ± std across seeds for the three multi-seed
  groups (cheetah, cartpole+fixed, walker+fixed).

Quick map from the old (pre-reorg) flat names, if you're cross-referencing an
older note: `cartpole_nesy` → `cartpole/nesy_blind`, `cartpole_aux_nesy[_sN]`
→ `cartpole/nesy_fixed_seed[0N]`, `cheetah_nesy[_sN]` →
`cheetah/nesy_seed[0N]`, `walker_nesy` → `walker/nesy_blind`,
`walker_aux_nesy[_sN]` → `walker/nesy_fixed_seed[0N]`, `hopper_nesy` →
`hopper/nesy_seed0`.
