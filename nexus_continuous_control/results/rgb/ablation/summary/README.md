# DEPRECATED: `method_comparison_nesy.png`

`method_comparison_nesy.png` (and its `--meta neural` sibling, produced by
`tools/plot_rgb_summary_figures.py`) draws a dashed "privileged upper bound"
that must not be presented. It is **superseded** by
`results/rgb/state_plus_rgb/figures/`.

## Why it is wrong

The "privileged state (cheats)" bar and the dashed upper-bound line are read
from `results/rgb/distill/combined.json`, i.e. from the DISTILLATION
experiment, and are then placed beside the in-loop pixel bars as if the two
were comparable. They are not:

* **Different environment.** Distillation trains its teacher on
  `configs/cartpole_balance_nesy.yaml`, which has `vision=False`. MuJoCo
  Playground's CartpoleBalance keys `ctrl_dt` (0.02 under vision),
  `episode_length` (250 under vision, 1000 otherwise), the REWARD FUNCTION
  (`_dense_vision_reward` vs `_dense_reward`) and the termination rule on
  `vision`. The in-loop bars beside it run the vision env. Different task.
* **Different observation.** The non-vision env feeds the actor the DM-suite
  featurised observation; the vision env feeds it `qpos+qvel`.
* **Different budget.** That teacher trained for 9,830,400 environment steps;
  the in-loop bars beside it had 2,048,000 -- roughly 5x fewer.
* **It is not even an upper bound.** The matched-budget state control measured
  in this campaign reaches an upright fraction of **1.000** at 2.05M steps,
  above the 0.743 the figure draws as the ceiling.

## What to use instead

`results/rgb/state_plus_rgb/figures/` -- `fig1_headline_learning_curves.png`
for baseline-vs-extension and `fig6_all_variants.png` for where every variant
lands. Those state baselines are measured in the SAME environment, at the SAME
budget, through the SAME evaluation code as the RGB arms, from configs that a
generator asserts differ in exactly one key.

The original image is preserved unaltered as
`method_comparison_nesy.SUPERSEDED.png`; the file under the original name now
carries a deprecation banner so it cannot be pasted into a talk by accident.
