# 30-episode re-evaluation of the state-vs-state+RGB campaign

**Nothing here was retrained.** These are the SAME 18 frozen policies as
`../state_plus_rgb/`, pickled in `~/runs_spr/` on the pool, re-loaded with
`rgb_pixel_ablation.py --load-policy` and re-scored with `--episodes 30`
instead of `--episodes 5`. Same environment, same rollout code, same metric
keys, same 250-step episodes, same reset keys. The ONLY difference from
`../state_plus_rgb/<env>/<arm>_seed<n>/pixel_ablation.json` is how many
episodes score each condition.

    tools/run_state_plus_rgb_eval30.sh walker:state_plus_rgb:0 ...   # all 18

Wall clock on the shared RTX 2080 Ti: ~2h50m for all 18 arms
(state+RGB arms score six conditions and take ~15 min each; state-only arms
score `intact` only and take ~2-4 min).

## Why

The campaign's first write-up quoted the 5-episode evaluation as its headline
number. At the effect sizes in play that metric has an achieved power of 0.16
where the training return has 0.78, and the two disagreed on WalkerWalk. More
episodes on the same weights is the direct fix, and it costs GPU minutes
rather than a retrain.

## What it changed

**WalkerWalk: the disagreement was resolution, not substance.** Going from 5
to 30 episodes moves the eval metric onto the training metric's side.

| | 5 episodes | 30 episodes |
|---|---|---|
| state only | 0.7138 +/- 0.0921 | 0.7130 +/- 0.0849 |
| state + RGB | 0.7855 +/- 0.0438 | 0.8157 +/- 0.0167 |
| difference | +0.0717 (+10.0%) | **+0.1027 (+14.4%)** |
| seed ranges | OVERLAP | **DO NOT OVERLAP** |
| Welch p | 0.31 | 0.17 |
| achieved power | 0.16 | 0.35 |
| paired episodes won | 11 / 15 | **76 / 90, p < 0.001** |

(+/- is the SAMPLE s.d., ddof=1, across the 3 seeds.)

**CheetahRun: confirmed at zero.** -0.0011 (-0.2%), Welch p = 0.97, 50 of 90
paired episodes. Both metrics at both episode counts agree there is nothing.

**CartpoleBalance: unchanged in substance.** 1.0000 / 1.0000 / 1.0000 against
0.7193 / 0.9999 / 0.2551. The one seed whose training stayed stable is at
parity with the baseline to within a single non-upright step in 7,500.

**The camera verdict did not move at all.** Walker's `frozen_first` still
costs 72.1% / 62.2% / 72.3% (94.9 / 74.2 / 80.2 at 5 episodes) while cartpole
and cheetah stay at the noise floor, and the old
`median{frozen_first, random_replay, zeros} > 30%` rule STILL labels walker
seed 0 "ignores" at 30 episodes. That rules out small-sample noise as the
explanation for the 2/3-vs-3/3 dispute: it is the rule, not the sample size.

## Files

`<env>/<arm>_seed<n>/pixel_ablation.{json,png}` and `run.log`, one per arm.
The JSONs also carry the fields added in the correction pass:
`upright_fraction_per_episode` (cartpole's headline metric, per episode, which
the 5-episode JSONs do not have and without which cartpole cannot be paired),
`*_std_sample` (ddof=1), `verdict_caveat`, `sd_convention`,
`eval_reset_key_formula` and `eval_pairing_note`.

There are no `training_curves.json` here: no training happened. The curves
live with the original runs in `../state_plus_rgb/`.

The analysis in `../state_plus_rgb/figures/README.md` uses this tree
ALL-OR-NOTHING: a partially finished sweep is ignored rather than producing a
column of "n/a" or silently comparing arms scored with different numbers of
episodes.
