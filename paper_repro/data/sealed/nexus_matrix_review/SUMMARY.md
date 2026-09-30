# Generated statistics — NEXUS final matrix (142 cells)

Packed 2026-09-08T18:28:50+00:00 on `melody-pc`.  
Primary metric: `primary_success_rate` at the training endpoint, 256 deterministic episodes, eval seed 30000, identical evaluator for every method including PPO.

Every table reports **per-seed min/max**, not means alone, and marks `n<=3` provisional. Comparisons are quotable only where `budget_matched` is true, or where the mismatch runs in the baseline's favour and is labelled.


**Sealed campaign: 100/142 cells complete.** Every cell that had an executable specification ran. The 42 missing rows are the `llm_pilot` and `llm_final` blocks, whose generated specifications were all rejected by the validator: that is a recorded negative result for RQ3, not unfinished work, because the kit forbids resampling until a proposal passes (`replacement_samples_allowed: false`).


**Second LLM cohort (`llm_v2`): 42 of those 42 cells trained so far** with a different, much larger generator model against a byte-identical validator and the same acceptance bar. It answers "does a bigger proposer clear the bar" as a parallel arm; it never replaces the sealed rows, and is tagged `(llm_v2)` wherever it appears below. Its budgets differ from the hand-written `nesy` reference, so any comparison between them is flagged budget-unmatched in section 2.


## 1. Arms — primary success by seed

| env | method | n | budget (steps) | min | max | mean | median | per-seed |
|---|---|---|---|---|---|---|---|---|
| CheetahRun | `llm_final:refined (llm_v2)` | 6 | 52428800 | 0.9277 | 0.9694 | 0.9565 | 0.9659 | 0.9694, 0.9650, 0.9277, 0.9668, 0.9671, 0.9428 |
| CheetahRun | `llm_final:initial (llm_v2)` | 6 | 52428800 | 0.8071 | 0.9712 | 0.9309 | 0.9609 | 0.9712, 0.9643, 0.9188, 0.8071, 0.9575, 0.9663 |
| CheetahRun | `llm_pilot:initial (llm_v2)` *(provisional)* | 3 | 13107200 | 0.8926 | 0.9585 | 0.9213 | 0.9127 | 0.9585, 0.8926, 0.9127 |
| CheetahRun | `llm_reference:nesy` | 5 | 52428800 | 0.8700 | 0.9633 | 0.9203 | 0.9073 | 0.8700, 0.9578, 0.9034, 0.9073, 0.9633 |
| CheetahRun | `llm_final:resample (llm_v2)` | 6 | 52428800 | 0.6103 | 0.9690 | 0.9009 | 0.9591 | 0.9690, 0.9397, 0.6103, 0.9538, 0.9685, 0.9644 |
| Go1JoystickFlatTerrain | `ppo` | 5 | 32768000 | 0.4104 | 0.4792 | 0.4447 | 0.4412 | 0.4751, 0.4792, 0.4104, 0.4412, 0.4177 |
| Go1JoystickFlatTerrain | `neural` | 5 | 32768000 | 0.2592 | 0.6207 | 0.3546 | 0.2902 | 0.2812, 0.2902, 0.3215, 0.2592, 0.6207 |
| Go1JoystickFlatTerrain | `nesy` | 5 | 32768000 | 0.2549 | 0.4401 | 0.3350 | 0.3167 | 0.3167, 0.2549, 0.4401, 0.3148, 0.3488 |
| Go1JoystickFlatTerrain | `symbolic` | 5 | 32768000 | 0.0813 | 0.3263 | 0.1750 | 0.1219 | 0.2386, 0.1071, 0.0813, 0.1219, 0.3263 |
| Go1JoystickFlatTerrain | `flat` | 5 | 32768000 | 0.0007 | 0.2249 | 0.0741 | 0.0031 | 0.0031, 0.0013, 0.1404, 0.2249, 0.0007 |
| Go1JoystickFlatTerrain | `hpqn` | 5 | 32768000 | 0.0007 | 0.0020 | 0.0012 | 0.0008 | 0.0008, 0.0017, 0.0008, 0.0020, 0.0007 |
| HopperHop | `neural` | 5 | 117964800 | 0.4119 | 0.4634 | 0.4398 | 0.4453 | 0.4470, 0.4453, 0.4119, 0.4634, 0.4315 |
| HopperHop | `nesy` | 5 | 117964800 | 0.4156 | 0.4398 | 0.4270 | 0.4301 | 0.4398, 0.4324, 0.4301, 0.4170, 0.4156 |
| HopperHop | `hpqn` | 5 | 117964800 | 0.2967 | 0.4264 | 0.3370 | 0.3192 | 0.2988, 0.3192, 0.4264, 0.2967, 0.3439 |
| HopperHop | `symbolic` | 5 | 117964800 | 0.0019 | 0.4456 | 0.1751 | 0.1895 | 0.0019, 0.0226, 0.2161, 0.4456, 0.1895 |
| HopperHop | `flat` | 5 | 117964800 | 0.0031 | 0.3822 | 0.1634 | 0.1174 | 0.0105, 0.0031, 0.3822, 0.3039, 0.1174 |
| HopperHop | `ppo` | 5 | 117964800 | 0.0016 | 0.0019 | 0.0018 | 0.0018 | 0.0019, 0.0016, 0.0018, 0.0019, 0.0018 |
| WalkerWalk | `llm_reference:nesy` | 5 | 52428800 | 0.9961 | 1.0000 | 0.9992 | 1.0000 | 1.0000, 1.0000, 0.9961, 1.0000, 1.0000 |
| WalkerWalk | `llm_final:initial (llm_v2)` | 6 | 52428800 | 0.1406 | 1.0000 | 0.6647 | 0.6426 | 0.5742, 0.7109, 1.0000, 1.0000, 0.5625, 0.1406 |
| WalkerWalk | `llm_pilot:initial (llm_v2)` *(provisional)* | 3 | 13107200 | 0.5312 | 0.5625 | 0.5417 | 0.5312 | 0.5312, 0.5312, 0.5625 |
| WalkerWalk | `llm_final:refined (llm_v2)` | 6 | 52428800 | 0.0000 | 0.7031 | 0.5371 | 0.6465 | 0.5742, 0.7031, 0.6445, 0.0000, 0.6523, 0.6484 |
| WalkerWalk | `llm_final:resample (llm_v2)` | 6 | 52428800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000 |

## 2. Within-environment comparisons

`sep A>B` is the campaign's strong form: min(A) > max(B) across every seed. `p_unpaired` is an exact permutation test on the difference of means; `p_paired` is an exact sign-flip test over seed-matched pairs — with 5 shared seeds the smallest attainable paired p is 0.0625, so a paired result should be read as direction plus effect size, not as significance.

| env | A | B | n_a | n_b | mean A | mean B | sep A>B | sep B>A | p_unpaired | paired Δ | p_paired | budget matched |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CheetahRun | `llm_final:initial (llm_v2)` | `llm_final:refined (llm_v2)` | 6 | 6 | 0.9309 | 0.9565 | no | no | 0.46320 | -0.0256 | 0.59375 | yes |
| CheetahRun | `llm_final:initial (llm_v2)` | `llm_final:resample (llm_v2)` | 6 | 6 | 0.9309 | 0.9009 | no | no | 0.79004 | 0.0299 | 0.65625 | yes |
| CheetahRun | `llm_final:initial (llm_v2)` | `llm_pilot:initial (llm_v2)` | 6 | 3 | 0.9309 | 0.9213 | no | no | 0.85714 | - | - | **NO** |
| CheetahRun | `llm_final:initial (llm_v2)` | `llm_reference:nesy` | 6 | 5 | 0.9309 | 0.9203 | no | no | 0.76190 | - | - | yes |
| CheetahRun | `llm_final:refined (llm_v2)` | `llm_final:resample (llm_v2)` | 6 | 6 | 0.9565 | 0.9009 | no | no | 0.58009 | 0.0555 | 0.34375 | yes |
| CheetahRun | `llm_final:refined (llm_v2)` | `llm_pilot:initial (llm_v2)` | 6 | 3 | 0.9565 | 0.9213 | no | no | 0.04762 | - | - | **NO** |
| CheetahRun | `llm_final:refined (llm_v2)` | `llm_reference:nesy` | 6 | 5 | 0.9565 | 0.9203 | no | no | 0.06926 | - | - | yes |
| CheetahRun | `llm_final:resample (llm_v2)` | `llm_pilot:initial (llm_v2)` | 6 | 3 | 0.9009 | 0.9213 | no | no | 0.97619 | - | - | **NO** |
| CheetahRun | `llm_final:resample (llm_v2)` | `llm_reference:nesy` | 6 | 5 | 0.9009 | 0.9203 | no | no | 0.98052 | - | - | yes |
| CheetahRun | `llm_pilot:initial (llm_v2)` | `llm_reference:nesy` | 3 | 5 | 0.9213 | 0.9203 | no | no | 0.96429 | - | - | **NO** |
| Go1JoystickFlatTerrain | `flat` | `hpqn` | 5 | 5 | 0.0741 | 0.0012 | no | no | 0.11905 | 0.0729 | 0.25000 | yes |
| Go1JoystickFlatTerrain | `flat` | `nesy` | 5 | 5 | 0.0741 | 0.3350 | no | YES | 0.00794 | -0.2610 | 0.06250 | yes |
| Go1JoystickFlatTerrain | `flat` | `neural` | 5 | 5 | 0.0741 | 0.3546 | no | YES | 0.00794 | -0.2805 | 0.06250 | yes |
| Go1JoystickFlatTerrain | `flat` | `ppo` | 5 | 5 | 0.0741 | 0.4447 | no | YES | 0.00794 | -0.3707 | 0.06250 | yes |
| Go1JoystickFlatTerrain | `flat` | `symbolic` | 5 | 5 | 0.0741 | 0.1750 | no | no | 0.17460 | -0.1010 | 0.31250 | yes |
| Go1JoystickFlatTerrain | `hpqn` | `nesy` | 5 | 5 | 0.0012 | 0.3350 | no | YES | 0.00794 | -0.3339 | 0.06250 | yes |
| Go1JoystickFlatTerrain | `hpqn` | `neural` | 5 | 5 | 0.0012 | 0.3546 | no | YES | 0.00794 | -0.3534 | 0.06250 | yes |
| Go1JoystickFlatTerrain | `hpqn` | `ppo` | 5 | 5 | 0.0012 | 0.4447 | no | YES | 0.00794 | -0.4435 | 0.06250 | yes |
| Go1JoystickFlatTerrain | `hpqn` | `symbolic` | 5 | 5 | 0.0012 | 0.1750 | no | YES | 0.00794 | -0.1739 | 0.06250 | yes |
| Go1JoystickFlatTerrain | `nesy` | `neural` | 5 | 5 | 0.3350 | 0.3546 | no | no | 0.88095 | -0.0195 | 0.93750 | yes |
| Go1JoystickFlatTerrain | `nesy` | `ppo` | 5 | 5 | 0.3350 | 0.4447 | no | no | 0.02381 | -0.1097 | 0.12500 | yes |
| Go1JoystickFlatTerrain | `nesy` | `symbolic` | 5 | 5 | 0.3350 | 0.1750 | no | no | 0.03175 | 0.1600 | 0.06250 | yes |
| Go1JoystickFlatTerrain | `neural` | `ppo` | 5 | 5 | 0.3546 | 0.4447 | no | no | 0.25397 | -0.0901 | 0.37500 | yes |
| Go1JoystickFlatTerrain | `neural` | `symbolic` | 5 | 5 | 0.3546 | 0.1750 | no | no | 0.05556 | 0.1795 | 0.06250 | yes |
| Go1JoystickFlatTerrain | `ppo` | `symbolic` | 5 | 5 | 0.4447 | 0.1750 | YES | no | 0.00794 | 0.2697 | 0.06250 | yes |
| HopperHop | `flat` | `hpqn` | 5 | 5 | 0.1634 | 0.3370 | no | no | 0.10317 | -0.1736 | 0.12500 | yes |
| HopperHop | `flat` | `nesy` | 5 | 5 | 0.1634 | 0.4270 | no | YES | 0.00794 | -0.2636 | 0.06250 | yes |
| HopperHop | `flat` | `neural` | 5 | 5 | 0.1634 | 0.4398 | no | YES | 0.00794 | -0.2764 | 0.06250 | yes |
| HopperHop | `flat` | `ppo` | 5 | 5 | 0.1634 | 0.0018 | YES | no | 0.00794 | 0.1616 | 0.06250 | yes |
| HopperHop | `flat` | `symbolic` | 5 | 5 | 0.1634 | 0.1751 | no | no | 0.93651 | -0.0117 | 0.87500 | yes |
| HopperHop | `hpqn` | `nesy` | 5 | 5 | 0.3370 | 0.4270 | no | no | 0.02381 | -0.0900 | 0.06250 | yes |
| HopperHop | `hpqn` | `neural` | 5 | 5 | 0.3370 | 0.4398 | no | no | 0.01587 | -0.1028 | 0.12500 | yes |
| HopperHop | `hpqn` | `ppo` | 5 | 5 | 0.3370 | 0.0018 | YES | no | 0.00794 | 0.3352 | 0.06250 | yes |
| HopperHop | `hpqn` | `symbolic` | 5 | 5 | 0.3370 | 0.1751 | no | no | 0.11111 | 0.1619 | 0.12500 | yes |
| HopperHop | `nesy` | `neural` | 5 | 5 | 0.4270 | 0.4398 | no | no | 0.20635 | -0.0128 | 0.31250 | yes |
| HopperHop | `nesy` | `ppo` | 5 | 5 | 0.4270 | 0.0018 | YES | no | 0.00794 | 0.4252 | 0.06250 | yes |
| HopperHop | `nesy` | `symbolic` | 5 | 5 | 0.4270 | 0.1751 | no | no | 0.04762 | 0.2519 | 0.12500 | yes |
| HopperHop | `neural` | `ppo` | 5 | 5 | 0.4398 | 0.0018 | YES | no | 0.00794 | 0.4380 | 0.06250 | yes |
| HopperHop | `neural` | `symbolic` | 5 | 5 | 0.4398 | 0.1751 | no | no | 0.03175 | 0.2647 | 0.06250 | yes |
| HopperHop | `ppo` | `symbolic` | 5 | 5 | 0.0018 | 0.1751 | no | no | 0.01587 | -0.1733 | 0.06250 | yes |
| WalkerWalk | `llm_final:initial (llm_v2)` | `llm_final:refined (llm_v2)` | 6 | 6 | 0.6647 | 0.5371 | no | no | 0.51299 | 0.1276 | 0.68750 | yes |
| WalkerWalk | `llm_final:initial (llm_v2)` | `llm_final:resample (llm_v2)` | 6 | 6 | 0.6647 | 0.0000 | YES | no | 0.00216 | 0.6647 | 0.03125 | yes |
| WalkerWalk | `llm_final:initial (llm_v2)` | `llm_pilot:initial (llm_v2)` | 6 | 3 | 0.6647 | 0.5417 | no | no | 0.47619 | - | - | **NO** |
| WalkerWalk | `llm_final:initial (llm_v2)` | `llm_reference:nesy` | 6 | 5 | 0.6647 | 0.9992 | no | no | 0.06061 | - | - | yes |
| WalkerWalk | `llm_final:refined (llm_v2)` | `llm_final:resample (llm_v2)` | 6 | 6 | 0.5371 | 0.0000 | no | no | 0.01515 | 0.5371 | 0.06250 | yes |
| WalkerWalk | `llm_final:refined (llm_v2)` | `llm_pilot:initial (llm_v2)` | 6 | 3 | 0.5371 | 0.5417 | no | no | 1.00000 | - | - | **NO** |
| WalkerWalk | `llm_final:refined (llm_v2)` | `llm_reference:nesy` | 6 | 5 | 0.5371 | 0.9992 | no | YES | 0.00216 | - | - | yes |
| WalkerWalk | `llm_final:resample (llm_v2)` | `llm_pilot:initial (llm_v2)` | 6 | 3 | 0.0000 | 0.5417 | no | YES | 0.01190 | - | - | **NO** |
| WalkerWalk | `llm_final:resample (llm_v2)` | `llm_reference:nesy` | 6 | 5 | 0.0000 | 0.9992 | no | YES | 0.00216 | - | - | yes |
| WalkerWalk | `llm_pilot:initial (llm_v2)` | `llm_reference:nesy` | 3 | 5 | 0.5417 | 0.9992 | no | YES | 0.01786 | - | - | **NO** |

## 3. RQ4 — vision arms (own metric: intact reward/step)

| env | method | n | min | max | mean | per-seed |
|---|---|---|---|---|---|---|
| CartpoleBalance | `constant` | 5 | 0.0859 | 0.0934 | 0.0900 | 0.0903, 0.0887, 0.0859, 0.0917, 0.0934 |
| CartpoleBalance | `pixels` | 5 | 0.0600 | 0.0994 | 0.0859 | 0.0978, 0.0600, 0.0749, 0.0994, 0.0975 |
| CartpoleBalance | `state` | 5 | 0.0882 | 0.0996 | 0.0964 | 0.0969, 0.0882, 0.0980, 0.0996, 0.0993 |
| WalkerWalk | `constant` | 5 | 0.6842 | 0.8101 | 0.7576 | 0.7790, 0.7689, 0.8101, 0.7457, 0.6842 |
| WalkerWalk | `pixels` | 5 | 0.5659 | 0.8169 | 0.7376 | 0.8066, 0.7989, 0.8169, 0.6999, 0.5659 |
| WalkerWalk | `state` | 5 | 0.5770 | 0.8085 | 0.7324 | 0.7857, 0.7959, 0.5770, 0.8085, 0.6951 |

| env | A | B | mean A | mean B | sep A>B | sep B>A | p_unpaired | paired Δ | p_paired |
|---|---|---|---|---|---|---|---|---|---|
| CartpoleBalance | `constant` | `pixels` | 0.0900 | 0.0859 | no | no | 0.63492 | 0.0041 | 0.75000 |
| CartpoleBalance | `constant` | `state` | 0.0900 | 0.0964 | no | no | 0.03968 | -0.0064 | 0.12500 |
| CartpoleBalance | `pixels` | `state` | 0.0859 | 0.0964 | no | no | 0.28571 | -0.0105 | 0.18750 |
| WalkerWalk | `constant` | `pixels` | 0.7576 | 0.7376 | no | no | 0.75397 | 0.0200 | 0.62500 |
| WalkerWalk | `constant` | `state` | 0.7576 | 0.7324 | no | no | 0.68254 | 0.0251 | 1.00000 |
| WalkerWalk | `pixels` | `state` | 0.7376 | 0.7324 | no | no | 0.80952 | 0.0052 | 0.81250 |

`state` / `pixels` / `constant` are the three vision conditions: privileged state only, state + live camera, state + a frozen (constant) image. `constant` isolates added CNN capacity from added information — that contrast is the whole point of RQ4.


## 4. RQ3 — specification pipeline disposition

- sealed `llm_pilot_disposition`: planned 6, executable 0, excluded 6; resampling-until-valid allowed: False
- sealed `llm_final_disposition`: planned 36, executable 0, excluded 36; resampling-until-valid allowed: False
- second cohort (`llm_v2`, different generator model): 18/18 specifications valid; see `llm.json` for per-spec attempts and skill names.
- second-cohort cells trained so far: 42
