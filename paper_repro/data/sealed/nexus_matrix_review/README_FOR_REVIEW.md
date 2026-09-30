# NEXUS final matrix — review pack

This zip is the complete numeric output of a pre-registered, sealed 142-cell
reinforcement-learning campaign, with the checkpoints stripped out. Everything a
reader needs to check the analysis or propose follow-up work is here; nothing in it
requires the 4.6 GB of model weights it was extracted from.

## What I want from a reviewer

1. **Are any of the pre-registered questions below already settled by these numbers,
   and are any of them NOT settled — i.e. which additional runs would actually change
   a conclusion, rather than narrow a confidence interval?**
2. **Which comparisons in `stats.json` / `SUMMARY.md` am I not entitled to make?**
   (Budget mismatches, unmatched denominators, metrics that measure something other
   than what their name suggests.)
3. **What is missing for a paper** — an ablation, a control, a baseline, a diagnostic
   — that is cheap relative to what has already been spent.
4. Concretely: **rank candidate follow-up runs by evidence gained per GPU-hour.**
   The machine is one RTX 5080; a core Hopper cell is ~13 min, a Go1 cell ~4 min, an
   LLM cell ~3 min, a vision cell ~18 min.

Please do not assume a positive result is wanted. Several headline claims in this
project have already reversed under more seeds, and reporting that a question is
unanswerable at this sample size is a useful answer.

## The pre-registered research questions

These were fixed before any cell ran, and no environment, algorithm or hyperparameter
search was added afterwards.

- **RQ1** — Are the named skills genuinely useful controllers, or just multiple actor
  heads? Matched-budget task learning, common skill objectives, and executable actor
  competence.
- **RQ2** — What does transparent selection buy *after* learning is finished? Frozen-weight
  actor forcing, actor removal, symbolic reselection, mask removal, command simplification,
  and perturbation.
- **RQ3** — Does a validated small-LLM specification pipeline produce useful skills, and
  does one feedback revision beat spending another generation call on a fresh proposal?
- **RQ4** — Does a *changing* camera stream improve control beyond the added CNN
  pathway/capacity? (Hence the `constant`-image control arm.)

## The design

| block | tasks | conditions | seeds | cells |
|---|---|---|---|---|
| Core | HopperHop, Go1JoystickFlatTerrain | flat AC-PQN, HPQN, NEXUS-neural, NEXUS-symbolic, NEXUS-NeSy, PPO | 0–4 | 60 |
| LLM hand reference | CheetahRun, WalkerWalk | manual NeSy | 0–4 | 10 |
| LLM feedback pilots | CheetahRun, WalkerWalk | initial proposal | 3 families | 6 |
| LLM final | CheetahRun, WalkerWalk | initial / one revision / fresh resample | 3 families × 2 seeds | 36 |
| Controlled vision | CartpoleBalance, WalkerWalk | state, state+images, state+constant-images | 0–4 | 30 |

Budgets: Hopper 117,964,800 transitions (900 updates); Go1 32,768,000 (250);
LLM final & reference 52,428,800 (400); LLM pilot 13,107,200 (100); vision 2,048,000 (250).
Snapshots at initialization and every 10% of one continuous run — **not** ten independently
trained endpoints; the optimizer and LR schedule are never restarted.

Contrast structure inside the core block: `neural` vs `hpqn` tests the contribution of
semantic skill rewards; `neural` vs `nesy` tests masks given the same skill specification;
`symbolic` vs `nesy` tests fixed versus learned resolution; `flat` is the single-actor
ablation and `ppo` the external reference. PPO is algorithm-budget matched, not
network-size or wall-clock matched.

## Evaluation, and what the metrics do and do not mean

The same state evaluator, task-policy module, first-episode-stopping convention and frozen
normalizer are used for every method **including PPO**. Native PPO reward logs are progress
information, not the cross-method score. Curve points use 64 episodes; endpoints use 256;
64 parallel eval envs, horizon 1000, eval seed 30000 reserved for final tests.

- Go1's `primary_success_rate` is the canonical thresholded tracking fraction: **a fraction
  of steps, not a probability of completing an episode.**
- Hopper's is the repository's upright/hop step criterion.
- `action_square_mean` is an effort proxy, **not** measured mechanical work or torque.
- Do not pool raw returns across environments.
- Every hand-written skill reward is evaluated on every policy's transitions, including
  PPO's, which was not trained on them — that is what makes the `common_skill_*` columns
  comparable across methods.

## Files

| file | what it is |
|---|---|
| `SUMMARY.md` | generated statistics: per-arm seed tables, every within-environment comparison, seed separation, exact permutation tests |
| `stats.json` | the same, machine-readable |
| `manifest.csv` | all 142 planned cells: identity, budget, status, actual steps, wall seconds, checkpoint hash |
| `final_metrics.csv` | endpoint metrics, long format (`run_id, step, metric, value`) — all ~40 metrics per run |
| `curves.csv` | all 11 snapshots per core run, same long format — learning curves for every metric |
| `probes.csv` | RQ2 frozen-weight interventions: `native`, `force_<k>` (run actor k for the whole rollout), `remove_<k>` (delete actor k from the selector), `selector_symbolic`, `selector_unmasked`. 128 paired episodes each |
| `shifts.csv` | RQ2 perturbation and command simplification: action noise 0/0.05/0.1/0.2 everywhere, plus Go1 `stop`/`forward`/`yaw`/`half`/`high`/`rough` command and terrain conditions |
| `rgb_ablation.csv`, `rgb_conditions.csv`, `rgb_curves.csv` | RQ4: per-run verdict fields and per-condition scores (`intact`, `frozen_first`, `random_replay`, `shuffle_frames`, `zeros`, `const_action`) |
| `llm.json` | RQ3: specification disposition for both cohorts — attempts, validity, skill names, which cells became executable |
| `llm_v2_metrics.csv` | present once the second LLM cohort has evaluated cells: same long format, with a `phase` column |
| `provenance.json` | source fingerprint, pinned package versions, readiness receipts, the matrix log's phase timeline |
| `figures/` | present only if packed with `--with-media`: rollout skill-usage strips and RGB ablation plots |

## Analysis conventions this project holds itself to

Stated so a reviewer can hold the numbers to them, and tell me where I have broken them.

- **Budget-matched comparisons only.** An arm beating another at a different step count is
  not a result. A cross-budget comparison is quotable only when the mismatch runs in the
  *baseline's* favour, and must be labelled mismatched every time it appears.
- **Seed separation over means.** Report `min(A) > max(B)` and an exact permutation p — not
  a t-test; the samples are small and non-normal. A mean win inside overlapping seeds is
  not a win.
- **Seed-matched beats sample size.** The same seed index across arms is far stronger than
  two independent samples.
- **Per-episode spread wherever the distribution is bimodal.** Several of these environments
  are all-or-nothing across seeds; a mean over such a distribution describes no episode that
  ever occurred.
- **n <= 3 is provisional**, said in the same sentence as the number.
- **Never report a number without its n and seed spread.**
- **Say plainly when a result reverses.** "Not proven" and "disproven" are different claims.

## Prior work this campaign supersedes but does not delete

An earlier exploratory phase ran roughly 660 training cells across seven environments
(CartpoleBalance, CheetahRun, WalkerWalk, HopperHop, PandaPickCube, Go1 flat and rough
terrain) with heterogeneous recipes and budget ladders up to 32x. Its headline findings:
the hierarchy beat its own flat ablation on the Go1 environments and hopper; a tuned
external PPO beat every NEXUS arm on six of seven environments at matched budget, losing
only on HopperHop; the symbolic (fixed-rule) variant was consistently the weakest arm; and
per-step re-selection failed on manipulation while a commitment interval fixed it.

Those recipes are **not** pooled into this matrix — that is deliberate, and it is why the
sample here is 5 seeds per cell on two focal tasks rather than a wide, uneven table. If
your recommendations depend on that earlier evidence, say so explicitly.
