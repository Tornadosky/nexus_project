# Experiment tracking: Weights & Biases (coexist design)

W&B is the **live** experiment-tracking layer. It does **not** replace the offline
results pipeline. The chain

```
pickle checkpoint -> tools/collect_nexus_results.py -> CSVs -> tools/phase2_validate_results.py
```

remains the **authoritative source of truth** for every research gate. W&B mirrors
the same metrics for dashboards and sweeps; if the two ever disagree, the offline
CSVs win.

## Why logging is post-hoc

`run_training` is `jax.jit` + `vmap`-over-seeds wrapping a `lax.scan` update loop,
so `wandb.log` cannot be called inside the loop. Training instead returns a stacked
metrics pytree (leading update axis; a seed axis when `NUM_SEEDS > 1`). After
training finishes, `nexus_continuous.tracking.log_training_run` replays that history
into **one W&B run per seed**, grouped by `<ENV_NAME>_<META_POLICY_TYPE>`.

The per-update reduction in `tracking/wandb_logger.py` is a deliberate mirror of
`tools/collect_nexus_results.py` (`_to_numpy` / `_series_from_metric`), so the live
curves match the CSVs leaf-for-leaf, including the `[updates, epochs, minibatches]`
and seed-axis shapes and the synthesised `env_step` axis.

## Enabling / disabling

Tracking is **on by default**. Disable it any of these ways (highest precedence
first):

| Method | Effect |
| --- | --- |
| `--no-wandb` CLI flag | off for that run |
| `WANDB: {enabled: false}` in config | off |
| `WANDB: {mode: disabled}` or `WANDB_MODE=disabled` | off |
| `WANDB_MODE=offline` / `WANDB: {mode: offline}` | **on**, logs locally for later `wandb sync` |

CPU smoke tests and CI should run with `--no-wandb` or `WANDB_MODE=disabled`.

Any W&B failure (missing package, no auth, network) is swallowed with a warning so
a training job is never broken by the tracking layer.

## Config block

All keys are optional; see the commented example in
`configs/go1_joystick_nesy_phase2.yaml`:

```yaml
WANDB:
  project: nexus-continuous-control   # or $WANDB_PROJECT
  entity: my-team                     # or $WANDB_ENTITY
  mode: online                        # online | offline | disabled
  group: Go1JoystickFlatTerrain_nesy  # default "<ENV_NAME>_<META_POLICY_TYPE>"
  tags: [phase2, robotics]
  enabled: true
```

The full resolved config plus the git `commit_hash` and `seed` are stored as
`wandb.config`. Deterministic-eval scalars are written to the run summary under
`eval/*`, with headline gate metrics (`primary_success_rate`,
`primary_goal_metric`, `episode_return_mean`, `mask/violation_rate`) also surfaced
at the top level for table/leaderboard columns.

## Hyperparameter sweeps

Sweeps run through `nexus_continuous.scripts.wandb_sweep_agent`, which merges the
agent's swept hyperparameters onto a base YAML config, trains, and replays history
into the active run. Example: `configs/sweeps/go1_joystick_nesy.yaml`.

```bash
wandb sweep configs/sweeps/go1_joystick_nesy.yaml      # prints a sweep id
wandb agent <entity>/<project>/<sweep_id>
```

The sweep optimises `primary_success_rate` (the same metric the offline gate uses).
The final pass/fail decision still comes from `phase2_validate_results.py`, not from
the W&B leaderboard.

## Relationship to the offline pipeline

- **W&B**: live curves, run comparison, sweeps, artifact lineage.
- **collector + validator**: deterministic CSVs and the authoritative gate verdict.

Both read the same training output, so adopting W&B costs nothing on the gate side.
