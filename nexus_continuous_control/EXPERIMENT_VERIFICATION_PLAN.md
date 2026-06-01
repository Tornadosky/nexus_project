# NEXUS Continuous-Control Verification Plan

This file is the checklist for validating the continuous-control NEXUS implementation. Run the commands below, collect the generated result bundle, and send that bundle back for analysis. The purpose is not just to check that training runs, but to verify that the implementation matches the intended NEXUS deliverables:

- hierarchical continuous-control NEXUS / AC-PQN in JAX;
- neural, symbolic, and neuro-symbolic meta-policy variants;
- hand-written skill rewards and hand-written meta-policy or mask functions;
- sensible behavior on 4-5 MuJoCo Playground continuous-control environments;
- metrics and plots sufficient to diagnose failure modes and decide the next implementation changes.

The expected final artifact from your side is a zip named like:

```text
nexus_results_for_chatgpt_<YYYYMMDD_HHMM>.zip
```

It should be created with `tools/collect_nexus_results.py` after the runs finish. The zip should contain CSV metrics, plots, logs, config snapshots, environment information, and diagnostics.

---

## 0. Result acceptance criteria

The project is considered ready for the next step when the collected results show the following.

### Required implementation checks

- [ ] All unit tests pass.
- [ ] At least one MuJoCo Playground environment can be created and stepped through from the training script.
- [ ] Smoke training completes for `CartpoleBalance` and `PandaPickCube` without NaNs, shape errors, or JIT errors.
- [ ] Checkpoints and metrics are saved for every run.
- [ ] The result collector successfully creates CSVs, plots, diagnostics, and a review zip.

### Required algorithmic checks

- [ ] Environment reward curves are finite and not constant for all updates.
- [ ] Critic, actor, and meta losses are finite.
- [ ] Critic TD error does not explode monotonically.
- [ ] Noise and meta epsilon schedules decay as configured.
- [ ] Skill usage sums to approximately 1.0 at every update.
- [ ] No skill is completely unused for the whole run unless the symbolic policy makes that expected and explainable.
- [ ] At least some skill-specific rewards improve or stabilize above their early-training values.
- [ ] Symbolic runs use the hand-written rules only; NeSy runs use the hand-written mask plus learned meta-Q; neural runs use learned meta-Q without symbolic filtering.

### Required environment coverage

Minimum project-finished set:

- [ ] `CartpoleBalance`
- [ ] `CheetahRun`
- [ ] `WalkerWalk`
- [ ] `HopperHop`
- [ ] `PandaPickCube`

Extension / stretch set:

- [ ] `Go1JoystickFlatTerrain`

### Required comparison set

- [ ] `CartpoleBalance`: neural, symbolic, and NeSy variants.
- [ ] `PandaPickCube`: neural, symbolic, and NeSy variants.
- [ ] `CheetahRun`, `WalkerWalk`, `HopperHop`: NeSy variant at minimum.
- [ ] `Go1JoystickFlatTerrain`: NeSy variant if compute allows.

---

## 1. Install and environment capture

Run from the root of the unzipped repository:

```bash
cd nexus_continuous_control
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .[dev,analysis]
```

Install the MuJoCo Playground / PureJAXQL stack according to your machine. The previous package README contains the normal install commands. After installing, capture the environment:

```bash
mkdir -p runs/verification/env_info
python -V | tee runs/verification/env_info/python_version.txt
pip freeze | tee runs/verification/env_info/pip_freeze.txt
python - <<'PY' | tee runs/verification/env_info/jax_devices.txt
import jax
print('jax_version:', jax.__version__)
print('backend:', jax.default_backend())
print('devices:', jax.devices())
PY
nvidia-smi > runs/verification/env_info/nvidia_smi.txt || true
env | sort > runs/verification/env_info/environment_variables.txt
```

Check the package and hand-written policy registry:

```bash
python -m nexus_continuous.scripts.train_nexus_playground --list-policies \
  | tee runs/verification/env_info/policies.json
```

Run unit tests:

```bash
pytest -q tests | tee runs/verification/env_info/pytest.log
```

Checklist:

- [ ] `pytest` passes.
- [ ] `policies.json` lists `cartpole_balance`, `cheetah_run`, `walker_walk`, `hopper_hop`, `panda_pick_cube`, and `go1_joystick`.
- [ ] `jax_devices.txt` shows the expected accelerator.
- [ ] `pip_freeze.txt` is present.

---

## 2. Smoke tests: prove the train loop works

These are intentionally small. They are not expected to solve the tasks. They are for catching JIT, shape, dependency, wrapper, and save/load problems quickly.

Run:

```bash
bash tools/run_smoke_tests.sh
```

Equivalent explicit commands:

```bash
mkdir -p runs/verification/smoke runs/verification/logs

python -m nexus_continuous.scripts.train_nexus_playground \
  --config configs/cartpole_balance_nesy.yaml \
  --override TOTAL_TIMESTEPS=131072 \
  --override NUM_ENVS=64 \
  --override NUM_STEPS=32 \
  --override NUM_EPOCHS=1 \
  --override NUM_MINIBATCHES=4 \
  --override SEED=0 \
  --override PRINT_EVERY=32768 \
  --override SAVE_PATH=runs/verification/smoke/cartpole_balance_nesy_smoke_seed0.pkl \
  2>&1 | tee runs/verification/logs/cartpole_balance_nesy_smoke_seed0.log

python -m nexus_continuous.scripts.train_nexus_playground \
  --config configs/panda_pick_cube_nesy.yaml \
  --override TOTAL_TIMESTEPS=131072 \
  --override NUM_ENVS=64 \
  --override NUM_STEPS=32 \
  --override NUM_EPOCHS=1 \
  --override NUM_MINIBATCHES=4 \
  --override SEED=0 \
  --override PRINT_EVERY=32768 \
  --override SAVE_PATH=runs/verification/smoke/panda_pick_cube_nesy_smoke_seed0.pkl \
  2>&1 | tee runs/verification/logs/panda_pick_cube_nesy_smoke_seed0.log
```

After smoke tests:

```bash
python tools/collect_nexus_results.py \
  --runs runs/verification \
  --out runs/verification_review_smoke \
  --zip runs/nexus_smoke_results_for_chatgpt.zip
```

Smoke checklist:

- [ ] Both smoke commands finish.
- [ ] No traceback in `runs/verification/logs/*smoke*.log`.
- [ ] Both `.pkl` files exist.
- [ ] `runs/nexus_smoke_results_for_chatgpt.zip` exists.
- [ ] `runs/verification_review_smoke/diagnostics.md` has no fatal errors.

Stop here and send the smoke zip back if either smoke run fails.

---

## 3. One-seed integration runs: verify all environments before expensive runs

Run one seed for every required environment/config. This catches environment-name mismatches and task-specific policy function bugs.

Run:

```bash
bash tools/run_one_seed_matrix.sh
```

Equivalent explicit command pattern:

```bash
python -m nexus_continuous.scripts.train_nexus_playground \
  --config <CONFIG> \
  --override SEED=0 \
  --override SAVE_PATH=runs/verification/one_seed/<RUN_ID>_seed0.pkl \
  2>&1 | tee runs/verification/logs/<RUN_ID>_seed0.log
```

The exact one-seed matrix is:

| Run ID | Config | Overrides |
|---|---|---|
| `cartpole_balance_nesy` | `configs/cartpole_balance_nesy.yaml` | none |
| `cartpole_balance_symbolic` | `configs/cartpole_balance_symbolic.yaml` | none |
| `cartpole_balance_neural` | `configs/cartpole_balance_nesy.yaml` | `META_POLICY_TYPE=neural` |
| `cheetah_run_nesy` | `configs/cheetah_run_nesy.yaml` | none |
| `walker_walk_nesy` | `configs/walker_walk_nesy.yaml` | none |
| `hopper_hop_nesy` | `configs/hopper_hop_nesy.yaml` | none |
| `panda_pick_cube_nesy` | `configs/panda_pick_cube_nesy.yaml` | none |
| `panda_pick_cube_symbolic` | `configs/panda_pick_cube_symbolic.yaml` | none |
| `panda_pick_cube_neural` | `configs/panda_pick_cube_nesy.yaml` | `META_POLICY_TYPE=neural` |

Collect:

```bash
python tools/collect_nexus_results.py \
  --runs runs/verification \
  --out runs/verification_review_one_seed \
  --zip runs/nexus_one_seed_results_for_chatgpt.zip
```

One-seed checklist:

- [ ] All nine one-seed runs finish.
- [ ] Every run has a `.pkl` and `.log` file.
- [ ] No run has NaN or inf metrics.
- [ ] Skill usage plots exist for every run.
- [ ] The result collector diagnostics has no fatal errors.

Stop here and send the one-seed zip back if any full-default one-seed run fails.

---

## 4. Main experiment matrix: three seeds

Run this once smoke and one-seed integration pass.

```bash
bash tools/run_main_experiments.sh
```

The script runs three seeds, `0 1 2`, for the required matrix:

| Run ID | Config | Env | Variant | Default timesteps |
|---|---|---|---|---:|
| `cartpole_balance_nesy` | `configs/cartpole_balance_nesy.yaml` | `CartpoleBalance` | NeSy | 2,000,000 |
| `cartpole_balance_symbolic` | `configs/cartpole_balance_symbolic.yaml` | `CartpoleBalance` | symbolic | 2,000,000 |
| `cartpole_balance_neural` | `configs/cartpole_balance_nesy.yaml` | `CartpoleBalance` | neural override | 2,000,000 |
| `cheetah_run_nesy` | `configs/cheetah_run_nesy.yaml` | `CheetahRun` | NeSy | 5,000,000 |
| `walker_walk_nesy` | `configs/walker_walk_nesy.yaml` | `WalkerWalk` | NeSy | 5,000,000 |
| `hopper_hop_nesy` | `configs/hopper_hop_nesy.yaml` | `HopperHop` | NeSy | 5,000,000 |
| `panda_pick_cube_nesy` | `configs/panda_pick_cube_nesy.yaml` | `PandaPickCube` | NeSy | 10,000,000 |
| `panda_pick_cube_symbolic` | `configs/panda_pick_cube_symbolic.yaml` | `PandaPickCube` | symbolic | 10,000,000 |
| `panda_pick_cube_neural` | `configs/panda_pick_cube_nesy.yaml` | `PandaPickCube` | neural override | 10,000,000 |

Equivalent explicit commands for one run/seed:

```bash
python -m nexus_continuous.scripts.train_nexus_playground \
  --config configs/cartpole_balance_nesy.yaml \
  --override SEED=0 \
  --override SAVE_PATH=runs/verification/main/cartpole_balance_nesy_seed0.pkl \
  2>&1 | tee runs/verification/logs/cartpole_balance_nesy_seed0.log
```

For neural overrides, use:

```bash
python -m nexus_continuous.scripts.train_nexus_playground \
  --config configs/cartpole_balance_nesy.yaml \
  --override META_POLICY_TYPE=neural \
  --override SEED=0 \
  --override SAVE_PATH=runs/verification/main/cartpole_balance_neural_seed0.pkl \
  2>&1 | tee runs/verification/logs/cartpole_balance_neural_seed0.log
```

Collect final main results:

```bash
python tools/collect_nexus_results.py \
  --runs runs/verification \
  --out runs/verification_review_main \
  --zip runs/nexus_main_results_for_chatgpt.zip
```

Main-run checklist:

- [ ] All 27 runs finish: 9 run IDs × 3 seeds.
- [ ] `final_summary.csv` contains all 27 rows.
- [ ] `metrics_wide.csv` contains every run and every update.
- [ ] Plots exist under `plots/by_run/`, `plots/by_env/`, and `plots/aggregate/`.
- [ ] `diagnostics.md` reports no fatal errors.
- [ ] Upload `runs/nexus_main_results_for_chatgpt.zip`.

---

## 5. Stretch run: Go1 joystick quadruped

Only run this after the main matrix passes, or if you specifically want the quadruped robotics demo.

```bash
bash tools/run_go1_extension.sh
```

Equivalent explicit commands:

```bash
for SEED in 0 1 2; do
  python -m nexus_continuous.scripts.train_nexus_playground \
    --config configs/go1_joystick_nesy.yaml \
    --override SEED=${SEED} \
    --override SAVE_PATH=runs/verification/extension/go1_joystick_nesy_seed${SEED}.pkl \
    2>&1 | tee runs/verification/logs/go1_joystick_nesy_seed${SEED}.log
done
```

Collect including extension:

```bash
python tools/collect_nexus_results.py \
  --runs runs/verification \
  --out runs/verification_review_with_go1 \
  --zip runs/nexus_results_with_go1_for_chatgpt.zip
```

Extension checklist:

- [ ] All three Go1 seeds finish, or failed logs are included.
- [ ] Skill usage includes `stand`, `track_velocity`, `turn`, and `recover`.
- [ ] If Go1 is unstable, include the failed logs and partial metrics; do not delete them.

---

## 6. What the result zip must contain

The result collector creates this structure:

```text
nexus_results_for_chatgpt_*.zip
  MANIFEST.json
  diagnostics.md
  final_summary.csv
  metrics_wide.csv
  metrics_long.csv
  run_inventory.csv
  env_info/
    python_version.txt
    pip_freeze.txt
    jax_devices.txt
    nvidia_smi.txt
    environment_variables.txt
    policies.json
    pytest.log
  logs/
    *.log
  configs/
    *_config.json
  plots/
    aggregate/
    by_env/
    by_run/
```

Required plots:

- [ ] Environment return / reward curves.
- [ ] Episode length and done fraction curves when available.
- [ ] Per-skill usage curves.
- [ ] Per-skill reward curves.
- [ ] Actor, critic, and meta losses.
- [ ] Critic and meta absolute TD error curves.
- [ ] Noise and meta epsilon schedules.
- [ ] Final return bar plots grouped by environment and variant.
- [ ] Final skill usage bar plots grouped by environment and variant.

Raw checkpoints are not included by default to keep the zip manageable. If a run fails in a way that needs deeper debugging, also send the corresponding `.pkl` file separately, or rerun the collector with:

```bash
python tools/collect_nexus_results.py \
  --runs runs/verification \
  --out runs/verification_review_debug \
  --zip runs/nexus_debug_results_for_chatgpt.zip \
  --include-checkpoints
```

---

## 7. Hand-written policies to verify

### `CartpoleBalance`

Skills:

```text
recover_balance
center_cart
damp_motion
```

Expected qualitative pattern:

- early training may use all skills due to meta exploration;
- symbolic policy should prioritize `recover_balance` when pole angle/angular velocity is large;
- `center_cart` should appear when cart position drifts;
- `damp_motion` should appear when cart or pole velocity is high.

### `CheetahRun`

Skills:

```text
accelerate_forward
stabilize_posture
energy_efficient_run
```

Expected qualitative pattern:

- `accelerate_forward` should dominate early if velocity is low;
- `stabilize_posture` should activate for high pitch or high joint-speed states;
- `energy_efficient_run` should appear more once forward speed is high.

### `WalkerWalk`

Skills:

```text
stand_recover
walk_forward
stabilize_gait
energy_efficient
```

Expected qualitative pattern:

- `stand_recover` should activate when height/orientation is bad;
- `walk_forward` should be common until forward velocity improves;
- `stabilize_gait` should occur during unstable gait phases;
- `energy_efficient` should appear after stable walking emerges.

### `HopperHop`

Skills:

```text
stand_recover
hop_forward
stabilize_landing
energy_efficient
```

Expected qualitative pattern:

- `stand_recover` should appear after falls or low height;
- `hop_forward` should dominate when forward velocity is low;
- `stabilize_landing` should appear around high pitch / high joint speed;
- `energy_efficient` should appear once the hopping behavior is stable.

### `PandaPickCube`

Skills:

```text
reach_cube
grasp_cube
lift_cube
place_or_stabilize
```

Expected qualitative pattern:

- `reach_cube` early and when tool center point is far from cube;
- `grasp_cube` once near the cube but not grasped;
- `lift_cube` once grasped and cube height is low;
- `place_or_stabilize` once cube is lifted.

### `Go1JoystickFlatTerrain`

Skills:

```text
stand
track_velocity
turn
recover
```

Expected qualitative pattern:

- `recover` when base height or orientation is bad;
- `turn` when yaw command magnitude is high;
- `track_velocity` when XY command is nonzero;
- `stand` near zero command and stable body state.

---

## 8. How I will read your results

When you upload the result zip, I will check:

1. **Run health**: logs, dependency versions, device, wall-clock hints, missing files, NaNs/infs, terminated runs.
2. **Learning signal**: environment reward, returned episode return, episode length, done fraction.
3. **Skill learning**: per-skill reward curves and whether different skills get nonzero learning signal.
4. **Meta-policy behavior**: skill usage distribution, symbolic vs neural vs NeSy differences, skill collapse.
5. **Optimization stability**: actor/critic/meta losses, absolute TD errors, noise and epsilon decay.
6. **Deliverable coverage**: required environments, variants, seeds, exact configs used.
7. **Next changes**: whether to adjust rewards, masks, actor update mode, behavior penalty, network size, learning rate, or environment observation adapters.

---

## 9. Failure triage: what to send if something breaks

If installation fails, send:

```text
runs/verification/env_info/pip_freeze.txt
runs/verification/env_info/python_version.txt
runs/verification/env_info/jax_devices.txt
full terminal error
```

If an environment name fails, send:

```text
failed command
failed log
MuJoCo Playground version / git commit
output of any registry/list-env command available in your install
```

If training starts but diverges, send:

```text
log file
.pkl checkpoint for that run
collector zip with partial metrics
GPU model and memory
exact command used
```

If only one environment fails, keep all successful runs and include the failed logs in the same result zip.
