# Continuous-control NEXUS / hierarchical AC-PQN for MuJoCo Playground


## Verification / experiment checklist

Use `EXPERIMENT_VERIFICATION_PLAN.md` as the source of truth for the validation runs. It contains the smoke tests, one-seed integration matrix, three-seed main experiment matrix, exact commands, required metrics, expected plots, and the result zip format to send back for analysis. The helper scripts are:

```bash
bash tools/run_smoke_tests.sh
bash tools/run_one_seed_matrix.sh
bash tools/run_main_experiments.sh
bash tools/run_go1_extension.sh
python tools/collect_nexus_results.py --runs runs/verification --out runs/verification_review_main --zip runs/nexus_main_results_for_chatgpt.zip
```

This folder contains a complete JAX implementation of a continuous-control NEXUS
agent for MuJoCo Playground. It implements the deliverables:

- hierarchical NEXUS in JAX for continuous actions;
- DDPG-style Actor-Critic PQN low-level skills;
- neural, symbolic, and neuro-symbolic (`nesy`) meta-policy modes;
- hand-written reward functions and meta-policy / mask functions;
- configs for 6 MuJoCo Playground environments.

The code is intentionally structured as a standalone experiment package.

## Folder layout

```text
nexus_continuous/
  algorithms/hierarchical_ac_pqn_playground.py   # main training algorithm
  networks.py                                    # actor, critic, meta-Q modules
  returns.py                                     # Q(lambda) targets
  envs/playground_adapter.py                     # PureJAXQL Playground wrapper adapter
  policies/*.py                                  # hand-written skills/rewards/rules
  scripts/train_nexus_playground.py              # CLI training entry point
  scripts/eval_policy.py                         # inspect rules on synthetic states
configs/*.yaml                                   # env-specific training configs
```

## Installation

### Standalone installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

# CPU smoke tests / development
pip install -e .[dev]

# GPU / Playground training. Current Playground source requires Python 3.11+.
# Use the JAX CUDA wheel appropriate for your machine.
pip install -U "jax[cuda12]" --index-url https://pypi.org/simple
pip install git+https://github.com/google-deepmind/mujoco_playground.git
pip install -e .[dev]
```

MuJoCo Playground's source install is often preferable for the newest tasks:

```bash
git clone https://github.com/google-deepmind/mujoco_playground.git
cd mujoco_playground
uv venv --python 3.12
source .venv/bin/activate
uv pip install -U "jax[cuda12]" --index-url https://pypi.org/simple
uv --no-config sync --all-extras
```

On Ampere/Ada GPUs, set full matmul precision before training:

```bash
export JAX_DEFAULT_MATMUL_PRECISION=highest
```

## Run quick checks

```bash
pytest tests
python -m nexus_continuous.scripts.eval_policy --policy cartpole_balance --obs '[0.0, 0.3, 0.0, 0.0]'
python -m nexus_continuous.scripts.train_nexus_playground \
  --config configs/cartpole_balance_nesy.yaml \
  --override TOTAL_TIMESTEPS=65536 \
  --override NUM_ENVS=64 \
  --override NUM_STEPS=32 \
  --override NUM_MINIBATCHES=8
```

The last command is a smoke run. It checks that the environment, wrappers,
networks, rewards, and update loop compile. It is not enough training for a good
policy.

## Run the main environment suite

The recommended first run order is from easiest to hardest:

```bash
# 1. Fast sanity check
python -m nexus_continuous.scripts.train_nexus_playground --config configs/cartpole_balance_nesy.yaml

# 2. Locomotion tasks
python -m nexus_continuous.scripts.train_nexus_playground --config configs/cheetah_run_nesy.yaml
python -m nexus_continuous.scripts.train_nexus_playground --config configs/walker_walk_nesy.yaml
python -m nexus_continuous.scripts.train_nexus_playground --config configs/hopper_hop_nesy.yaml

# 3. Manipulation
python -m nexus_continuous.scripts.train_nexus_playground --config configs/panda_pick_cube_nesy.yaml

# 4. Larger robot demo
python -m nexus_continuous.scripts.train_nexus_playground --config configs/go1_joystick_nesy.yaml
```

Symbolic-only variants are included for Cartpole and Panda:

```bash
python -m nexus_continuous.scripts.train_nexus_playground --config configs/cartpole_balance_symbolic.yaml
python -m nexus_continuous.scripts.train_nexus_playground --config configs/panda_pick_cube_symbolic.yaml
```

Override `META_POLICY_TYPE=neural` to run the purely neural meta-policy version:

```bash
python -m nexus_continuous.scripts.train_nexus_playground \
  --config configs/walker_walk_nesy.yaml \
  --override META_POLICY_TYPE=neural
```

## Experiment tracking (Weights & Biases, coexist)

Live W&B tracking is on by default and logs one run per seed after training
finishes (post-hoc replay of the stacked metrics history). It is a *coexist* layer:
the offline `collect_nexus_results.py` -> CSV -> `phase2_validate_results.py`
pipeline stays authoritative for all research gates. Disable per run with
`--no-wandb`, or globally with `WANDB_MODE=disabled`. Sweeps run through
`nexus_continuous.scripts.wandb_sweep_agent` (see `configs/sweeps/`). Full details
in `docs/reports/wandb_tracking.md`.

```bash
# Tracked run (default)
python -m nexus_continuous.scripts.train_nexus_playground --config configs/cheetah_run_nesy.yaml
# Untracked (CI / smoke)
python -m nexus_continuous.scripts.train_nexus_playground --config configs/cheetah_run_nesy.yaml --no-wandb
```

## What the algorithm does

At every environment step:

1. All skill actors propose continuous actions.
2. The meta-policy selects a skill.
3. The selected skill action is perturbed with Gaussian exploration noise and clipped.
4. The environment step returns the real task reward.
5. The selected reward module also computes one reward per skill.
6. Skill critics are trained with Q(lambda) targets from skill rewards.
7. The learned meta-Q, when used, is trained with Q(lambda) targets from environment reward.
8. Skill actors maximize their own critics.

Meta-policy modes:

- `neural`: `argmax_i Q_meta(s, i)` plus epsilon exploration.
- `symbolic`: fixed hand-written priority rules, no meta-Q update.
- `nesy`: hand-written boolean skill mask, then `argmax` over masked meta-Q values.

The NeSy implementation uses `where(mask, q, -1e9)`, not multiplication. That is
important because continuous-control returns can be negative; multiplying masked
skills by zero can accidentally select an unavailable skill.

## Hand-written policies and rewards

### `CartpoleBalance` / policy `cartpole_balance`

Skills:

- `recover_balance`: reduce pole angle and angular velocity.
- `center_cart`: keep cart near the track center.
- `damp_motion`: reduce cart/pole velocities after balance is mostly recovered.

Symbolic rule:

```text
if |pole_angle| > 0.20 or |pole_angular_velocity| > 1.5: recover_balance
elif |cart_position| > 0.35: center_cart
elif |cart_velocity| + |pole_angular_velocity| > 1.0: damp_motion
else: recover_balance
```

### `CheetahRun` / policy `cheetah_run`

Skills:

- `accelerate_forward`: maximize forward speed.
- `stabilize_posture`: avoid extreme torso pitch and joint velocity.
- `energy_efficient_run`: keep speed while reducing action cost.

Symbolic rule:

```text
if |torso_pitch| > 0.55 or joint_speed > 8: stabilize_posture
elif forward_velocity > 6: energy_efficient_run
else: accelerate_forward
```

### `WalkerWalk` / policy `walker_walk`

Skills:

- `stand_recover`: keep torso high and upright.
- `walk_forward`: track a modest forward walking speed.
- `stabilize_gait`: reduce pitch and joint velocity spikes.
- `energy_efficient`: preserve walking with smaller torques.

Symbolic rule:

```text
if height < 0.85 or |pitch| > 0.45: stand_recover
elif forward_velocity < 1.2: walk_forward
elif |pitch| > 0.25 or joint_speed > 8: stabilize_gait
else: energy_efficient
```

### `HopperHop` / policy `hopper_hop`

Skills:

- `stand_recover`: recover height/uprightness.
- `hop_forward`: generate forward hopping velocity.
- `stabilize_landing`: reduce pitch and vertical/joint velocity spikes.
- `energy_efficient`: maintain speed with smaller torques.

Symbolic rule:

```text
if height < 0.9 or |pitch| > 0.45: stand_recover
elif forward_velocity < 1.5: hop_forward
elif |pitch| > 0.25 or joint_speed > 10: stabilize_landing
else: energy_efficient
```

### `PandaPickCube` / policy `panda_pick_cube`

Skills:

- `reach_cube`: move gripper/TCP near the cube.
- `grasp_cube`: close and secure the cube.
- `lift_cube`: raise the cube above the table.
- `place_or_stabilize`: move the lifted cube toward the target or hold it.

Symbolic rule:

```text
if distance(tcp, cube) > 0.06: reach_cube
elif not grasped: grasp_cube
elif cube_height < 0.12: lift_cube
else: place_or_stabilize
```

### `Go1JoystickFlatTerrain` / policy `go1_joystick`

Skills:

- `stand`: hold a stable quadruped stance.
- `track_velocity`: follow commanded planar velocity.
- `turn`: follow commanded yaw rate.
- `recover`: recover from low body height or large roll/pitch.

Symbolic rule:

```text
if base_height < 0.22 or |roll| > 0.6 or |pitch| > 0.6: recover
elif |command_yaw| > 0.15: turn
elif norm(command_xy) > 0.10: track_velocity
else: stand
```

## RGB skill-agent extension (implemented)

The RGB extension is implemented behind a single `USE_RGB` config flag (default
`false`). It is an **asymmetric / privileged-critic** design (Pinto et al. 2017):
the skill **actors** read camera pixels, while the critics, meta-Q, symbolic
rules, and skill rewards stay on privileged state — so the interpretable
high-level layer is unchanged and only low-level execution becomes pixel-based.

```text
actor_i(pixels, proprioception) -> action     # VisionSkillActor (CNN, nexus_continuous/vision.py)
critic_i(state, action) -> Q_i                 # unchanged SkillCritic (privileged state)
meta-policy(state/symbols) -> skill            # unchanged symbolic/nesy layer
```

- **Switch:** `USE_RGB: true` (config `configs/cartpole_balance_nesy_rgb.yaml`) or
  `--override USE_RGB=true`. When off, the state path is byte-identical.
- **Encoder:** dtype-driven normalization, orthogonal conv init, LayerNorm→tanh
  trunk (DrQ-v2 / SAC+AE conventions).
- **Env coverage:** the framework's own vision pipeline covers only
  `CartpoleBalance`; `nexus_continuous/envs/dm_control_vision.py` ports the same
  MJWarp render path to `CheetahRun`/`WalkerWalk`/`HopperHop`, so in-loop RGB
  training runs on all 4 environments (see `results/rgb/ablation/`).
- **Status:** the code is verified (`pytest tests/test_vision_rgb_smoke.py` drives
  the full `USE_RGB` training path on a fake pixel env). The **in-loop pixel RL**
  path (skills trained directly from MJWarp-rendered pixels) also **runs** on the
  student pool: install the version-aligned renderer `pip install mujoco-warp==3.11.0`
  (matches mujoco/mujoco-mjx 3.11), and `build_playground_env` applies two runtime
  shims automatically — `ensure_mjwarp_graphmode()` and `ensure_mjx_render_compat()`
  (the latter fixes a mujoco-mjx 3.11 `mjx.render` tuple-arity drift). The earlier
  Colab block was a mujoco/warp **version desync**, not a fundamental limitation.
- **IMPORTANT — "runs and completes" is not the same as "uses the camera."**
  A pixel-dependence ablation campaign (`nexus_continuous/scripts/rgb_pixel_ablation.py`:
  corrupt only the actor's image input and check whether performance survives)
  found the in-loop `CartpoleBalance` policy was **BLIND** — a state-based
  privileged meta-policy was solving the task alone, so the pixel actor never
  learned to see, despite training completing and scoring well. `CheetahRun`
  and `WalkerWalk` were independently verified **genuinely pixel-driven**
  (93-99% performance collapse when the camera is corrupted, confirmed on 3
  seeds each); `HopperHop` is inconclusive (score is noise around zero). A fix
  (auxiliary pixel→state loss + a longer meta-decision interval) was found and
  verified to repair the cartpole blindness, and separately rescued 3
  previously-saturated WalkerWalk skills. Full findings, every figure, every
  number: [`results/rgb/ablation/`](results/rgb/ablation/) and
  [`docs/reports/rgb_extension_team_briefing.txt`](docs/reports/rgb_extension_team_briefing.txt).
  The headline **distillation** result below predates and is unaffected by this
  finding (it was independently verified pixel-driven from the start, r≈0.99
  held-out fidelity) and remains a primary deliverable in its own right.

### Distillation result (report-grade, reproduced on GPU)

`nexus_continuous/scripts/rgb_distill_nexus.py` is the report deliverable. It
trains the **real** state NEXUS hierarchy (meta variant via `--meta`:
`nesy`/`neural`/`symbolic`), rolls it out to record `(rendered 64×64 frame,
meta-selected skill, action)`, behavior-clones each skill into a
`VisionSkillActor`, then runs a **closed-loop pixel-vs-state** comparison in which
the unchanged meta selects skills from privileged state and the pixel students act.
`nexus_continuous/scripts/rgb_report.py` aggregates the per-variant runs into one
figure + table.

Results (CartpoleBalance, 3 seeds each; artifacts in [`results/rgb/`](results/rgb/),
figure `results/rgb/comparison.png`):

| meta | state success | pixel success | retention | pixel-fallback |
|------|---------------|---------------|-----------|----------------|
| nesy     | 0.743 ± 0.221 | 0.345 ± 0.092 | 0.46 | 0.008 |
| neural   | 1.000 ± 0.000 | 0.520 ± 0.038 | 0.52 | 0.008 |
| symbolic | 0.311 ± 0.260 | 0.157 ± 0.094 | 0.51 | 0.023 |

Per-skill distillation is near-exact for well-sampled skills (BC MSE 1e-4–2e-3);
the ~50 % closed-loop gap is the expected **compounding-error / partial-
observability** cost of acting from a 3-frame 64×64 grayscale stack instead of
privileged state (Ross et al. 2011). Pixel-fallback < 2.5 % confirms the pixel
hierarchy is genuinely pixel-driven, not privileged-leaking. **Headline finding:
distillation retains ~50 % of privileged performance across all three NEXUS meta
variants — the RGB skill extension is meta-agnostic.**

`nexus_continuous/scripts/rgb_visualize.py` produces the qualitative artifacts for a
single run — an annotated rollout video, the 64×64 observation filmstrip, the
skill-activation timeline, and a held-out fidelity scatter (see `results/rgb/viz/`).

## Reproducing results for the report (graphs + videos)

```bash
# 1. Code-correctness gate (no GPU): the RGB training path + vision modules.
JAX_PLATFORMS=cpu pytest tests/test_vision_rgb_smoke.py tests/test_vision_shapes.py

# 2. State-based NEXUS training (the main deliverable) — saves metrics to plot.
python -m nexus_continuous.scripts.train_nexus_playground \
  --config configs/cartpole_balance_nesy.yaml --save runs/cartpole_state.pkl
# metrics live in the checkpoint: load runs/cartpole_state.pkl -> ['metrics'] and plot
# 'rollout/episode_return', and ['eval_metrics'] for the deterministic eval summary.

# 3. RGB pixel-skill result (report-grade, headless GPU): train the real NEXUS
#    teacher, distill to pixel skills, and compare pixel-vs-state — per variant.
for META in nesy neural symbolic; do
  python -m nexus_continuous.scripts.rgb_distill_nexus \
    --config configs/cartpole_balance_${META}.yaml --meta ${META} \
    --seeds 0,1,2 --out runs/rgb_nexus_${META}
done
python -m nexus_continuous.scripts.rgb_report --runs runs \
  --metas nesy,neural,symbolic --out runs/rgb_report   # -> comparison.png + table

# 3b. Qualitative artifacts (annotated video + skill timeline + filmstrip) for one run:
python -m nexus_continuous.scripts.rgb_visualize \
  --config configs/cartpole_balance_neural.yaml --meta neural --seed 0 --out runs/rgb_viz
```

### Headless rendering on a shared GPU (no display, no root)

CartpoleBalance needs an offscreen GL context. When the machine's GPU render node
(`/dev/dri`) is not group-accessible and you have no sudo (typical on a shared
student pool), skip EGL-on-GPU and OSMesa (dropped from recent `mesalib`) and use
**software EGL (llvmpipe)** — a pure-CPU offscreen renderer that coexists with
jax-on-GPU in the same process. Install userspace Mesa once via micromamba, then:

```bash
micromamba create -n mesa -c conda-forge mesalib -y   # ships libEGL_mesa + llvmpipe
M=$HOME/micromamba/envs/mesa
export MUJOCO_GL=egl LD_LIBRARY_PATH=$M/lib:$LD_LIBRARY_PATH
export LIBGL_DRIVERS_PATH=$M/lib/dri MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
export __EGL_VENDOR_LIBRARY_DIRS=$M/share/glvnd/egl_vendor.d EGL_PLATFORM=surfaceless
# now `MUJOCO_GL=egl` renders on CPU; jax still uses the GPU (CudaDevice).
```

## Colab notebooks (`notebooks/`)

| Notebook | Purpose |
|---|---|
| `run_environments_colab.ipynb` | Train the NEXUS suite — select 1 env or all 6; per-env curves + eval table. |

(The RGB extension now runs headless on a GPU via the scripts above; see the
"RGB skill-agent extension" section and `results/rgb/`. Code-correctness is gated by
`pytest tests/test_vision_rgb_smoke.py tests/test_vision_shapes.py`.)

Open them in Colab (GPU runtime); each clones this repo and runs end-to-end.

- **Graphs:** `rgb_distill_nexus.py` → `results/rgb/comparison.png` + per-variant
  `state_vs_pixel.png`; `rgb_visualize.py` → skill-activation timeline, observation
  filmstrip, held-out fidelity scatter. State training curves: plot the `.pkl` `metrics`.
- **Videos:** `rgb_visualize.py` → `rollout_pixel.mp4` / `.gif` (the pixel hierarchy
  acting, each frame annotated with the active skill).
- **Honest scope for the writeup:** the RGB result has two independently-verified
  legs. (1) A *quantitative distillation study* — the real NEXUS hierarchy's
  disentangled skills are behavior-cloned to 64×64 pixels and retain ~50 % of
  privileged closed-loop success across all three meta variants on CartpoleBalance
  (see `results/rgb/`), verified genuinely pixel-driven (r≈0.99 held-out fidelity).
  (2) *In-loop pixel RL* (skills trained directly from MJWarp pixels) works on
  CheetahRun and WalkerWalk (verified pixel-driven, 3 seeds each) but the
  CartpoleBalance in-loop result was found BLIND by ablation and required a
  targeted fix to become genuinely pixel-driven — see
  `results/rgb/ablation/` and `docs/reports/rgb_extension_team_briefing.txt`
  for the full campaign. In both cases the interpretable meta-policy remains
  state-based by design (privileged-critic asymmetry).

## Notes on observation features

The policy modules first look for semantic keys in `info`, such as `x_velocity`,
`tcp_pos`, `cube_pos`, `target_pos`, `base_height`, or `command_yaw`. If a key is
missing, they fall back to conservative observation indices. For a final paper
run, inspect the installed Playground environment's observation schema and, if
needed, tighten the feature extraction in the corresponding policy module.
