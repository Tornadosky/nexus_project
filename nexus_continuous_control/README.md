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

## RGB skill-agent extension

The current implementation is state-based. To use RGB for skill actors, keep the
symbolic/meta layer state-based and replace `SkillActor` with an image encoder +
proprioception MLP. The safest first version is a privileged critic setup:

```text
actor_i(rgb, proprioception) -> action
critic_i(state, action) -> Q_i
meta-policy(state/symbols) -> skill
```

This preserves interpretable high-level rules while allowing the skill agents to
learn from pixels.

## Notes on observation features

The policy modules first look for semantic keys in `info`, such as `x_velocity`,
`tcp_pos`, `cube_pos`, `target_pos`, `base_height`, or `command_yaw`. If a key is
missing, they fall back to conservative observation indices. For a final paper
run, inspect the installed Playground environment's observation schema and, if
needed, tighten the feature extraction in the corresponding policy module.
