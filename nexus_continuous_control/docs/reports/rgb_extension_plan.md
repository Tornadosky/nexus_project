# RGB skill-agent extension — status & implementation plan

**Status: vision modules + the integrated `USE_RGB` trainer path are CPU-smoke-tested
with a fake pixel env (`tests/test_vision_rgb_smoke.py`); the real in-loop MJWarp
render path has NOT been executed end-to-end (GPU + warp-lang dependency, see
below). Treat any result at the shipped 200-update budget as a feasibility/
"does-anything-move" check, not evidence of learning.**

The mentor's optional extension is "use RGB inputs for the skill agents." This doc
reflects what the *vendored framework actually provides* (verified June 2026), the
privileged-critic design, and the precise integration points.

## ⚠️ Correction to the previous plan
The earlier version of this doc claimed the blocker was that "in-loop vectorized
MJX rendering is expensive and not currently emitted" and proposed hand-rolling a
render call into `_PlaygroundVecWrapper`. **That premise was wrong.** MuJoCo
Playground renders pixels **in-loop, in pure MJX via the MJWarp batch renderer,
with no Madrona dependency.** You enable it with a config flag, not custom code.

The real constraint is the opposite: rendering is easy, **availability is the
limit**. Among our envs, only **`CartpoleBalance`** implements the vision pipeline
(`default_vision_config` + `mjx.create_render_context` in
`vendor/.../dm_control_suite/cartpole.py`). `CheetahRun`/`WalkerWalk` have cameras
in their XML but **no render context** — supporting them means porting cartpole's
vision code (`default_vision_config`, the `_rc` render context, and a
`_dense_vision_reward`) into `cheetah.py` / `walker.py`.

## What is ready
- `nexus_continuous/vision.py`: `RGBEncoder` (3-conv CNN) and
  `VisionSkillActor(pixels, proprioception) -> action`. Shape- and vmap-verified
  (`tests/test_vision_shapes.py`). Normalization is now **dtype-driven** (DrQ-v2
  convention): integer frames → `/255 - 0.5`, float frames pass through unchanged
  (Playground already emits float grayscale ~[-0.5, 0.5]). The encoder ends in a
  `LayerNorm → tanh` bounded trunk and uses orthogonal conv init.
- `tests/test_vision_rgb_smoke.py`: drives the real `make_train`/`run_training`
  `USE_RGB` path on a fake pixel env (no renderer needed) for 2 updates + eval,
  asserting finite losses/returns. Covers the actor-apply pixel branch, the RGB
  init path, augmentation, the `(train_state, rng)` minibatch carry, the
  `_drop_actor_pixels` next_obs handling, and the eval pixel path.

## Implementation grounding (best-practice review, June 2026)
Reviewed against DrQ (arXiv:2004.13649), DrQ-v2 (2107.09645), Asymmetric
Actor-Critic (Pinto et al. 2018, 1710.06542), SAC+AE (1910.01741), and the
MuJoCo Playground vision recipe. Key conclusions:
- **The privileged-critic + actor-only-encoder design is a published, working
  pattern** (Pinto 2018; RSS-2024 "Agile Flight from Pixels"), not a
  misconfiguration. DrQ/DrQ-v2/SAC+AE *stop* the actor→encoder gradient, but that
  rule assumes a SHARED encoder feeding a pixel critic — it does not transfer
  here: our critic is state-only and has no encoder, so the actor's policy
  gradient is the only (and correct) source for the CNN.
- **DrQ random-shift augmentation is load-bearing** in this regime (it stands in
  for the encoder regularization the absent critic-loss would otherwise give).
  Keep it on the actor pixels only (the state critic has no pixels to augment).
- **Staged fallback if pixels-only stalls:** add an L2 pixel→state regression
  head (predict qpos/qvel from the encoder latent), NOT image reconstruction —
  it is dense, low-dimensional, and task-aligned (Pinto's "bottleneck" aux loss;
  SAC+AE deterministic-regressor recipe). Try WITHOUT it first.
- LR set to 1e-4 (DrQ-v2 scale) for the RGB config; a deterministic-policy-gradient
  actor that also trains the CNN is steadier at 1e-4 than the state task's 3e-4.

## How the framework emits pixels (verified)
- `cfg = registry.get_default_config("CartpoleBalance"); cfg.vision = True;
  cfg.vision_config.nworld = NUM_ENVS` → obs becomes `{"pixels/view_0": [N,64,64,3]}`
  float32, a 3-frame grayscale stack (so velocity is encoded in the image).
- `vision=True` also forces **`episode_length=250`**, `ctrl_dt=0.02`, and
  `_dense_vision_reward`. Vision-cartpole returns are on a **~0–30 scale**, not the
  state task's ~1000 — never compare the two return numbers directly.
- `nworld` is the render-context batch size and **must equal `NUM_ENVS`**.
- Our trainer already wraps envs with the *same* `wrap_for_brax_training`
  (`playground_adapter.py:491`) that the framework's own `vision.ipynb` uses for
  vision training — so the batch plumbing is already compatible. The pixel batch
  flows through `reset`/`step` as `state.obs["pixels/view_0"]` with a leading
  `[N, ...]` axis; `step` is **not** separately vmapped (the render context owns
  the batch).

## Recommended design (privileged critic)
Only the **skill actors** see pixels. The **critics, meta-Q, symbolic rules, and
skill rewards stay state-based** (privileged) — preserving interpretability and the
validated training machinery:

```
actor_i(pixels, proprio) -> action     # VisionSkillActor (pixels)
critic_i(state, action)  -> Q_i        # unchanged SkillCritic (state)
meta(state) -> skill ; skill_rewards(state) ; skill_mask(state)   # unchanged
```

Only the actor's *input* changes; targets, meta-Q, loss structure are unchanged.

## Step 0 — feasibility spike (DO THIS FIRST)
`nexus_rgb_feasibility_colab.ipynb` (project root) measures RGB env-steps/sec vs
`NUM_ENVS` and the render overhead factor, and confirms the pixel shape/range. The
numbers decide the feasible `NUM_ENVS` and whether to commit to in-loop training or
the distillation fallback. **Do not write integration code before this passes.**

## Integration points (all behind `USE_RGB`, default off)

1. **Enable vision in the env build** — `envs/playground_adapter.py` (~L480-496):
   when `config.get("USE_RGB")`, set `env_config.vision = True` and
   `env_config.vision_config.nworld = config["NUM_ENVS"]` before `registry.load`.
   Note the forced `episode_length=250`.
2. **`_PlaygroundVecWrapper._get_obs`** (L79-93): add a vision branch.
   `obs["pixels/view_0"]` has no state vector, so **reconstruct the proprio/state
   vector from `state.data`** (cart pos, pole cos/sin, qvel — the same quantities
   the non-vision `_get_obs` returns; `_semantic_state_info` already reads them).
   Emit: `{"actor_pixels": pixels, "actor": proprio, "critic": proprio,
   "raw_actor": proprio, "raw_critic": proprio, "policy_info": semantic}`.
3. **`get_actor_pixels(obs)`** — new helper next to `get_actor_obs`.
4. **Actor construction** (~L225): branch `SkillActor` → `VisionSkillActor` on
   `USE_RGB` (vmap over the skill axis exactly as today).
5. **`_actor_apply` / `one_skill_q` / `skill_actor_bootstrap_values` /
   `_select_action`**: when `USE_RGB`, call `actor.apply({"params": p}, pixels,
   proprio)` with `proprio = get_actor_obs(obs)`, `pixels = get_actor_pixels(obs)`.
6. **`init`** the vision actor with a dummy `(pixels, proprio)` pair.
7. **Normalization:** normalize the *proprio* vector as today; do **not** normalize
   pixels (the encoder handles their range).

## Suggested config
```yaml
USE_RGB: true
ENV_NAME: CartpoleBalance
NUM_ENVS: 256          # set from the spike; nworld is tied to this
ACTOR_HIDDEN_SIZES: [256, 256]
RGB_EMBED_DIM: 128
# episode_length is forced to 250 by vision=True; budget in updates accordingly.
```

## Validation order (on GPU)
1. Spike (`nexus_rgb_feasibility_colab.ipynb`) — render cost + obs sanity.
2. Overfit `CartpoleBalance` `USE_RGB=true` at small `NUM_ENVS`; confirm the vision
   actor learns *anything* (vision-scale return rises off the floor).
3. Compare RGB vs state skills on the same task. Expect slower convergence, not
   higher return — the paper frames RGB as a *feasibility* result, not a win.

## Fallback if in-loop rendering is too slow (or to cover cheetah/walker)
Train skills state-based (already working), then **distill** into `VisionSkillActor`
via behavioral cloning on rendered rollouts (offline, no in-loop render cost). This
still demonstrates "skills from pixels" and works on any env, including the
now-working CheetahRun.
