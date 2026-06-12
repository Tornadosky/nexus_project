# RGB skill-agent extension — status & implementation plan

**Status: vision modules validated; in-loop training wiring is a GPU-iteration task (not done blind).**

The mentor's optional extension is "use RGB inputs for the skill agents." The
building block exists and is shape-verified, but full integration into the
training loop must be done and tuned on a GPU. This doc explains exactly what is
ready, what is missing, and the precise integration points — so it can be
finished without re-deriving the design.

## What is ready
- `nexus_continuous/vision.py`: `RGBEncoder` (3-conv CNN) and
  `VisionSkillActor(pixels, proprioception) -> action`.
- `tests/test_vision_shapes.py`: confirms the encoder/actor produce correctly
  shaped, in-range actions **and vmap over the skill axis** exactly like the
  state-based `SkillActor` (the property the trainer relies on).

## Why it is not wired into the trainer yet (the honest blocker)
The actor is the easy half. The hard half is the **pixel source**: the training
loop runs ~1–2k MJX environments in parallel and the actor needs a rendered
`pixels` field in the observation **every step**. In-loop, vectorized MJX
rendering at that scale is expensive and not currently emitted by
`_PlaygroundVecWrapper`. Wiring a half-working pixel path into the hot loop
blind would risk the validated state-based trainer for no gain, so it is left as
a guarded, opt-in task with the design below.

## Recommended design (privileged critic)
Keep the symbolic/meta layer and the **critics state-based** (privileged), only
the **skill actors** see pixels. This preserves interpretability and the working
critic/meta machinery:

```
actor_i(pixels, proprioception) -> action      # VisionSkillActor (pixels)
critic_i(state, action)         -> Q_i         # unchanged SkillCritic (state)
meta-policy(state / symbols)    -> skill        # unchanged
```

Only the actor's *input* changes; targets, meta-Q, and the loss structure stay.

## Exact integration points (all behind a `USE_RGB` flag, default off)
File: `nexus_continuous/algorithms/hierarchical_ac_pqn_playground.py`

1. **Env obs must carry pixels.** In `envs/playground_adapter.py`
   `_PlaygroundVecWrapper`, add a `pixels` key to the obs dict (render the MJX
   state to a small frame, e.g. 64×64). This is the expensive, must-validate
   piece. Add `get_actor_pixels(obs)` next to `get_actor_obs`.
2. **Actor construction.** Where `actor = SkillActor(...)` is built (~L225),
   branch on `config.get("USE_RGB", False)` to build `VisionSkillActor` instead.
3. **`_actor_apply`.** Change to pass pixels + proprio when `USE_RGB`:
   `actor.apply({"params": p}, pixels, proprio)`. Proprio = `get_actor_obs`,
   pixels = `get_actor_pixels`.
4. **`skill_actor_bootstrap_values`** (module fn) and the **actor loss**
   `one_skill_q` both call the actor — thread pixels through the same way (they
   already receive `obs_actor`; add an `obs_pixels` arg).
5. **`_select_action`** — pass pixels to the actor apply.
6. **`init`** the vision actor with a dummy `(pixels, proprio)` pair.

Everything else (critics, meta-Q, Q(λ) targets, normalization of the *proprio*
vector, eval) is unchanged. Normalize proprioception as today; do **not**
normalize pixels (the encoder handles 0..255).

## Suggested config
```yaml
USE_RGB: true
RGB_HEIGHT: 64
RGB_WIDTH: 64
RGB_EMBED_DIM: 128
# start from a single env that renders cleanly (e.g. CheetahRun) and a SMALL
# NUM_ENVS until the render cost is measured.
```

## Validation order (on GPU / Colab)
1. Confirm the wrapper can emit a `pixels` batch and measure the per-step render
   cost (this decides whether in-loop RGB is feasible at the current NUM_ENVS).
2. Overfit one env (CheetahRun) with `USE_RGB=true`, tiny NUM_ENVS; check the
   vision actor learns *anything* (return rises) vs the state-based actor.
3. Compare RGB vs state skills on return/success; expect slower, not better —
   the paper frames RGB as feasibility, not a performance win.

## Fallback if in-loop rendering is too slow
Pretrain skills state-based, then **distill** into vision actors offline
(behavioral cloning on rendered rollouts): cheaper and avoids in-loop rendering.
This still demonstrates "skills from pixels" for the report.
