# LLM Skill Generation — findings (Goal 2)

Date: 2026-06. Can an LLM propose sensible interpretable skills + rewards for the
continuous-control NEXUS agent? Short answer: **yes, reliably.**

## Setup

- **Model / access:** Vertex AI `gemini-2.5-flash`, project from active gcloud
  config (`project-5b12e62b-…`), region `us-central1`, auth via
  `gcloud auth print-access-token`. (`gemini-2.0-flash-001` and
  `gemini-1.5-flash-002` return 404 in this project; 2.5-flash works.)
- **Tool:** `tools/llm_skill_gen.py` — fills the skill-proposal prompt
  (`nexus_continuous/llm/prompts.md`) with a per-env named-field state schema +
  task, calls Gemini in JSON mode (`responseMimeType=application/json`), parses
  into `nexus_continuous.llm.schema.NexusSkillSet`, and validates. **No LLM code
  is executed** — only the typed JSON skill spec is consumed, matching the repo's
  safety design.
- **Artifacts:** one generated skill set per env in
  `docs/reports/llm_generated_skills/*.json` (T=0.4 sample).

## Results

All 6 environments produced **3–5 schema-valid skills, 100 % of the time**, with
activation rules over the correct named fields and appropriate reward-term types.

| Env | LLM skills | vs hand-written |
| --- | --- | --- |
| PandaPickCube | reach_cube, grasp_cube, lift_cube, move_to_target | **matches** reach/grasp/lift/place exactly |
| HopperHop | recover_and_stand, maintain_stance, initiate_forward_hop, sustain_forward_hop | matches stand/hop family; **inferred the env's true 0.6 standing height** |
| CartpoleBalance | recover_pole, balance_pole, center_cart, maintain_stable | recover/center + an extra fine-balance skill |
| CheetahRun | sprint_forward, recover_balance, maintain_pace, stand_upright | accelerate/stabilize/efficient + idle |
| WalkerWalk | stand_up, maintain_balance, walk_forward, stop_and_stand | stand/walk/stabilize family |
| Go1JoystickFlatTerrain | stand_still, turn_in_place, navigate_linear, navigate_full | stand/turn/track + combined; mutually-exclusive by command |

**Quality highlights**
- **PandaPickCube**: the LLM independently reproduces the hand-written
  reach→grasp→lift→place decomposition, with sensible gates
  (`gripper_open`, `dist_tcp_cube<0.05`, `cube_height`) and reward types
  (`positive_velocity`/`target_height` for lift, `negative_distance` for reach).
- **HopperHop**: generated `recover_and_stand` activated by
  `torso_height < 0.6 OR abs(torso_pitch) > 0.7` — independently matching the
  env's actual `_STAND_HEIGHT=0.6` (the exact calibration that had to be debugged
  by hand for the success metric), and a clean stand→initiate-hop→sustain-hop
  progression.
- **Go1**: partitions behaviour by command type with mutually-exclusive
  activation rules — a clean, interpretable meta-policy.

**Robustness (various prompts/temperatures)**
- At T=0.4, 0.8 and 0.9 the decompositions are stable: Panda is essentially
  always reach/grasp/lift/move-to-target; Hopper is always a
  recover→maintain→initiate-hop→sustain-hop progression with the 0.6 threshold.
  Skill count occasionally varies (4 vs 5) but structure and reward types are
  consistent. No schema violations observed across ~14 generations.

## Assessment

The LLM is a viable skill proposer: it returns interpretable, schema-valid,
task-appropriate skills that match or sensibly extend the hand-written ones, and
it correctly infers environment-specific thresholds. This validates the repo's
LLM-NeSy design (JSON schema + reviewed conversion) end-to-end at the *proposal*
stage.

## Head-to-head training comparison (done)

`nexus_continuous/llm/interpreter.py` compiles a `NexusSkillSet` into runnable
JAX skill-reward / mask / symbolic-policy functions (safe AST evaluation of
activation rules over named fields; the reward vocabulary maps to JAX ops). The
LLM skills are then trained with the *same* trainer, env, config, seed and the
*same* (hand-written) `task_metrics`, so success is measured identically.
Validated by `tests/test_llm_interpreter.py` (5 tests). Scope: the four
scalar-feature envs (panda/go1 vector fields are future work).

GPU, nesy, seed 0, 180-300 updates:

| Env | LLM-skills success | hand-written success | LLM skill usage |
| --- | --- | --- | --- |
| Cartpole | **0.633** (ret 729) | 0.787 (ret 900) | diverse: 0.34/0.16/0.32/0.17 |
| Cheetah | **0.870** (ret 235) | 0.956 (ret 634) | run_fast 0.99 |
| Walker | 0.001 (ret 21) | 0.271 (ret 798) | **stand_up 1.00** (degenerate) |
| Hopper | 0.000 (ret 0.01) | 0.009 (ret 8) | **recover 1.00** (degenerate) |
| Panda | 0.000 (cube not lifted) | 0.16 (lift used 0.64) | **reach 0.99** (degenerate; lift never used) |
| Go1 | 0.003 (ret 0, falls) | 0.19 (ret 9) | follow_velocity 0.95 (NOT degenerate) |

The **Panda** case is the sharpest illustration of finding (2): the LLM's
`lift_cube` skill exists and is sensible, but its mask never unlocks it (grasp
requires `dist_tcp_cube<=0.05`, which the policy rarely reaches), so the cube is
never lifted. The interpreter's `field_fn` hook computes panda's derived scalar
fields (distances, cube height) from the adapter's position vectors so the same
scalar pipeline applies.

The **Go1** case isolates finding (3): the LLM does NOT degenerate here (it picks
`follow_velocity_commands` 95 % of the time, which is correct), yet still fails
(it falls -> return 0) purely because its reward terms do not keep the base
upright. So reward calibration is a limiter independent of the mask issue. All 6
environments are now covered by the comparison (`tools/llm_skill_compare.py`).

Two findings:

1. **LLM skills are trainable and competitive where the target regime is
   reachable** -- Cartpole 0.63 (vs 0.79) and Cheetah 0.87 (vs 0.96), with
   plausible multi-skill usage. The JSON->JAX pipeline works end to end.
2. **LLM activation rules over-restrict the NeSy mask.** The LLM writes
   *mutually-exclusive* activation rules (ideal for a symbolic priority policy).
   Used directly as the NeSy mask they collapse to the single "recover/stand"
   skill on walker/hopper, because while the agent is stuck "fallen" only that
   skill is unmasked (the same chicken-and-egg as the symbolic variant). Running
   the *identical* LLM skills in **neural** mode (no mask) restores diverse
   usage -- walker 1.00->0.57/0.15/0.04/0.24, hopper 0.997->0.42/0.19/0.09/0.30
   -- confirming the mask is the cause.
3. **LLM reward calibration is weaker than the tuned hand-written rewards.** Even
   with diverse neural-mode skill usage, the LLM-reward policies reach far lower
   env return (walker 11 vs 889; hopper 0.1 vs 39): the LLM rewards are a sound
   first draft but not as well shaped as the human-tuned ones.

**Mask-relaxation experiment.** `interpreter.make_policy_module(..., mask_mode=
"progressive")` allows, in addition to active skills, every skill up to
(highest-active-index + 1) -- so the meta can try the next step of a progression
instead of being stuck. On Panda this restored reach->grasp progression
(grasp usage 0.006 -> 0.22) but lift still rarely unlocked because the LLM's
grasp gate (`dist_tcp_cube<=0.05`) is too tight, so success stayed ~0. So mask
relaxation helps the *skill-usage* degeneracy partially; the remaining limiter
is the LLM's tight activation thresholds + reward weights.

**Prompt-refinement experiment (various prompts).** A `--style refined` prompt
adds explicit guidance: activation rules are an availability mask so make them
*permissive and overlapping* (generous thresholds, goal skills available early),
and reward terms must give a *dense* gradient. Re-generated and trained:

| Env | standard LLM | refined LLM | hand-written |
| --- | --- | --- | --- |
| Panda | lift usage 0.00, ret 42, succ 0 | **lift usage 0.83, ret 223**, succ 0 | lift 0.61, ret 329, succ 0.22 |
| Walker | succ 0.001, ret 21 | **succ 0.058, ret 170** | succ 0.287, ret 530 |

The refined prompt **directly fixes the mask-degeneracy** -- panda's `grasp` gate
loosened to `dist_tcp_cube<0.15` (matching the hand-written `GRASP_RADIUS`) and
`lift` became available early, so lift usage went 0 -> 0.83 and return 5x. Walker
success rose 50x (0.001 -> 0.058). Both move toward but do not match the
hand-written policy: the residual gaps are reward calibration and step sequencing
(refined panda now over-selects `lift` and skips `grasp`, so it lifts an empty
gripper and the cube still is not raised). So **prompt engineering is a strong,
cheap lever** on LLM-skill quality, and the LLM is a good proposer that still
benefits from a round of human reward/threshold tuning. Refined skill sets are
saved as `*_refined.json`.

Recommendations: (a) for LLM-in-NeSy, relax the mask compiled from LLM rules
(`mask_mode="progressive"`, and/or loosen activation thresholds by a margin);
(b) treat LLM output as a strong *starting point* for the hand-written policy --
it nails the skill decomposition, the named fields, and the rough thresholds
(it even inferred hopper's 0.6 standing height), and needs threshold/reward-weight
tuning to match the human-tuned policies. Net: the LLM is a viable skill *and*
reward proposer, and the full NEXUS LLM-NeSy loop is demonstrated end to end
(propose JSON -> compile to JAX -> train -> compare) across **all 6
environments**.
