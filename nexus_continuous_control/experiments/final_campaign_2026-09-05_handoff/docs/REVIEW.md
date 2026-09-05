# Candid review and exact changes

The earlier paper was organized around an implementation/evidence audit. That exposed real
problems but did not establish an interesting scientific mechanism. Repeated endpoint bars,
weakly connected questions, and discussion of skill names/usage were not enough. The replacement
campaign makes learning, executable skills, frozen-weight steering, and controlled automation/
perception the core of the paper. It does not promise that NEXUS wins every comparison.

## Source-derived findings versus new work

| Existing source / behavior | New campaign code / interpretation |
|---|---|
| `scripts/train_nexus_playground.py` saves after training only | `train_state.py` adds actor/meta/normalizer snapshots in a loaded copy of `make_train`; one continuous optimization schedule; final full runner preserved |
| `tools/train_ppo_baseline.py` can inherit `flat_baseline` as task policy, stores shipped rather than fully effective configuration, and has no budget-exact schedule | `train_ppo.py` exports effective kwargs, uses 11 eval points/one reset, and asserts actual final steps; common evaluator requires explicit TASK_POLICY |
| `tools/robustness_eval.py` returns summary only and broadcasts batch skill shares | `evaluate.py` preserves per-episode data and true per-episode skill counts; all policies receive the same hand-written skill reward functions |
| Legacy actor-deletion fallback can let an excluded skill back into an empty mask | New deletion permanently excludes it and measures relaxation over remaining actors |
| `llm/interpreter.py` silently maps unsupported reward types/unknown fields to ineffective values; weight=0 becomes 1 | `llm_specs.py` validates vocabulary/fields/rules and rejects ignored arguments/zero weights BEFORE execution; existing DSL semantics are preserved |
| Historical refinement has one trajectory and a different initial proposal from the five-seed initial comparison | Same-family initial/refined/resample protocol, fixed skill count and same final evaluator; generation family is the replication unit |
| `origin-main/.../rgb_pixel_ablation.py` provides a genuine vision-environment-matched state control | Keep it. Add a same-CNN constant-image training control; preserve state critics/meta |
| RGB harness can continue across resets within a fixed horizon | Label results as 250-step windows, not first-episode returns. Do not silently redefine historical numbers |
| Go1 rough-terrain height is world-z, not local terrain clearance | Rough transfer reports tracking errors and named orientation criteria, not a claimed fall probability |

No original files are overwritten. All experiment wrappers are NEW code. The optional loaded
source instrumentation fails when its anchor is not unique, rather than guessing how to patch an
unknown revision. Generated files are saved under new directories and use no-overwrite or atomic
write policies. Progress logs may be atomically updated inside their own new run directory.

## Validation boundary

The matrix arithmetic, Python syntax, schema rejection, no-overwrite behavior, and source anchors
were checked in the assistant container. That container does not contain the supplied vendor
submodules or a usable project GPU simulator stack. Training/evaluation, ROCm execution,
NVIDIA rendering, and Brax checkpoint reconstruction are NOT claimed to have been runtime-tested.
The runbook's smoke checks are mandatory before allocating the production wave. Compact snapshots
support evaluation, not resumption; the wrappers do not implement a mid-run resume facility.

## Source basis

Uploaded `nexus_llm_context.zip`:
- `code/current/nexus_continuous_control/` for state trainer, hand policies, PPO and robustness tools.
- `code/origin-main/nexus_continuous_control/` for the controlled vision implementation.
- `code/llm-extension/nexus_continuous_control/` and the LLM report for historical generation/refinement limitations.
- Archive README for omitted weights/vendor submodules.

Uploaded *From Objects to Skills: Interpretable Metapolicies for Neural Control*: Figures 3–8,
Sections 4.2–4.3, discrete-action limitation, and Appendix E motivate common skill scores,
real rule/decision visualizations, return-versus-aligned-goal reporting, frozen-weight changes,
and disclosure of manual adaptation of generated specifications. They do not establish that the
same benefits transfer to robotics; that is what this campaign tests.

Current external primary sources checked September 5, 2026:
- MPCDF Viper-GPU User Guide: https://docs.mpcdf.mpg.de/doc/computing/viper-gpu-user-guide.html
- Brax PPO implementation: https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py
- Qwen model card: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- MuJoCo Warp hardware/backend scope: https://github.com/google-deepmind/mujoco_warp

Live upstream pages are external reference material, NOT a replacement for pinning the actual
installed vendor revisions used in the campaign.
