# NEXUS final experiment protocol

Status: proposed, fixed scientific matrix; source-inspected and CPU/static-tested launcher.
GPU simulation/training has NOT been executed by the assistant. Freeze the actual upstream/vendor
commits and pass the hardware checks before the production wave. No changes to your source files
are made by the wrappers. The maximum is 142 research training cells and 7,073,792,000 training transitions; mandatory tiny
plumbing-only smoke checks are separate and never included in scientific results;
checkpoint-only evaluation and compilation add compute, not training steps.

## Scientific story and stopping rule

The question is not merely whether NEXUS earns more reward. It is whether a continuous-action
hierarchy exposes useful, behaviorally distinct controllers that can be inspected and redirected,
and whether automated specifications and images preserve or improve that benefit.

RQ1: Are the named skills genuinely useful controllers, or just multiple actor heads? Compare
matched-budget task learning, common skill objectives, and executable actor competence.
RQ2: What does transparent selection buy after learning is finished? Test frozen-weight actor
removal, mask removal, symbolic reselection, command simplification, and perturbations.
RQ3: Does a validated small-LLM specification pipeline produce useful skills, and does one
feedback revision outperform spending another generation call on a fresh proposal?
RQ4: Does a changing camera stream improve control beyond the added CNN pathway/capacity?

These are questions, not promised positive findings. A bounded campaign can settle the evidence
available for these questions; it cannot ensure narrow confidence intervals or favorable results.
Report failures, ties, trade-offs, and wide uncertainty. Do not train until a desired headline appears.
No new environments, hyperparameter searches, model leaderboards, pure-pixel systems, obstacle
courses, or extra reinforcement-learning algorithms are added after freeze.

## Matrix and budgets

| Block | Tasks | Conditions | Replication | Maximum training jobs |
|---|---|---|---|---:|
| Core | HopperHop, Go1JoystickFlatTerrain | flat AC-PQN, HPQN, NEXUS neural, symbolic, NeSy, PPO | seeds 0..4 | 60 |
| LLM hand reference | CheetahRun, WalkerWalk | manual NeSy | seeds 0..4 | 10 |
| LLM feedback pilots | CheetahRun, WalkerWalk | initial proposal | families 0..2; seed 900+family | 6 |
| LLM final comparison | CheetahRun, WalkerWalk | initial, one feedback revision, fresh resample | 3 families × RL seeds 0,1 | 36 |
| Controlled vision | CartpoleBalance, WalkerWalk | state, state+images, state+constant-images | seeds 0..4 | 30 |

Core Hopper: 117,964,800 transitions, 900 NEXUS updates. Core Go1: 32,768,000 transitions,
250 NEXUS updates. Save initialization and every 10% of training. The corresponding intervals
are 11,796,480 and 3,276,800 transitions. These are snapshots of ONE continuous training run,
not ten independently trained endpoints. Do not restart the optimizer or learning-rate schedule.

PPO is the external deep-RL reference already integrated in the repository and present in the
supervisor's original comparison. HPQN uses multiple actors with the same environment reward
and a neural meta-policy. It is the capacity/hierarchy control, not another hand-shaped method.
NEXUS neural versus HPQN tests the contribution of semantic skill rewards; neural versus NeSy
tests masks with the same skill specification; symbolic versus NeSy tests fixed versus learned
resolution. This is not a complete factorial study of every possible interaction.

LLM final budgets: 52,428,800 transitions per run (400 updates). Pilot budgets: 13,107,200
(100 updates). Vision: 2,048,000 transitions (250 × 128 × 64), preserving the controlled recipe.
The existing five-task exploratory reports remain descriptive background/appendix evidence;
do not pool their heterogeneous recipes into the new matched-budget comparison.

## State-training parameters

For all five internal core methods: 2048 parallel environments; rollout length 64; 4 epochs;
64 minibatches; actor and critic MLPs [256,256]; meta MLP [128,128]; ReLU; layer normalization;
2 critics with mean aggregation; actor init scale .01, critic/meta init scale 1; gamma .99;
skill lambda .65; meta lambda .8; learning rate linearly 3e-4 to 5e-5 across the FULL declared
training budget; max gradient norm 1; behavior penalty 5e-4; actor updates on all states;
observation normalization on, reward normalization off; meta-decision interval 1; JAX environment
implementation; no images. No skill-dependent gradient clipping or linspace noise.

Hopper exploration: action-noise scale .35 -> .05 over 100% of training; meta epsilon 1 -> .05
over 60%. Go1: noise .25 -> .02 over 80%; meta epsilon 1 -> .02 over 60%. These scales multiply
half the action range, not a torque in physical units. Cheetah/Walker LLM runs preserve the
canonical noise .35 -> .03 over 80%, meta epsilon 1 -> .02 over 60%. All remaining values are
explicitly preserved in the 142 supplied YAML files.

PPO Hopper: 2048 environments, unroll 30, batch 1024, 32 minibatches, 16 updates/batch;
lr .001, gamma .995, entropy coefficient .01, reward scale 10. Go1: 8192 environments,
unroll 20, batch 256, 32 minibatches, 4 updates/batch; lr .0003, gamma .97, entropy .01,
reward scale 1, max gradient norm 1; actor/value layers [512,256,128] with state/privileged_state
observation keys. Both: episode length 1000, action repeat 1, normalized observations,
11 evaluations including initialization, one reset per evaluation period, 64 native eval envs.
Other settings come from the pinned shipped configuration and are exported BEFORE training.
The wrapper refuses unsupported Brax arguments and verifies actual final step counts.
PPO is algorithm-budget matched, not network-size or wall-clock matched.

## Common evaluation

The same state evaluator, task-policy module, first-episode stopping convention, and frozen
normalizer are used for every method including PPO. `TASK_POLICY` is explicit; never inherit
`flat_baseline` as a task metric. Native PPO rewards are useful progress logs but are NOT the
final cross-method evaluator. Evaluation random seed 30000 is reserved for final tests.
Intermediate curve points: 64 episodes; final endpoint: 256. All use 64 parallel evaluation
environments and horizon 1000. Each curve panel therefore has 11 points per training seed.

Each saved point yields environment episode return, episode length, canonical task metrics,
physical diagnostics, and ALL hand-written skill rewards evaluated on that policy's transitions.
Multiplying each episode's mean per-step skill reward by its own length gives its common skill
return. A PPO controller is scored with the same four rewards even though it was not trained on
them. Do not compare one method's private training reward to another method's task reward.

Hopper: return, forward velocity, and fraction of steps satisfying the repository's upright/hop
criterion. Go1: return, linear velocity error in m/s, yaw error in rad/s, orientation criteria,
and the canonical thresholded tracking fraction. The latter is a fraction of steps, NOT the
probability of successfully completing an episode. Preserve the original definitions and names
in raw exports; use precise labels in figures. Actions squared are an effort proxy, NOT measured
mechanical energy or torque work. Do not pool raw returns across different environments.

## Frozen-weight skill and selector tests: no training

On HPQN and all three NEXUS variants, both focal tasks, all five seeds: evaluate native control
and force each of four actors for an entire rollout. Use 128 episodes per condition with the
same reset seeds. This makes a 4×4 matrix: executed actor versus canonical skill objective.
Include physical diagnostics alongside the reward matrix. HPQN rows must be labelled anonymous
heads 0..3: inherited code labels do not give them the semantics of hand-shaped skills. Report score per episode and per
step; survival/horizon differences can otherwise dominate the totals. A semantic label or usage
fraction alone is not competence evidence. Forced evaluation from ordinary resets is NOT a
proof of recovery from a fallen-state distribution or independent disentanglement.

On neural and NeSy: remove each actor separately from the selector, same 128 paired episodes.
The new selector never re-enables a removed actor. When no other eligible actor exists, it
relaxes the mask over remaining actors and logs the relaxation frequency. This is an explicitly
defined intervention, not a silent fallback to the deleted controller.

On NeSy only: same weights with unmasked learned meta-Q, and with the hand-written symbolic
selector. Label these `frozen selector interventions`; they are not independently trained neural
or symbolic baselines. Use 128 episodes. A deterioration under mask removal is evidence that
that trained controller depends on its mask, not proof of better optimal asymptotic learning.

## Distribution changes: no training

All six core methods, both tasks, five training seeds, 128 paired episodes each:
independent per-step Gaussian action-noise scales {0,.05,.10,.20}, clipped to action bounds.
Do not call these external pushes; they are actuation perturbations.

For Go1 add the following command-range vectors using the supported `command_config.a` override:
normal [1.5,.8,1.2]; stop [0,0,0]; half range [.75,.4,.6]; high range [2.25,1.2,1.8];
forward only [1.5,0,0]; yaw only [0,0,1.2]. The normal condition is the same noise=0 baseline.
These are distributions/ranges, NOT fixed commanded trajectories. Verify actual sampled command
magnitudes in the exported diagnostics. Small commands make a fixed success threshold easier;
show velocity/yaw errors in physical units and the thresholded fraction together.

Also evaluate flat-trained weights on Go1JoystickRoughTerrain, no adaptation. Explicitly confirm
observation dimensions/semantics and unchanged normalizer. The existing height feature is world-z,
not terrain-relative clearance. For rough terrain emphasize tracking error and the explicitly
named orientation fraction. Do not interpret the legacy height-based `no_fall_rate` as a robust
fall probability. Failed observation compatibility is an engineering failure, not a low score.

## LLM protocol

Use the repository-default model family Qwen/Qwen2.5-1.5B-Instruct, resolving one exact revision
before generation. Temperature .7, top-p .9, max new tokens 4096, no model sweep. These are NEW
frozen generation settings, not a claim to reproduce undocumented historical calls.

For each task generate three independent initial families with seeds 2000,2001,2002. Generate
fresh resamples with 3000,3001,3002 WITHOUT reading feedback. Train each initial family once at
the pilot budget and training seed 900+family. Evaluate that pilot on 64 validation episodes,
seed 20000. Generate exactly one feedback revision of the same initial JSON, using its pilot
summary, with seeds 4000,4001,4002. No selection based on final-test performance.

Train initial/refined/resample independently FROM SCRATCH at the final budget with RL seeds
0 and 1. Do not carry weights between changed skillsets. Preserve three skills in Cheetah and
four in Walker in every condition. Hand-written references use five seeds. Every final model
receives the same 256-episode final evaluation, seed 30000. Average the two RL seeds within
proposal family first; the generation replication count is THREE, not six. With so few proposal
families, show individual families and avoid claims of broad LLM superiority.

The validator rejects unsupported reward types, unavailable fields, ignored arguments, invalid
rule ASTs, non-finite constants and zero weights (the old interpreter converts zero to one).
One always-eligible skill avoids hidden all-false fallback. There are at most two bounded
syntax/type repair calls, recorded with raw responses. Failed proposals remain failures and are
not replaced with extra sampled families. Validation is a type/execution check, not proof of good
reward design. Report proposal validity as well as conditional training performance. If generation
fails, the end-to-end pipeline has failed; do not invent a refinement benefit.

Feedback costs matter: initial deployment uses one final-budget training; refinement also uses a
quarter-budget pilot. Report that extra RL cost, generation calls/tokens, and failed attempts.
The resampling condition can be charged the same pilot budget for an equal-allocation comparison,
but also show its actual cheaper execution cost since it does not use that pilot's feedback.
Do not claim that refinement is more sample-efficient merely because final policies get equal
training steps. Historical invalid skillsets can be audited without retraining, but the repaired
pipeline versus old figures is not an isolated validator ablation.

## Controlled image input

All arms use the SAME vision-enabled environment and state representation. State-only sets
RGB_ACTOR=false, not USE_RGB=false. Informative images use RGB_ACTOR=true, RGB_PROPRIO=full,
64×64 three-channel grayscale history, RGB_EMBED_DIM=128, augmentation pad 4. Critics and
meta-policy stay state-based; private skill encoders, no shared-encoder or pixel-meta extension.
The third arm preserves the identical CNN architecture but feeds zero image tensors throughout
training and evaluation. This is different from corrupting images only at evaluation. It matches architecture and parameter
count, not every aspect of functional capacity; do not make that stronger claim.

Use the existing 250-update, 128-environment controlled configs and seeds 0..4 on both tasks.
Run 64 paired 250-step windows under intact, frozen-first, replayed-image, shuffled-frame,
zero-image, and constant-action conditions where defined. The original harness may include
resets within each window: label the measurement `250-step window return` or `reward/step`,
not complete-episode return. State-only is evaluated under intact input only. The constant-image
CNN is still run through the pixel tests as a plumbing check; most should have no image effect.

Compare informative versus state-only AND informative versus constant-input CNN. Improvement
only over state-only does not isolate the changing images from the extra pathway. A corruption
drop shows input dependence, not useful visual information or sim-to-real generalization by
itself. Full state is already supplied; avoid an information-theoretic claim that images reveal
state that the actor otherwise cannot access. Keep the negative Cartpole result visible.

## Predeclared figure set

F1: real Go1 frames with rule truth values/meta-Q, plus a short skill timeline. Use training seed
0 and evaluation seed 41000. Show the first eligible ambiguous decision frames, separated by
at least 60 steps, rather than searching for the best policy seed. Failure frames are valid.

F2: Hopper and Go1 task-return learning curves plus aligned physical/task-metric curves against
actual transitions. Distinguish the six methods; show seed traces and training-seed uncertainty.
Supplement with the original-paper-style common skill-return curves in the appendix.

F3: forced-actor × objective heatmap, alongside paired actor-deletion effects. Display raw physical
units in annotations or a companion table. Any color normalization is within one objective only;
no synthetic aggregate of unrelated units. This is the mechanism/skill-competence figure.

F4: Go1 command-condition heatmap or paired degradation plot, and return-versus-tracking scatter.
This can reveal reward/behavior ranking reversals and the difference between robustness and
standing still. Put the full action-noise curves and Hopper robustness in the appendix.

F5: an extensions figure with family-paired initial/refined/resample outcomes and paired
state/pixels/constant-CNN outcomes. Include an actual camera frame and its corrupted counterpart.
Show all families/seeds. Move the full validity table, RGB corruption forest, and extra task curves
to the appendix. Five main-text pages do not have room for every generated figure.

Do not replace real simulation screenshots with generated illustrations. Do not make radar plots
of incomparable reward scales. Colored bars alone are not a scientific contribution.

## Replication, completion, and compute

Training seed is the replication unit for core/RGB; proposal family is the upper-level unit for
LLM generation. Show all seeds and bootstrap 95% intervals with 10,000 resamples where useful.
Episode count is not a substitute for independent trained policies. With five seeds (or three
families) intervals remain limited; a nonsignificant difference is not evidence of equivalence.
State the prespecified contrasts; do not inspect dozens of panels and report only the best one.

Reuse only checkpoints with weights, normalization, exact architecture/rewards/rules, full effective
config, actual budget, source/vendor identity, and appropriate seed. Reevaluate them with the new
metric contract. Final-only weights cannot reconstruct historical learning curves. The supplied
archive omits NEXUS weights/vendor trees; some LLM pickles contain only metrics. The plan's
maximum assumes no reusable weights; controlled RGB seeds 0..2 are the clearest reuse candidates
when their metadata and companion configuration can be verified.

Planning envelope: up to 1200 APU-hours and 240 NVIDIA GPU-hours, with roughly 30% reserved for
compilation/evaluation/infrastructure retries. This is an allocation, NOT a measured runtime.
Use up to eight independent one-APU jobs, not an unimplemented distributed trainer. Use NVIDIA
for the vision renderer; retain the proven JAX/ROCm/CUDA environments. Day 1: inventory, tiny
plumbing tests, full-size first production jobs for memory/throughput checks. Days 2–6: core,
LLM pilots/final stages, RGB. Days 7–8: checkpoint-only suites and one identical infrastructure
retry per failed job. Days 9–10: data lock, figures, writing. Numerical learning failures are
outcomes, not permission to tune or discard a seed. Do not consume the reserve on optional tasks.

Before launch, verify budget feasibility using observed full-size snapshot timings. If the
allocation/queue cannot support the declared matrix, change and record the protocol BEFORE
looking at scientific outcomes. No honest plan can guarantee wall time without this check.
Once frozen, absence of a favorable result is never a reason for additional training.

Final completion means the declared outputs or explicitly documented failures are accounted for,
raw tables/weights/configs/versions/prompts are archived twice, all claims match those data, and
the author-contribution/LLM-use disclosure and minimal upstream PR are complete. Those last
writing and software-delivery obligations are not replaced by running more experiments.
