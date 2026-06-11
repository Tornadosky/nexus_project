# Env-reliability loop — working notes

Loop: hourly cron `86c0e182` at :07 (session-only; previous session's cron
`bd55f168` died with that session ~05:00 2026-06-10, stranding batch A after
3/10 runs and before batch B started). RELAUNCHED 23:35 2026-06-10: batch A
driver = bg task bt0jhadgi (resumed at panda s0 redo; skips the 3 done .pkl),
batch B driver = bg task bsrq5opd1 (pgrep-gated on batch A), file monitor =
task bdw5pig9g (.pkl completions + .err crash signatures). Goal: make all envs reliably learn
(walker walks, panda lifts, hopper reliable, go1 tracking, symbolic degeneracy).
Baselines = phase-2 final matrix (docs/reports/continuous_nexus_phase2_results.md)
and docs/reports/environment_debug_report.md.

## Key prior facts (don't re-derive)
- Training only in WSL distro **Ubuntu** (`wsl -d Ubuntu`), venv
  `.venv-wsl312/bin/python` inside repo (default distro Ubuntu-20.04 is wrong).
- Meta-Q trains on env reward; skill critics on hand-written skill rewards;
  meta re-decides every step (until META_DECISION_INTERVAL added, see below).
- Final-matrix budgets were tiny: walker/cheetah 38 updates, hopper 76,
  panda 152, go1 228 (updates = TOTAL_TIMESTEPS/NUM_ENVS/NUM_STEPS).
- Walker env reward = stand*(5*move+1)/6, move=0 below 0.5 m/s → env reward
  already strongly favors true walking; standing caps at ~0.17/step.
- Panda env reward = gripper_box + box_target*reached_box + posture terms;
  holding cube on table pays ~steady income; lifting pays more via box_target.
- Phase-2 baselines: walker neural ret 569/fwd-vel −0.01; panda nesy lift 0.177,
  neural 0.000; go1 nesy succ 0.229 no-fall 0.908 vel-err 0.844; cartpole
  symbolic ret 142.5 succ 0.052 (100% recover skill), nesy ret 336.8 succ 0.131.
- Hopper stock 400up 3-seed returns {9.6, 8.6, 276.9} (~1/3 succeed).

## Running experiments
- **Batch A** (background task bkwidwpqj, runs/loop_fix/batchA/, started 02:48):
  pure budget scaling, stock code. Queue: walker nesy s0 → hopper s0 → walker
  neural s0 → panda s0 → hopper s1 → walker nesy s1 → walker neural s1 →
  hopper s2 → panda s1 → go1 s0. Walker 400up; hopper 400up @2048envs +
  NOISE_FINISH=0.05 NOISE_DECAY=1.0 META_EPS_FINISH=0.05; panda 400up;
  go1 500up. Each writes .out/.err/.pkl.
- **Cartpole dampfix probes** (runs/loop_fix/probes/): code change applied
  (see below). seed0 symbolic/nesy done PRE-algorithm-edit (valid policy change,
  old algo); neural s0 + symbolic/nesy s1,s2 re-running (task bjfl3859u).

## Code changes this loop (uncommitted)
1. `policies/cartpole_balance.py`: recover skill reward gets −0.1|ang_vel|
   damping (recover actor used to overshoot forever → symbolic one-skill trap);
   symbolic rule rest-branch now → damp_motion (was 0/recover, contradicting
   explain_policy).
2. `algorithms/hierarchical_ac_pqn_playground.py`: new opt-in
   `META_DECISION_INTERVAL: K` — meta holds skill K steps (re-decides on done
   or when nesy mask invalidates held skill). Default 1 = behavior unchanged.
   Full pytest (43) green; CPU smoke for K=1 and K=4 green. NOT yet used in
   any GPU run; intended for walker (gait dithering) and panda-neural
   (lift never credited) if batch A shows budget alone insufficient.

## Probe results so far (cartpole dampfix, stock budget 30up, seeds 0-2)
- symbolic: ret {221,234,230} succ {0.099,0.126,0.124} → mean 228/0.116 vs
  baseline 142.5/0.052. Clear improvement, but usage still ~95% recover.
- nesy: ret {202,320,326} succ {0.064,0.036,0.143} → mean 283/0.081 vs
  336.8/0.131 — mildly down (same seeds as baseline); not conclusive.
- neural: s0 ret 232 succ 0.053 vs 271.8/0.083 — mildly down.
- KEY REFRAME: CartpoleBalance resets NEAR-UPRIGHT (±0.034 rad), yet eval
  angle_abs_mean is 1.5–2.1 → every variant drops the pole early and spends
  the episode fallen. 95% recover usage is then partly legitimate; the real
  problem is control quality at 30 updates. Budget probe launched:
  cartpole 150up (9.83M steps) × {symbolic,nesy,neural} × seeds 0-2 with
  dampfix in place (task bhfn52obl, files *_dampfix_150up_seed*.out).
  Decision: if 150up nesy/neural ≥ baselines and symbolic usage diversifies →
  adopt budget+dampfix; if nesy still below stock baseline → reduce damping
  to 0.03-0.05 or revert policy change.

## Results: cartpole 150up (5x budget) with dampfix, seeds 0-2
- nesy: ret {855,662,616} succ {0.70,0.39,0.25} mean 0.45 (baseline 0.131!).
- neural: ret {441,801,603} succ {0.29,0.47,0.37} mean 0.38 (baseline 0.083!).
- symbolic: ret {169,271,256} succ mean 0.107 — flat vs 30up, still ~93% recover.
- CONCLUSION: cartpole learned variants were undertrained; adopt ~150up budget.
  Dampfix regression concern moot. Symbolic rule thresholds retuned
  (urgent 0.20->0.35 rad, 1.5->2.5 rad/s; nesy meta picks damp in ~60% of
  "urgent" states) — probe running (task bloqa6ufa).

## Results: walker nesy 400up seed1 (batch A, done 2026-06-11 00:08)
- ret 322.8, stand 0.764, walk_succ 0.270, **fwd-vel −0.0095 ≈ 0**. Replicates
  seed0 (430/0.77/0.36/−0.005): budget alone does NOT produce locomotion on
  ANY seed so far — sway-in-place is the converged behavior. K10 skill-commit
  (batch B) is now the live hypothesis; if K10 also fails, next levers are
  stronger walk-skill speed reward or longer skill commitment K∈{20,50}.
- METRIC NOTE (truthfulness): walk_success_rate (instantaneous stand & vel>0.5,
  step-averaged) reads 0.27 while net velocity is 0 — it counts the forward
  half of each sway cycle. fwd-vel mean (= primary_goal_metric) is the honest
  number; never report walker walk_success without it. TODO (eval-only code
  change, cheap): add per-episode net-walk success = (episode mean fwd-vel >
  0.5) using the existing eval episode table, so "walks" means net locomotion.

## Results: walker neural 400up seed1 (batch A, done 2026-06-11 00:19)
- ret 866.6, stand 0.936, walk_succ 0.238, **fwd-vel −0.0008**. Budget probe
  CLOSED at 4/4 runs (nesy s0 −0.005, nesy s1 −0.0095, neural s0 +0.00002,
  neural s1 −0.0008): 10x budget reliably produces standing+swaying with
  exactly zero net locomotion, both meta types. Verdict: budget is NOT the
  walker limiter; the sway equilibrium is an attractor. Next decision comes
  from batch B walker K10 runs (skill-commit). If K10 fwd-vel also ~0, go to
  reward shaping (walk_forward skill speed term) — do not spend more compute
  on stock-config walker seeds.

## Results: walker nesy 400up seed0 (batch A)
- ret 430 (38up baseline 198), stand 0.77 (was 0.128), walk_succ 0.36,
  **forward_velocity_mean -0.005 — still sways, no net locomotion at 10x
  budget**. → skill-commit experiment triggered: walker nesy+neural 400up with
  META_DECISION_INTERVAL=10, seed0 (task bloqa6ufa, *_K10_seed0.out).

## Results: hopper explore-variant 400up seed0 (batch A)
- ret 141, hop_succ 0.188, fwd-vel 0.93 — LEARNED (stock seed0 at same budget
  collapsed to 9.6). Seeds 1,2 pending in batch A.

## Results: hopper explore-variant 400up seed1 (batch A, done 2026-06-10 23:57)
- ret 309.6, hop_succ 0.389, fwd-vel 1.24, upright 0.42, full 1000-step
  episodes. STRONG. Explore-variant now 2/2 learned seeds (vs stock 3-seed
  {9.6, 8.6, 276.9} = 1/3). Metric sanity ok: hop_succ ≤ upright, fwd-vel >
  success speed 1.0.
- If seed2 also learns (or even hops weakly), ADOPT explore overrides
  (NUM_ENVS=2048, NOISE_FINISH=0.05, NOISE_DECAY=1.0, META_EPS_FINISH=0.05)
  into configs/hopper_hop_nesy.yaml and document that the "bimodal collapse"
  was an exploration-schedule artifact, not intrinsic — the strongest possible
  answer to the phase-2 HopperHop drop.

## Results: hopper explore-variant 400up seed2 (batch A, done 2026-06-11 00:50)
- ret 129.8, hop_succ 0.171, fwd-vel 0.96, upright 0.20. LEARNED. **Explore
  variant final: 3/3 seeds hop** (ret {141, 310, 130}, succ {0.19, 0.39,
  0.17}) vs stock 1/3 ({9.6, 8.6, 276.9}). Sanity: hop_succ ≤ upright on all
  seeds; fwd-vel 0.93-1.24 consistent with partial-credit hop band.
- ADOPTED into configs/hopper_hop_nesy.yaml (TOTAL_TIMESTEPS=52428800,
  NUM_ENVS=2048, NOISE_FINISH=0.05, NOISE_DECAY=1.0, META_EPS_FINISH=0.05)
  with provenance comments. CONCLUSION for reports: the phase-2 "bimodal
  collapse" was an exploration-schedule artifact (premature noise decay), not
  an intrinsic env property; HopperHop can rejoin the main matrix with this
  config. environment_debug_report.md's "intrinsic instability" verdict needs
  a follow-up paragraph when finalizing (still true at OLD schedule; fixed by
  sustained exploration).

## Code changes round 2 (uncommitted, tests 43/43 green)
3. `policies/hopper_hop.py` symbolic rule: recover threshold height<0.9 →
   <ENV_STAND_HEIGHT(0.6) and wrap-safe cos(pitch) (old rule yanked control to
   stand_recover mid-hop-cycle → degeneracy). Symbolic-only change.
4. `policies/walker_walk.py` symbolic rule: wrap-safe cos(pitch) thresholds.
5. `tests/test_policy_obs_raw.py`: urgent-angle probe 0.30→0.50 (threshold
   retune made 0.30 non-urgent; test intent = raw-vs-normalized invariance).
- TODO when GPU frees: probe hopper symbolic 400up s0 + walker symbolic 400up
  s0 with fixed rules (compare vs "100% stand, succ 0" audit numbers).

## Code changes round 3 (uncommitted, 2026-06-11 ~00:15; pytest 43/43 + walker
## CPU eval smoke green)
6. `algorithms/hierarchical_ac_pqn_playground.py`: `_walker_episode_overrides`
   in deterministic eval — new `walker/net_walk_success_rate` = per-episode
   (mean fwd-vel > 0.5 & stand > 0.5); walker `primary_success_rate` now uses
   it (honest net locomotion); instantaneous `walk_success_rate` kept for
   baseline comparability. Eval-only, training-neutral; no-op for non-walker
   envs (key-gated). Applies to all batch B walker runs (batch A's walker runs
   already finished on old code — for those, read fwd-vel mean instead).
   When comparing batch B walker numbers to phase-2 baseline (0.118) use
   walk_success_rate, not the new primary.

## Ops incident (resolved ~05:25)
- The probes queue (walker K10 @0.15 mem slice) ran CONCURRENTLY with batch A's
  walker-neural from 03:14 → GPU thrash, both crawled ~2h with no progress.
  Killed probes queue; batch A continues alone. LESSON: do not run heavy
  (2048-env) jobs on a small slice next to batch jobs; cartpole-sized probes
  only.
- **Batch B** (task bux9mruus, runs/loop_fix/batchB/) gated on batch A driver
  exit, then sequential at full GPU: walker nesy/neural K10 400up s0;
  cartpole symbolic retune 150up s0-2; hopper symbolic rulefix 400up s0
  (explore overrides, 2048envs); walker symbolic rulefix 400up s0;
  panda neural antitrap (K10 + META_EPS_FINISH=0.15, DECAY=1.0) 400up s0.

## Results: panda nesy 400up seed0 (batch A, relaunched, done 2026-06-10 23:47)
- ret 445 (152up baseline 332), reach 1.0, closed_near_cube 1.0, but
  **lift_succ 0.023 vs baseline 0.177; cube_height_max 0.0366 (table=0.03),
  height_delta 0.0066**. Return UP, lift DOWN → 2.6x budget pushes the policy
  DEEPER into the table-income local optimum (gripper_box+reached_box steady
  income), it does NOT escape it. Metrics internally consistent (lift≈0 matches
  ~0 height delta; episode_length 146.7 sane).
- DECISION-TREE UPDATE: "panda budget helps" branch is dead. The limiter is the
  credit structure, not budget. Hypotheses now ranked: (1) batch B antitrap
  probe (K10 commit + sustained meta-eps, neural) — checks whether sustained
  lift attempts get credited once the meta can't dither back to grasp-income
  every step; (2) if antitrap fails: skill-reward shaping (bonus on height
  delta) or mask change forcing lift after sustained closed_near_cube; do NOT
  just add seeds at 400up. Also worth one probe: panda nesy ~150-200up may be
  the lift sweet spot (phase-2's 0.177 at 152up) — check before claiming any
  config "best".

## Results: panda nesy 400up seed1 (batch A, done 2026-06-11 01:1x)
- **lift_succ 0.344** (>0.3!), height_max 0.145 m (delta +0.115), place 0.008,
  ret 337, episode_len 124. Sanity ok (delta consistent with lift rate;
  reach/closed_near_cube 1.0).
- With seed0's 0.023 the 400up nesy picture is **seed-bimodal {0.02, 0.34}**:
  same discovery-vs-trap pattern as hopper (table-income local optimum instead
  of never-stands). Seed0's HIGHER return (445 vs 337) at near-zero lift
  confirms the env return mildly favors the trap on this horizon — never gate
  panda on return; gate on lift_success/height delta.
- Hypothesis (hopper-informed): sustained meta-exploration should de-bimodalize
  panda nesy like sustained noise fixed hopper. Batch B antitrap probe tests
  the neural variant (K10 + META_EPS_FINISH=0.15, DECAY=1.0). If it unsticks
  neural lift, run panda NESY with sustained meta-eps (no K10) x2 seeds before
  adopting anything.

## Results: go1 nesy 500up seed0 (batch A FINAL run, done 2026-06-11 00:48)
- ret 4.93, **no_fall 0.969** (phase-2: 0.908), tracking_succ 0.189 (phase-2:
  0.229), vel_err 0.892 (0.844), yaw_err 0.454 (0.445), episode_len 942.
  Sanity ok (len<1000 consistent with 3% falls).
- Verdict: 2.2x budget makes go1 MORE stable but NO better at command
  tracking — same "safe attractor" pattern as walker (sway) and panda-s0
  (table-hold). Budget branch dead for go1 too. Tracking needs either
  command-conditioned skill rewards w/ stronger vel-tracking term or
  K10-style commitment; park go1 until batch B's walker K10 verdict — if K10
  unlocks walker locomotion, a go1 K10 probe is the next cheap test.

## Results: walker nesy K10 400up seed0 (batch B run 1, done 2026-06-11 ~01:2x)
- ret 188.5 (K1 seed0: 430), stand 0.716, instantaneous walk_succ 0.152,
  fwd-vel −0.013, **net_walk_success 0** (new honest metric live in
  production, working as designed: primary_success now 0 instead of 0.15).
- K10 did NOT unlock locomotion for nesy and HALVED return — 10-step
  commitment seems to hurt the stabilization half of the policy without
  buying a gait. Await neural K10 (running) before declaring the K10 branch
  dead for walker.
- NEXT LEVER if neural K10 also fails: the hopper recipe was never tried on
  walker — sway is a discovery local-optimum exactly like hopper's
  never-stands. Cheap config-only probe: walker nesy 400up NOISE_FINISH=0.05
  NOISE_DECAY=1.0 META_EPS_FINISH=0.05 (queue after batch B; no code change).
  Reward shaping (walk_forward speed term) is the lever after that.

## Results: walker neural K10 400up seed0 (batch B run 2, done 2026-06-11 ~01:5x)
- ret 765, stand 0.949, fwd-vel −0.0012, net_walk 0 (instantaneous walk_succ
  0.318 — sway again). **K10 BRANCH DEAD for walker** (nesy: locomotion no +
  return halved; neural: locomotion no, return ~unchanged 765 vs 897).
- → BATCH C launched (bg task bdrss0u7u, runs/loop_fix/batchC/, gated on
  batch B driver; monitor byekzzvd8): walker hopper-recipe probe
  (NOISE_FINISH=0.05 NOISE_DECAY=1.0 META_EPS_FINISH=0.05, 400up, nesy
  s0/s1 + neural s0) and panda nesy sustained-eps (META_EPS_FINISH=0.05
  DECAY=1.0, seeds 0 & 2). Script: tools/run_loop_fix_batchC.sh.

## Results: cartpole symbolic retune 150up seed0 (batch B run 3)
- ret 233.4, succ 0.083, upright_frac 0.162, angle_abs 1.28. vs dampfix-only
  150up symbolic {169,271,256}/succ-mean 0.107: NOT an improvement so far
  (s0: 0.083 < 0.107). Pole still falls early (angle_abs 1.28). Await s1/s2;
  if mean ≤ dampfix-only, REVERT the threshold retune (keep dampfix + rest-
  branch→damp) and accept+document symbolic one-skill degeneracy as a
  fixed-rule-baseline limitation (the learned variants are the claim).

## Results: cartpole symbolic retune CLOSED (s1 0.076, s2 0.050; done ~02:3x)
- Retune final {0.083, 0.076, 0.050} mean 0.070 < dampfix-only 0.107.
  **REVERTED** thresholds to 0.20/1.5 in policies/cartpole_balance.py
  (comment documents the A/B); kept dampfix reward + rest-branch→damp (those
  ARE validated: symbolic 142.5→~230 ret, succ 0.052→~0.107). Test probe
  comment updated; pytest 43/43 green. Final cartpole symbolic story:
  dampfix+150up doubles the baseline but the fixed rule remains recover-
  heavy — document as baseline limitation, learned variants are the result
  (nesy 0.45 / neural 0.38 at 150up).

## BATCH A COMPLETE (7/7 relaunched runs done, driver exit 0, 2026-06-11
## 00:48). Summary: hopper FIXED (3/3, adopted); panda bimodal {0.02,0.34};
## walker budget-dead (4/4 sway); go1 budget-dead (stability up, tracking
## down). Batch B (K10/symbolic/antitrap) takes GPU next.

## Results: hopper symbolic rulefix 400up seed0 (batch B, done ~03:2x)
- ret 22.2, hop_succ 0.028, upright 0.095, fwd-vel 0.05 (explore overrides
  active). vs old-rule audit "100% stand, succ 0.000": the env-aligned
  wrap-safe rule yields nonzero hopping (0 → 0.028) — directionally right,
  still a weak baseline (nesy same recipe: 0.17-0.39). Chicken-and-egg
  stands: a fixed rule cannot bootstrap discovery. KEEP the rule fix
  (correctness + small gain); present hopper symbolic as weak-baseline,
  consistent with the cartpole symbolic story.

## Results: walker symbolic rulefix 400up seed0 (batch B, done ~03:5x)
- ret 644.6 (std 328!), stand 0.738, walk_succ 0.218, fwd-vel −0.007,
  net_walk 0. vs audit old rule "90% stand, succ 0.01": stands comparably,
  still zero locomotion. KEEP rule fix (correctness); walker symbolic stays
  weak-baseline like hopper/cartpole symbolic.
- **KEY REWARD INSIGHT (explains the sway attractor everywhere):** avg env
  reward 0.64/step with NET velocity 0 is only possible because the env's
  `move` tolerance term credits the positive half of each sway cycle and
  gives 0 (not negative) for the backward half — fast symmetric sway pays
  ~0.5-0.65/step vs ~1.0 for true walking but is trivially discoverable.
  So env RETURN is not a walking gate either; only net velocity /
  net_walk_success is honest. All four budget-scaled walker runs and this
  one converged to exactly this exploit. Discovery premium for true gait is
  only ~0.35/step — sustained exploration (batch C) must bridge it.

## Results: panda neural antitrap 400up seed0 (batch B FINAL, done ~04:1x)
- **lift 0.000**, height delta 0.0015, ret 456.7 (highest panda return yet —
  the MOST efficient table-farmer so far), reach/closed 1.0. K10 + sustained
  meta-eps (0.15, no decay) did NOT unstick neural panda. ANTITRAP-VIA-
  COMMITMENT BRANCH DEAD for panda neural. The neural meta has no mask to
  even expose lift as distinct — exploration alone can't credit it.
- Remaining panda hope: nesy sustained-eps (batch C s0/s2) — mechanism
  differs (mask forces reach→grasp→lift progression; sustained eps keeps
  sampling lift while masked-allowed). If that also fails to de-bimodalize,
  the conclusion is: panda lift needs the NeSy mask AND luck — report nesy
  multi-seed success fraction {0.02, 0.34, +batchC seeds} and the trap
  mechanics as a finding (return anti-correlates with lifting).

## Results: walker nesy explore 400up seed0 (batch C run 1, done ~04:4x)
- ret 231, stand 0.887, fwd-vel −0.0045, **net_walk 0**. The hopper recipe
  does NOT transfer to walker: sustained exploration alone cannot escape the
  sway exploit (it pays from step one; hopper's collapse paid nothing, which
  is why exploration sufficed there). Config-level levers now exhausted:
  budget 4/4 dead, K10 2/2 dead, explore 1/1 dead.

## Code changes round 4 (2026-06-11 ~04:5x; pytest 43/43 + CPU smoke green)
7. `policies/walker_walk.py` skill_rewards: walk skill now carries posture
   terms — `walk = 0.5*(height_reward+upright) + x_velocity + 0.5*speed_track
   - ctrl` (was posture-free). Rationale: a posture-free walk actor can only
   chase signed velocity by lunging/falling; meta learns to avoid the skill;
   policy settles into the sway exploit. Additive (not multiplicative) so a
   stumble retains recovery gradient. BATCH D launched (bg task bbsggbczf,
   monitor bpbodp7lx, gated on batch C): walker shaped+explore nesy s0/s1 +
   neural s0; pre-registered gate net_walk > 0.3 on >=1 seed.
- **LABEL CONFOUND (important):** batch C's two remaining walker runs
  (walker_neural_explore_400up_seed0, walker_nesy_explore_400up_seed1) start
  AFTER this code change, so they run the SHAPED reward despite their
  "explore" filenames — treat them as extra batch D samples, NOT as
  explore-alone evidence. Explore-alone evidence = batch C nesy seed0 only.

## Results: panda nesy sustained-eps 400up seed0 (batch C, done ~05:1x)
- **lift 0.109** (same seed stock: 0.023 — ~5x de-trapping), place 0.023
  (first nonzero on s0), height_max 0.053, ret 452.8. Sanity: place ≤ lift ≤
  closed_near_cube, height consistent with lift rate. Sustained meta-eps
  (FINISH=0.05, DECAY=1.0) keeps re-sampling lift while the mask allows it —
  exactly the predicted mechanism; the meta cannot fully deconverge into the
  table trap.
- Still below the lucky stock seed1 (0.344). Await susteps seed2 (queued):
  if it also lands ≥0.1, ADOPT sustained meta-eps into
  configs/panda_pick_cube_nesy.yaml and report panda nesy as "lifts on all
  seeds with sustained skill-exploration" (multi-seed mean), vs neural 0.000
  everywhere — the cleanest NeSy-mask evidence in the paper.

## Results: walker SHAPED neural 400up seed0 (batch C relabeled, done ~05:4x)
- (= first shaped+explore sample, neural meta) ret 930.5 — highest walker
  return of the whole campaign — stand 0.965, fwd-vel +0.001, **net_walk 0**.
  Shaping made sway MORE stable for the neural meta but did not unlock gait.
  One negative sample; the nesy shaped runs (mask focuses fallen-state data
  onto stand_recover, so walk-skill gradients are cleaner) are the remaining
  arbiters: batch C walker_nesy_explore_seed1 (shaped, mislabeled) + batch D
  nesy s0/s1.
- If all shaped nesy seeds also net_walk 0 → close the walker-walks goal as
  NOT ACHIEVED at this architecture scale; deliverable = the negative-result
  dossier (5 levers x N seeds, every run high-return/zero-locomotion, sway
  exploit mechanics) + walker presented as stand/stabilize success with
  locomotion limitation. That is a *good documented result*, not a shrug.

## Results: panda nesy sustained-eps seed2 + ADOPTION (done ~06:1x)
- **lift 0.172**, place 0.008, height_max 0.065, ret 372.7. Sanity coherent.
- Sustained-eps verdict: {s0 0.109, s2 0.172} — both ≥ 0.1 incl. the
  previously-trapped seed; stock was bimodal {0.023, 0.344}. Trades the
  lucky-seed ceiling for reliability. ADOPTED into
  configs/panda_pick_cube_nesy.yaml (TOTAL_TIMESTEPS=26214400,
  META_EPS_FINISH=0.05, META_EPS_DECAY=1.0) with provenance comments.
- Paper line: panda nesy lifts on every seed tested with sustained
  skill-exploration; panda neural lifts on NO seed at any budget/exploration
  setting (incl. K10+eps antitrap) — mask-dependence cleanly isolated.

## Results: walker SHAPED nesy s1 (batch C relabeled, FINAL batch C run,
## driver exit 0, done ~06:5x)
- ret 607, stand 0.870, walk_succ(inst) 0.337, fwd-vel −0.0044,
  **net_walk 0**. Shaped scoreboard: neural s0 0, nesy s1 0. Batch D
  (shaped nesy s0, neural s0 dup, nesy s1 dup) running next; walker closes
  as documented limitation if those also read net_walk 0.

## Results: walker SHAPED nesy s0 (batch D run 1, done ~07:2x)
- ret 323, stand 0.787, walk_succ(inst) 0.366, fwd-vel −0.005, **net_walk 0**.
- Shaped scoreboard: **0/3 distinct combos** (neural s0, nesy s0, nesy s1).
  Remaining batch D runs are nondeterminism replicas of failed combos —
  verdict effectively in unless one flukes >0.3. On batch D driver exit:
  write final walker verdict (documented limitation, 5-lever dossier) into
  docs/reports/env_reliability_campaign.md, update
  environment_debug_report.md hopper paragraph, decide walker shaping
  keep-or-revert (lean KEEP: posture-carrying walk reward is more correct,
  never regressed return/stand, and the comment documents the negative), run
  full pytest, summarize campaign.

## CAMPAIGN CLOSED (2026-06-11 ~08:0x). Batch D final run net_walk 0 →
## shaping 0/5; walker closed as documented limitation (5 levers, 11 runs).
## All drivers exited 0; monitors stopped; campaign report finalized
## (docs/reports/env_reliability_campaign.md); environment_debug_report.md
## hopper paragraph superseded with the 3/3 fix; pytest final pass below.
## Adopted configs: hopper (explore recipe), panda nesy (sustained-eps),
## cartpole x3 (150up). Kept code: cartpole dampfix+rest-branch, hopper/
## walker wrap-safe rules, walker net_walk metric + posture shaping.
## Reverted: cartpole threshold retune. Remaining (out of loop scope):
## commit the working tree; rerun final research matrix with adopted
## configs; go1 tracking has no validated lever (documented limitation).

## BATCH B COMPLETE (8/8, driver exit 0, 2026-06-11 ~04:1x). Kept: hopper+
## walker symbolic rule fixes (correctness; small or no gain). Reverted:
## cartpole threshold retune. Dead: walker K10 (both metas), panda neural
## antitrap. Live: batch C (walker explore recipe; panda nesy sustained-eps).

## Adopted: cartpole 150up budget (2026-06-11 ~03:1x)
- configs/cartpole_balance_{nesy,neural,symbolic}.yaml TOTAL_TIMESTEPS
  2000000 → 9830400 with provenance comments (validated probes: nesy 0.45,
  neural 0.38, symbolic 0.107). YAML parse-checked; pytest 43/43.
- NOTE for final matrix: flat-baseline cartpole runs must use the same
  budget for fair ratios (no cartpole_balance_flat.yaml exists — flat is
  presumably a META_POLICY_TYPE override; use the same config).

## Decision tree for next wakeups
- Batch A walker: fwd-vel ≥0.3 → budget was the issue, rerun more seeds &
  update configs; fwd-vel ~0 → run walker with META_DECISION_INTERVAL ∈ {5,10}
  and/or stronger walk-skill speed reward.
- Hopper explore-variant: ≥2/3 seeds return >150 → adopt config; else try
  2048envs + longer (600up) or reward discovery bonus.
- Panda batch A 400up: nesy lift >0.3 → budget helps, add seeds; neural still
  0 → test META_DECISION_INTERVAL=10 + META_EPS_FINISH=0.15 for neural.
- Cartpole dampfix: if nesy s1/s2 confirm regression → reduce damping to 0.03
  or revert; if symbolic stays degenerate → try META_DECISION_INTERVAL for
  symbolic too (hysteresis effect) or accept+document.
- Go1: await 500up result; compare succ vs 0.229.
