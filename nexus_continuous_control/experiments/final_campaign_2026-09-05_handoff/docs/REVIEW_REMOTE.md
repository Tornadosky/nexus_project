# Candid review: what was wrong, what is corrected, and what is unfinished

## Earlier assumptions that were too strong

**“The package is ready to run on the actual machines.”** The earlier package had only CPU/static validation. That did not establish remote installation, GPU restore, MJWarp rendering, production memory use, or a compatible deployed PPO implementation. The live audit found multiple concrete gaps. The package now records checks by level and retains installation/smoke gates instead of presenting a launch command as execution proof.

**“Use the latest code.”** A blind move to GitHub main would discard or bypass useful uncommitted core functionality. The local core includes the HPQN shared-reward control needed by the proposed matrix; the newer RGB tree serves another purpose. `freeze_sources.py` now preserves distinct core-working-tree and RGB-Git-object snapshots, without changing Git branches. Six critical hashes and the HEAD must still match the audit.

**“Up to 12 RGB runs can probably be reused.”** No RGB weights were found in the searched runs or tracked main tree. That discount is withdrawn. All 142 primary rows remain new unless an exact later reuse proof is supplied. File presence, same environment name, or same nominal budget is not enough.

**“Local and Viper runs can be pooled as one cohort.”** Their numerical stacks differ, and Viper's old code lacks the PPO launcher. All 112 non-RGB primary training rows now share a new Viper source/runtime deployment. WSL core is smoke/debug only. RGB stays within one verified NVIDIA stack.

**“Available WSL space makes local training safe.”** C: is 97% used with about 30 GB available. The logical size of a WSL filesystem is not physical backing capacity. The profile adds an explicit conservative guard; no cleanup is performed automatically.

## Code changes in this handoff

| Before | After / location |
|---|---|
| Generic repository/interpreter placeholders | Observed WSL and Viper profiles in `deploy/`; unverified LLM/RGB dependencies remain explicit gates |
| Source could drift behind imported packages | `freeze_sources.py`, source manifest, six-file contract, and imported-source checks |
| Completed filename could be mistaken for the intended run | `agent.py` binds run identity to source/config/spec/environment hashes before accepting a completed file |
| Any direct launcher could be used before installation | `INSTALLING` check in agent/common worker path and LLM entry point |
| CPU tests might be read as GPU readiness | `verify_installation.py` labels its result code-only; runtime smokes remain required |
| Uncontrolled simultaneous local jobs | Shared `/tmp/nexus_local_gpu.lock`, including generation |
| Old four-hour wall time inherited without measured runtime | Production request up to the verified 24-hour partition limit, one-hour smoke request; timing still unmeasured |
| Broad queue first | Specific row execution and same-task/method smoke receipts; dry-run default |

Intermediate NEXUS snapshots remain evaluation-only, not resumable optimizer state. This limitation was not fixed. The final full runner is saved, but a timed-out run may need an identical restart in a separate output root. No claim of automatic recovery is made.

## Remote operations stopped short of completion

Two connector operations were blocked. They were not retried through another access path. The first prevented reading probe summaries. The second prevented further audit copying/deployment work. The two Python diagnostic sessions were exited; no training was started.

The on-machine partial package and the downloadable portable package are not asserted byte-identical. Use the portable package in a fresh sibling; preserve the partial folder and original audit. Do not treat unread probe logs as passes, and do not claim the Viper package exists before it is copied and verified.

## What the tests actually establish

The supplied `validation.json` records assistant-sandbox tests only. Source/config checks against the uploaded reference can establish wrapper anchors and matrix reproducibility, not the behavior of the live GPU stack. The live checkpoint audit establishes deserialization, metadata, and byte identity, not that every policy restores or performs correctly.

The full code is in `ALL_CODE.md`; the entire phased command sequence is in `docs/RUNBOOK.md`. No instruction relies on “same as the previous answer.”
