# Live project audit — 5 September 2026

## Outcome and evidence boundary

The repository/GitHub, WSL runtime, checkpoint metadata, and existing Viper deployment were inspected through Remote Desktop Commander. The audit is complete enough to identify the required sources, available weights, and operational blockers. **The new remote installation is not complete and is not cleared for training.** Two connector operations were blocked. Remote writes stopped; the partial repository package retains its `INSTALLING` guard.

This report transcribes successful tool responses. `audit/observed_state.json` and `audit/github_heads.csv` preserve the key observations. They are not substitutes for the full original remote logs. The raw completed audit remains at:

```text
/home/smirn/nexus_campaign_audit_2026-09-05_16-08/
    initial_audit.json
    checkpoint_files.json
    checkpoint_metadata.jsonl
    checkpoint_scan.log
    environment_probe.log
    installed_simulator.json
    viper_inventory.json
    viper_details.json
```

The full per-file checkpoint JSONL contains original paths, SHA-256 values, configurations, actor update/timestep counters, parameter shapes, metric shapes, and recorded normalization status. **It was not copied back into this download.** There is no fabricated 913-row checkpoint index here. The optional summarizer is supplied, but was not run after the connector block.

## 1. Git: no single branch contains everything needed

The working repository is `/mnt/c/Users/smirn/VSCodeProjects/nexus_project`. Its checked-out `main` is `0a8a1615d45bbadef4a52828a3e32b5f0865fde1`. Live `git ls-remote --heads origin` confirmed `main` at `7557d5d9b9c75fbe93091ead6ae525a1c377cdf6`, 58 commits ahead. All returned live branch heads matched the local remote-tracking refs at the time of the check.

| Branch | Live short SHA | Relevant finding |
|---|---|---|
| main | 7557d5d | Contains the controlled RGB harness; not the working-tree core fixes |
| rgb-pixel-ablation | 1ffa6a7 | Already an ancestor of main |
| rgb-plus-state | 7542634 | Already an ancestor of main |
| rgb-shared-encoder | 3738423 | Already an ancestor of main |
| llm-extension | 549afd2 | Not an ancestor of main; inspected, not merged |
| codex/finalize-continuous-nexus | d4ee96a | Historical handoff branch |
| codex/phase2-eval-finalize | 196ec05 | Historical deterministic-evaluation branch |
| feat_wandb | aa9c27b | Historical reliability/experiment branch |
| rosela | 8375952 | Historical notebook branch |

Raw status lists many modifications, largely line endings. `git diff --ignore-space-at-eol --stat` reduces this to eight paths, including three submodule dirty flags. The submodules have no semantic diff under that check. There are also many untracked experiment scripts and policy variants. No pull, merge, reset, branch switch, worktree prune, commit, or push was performed.

The actual core trainer has local functionality that the newer RGB-integrated source does not preserve, including the shared-skill-reward control required by HPQN. The local common-policy/registry/evaluator edits also matter. A newer commit date does not make a blind pull scientifically or operationally safe.

**Source decision:** freeze the actual working core and separately export RGB from Git object `7557d5d…`. Six critical working-file hashes match the uploaded `code/current` reference. `audit/source_contract.json` records those exact hashes. `freeze_sources.py` refuses to proceed if the audited HEAD or critical files have changed. Other copied source files receive a fresh manifest; this is not a claim that every local file was previously hashed against GitHub.

All three submodule gitlinks agree between the inspected local and remote-main trees: Playground `33f1b284…`, PureJaxQL `47af6d7…`, Symbolic Options `cabc240…`. The functioning Python environment imports installed Playground, not the vendor directory. Installed Playground's provenance identifies commit `33f1b284…`.

## 2. Checkpoints: many real weights, no approved replacements for the new primary cohort

| Observation | Verified result |
|---|---:|
| Inspected checkpoint-format files under core/runs | 913 |
| Total bytes | 42,883,762,520 |
| Files decoded without errors | 913 |
| Files containing weights | 907 |
| Weight files with recorded normalization under the scanner schema | 782 |
| Weight files lacking that recorded normalization | 125 |
| Metrics-only files | 6 |
| Unique file hashes | 893 |
| Byte-identical duplicate pairs | 20 |
| Recorded actual counters matching either new core budget | 0 |
| RGB/pixel/camera-named weights in the inspected runs | 0 |
| Approved primary-run reuse | 0 |

The six metrics-only files are the three `go1_flat_llm_s*.pkl` and three `hopper_hop_llm_s*.pkl` files in `runs/llm_budget`. They have configuration/skills/metrics but no policy weights. They cannot be restored as controllers.

The largest file groups are `verify` (401), `viper` (261), `final_research_matrix` (60), `phase2_final_matrix` (51), `verification` (41), and `ppo` (24). Some Cheetah/Walker checkpoints have the new LLM-reference transition budget, but budget coincidence alone is not configuration, source, normalizer, simulator, or seed provenance. These were not approved as matched primary replacements.

No inspected actual counter equals Hopper's 117,964,800 or Go1's 32,768,000. Old PPO files include nominal `num_timesteps` but lack the new actual-step record, so absence of a matching verified counter is not proof of their exact consumed steps. Final-only weights also cannot recreate missing intermediate held-out evaluation curves.

**Correction to the earlier answer:** the possible 12-run RGB reuse discount is withdrawn. No such weights were located in the searched local runs or tracked GitHub-main tree. They could still exist on an uninspected machine or storage account; this audit does not rule that out.

The 907 weighted files remain useful as historical/evaluation candidates. The audit deserialized and hashed them; it did **not** run model-forward restoration or simulator rollouts. Duplicate files are not additional seeds. Missing normalization requires investigation, not blind default normalization. Nothing was deleted or deduplicated.

## 3. Actual WSL environment and storage

Use WSL distribution **Ubuntu**, not the default Ubuntu-20.04 distribution. The actual project interpreter is:

```text
/mnt/c/Users/smirn/VSCodeProjects/nexus_project/nexus_continuous_control/.venv-wsl312/bin/python
```

Python is 3.12.13. JAX/JAXlib 0.10.1, Flax 0.12.7, Optax 0.2.8, Brax 0.14.2, MuJoCo/MJX 3.9.0, Playground 0.2.0, NumPy 2.4.6, and Warp 1.13.0 were observed. JAX detected the RTX 4060 Ti and a GPU backend. The system Python and `/home/smirn/nexus_venv` are not the active project interpreter.

Torch, Transformers, and Hugging Face Hub are absent from that interpreter. Accelerate was not verified. Use a separately verified generation environment; do not install a new ML stack into the functioning JAX environment merely to satisfy imports. The RGB code uses the MJWarp rendering path; missing Madrona is not evidence of an RGB blocker. Actual RGB render-API and rendering readiness remain unverified.

The C: project volume reports about 30 GB available and 97% used. WSL's virtual filesystem reports much more logical free space; that does not establish the VHD's physical backing capacity. A conservative 50 GiB production-space guard is included. It is an operational guard, not a measured exact storage requirement. Do not delete checkpoints to pass it. Physical storage and VHD backing must be checked before local RGB training.

`/etc/wsl.conf` has systemd enabled. Host `.wslconfig` was not inspected. No WSL settings changed. An auxiliary Ubuntu-20.04 Python diagnostic session was started and exited; no claim is made that the distribution returned to its original stopped state. No unrelated Windows documents were browsed.

Only the RTX 4060 Ti machine was connected through MCP. The RTX 5080 environment and storage were not verified.

## 4. Viper: reachable, but do not train using the old deployed copy

Read-only SSH to `viper11` succeeded. `/ptmp/akalenik/nexus/repo` is a copied code tree, not a Git checkout. Its trainer and evaluator hashes differ from local; `tools/train_ppo_baseline.py` is absent. The six-file comparison confirms that some shared files agree while those important parts do not. No attempt was made to repair that legacy deployment in place.

The existing interpreter is `/ptmp/akalenik/jaxrocm_venv/bin/python`: Python 3.13.5, JAX/JAXlib and ROCm7 plugins 0.10.2, MuJoCo/MJX 3.10.0, and Brax 0.14.2. This differs from the local numerical stack. The old site overlay is `/ptmp/akalenik/nexus/site`.

There are 261 checkpoint files, totaling 7,922,697,103 bytes, under the existing Viper runs directory. The equal local folder count does not prove byte-identical files: that cross-machine hash comparison was not performed. The user's queue was empty when checked. No Slurm allocation, smoke training, full training, or new Viper package transfer occurred.

The existing resource recipe uses account `mage_apu`, partition `apu`, one APU, 24 CPUs, 108000 MB, `MUJOCO_GL=disable`, and the ROCm command-buffer workaround. The partition reports a 24-hour maximum. The portable wrapper requests that maximum for production and one hour for smokes; this is a scheduling safeguard, not a runtime prediction. Longer requests may increase queue delay. Intermediate snapshots cannot currently resume training, so the first production-size run must establish timing before broad submission.

Filesystem free space does not establish the user's quota or remaining project allocation. Both remain to be verified. Eight concurrent APUs are a requested cap, not a reservation.

## 5. Matrix decision and actual installation state

Scientific rows remain **60 core + 52 LLM-related + 30 RGB = 142 maximum new runs**, totaling 7,073,792,000 requested transitions. There is no justified numeric reduction at this point and no new scientific sweep.

All 112 core/PPO/LLM-training rows are assigned to one newly frozen Viper source/runtime cohort. WSL core is smoke/debug only. The 30 RGB rows stay together on one verified NVIDIA rendering stack; the 5080 is not silently mixed in. Existing weights are not automatically discarded, but no primary row is marked complete because a filename resembles it.

The partial remote folder is:

```text
/mnt/c/Users/smirn/VSCodeProjects/nexus_project/
  nexus_continuous_control/experiments/final_campaign_2026-09-05/
```

Source freezing, matrix generation, and many launcher/evaluator files were written there. The `INSTALLING` guard remains. Runtime import probes were started, but their results were not read after the connector block. Documentation, final integrity checks, full audit copying, packaging, and Viper deployment were not completed there.

The downloadable handoff is a **separate complete source package**, not a byte-identical copy of that partial directory. It deliberately contains no claimed live source snapshot or copied weights. Install it into the new sibling `final_campaign_2026-09-05_handoff`, freeze the actual source, and run the provided checks. Do not merge the two package folders or remove the original guard manually.

## Acceptance criteria for the next agent

Installation must pass critical-source hashes, all-source manifest checks, exact regeneration of all 142 configurations, Python/static tests, shell syntax, imports/API checks, and then GPU training/save/restore/rendering smokes for each declared code path. Production-size memory/timing and physical space/quota checks are separate requirements. The code-only installer does not authorize production training.

No learning, robustness, LLM-refinement, or visual-control scientific conclusion follows from this audit. The controlled campaign remains a plan, not executed experimental evidence.
