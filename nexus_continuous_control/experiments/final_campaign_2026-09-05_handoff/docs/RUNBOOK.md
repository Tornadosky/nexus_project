# Complete machine-specific runbook

This is the active runbook. `RUNBOOK_PRE_AUDIT.md` is historical and must not be used for deployment. These commands are for the user's authorized coding agent to execute later. They were not run on the remote machine during delivery of this portable handoff.

## 0. Stop conditions

Do not start production before installation, runtime, restore/render, storage, and quota checks. Do not upgrade existing ML environments in place. Do not modify old code/weights or clear partial output folders. No automatic Git update is required. The original remote partial folder stays unchanged and guarded.

An error means inspect the evidence, not replace expected hashes or remove guards. Source or environment changes require a reviewed campaign revision and fresh smoke receipts. An identical infrastructure retry uses a separate `--results` root and must be recorded; the wrappers deliberately refuse partial-output overwrite.

## 1. Install the portable handoff inside the repository, without training

Place the downloaded ZIP in `/home/smirn/Downloads/`. Start **Ubuntu** from PowerShell only when not already in it:

```powershell
wsl.exe -d Ubuntu
```

Then use this complete WSL sequence. It creates a new sibling and will not merge with an existing directory:

```bash
set -euo pipefail
export REPO=/mnt/c/Users/smirn/VSCodeProjects/nexus_project
export PY="$REPO/nexus_continuous_control/.venv-wsl312/bin/python"
export ZIP=/home/smirn/Downloads/nexus_campaign_handoff_2026-09-05_17-00.zip
export PARENT="$REPO/nexus_continuous_control/experiments"
export KIT="$PARENT/final_campaign_2026-09-05_handoff"
test -f "$ZIP"
test -x "$PY"
test ! -e "$KIT"
mkdir -p "$PARENT"
unzip "$ZIP" -d "$PARENT"
cd "$KIT"
sha256sum -c SHA256SUMS.txt
"$PY" scripts/freeze_sources.py --repo "$REPO" --package "$KIT"
"$PY" scripts/verify_installation.py --finish
"$PY" scripts/agent.py status --profile wsl_rgb
"$PY" scripts/agent.py plan --profile wsl_rgb --group rgb
```

The source freezer checks the audited local HEAD and six critical normalized hashes, exports RGB from the recorded Git object, and creates its own all-source manifest. It does not fetch, checkout, commit, or change the original source. The static verifier regenerates the entire plan in a new check directory and compares it to the supplied plan. Only after success does it rename its own `INSTALLING` file to `INSTALLING.closed_after_static_checks`. This is **not** production clearance.

Save an installation checkpoint before further work:

```bash
set -euo pipefail
export REPO=/mnt/c/Users/smirn/VSCodeProjects/nexus_project
export KIT="$REPO/nexus_continuous_control/experiments/final_campaign_2026-09-05_handoff"
export ARCHIVE=/home/smirn/nexus_campaign_installed_$(date -u +%Y-%m-%d_%H-%M-%S).tar.gz
test ! -e "$ARCHIVE"
tar -czf "$ARCHIVE" -C "$(dirname "$KIT")" "$(basename "$KIT")"
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
```

This is a small code/source/config checkpoint, not a backup of old or new model weights. It is not an independent-disk backup when both locations share the same backing drive.

## 2. Fresh runtime/API checks on WSL

These checks import code only; they do not render or train. They write new uniquely named evidence, so unread earlier probe files are not overwritten.

```bash
set -euo pipefail
export REPO=/mnt/c/Users/smirn/VSCodeProjects/nexus_project
export KIT="$REPO/nexus_continuous_control/experiments/final_campaign_2026-09-05_handoff"
export PY="$REPO/nexus_continuous_control/.venv-wsl312/bin/python"
export CHECK="$KIT/installation_checks/runtime_$(date -u +%Y-%m-%d_%H-%M-%S)"
mkdir "$CHECK"
JAX_PLATFORMS=cpu "$PY" "$KIT/scripts/probe_runtime.py" \
 --repo "$KIT/sources/core" --kind state --out "$CHECK/state.json"
JAX_PLATFORMS=cpu "$PY" "$KIT/scripts/probe_runtime.py" \
 --repo "$KIT/sources/core" --kind ppo --out "$CHECK/ppo.json"
JAX_PLATFORMS=cpu "$PY" "$KIT/scripts/probe_runtime.py" \
 --repo "$KIT/sources/rgb" --kind rgb --out "$CHECK/rgb.json"
```

Any failing import/API check stops this sequence. Preserve its log. The RGB source uses MJWarp, not Madrona. The installed API was not verified during the audit; do not assume that a MuJoCo version number alone establishes compatibility. A different RGB interpreter must be verified separately and recorded in `deploy/wsl_rgb.json` **before all three controlled arms**. All arms must use that same frozen environment. No automatic package-install command is supplied for an environment whose required compatibility has not been established.

## 3. Deploy the frozen package to a new Viper directory

These are ordinary authorized deployment commands, not an attempt to retry a blocked MCP operation. They transfer this newly installed package, not files denied by the connector. No old Viper source/run is changed.

```bash
set -euo pipefail
export REPO=/mnt/c/Users/smirn/VSCodeProjects/nexus_project
export KIT="$REPO/nexus_continuous_control/experiments/final_campaign_2026-09-05_handoff"
export VIPER_KIT=/ptmp/akalenik/nexus/final_campaign_2026-09-05_handoff
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes viper11 \
 "test ! -e '$VIPER_KIT' && mkdir '$VIPER_KIT'"
rsync -a --ignore-existing --exclude='__pycache__/' "$KIT/" "viper11:$VIPER_KIT/"
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes viper11
```

Now on Viper (login node; no training here):

```bash
set -euo pipefail
export KIT=/ptmp/akalenik/nexus/final_campaign_2026-09-05_handoff
export PY=/ptmp/akalenik/jaxrocm_venv/bin/python
export PYTHONPATH=/ptmp/akalenik/nexus/site
export OUT=/ptmp/akalenik/nexus/final_campaign_2026-09-05_outputs
export CHECK="$OUT/setup_$(date -u +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$CHECK" "$OUT/logs" "$OUT/specs"
cd "$KIT"
sha256sum -c SHA256SUMS.txt
"$PY" scripts/agent.py status --profile viper
JAX_PLATFORMS=cpu "$PY" scripts/probe_runtime.py \
 --repo "$KIT/sources/core" --kind state --out "$CHECK/state.json"
JAX_PLATFORMS=cpu "$PY" scripts/probe_runtime.py \
 --repo "$KIT/sources/core" --kind ppo --out "$CHECK/ppo.json"
"$PY" scripts/write_job_lists.py --out "$OUT/lists"
```

The code checksum manifest excludes the install guard, because the static installer archives it. Agent status verifies frozen source hashes as well. The new working path is not the old `/ptmp/akalenik/nexus/repo`. The old site overlay is retained intentionally; verify its installed-source provenance and do not silently upgrade it.

Confirm project allocation/user quota and required simulator assets before GPU jobs. This audit verified the existing account/partition recipe, not remaining entitlement. No GPU initialization or learning should run on a login node.

## 4. Viper smoke → restore/intervention tests → production

On Viper after section 3:

```bash
set -euo pipefail
export KIT=/ptmp/akalenik/nexus/final_campaign_2026-09-05_handoff
# Prints the 12 core smoke submissions without submitting:
bash "$KIT/scripts/submit_group.sh" core --smoke
# Explicitly submits one task/method smoke per array index, at most eight at once:
bash "$KIT/scripts/submit_group.sh" core --smoke --submit
```

Wait for all 12. The state smoke must save initialization plus two updates; PPO must satisfy its exact smoke-step assertion. Inspect errors, finite outputs, normalization, and checkpoints. The launcher writes receipts only after a successful smoke save. **Receipts alone do not test the evaluator.** Restore and intervene in an allocated one-APU shell:

```bash
salloc --account=mage_apu --partition=apu --ntasks=1 --gres=gpu:1 \
 --cpus-per-task=24 --mem=108000 --time=01:00:00
srun --pty bash --noprofile --norc
```

Inside that allocated shell:

```bash
set -euo pipefail
export KIT=/ptmp/akalenik/nexus/final_campaign_2026-09-05_handoff
export PY=/ptmp/akalenik/jaxrocm_venv/bin/python
export OUT=/ptmp/akalenik/nexus/final_campaign_2026-09-05_outputs
export PYTHONPATH=/ptmp/akalenik/nexus/site
export MUJOCO_GL=disable XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export CHECK="$OUT/restore_$(date -u +%Y-%m-%d_%H-%M-%S)"
"$PY" "$KIT/scripts/probe_runtime.py" --repo "$KIT/sources/core" \
 --kind state --device --out "$CHECK/device.json"
for task in hopper go1; do
 for method in flat hpqn neural symbolic nesy ppo; do
  "$PY" "$KIT/scripts/evaluate.py" --repo "$KIT/sources/core" \
   --checkpoint "$OUT/results/core__${task}__${method}__s0__smoke/final.pkl" \
   --episodes 4 --num-envs 4 --max-steps 16 --out "$CHECK/${task}_${method}"
 done
done
"$PY" "$KIT/scripts/evaluate.py" --repo "$KIT/sources/core" \
 --checkpoint "$OUT/results/core__hopper__nesy__s0__smoke/final.pkl" \
 --episodes 4 --num-envs 4 --max-steps 16 --remove 0 --out "$CHECK/hopper_remove0"
"$PY" "$KIT/scripts/evaluate.py" --repo "$KIT/sources/core" \
 --checkpoint "$OUT/results/core__go1__nesy__s0__smoke/final.pkl" \
 --episodes 4 --num-envs 4 --max-steps 16 --command-range 0 0 0 --out "$CHECK/go1_stop"
"$PY" "$KIT/scripts/evaluate.py" --repo "$KIT/sources/core" \
 --checkpoint "$OUT/results/core__go1__nesy__s0__smoke/final.pkl" \
 --episodes 4 --num-envs 4 --max-steps 16 --env-name Go1JoystickRoughTerrain \
 --out "$CHECK/go1_rough"
exit
exit
```

The two exits leave the srun shell and allocation shell. No production run is started by this block. Verify removal cannot reintroduce the deleted actor and the stop condition actually samples zero commands.

Before broad production, run one chosen seed-0 production row for each resource-heavy task/engine in a suitable allocation using the same `agent.py ... --execute` command. It counts toward the fixed matrix; it is not a new scientific experiment. Preserve the full budget and schedule. Use its snapshot timings/peak memory to decide whether the rest fit the 24-hour request. Do not launch 60 long runs on the strength of a tiny smoke.

After those checks, on the Viper login node:

```bash
set -euo pipefail
export KIT=/ptmp/akalenik/nexus/final_campaign_2026-09-05_handoff
bash "$KIT/scripts/submit_group.sh" core
bash "$KIT/scripts/submit_group.sh" core --submit
```

Already verified complete same-identity rows are skipped. Partial rows are never overwritten. The wrapper requests one APU, 24 CPUs, 108000 MB, and up to 24 hours; small smokes request one hour. Maximum eight concurrent array jobs is a cap, not guaranteed availability. Never request eight APUs for one single-seed trainer.

## 5. LLM generation, pilots, feedback, final comparison

The audit did not find a verified generator interpreter. The following gate requires `LLM_PY` to be supplied by the agent **after** preparing and verifying a separate environment. It must not be the functioning core venv. This is the one genuinely unresolved interpreter path, not a fabricated installation claim.

On WSL, after the separate environment has been verified:

```bash
set -euo pipefail
: "${LLM_PY:?Set the exact independently verified Torch/Transformers interpreter}"
export REPO=/mnt/c/Users/smirn/VSCodeProjects/nexus_project
export KIT="$REPO/nexus_continuous_control/experiments/final_campaign_2026-09-05_handoff"
export SPECS=/home/smirn/nexus_campaign_specs_2026-09-05
export EVIDENCE=/home/smirn/nexus_campaign_evidence_2026-09-05
"$LLM_PY" -c 'import torch,transformers,huggingface_hub,accelerate; print(torch.__version__,torch.cuda.is_available(),transformers.__version__)'
bash "$KIT/scripts/llm_phases.sh" lock
bash "$KIT/scripts/llm_phases.sh" initial
bash "$KIT/scripts/llm_phases.sh" resample
rsync -a --ignore-existing "$SPECS/" \
 viper11:/ptmp/akalenik/nexus/final_campaign_2026-09-05_outputs/specs/
```

Keep raw model outputs, bounded repairs, the exact model revision, and environment lock. A generation error stops the phase and remains a result; do not resample until successful. The agent must record failed families rather than submit nonexistent JSONs.

On Viper:

```bash
set -euo pipefail
export KIT=/ptmp/akalenik/nexus/final_campaign_2026-09-05_handoff
bash "$KIT/scripts/submit_group.sh" llm_reference --smoke --submit
bash "$KIT/scripts/submit_group.sh" llm_pilot --smoke --submit
```

Wait, inspect and restore the generated-policy/reference smoke files using the common evaluator exactly as in section 4. Both tasks need a passing path. Then:

```bash
set -euo pipefail
export KIT=/ptmp/akalenik/nexus/final_campaign_2026-09-05_handoff
bash "$KIT/scripts/submit_group.sh" llm_reference --submit
bash "$KIT/scripts/submit_group.sh" llm_pilot --submit
```

After all valid pilot rows finish, submit validation, not final-test evaluation:

```bash
set -euo pipefail
export KIT=/ptmp/akalenik/nexus/final_campaign_2026-09-05_handoff
export CAMPAIGN_ROOT="$KIT" EVAL_SUITE=pilot EVAL_SHARDS=6
sbatch --array=0-5%6 --export=ALL "$KIT/scripts/eval_viper.sbatch"
```

Once validation finishes, WSL receives only pilot summary files for refinement:

```bash
set -euo pipefail
: "${LLM_PY:?Previously verified separate generator interpreter}"
export REPO=/mnt/c/Users/smirn/VSCodeProjects/nexus_project
export KIT="$REPO/nexus_continuous_control/experiments/final_campaign_2026-09-05_handoff"
export SPECS=/home/smirn/nexus_campaign_specs_2026-09-05
export EVIDENCE=/home/smirn/nexus_campaign_evidence_2026-09-05
rsync -a --ignore-existing --include='*/' --include='summary.json' --exclude='*' \
 viper11:/ptmp/akalenik/nexus/final_campaign_2026-09-05_outputs/evidence/pilot/ \
 "$EVIDENCE/pilot/"
bash "$KIT/scripts/llm_phases.sh" refine
rsync -a --ignore-existing "$SPECS/" \
 viper11:/ptmp/akalenik/nexus/final_campaign_2026-09-05_outputs/specs/
```

The `--ignore-existing` copies preserve earlier proposals. Verify copied hashes; a hash mismatch is an error, not permission to overwrite a used specification. On Viper, after all expected valid specifications are present:

```bash
set -euo pipefail
export KIT=/ptmp/akalenik/nexus/final_campaign_2026-09-05_handoff
bash "$KIT/scripts/submit_group.sh" llm_final --submit
```

This is 52 maximum RL runs including references/pilots, not 52 additional runs after those phases. Final conditions start from scratch. Generation failures can leave declared missing cells; do not invent metrics or change the scientific design to hide them.

## 6. RGB smoke and serial production

Resolve the render/API and physical-storage gates first. The recorded profile initially points at the actual core interpreter for inspection, **not** as proof it supports rendering. If a separate RGB environment is required, record its exact Python path in `deploy/wsl_rgb.json`, verify it, and use it for all arms. Keep core dependencies untouched.

With the selected verified interpreter from the profile:

```bash
set -euo pipefail
export REPO=/mnt/c/Users/smirn/VSCodeProjects/nexus_project
export KIT="$REPO/nexus_continuous_control/experiments/final_campaign_2026-09-05_handoff"
export PY="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["python"])' "$KIT/deploy/wsl_rgb.json")"
for task in cartpole walker; do
 for arm in state pixels constant; do
  "$PY" "$KIT/scripts/agent.py" run --profile wsl_rgb \
   --id "rgb__${task}__${arm}__s0" --smoke --execute
 done
done
```

Inspect actual frames, constant tensors, preserved vision task in the state control, saved/restored policies, and corruption outputs. Fixed evaluation windows may span resets; label them accordingly. Then measure a first full production row, retain it as part of the matrix, and only afterward run the serial queue:

```bash
set -euo pipefail
export REPO=/mnt/c/Users/smirn/VSCodeProjects/nexus_project
export KIT="$REPO/nexus_continuous_control/experiments/final_campaign_2026-09-05_handoff"
export PY="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["python"])' "$KIT/deploy/wsl_rgb.json")"
bash "$KIT/scripts/run_rgb_serial.sh"
```

The shared GPU lock prevents overlap with the existing local project queues and LLM-generation script. The RTX 5080 is deliberately not assigned until separately inspected; no automatic two-machine sharding is used.

## 7. Final checkpoint-only evaluation suites

After the corresponding training phases finish, submit **one suite at a time** on Viper. Wait for each before the next to respect the overall eight-job cap.

```bash
set -euo pipefail
export KIT=/ptmp/akalenik/nexus/final_campaign_2026-09-05_handoff
export CAMPAIGN_ROOT="$KIT" EVAL_SHARDS=8
export EVAL_SUITE=curves
sbatch --array=0-7%8 --export=ALL "$KIT/scripts/eval_viper.sbatch"
```

After completion, repeat the complete block with `EVAL_SUITE=probes`, then `shifts`, then `llm`. These are required checkpoint-only suites, not new training. `curves` requires 11 snapshots for every core run, not separately trained endpoints. Each evaluation writes `summary.json`, `metadata.json`, and `episodes.npz` with the checkpoint hash. Missing weights stop rather than silently skip a scientific condition.

For an explicitly requested independent suite, the full command is:

```bash
set -euo pipefail
export KIT=/ptmp/akalenik/nexus/final_campaign_2026-09-05_handoff
export CAMPAIGN_ROOT="$KIT" EVAL_SHARDS=8 EVAL_SUITE=probes
sbatch --array=0-7%8 --export=ALL "$KIT/scripts/eval_viper.sbatch"
# After that array finishes, run:
export EVAL_SUITE=shifts
sbatch --array=0-7%8 --export=ALL "$KIT/scripts/eval_viper.sbatch"
# After that array finishes and LLM training is complete, run:
export EVAL_SUITE=llm
sbatch --array=0-7%8 --export=ALL "$KIT/scripts/eval_viper.sbatch"
```

Do not paste all three submissions at once; the waiting points are intentional. The raw suite implementation is in `evaluate_campaign.py` for agent automation with explicit scheduler dependencies.

## 8. Media, data lock, and final report

Use the frozen repository `tools/fig6_panels.py` and `tools/render_rollout.py` only after checking their `--help`, normalizer restoration, and renderer compatibility. Preselect training seed 0 and rendering seed 41000, not the best-looking successful agent. The core Viper simulator differs from the local simulator: rendering a restored policy locally is a separate deployment unless an equivalent environment is established. Label such media accurately and never use its returns as the Viper evaluation statistics.

Keep source/identity manifests, all JSON/NPZ/weights, generation outputs, failures, and actual-step logs. Make a second verified copy on independent durable storage. No independently verified backup destination was available in the audit; do not claim two folders on the same drive are two backups. Do not delete old runs to create space.

After these fixed experiments, stop training. Final work is analysis, figure composition, narrative, limitations, contributions/LLM-use disclosure, and a minimal core-port PR on a separately reviewed branch.
