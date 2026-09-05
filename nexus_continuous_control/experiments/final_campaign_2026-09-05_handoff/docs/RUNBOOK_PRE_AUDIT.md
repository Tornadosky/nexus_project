> HISTORICAL ONLY. Superseded by RUNBOOK.md and AUDIT_REPORT.md. Do not execute these old deployment commands.

# Complete launch sequence

Do not start the production array until the smoke checks and source/version inventory succeed.
These are new wrappers, not claimed existing repository commands. The paths below are the only
machine-specific inputs. Use the existing functioning project Python environments; this package
does not reinstall JAX, ROCm, CUDA, MuJoCo, Torch, or vendor submodules. A missing vendor tree or
GPU dependency must be restored from your actual working repository/environment, not replaced
by an unpinned `pip install` from the latest GitHub branch.

## 1. Select paths, inventory, and validate

Extract the ZIP into a NEW directory. The archive contains `nexus_final_campaign/`.
On a Linux/WSL machine with your core and RGB checkouts available:

```bash
set -euo pipefail
export KIT="$PWD/nexus_final_campaign"
export CORE="/absolute/path/to/core/nexus_continuous_control"
export RGB="/absolute/path/to/rgb/nexus_continuous_control"
export PY_CORE="/absolute/path/to/working/core/python"
export PY_RGB="/absolute/path/to/working/rgb/python"
export PY_LLM="/absolute/path/to/working/huggingface/python"
export RESULTS="$PWD/nexus_final_results"
export EVIDENCE="$PWD/nexus_final_evidence"
export SPECS="$PWD/nexus_final_specs"
mkdir -p "$RESULTS" "$EVIDENCE" "$SPECS" "$KIT/logs"

"$PY_CORE" -c 'import jax,flax,optax,mujoco,mujoco_playground,brax,yaml; print(jax.__version__,jax.devices())'
"$PY_RGB" -c 'import jax,mujoco,mujoco_playground; print(jax.__version__,jax.devices())'
"$PY_LLM" -c 'import torch,transformers,huggingface_hub; print(torch.__version__,torch.cuda.is_available(),transformers.__version__)'

"$PY_CORE" -m unittest discover -s "$KIT/tests" -v
JAX_PLATFORMS=cpu "$PY_CORE" "$KIT/scripts/inventory.py" --repo "$CORE" \
  --out "$EVIDENCE/core_inventory.json"
JAX_PLATFORMS=cpu "$PY_RGB" "$KIT/scripts/inventory.py" --repo "$RGB" \
  --out "$EVIDENCE/rgb_inventory.json"

# Pickle is unsafe for untrusted files. Run this ONLY on your own saved checkpoints.
JAX_PLATFORMS=cpu "$PY_CORE" "$KIT/scripts/inventory.py" --repo "$CORE" \
  --checkpoint-root "/absolute/path/to/your/old/runs" --trusted-checkpoints \
  --out "$EVIDENCE/checkpoint_inventory.json"

"$PY_CORE" "$KIT/scripts/campaign.py" --matrix "$KIT/plan/matrix.json" \
  --core-repo "$CORE" --results "$RESULTS" --groups core --list
```

The supplied `plan/` is already generated. Do not run `make_matrix.py --out plan` over it.
To regenerate from the confirmed snapshots, use a NEW directory:

```bash
"$PY_CORE" "$KIT/scripts/make_matrix.py" --core-repo "$CORE" --rgb-repo "$RGB" \
  --out "$KIT/plan_rebuilt"
```

Use `plan_rebuilt/matrix.json` consistently instead of `plan/matrix.json` only after comparing
and freezing it. The standard commands below use the supplied `plan/`.

Checkpoint reuse is manual approval, not automatic filename matching. Verify exact effective
configuration, actual budget, source/vendor identity, seed, actor shape, and normalization. A
metrics-only pickle is not reusable. Legacy final-only checkpoints can support reevaluation,
but not missing intermediate learning curves. Do not mix the legacy and new core campaigns.

## 2. Mandatory small plumbing tests

Use the appropriate machine's already working environment. These are labelled SMOKE and never
enter the paper's statistics. Preserve these logs. An existing partial output directory causes
a hard stop; use a new directory for a documented retry rather than deleting evidence.

```bash
"$PY_CORE" "$KIT/scripts/campaign.py" --matrix "$KIT/plan/matrix.json" \
  --core-repo "$CORE" --results "$RESULTS" --groups core \
  --id core__hopper__nesy__s0 --smoke
"$PY_CORE" "$KIT/scripts/campaign.py" --matrix "$KIT/plan/matrix.json" \
  --core-repo "$CORE" --results "$RESULTS" --groups core \
  --id core__go1__nesy__s0 --smoke
"$PY_CORE" "$KIT/scripts/campaign.py" --matrix "$KIT/plan/matrix.json" \
  --core-repo "$CORE" --results "$RESULTS" --groups core \
  --id core__hopper__ppo__s0 --smoke
"$PY_CORE" "$KIT/scripts/campaign.py" --matrix "$KIT/plan/matrix.json" \
  --core-repo "$CORE" --results "$RESULTS" --groups core \
  --id core__go1__ppo__s0 --smoke

"$PY_CORE" "$KIT/scripts/evaluate.py" --repo "$CORE" \
  --checkpoint "$RESULTS/core__hopper__nesy__s0__smoke/final.pkl" \
  --episodes 4 --num-envs 4 --max-steps 16 \
  --out "$EVIDENCE/smoke/hopper_native"
"$PY_CORE" "$KIT/scripts/evaluate.py" --repo "$CORE" \
  --checkpoint "$RESULTS/core__hopper__nesy__s0__smoke/final.pkl" \
  --episodes 4 --num-envs 4 --max-steps 16 --remove 0 \
  --out "$EVIDENCE/smoke/hopper_remove0"
"$PY_CORE" "$KIT/scripts/evaluate.py" --repo "$CORE" \
  --checkpoint "$RESULTS/core__go1__ppo__s0__smoke/final.pkl" \
  --episodes 4 --num-envs 4 --max-steps 16 \
  --out "$EVIDENCE/smoke/go1_ppo_restore"

"$PY_RGB" "$KIT/scripts/campaign.py" --matrix "$KIT/plan/matrix.json" \
  --rgb-repo "$RGB" --results "$RESULTS" --groups rgb \
  --id rgb__walker__constant__s0 --smoke
```

Finish the remaining architecture and intervention plumbing checks:

```bash
for task in cartpole walker; do
  for arm in state pixels constant; do
    "$PY_RGB" "$KIT/scripts/campaign.py" --matrix "$KIT/plan/matrix.json" \
      --rgb-repo "$RGB" --results "$RESULTS" --groups rgb \
      --id "rgb__${task}__${arm}__s0" --smoke
  done
done
for mode in force remove; do
  "$PY_CORE" "$KIT/scripts/evaluate.py" --repo "$CORE" \
    --checkpoint "$RESULTS/core__go1__nesy__s0__smoke/final.pkl" \
    --episodes 4 --num-envs 4 --max-steps 16 "--$mode" 0 \
    --out "$EVIDENCE/smoke/go1_${mode}0"
done
for selector in unmasked symbolic; do
  "$PY_CORE" "$KIT/scripts/evaluate.py" --repo "$CORE" \
    --checkpoint "$RESULTS/core__go1__nesy__s0__smoke/final.pkl" \
    --episodes 4 --num-envs 4 --max-steps 16 --selector "$selector" \
    --out "$EVIDENCE/smoke/go1_${selector}"
done
"$PY_CORE" "$KIT/scripts/evaluate.py" --repo "$CORE" \
  --checkpoint "$RESULTS/core__go1__nesy__s0__smoke/final.pkl" \
  --episodes 4 --num-envs 4 --max-steps 16 --command-range 0 0 0 \
  --out "$EVIDENCE/smoke/go1_stop"
"$PY_CORE" "$KIT/scripts/evaluate.py" --repo "$CORE" \
  --checkpoint "$RESULTS/core__go1__nesy__s0__smoke/final.pkl" \
  --episodes 4 --num-envs 4 --max-steps 16 --env-name Go1JoystickRoughTerrain \
  --out "$EVIDENCE/smoke/go1_rough"
```

The tiny tests establish plumbing only, NOT production-size memory or runtime. Use the first full-sized production jobs' snapshot timings
for that check. Preserve full budgets and schedules in those production jobs.

## 3. Core production

The exact one-job command is:

```bash
"$PY_CORE" "$KIT/scripts/campaign.py" --matrix "$KIT/plan/matrix.json" \
  --core-repo "$CORE" --results "$RESULTS" --groups core \
  --id core__hopper__nesy__s0
```

For a sequential local queue, omit `--id`. `--dry-run` prints the exact underlying new wrapper
command without running it. To split a local queue across two separate GPUs/machines, use
`--shards 2 --shard-index 0` and `--shards 2 --shard-index 1`. Each machine must use matching
source/configuration/environment versions. Do not simultaneously assign the same row twice.
Keep hardware balanced across methods; preferably use the same hardware for a task's core runs.

For Viper, copy this package and the frozen specs/source to the cluster using your existing
connection. Run from the package directory on a login node. YOUR uploaded project job uses the
following environment (paths may differ in your current working installation):

```bash
set -euo pipefail
export CAMPAIGN_ROOT="/absolute/cluster/path/nexus_final_campaign"
export CORE_REPO="/ptmp/akalenik/nexus/repo"
export PYTHON="/ptmp/akalenik/jaxrocm_venv/bin/python"
export RESULTS="/ptmp/akalenik/nexus/final_campaign_results"
export EVIDENCE="/ptmp/akalenik/nexus/final_campaign_evidence"
export SPECS="/ptmp/akalenik/nexus/final_campaign_specs"
export ENV_INIT="$CAMPAIGN_ROOT/viper_environment.sh"
cd "$CAMPAIGN_ROOT"
mkdir -p logs "$RESULTS" "$EVIDENCE" "$SPECS"

# This is the wheel-bundled setup in the uploaded project, NOT a recommendation
# to mix it with the site's newer module-based ROCm stack.
cat > "$ENV_INIT" <<'SH'
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export MUJOCO_GL=disable
export PYTHONPATH=/ptmp/akalenik/nexus/site:"$CORE_REPO":${PYTHONPATH:-}
SH

"$PYTHON" - <<'PY'
import json
from pathlib import Path
rows=json.loads(Path('plan/matrix.json').read_text())
for group in sorted({r['group'] for r in rows}):
    Path(group+'_ids.txt').write_text(','.join(str(i) for i,r in enumerate(rows) if r['group']==group))
PY

# mage_apu is the account used by the uploaded project script.
# Substitute your approved account if it differs; the resource request is one APU per run.
sbatch --account=mage_apu --array="$(cat core_ids.txt)%8" scripts/viper.sbatch
```

The scheduler supplies one accelerator; do not request eight for a single-seed trainer. The
maximum current shared-job duration is 24 hours. If first production timings predict longer,
stop the rest of the launch before its results are inspected and resolve the implementation/
allocation problem. The compact snapshots are evaluation checkpoints, not restart checkpoints.

Compute nodes have no internet. Download model/assets and initialize the pinned submodules on
the login/local machine first. Do not run training on Viper login nodes.

## 4. Generate LLM proposals, train pilots, then refine

On the NVIDIA/LLM machine, lock/download the one model revision once:

```bash
"$PY_LLM" "$KIT/scripts/llm_specs.py" lock-model --out "$SPECS/model_lock.json"
for task in cheetah walker; do
  for g in 0 1 2; do
    for condition in initial resample; do
      "$PY_LLM" "$KIT/scripts/llm_specs.py" generate \
        --model-lock "$SPECS/model_lock.json" --task "$task" --family "$g" \
        --condition "$condition" --out "$SPECS/$task/g$g/$condition.json"
    done
  done
done
```

A generation failure is preserved in `.generation.json`, returns nonzero, and must be recorded.
Do not silently draw another family. On valid proposals, copy `SPECS` to the cluster before the
pilot wave (using your existing SSH host alias; no deletion flag):

```bash
rsync -av --progress "$SPECS/" "$VIPER_HOST:$REMOTE_SPECS/"
```

Then, on Viper, with the variables from section 3 still set:

```bash
cd "$CAMPAIGN_ROOT"
sbatch --account=mage_apu --array="$(cat llm_reference_ids.txt)%8" scripts/viper.sbatch
sbatch --account=mage_apu --array="$(cat llm_pilot_ids.txt)%8" scripts/viper.sbatch
```

When ALL six pilots are verified complete, run their validation (not final-test evaluation):

```bash
export EVAL_SUITE=pilot
sbatch --account=mage_apu --array=0-5 scripts/eval_viper.sbatch
```

The six pilot jobs occupy shards 0..5 under the fixed eight-way partition. Do not submit
empty shards 6 or 7 for this suite. This array runs on compute nodes, not the login node.

Bring only the pilot summaries back to the LLM machine:

```bash
rsync -av --include='*/' --include='summary.json' --exclude='*' \
  "$VIPER_HOST:$REMOTE_EVIDENCE/pilot/" "$EVIDENCE/pilot/"
for task in cheetah walker; do
  for g in 0 1 2; do
    seed=$((900 + g))
    "$PY_LLM" "$KIT/scripts/llm_specs.py" generate \
      --model-lock "$SPECS/model_lock.json" --task "$task" --family "$g" \
      --condition refined --initial "$SPECS/$task/g$g/initial.json" \
      --feedback "$EVIDENCE/pilot/llm_pilot__${task}__initial__g${g}__s${seed}/validation/summary.json" \
      --out "$SPECS/$task/g$g/refined.json"
  done
done
rsync -av --progress "$SPECS/" "$VIPER_HOST:$REMOTE_SPECS/"
```

Then launch the frozen final three-condition comparison on Viper:

```bash
cd "$CAMPAIGN_ROOT"
sbatch --account=mage_apu --array="$(cat llm_final_ids.txt)%8" scripts/viper.sbatch
```

Without Viper, the identical phase commands are:

```bash
"$PY_CORE" "$KIT/scripts/campaign.py" --matrix "$KIT/plan/matrix.json" \
 --core-repo "$CORE" --results "$RESULTS" --specs "$SPECS" --groups llm_reference,llm_pilot
"$PY_CORE" "$KIT/scripts/evaluate_campaign.py" --matrix "$KIT/plan/matrix.json" \
 --repo "$CORE" --results "$RESULTS" --out "$EVIDENCE" --suite pilot
# Perform the bounded refinement-generation loop above before this next command.
"$PY_CORE" "$KIT/scripts/campaign.py" --matrix "$KIT/plan/matrix.json" \
 --core-repo "$CORE" --results "$RESULTS" --specs "$SPECS" --groups llm_final
```

## 5. Vision production on NVIDIA

All three arms keep USE_RGB=true. The matched state/pixel configs are supplied. Run two disjoint
queues on the two NVIDIA machines (same source and compatible, frozen dependencies):

```bash
# NVIDIA worker 0:
CUDA_VISIBLE_DEVICES=0 "$PY_RGB" "$KIT/scripts/campaign.py" \
 --matrix "$KIT/plan/matrix.json" --rgb-repo "$RGB" --results "$RESULTS" \
 --groups rgb --shards 2 --shard-index 0

# NVIDIA worker 1, in its own terminal/machine:
CUDA_VISIBLE_DEVICES=0 "$PY_RGB" "$KIT/scripts/campaign.py" \
 --matrix "$KIT/plan/matrix.json" --rgb-repo "$RGB" --results "$RESULTS" \
 --groups rgb --shards 2 --shard-index 1
```

When both NVIDIA cards are in one host, set the second worker to `CUDA_VISIBLE_DEVICES=1`.
The two `=0` examples above assume different machines.

The wrapper saves trained weights and runs all relevant fixed-image tests with 64 windows.
For approved legacy controlled RGB weights, evaluate without training using the exact arm config:

```bash
"$PY_RGB" "$KIT/scripts/rgb_run.py" --repo "$RGB" \
 --config "$KIT/plan/configs/rgb__walker__pixels__s0.yaml" \
 --load-policy "/absolute/path/to/verified/old/walker_pixels_seed0.pkl" \
 --reuse-proof "/absolute/path/to/verified/old/walker_pixels_seed0_proof.json" \
 --out "$RESULTS/rgb__walker__pixels__s0"
```

For legacy pickles, the proof JSON must contain `checkpoint_sha256`, `actual_steps`, `config`
(the OLD run's resolved configuration), and `provenance_description` identifying preserved logs,
commands and source versions. Build it from the original records, not by copying desired values
from the new matrix. The wrapper rejects a hash, budget, or configuration mismatch. A checkpoint
written by this new package already contains its config and actual steps and does not need a proof.

This reuse command assumes you have verified the old run's source identity; the legacy
pickle alone does not establish that provenance. The original file is preserved. The new saved
checkpoint gains the verified config and constant-input metadata. Do not apply this to
metrics-only pickles or an old environment-mismatched state control.

## 6. Final checkpoint-only suites

Local/sequential commands, to execute in a GPU environment:

```bash
for suite in curves probes shifts llm; do
  "$PY_CORE" "$KIT/scripts/evaluate_campaign.py" --matrix "$KIT/plan/matrix.json" \
    --repo "$CORE" --results "$RESULTS" --out "$EVIDENCE" --suite "$suite"
done
```

On Viper submit one array per suite, AFTER the corresponding training phase is complete:

```bash
cd "$CAMPAIGN_ROOT"
export EVAL_SUITE=curves
sbatch --account=mage_apu --array=0-7 scripts/eval_viper.sbatch
export EVAL_SUITE=probes
sbatch --account=mage_apu --array=0-7 scripts/eval_viper.sbatch
export EVAL_SUITE=shifts
sbatch --account=mage_apu --array=0-7 scripts/eval_viper.sbatch
export EVAL_SUITE=llm
sbatch --account=mage_apu --array=0-7 scripts/eval_viper.sbatch
```

Each successful evaluation produces `summary.json`, `metadata.json`, and `episodes.npz`.
Missing checkpoints fail explicitly. Every curve snapshot is from the same underlying run.
The evaluator never calls a training function.

## 7. Real media and two-copy data lock

On a NVIDIA machine configured for EGL rendering, use the existing repository media tools.
These commands render fixed-seed rollouts; they do not train or synthesize experimental images.

```bash
mkdir -p "$EVIDENCE/media"
cd "$CORE"
PYTHONPATH="$CORE:$CORE/tools:${PYTHONPATH:-}" MUJOCO_GL=egl "$PY_CORE" tools/fig6_panels.py \
 --checkpoint "$RESULTS/core__go1__nesy__s0/final.pkl" \
 --seed 41000 --steps 1000 --panels 3 --min-gap 60 --width 640 --height 480 \
 --out "$EVIDENCE/media/go1_rules"
PYTHONPATH="$CORE:$CORE/tools:${PYTHONPATH:-}" MUJOCO_GL=egl "$PY_CORE" tools/render_rollout.py \
 --checkpoint "$RESULTS/core__go1__nesy__s0/final.pkl" \
 --seed 41000 --steps 1000 --width 640 --height 480 --fps 30 --strip \
 --out "$EVIDENCE/media/go1_nesy_seed0.mp4"
PYTHONPATH="$CORE:$CORE/tools:${PYTHONPATH:-}" MUJOCO_GL=egl "$PY_CORE" tools/render_rollout.py \
 --checkpoint "$RESULTS/core__hopper__nesy__s0/final.pkl" \
 --seed 41000 --steps 1000 --width 640 --height 480 --fps 30 --strip \
 --out "$EVIDENCE/media/hopper_nesy_seed0.mp4"

# Choose a distinct durable disk/location. These commands do not delete files.
export BACKUP_ROOT="/absolute/path/to/independent/backup/nexus_final"
mkdir -p "$BACKUP_ROOT/results" "$BACKUP_ROOT/evidence" "$BACKUP_ROOT/specs"
rsync -a --checksum "$RESULTS/" "$BACKUP_ROOT/results/"
rsync -a --checksum "$EVIDENCE/" "$BACKUP_ROOT/evidence/"
rsync -a --checksum "$SPECS/" "$BACKUP_ROOT/specs/"
```

Do not count two directories on the same temporary filesystem as independent backups. Viper
`/u` and `/ptmp` have no automatic system backup; `/ptmp` is subject to age-based deletion.
Stop all new training after the declared campaign. Figure layout, statistics, prose, source PR,
and authorship disclosure can be completed from these frozen artifacts without another run.
