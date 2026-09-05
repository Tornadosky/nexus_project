# Coding-agent handoff — finish installation, then run the fixed campaign

## Read this first

The live audit was completed through Remote Desktop Commander, but deployment was interrupted by two blocked connector operations. **Do not launch from the partial directory** `nexus_continuous_control/experiments/final_campaign_2026-09-05`. Its `INSTALLING` guard remains intentionally. Do not retry blocked connector operations through another access path.

Use this portable handoff for a fresh, authorized local installation in the sibling `nexus_continuous_control/experiments/final_campaign_2026-09-05_handoff`. It includes all campaign code/configurations but deliberately does not claim to include a recovered live source tree. Follow `docs/RUNBOOK.md` in order. No training has been run as part of the audit.

## Assignment

Preserve the user's dirty repository, every old checkpoint, all partial outputs, and the raw audit at `/home/smirn/nexus_campaign_audit_2026-09-05_16-08`. No `git pull`, `reset`, `clean`, checkout, merge, dependency upgrade, old-run deletion, or force push is authorized by this handoff. Source freeze uses the actual working core and RGB Git object `7557d5d…` without switching branches. The working HEAD must still be `0a8a161…`, and six critical file hashes must match.

Finish code-only installation checks first. They verify source integrity and exact regeneration of all 142 run configurations. They do not establish GPU training/rendering/restore compatibility. Use the recorded working interpreter, not system Python or the old home-directory venv.

The source/API probes started during the audit have **unread results**. Do not treat them as passes. Run fresh, uniquely named checks after installing. Resolve any compatibility problem before allocating the full campaign. Record each change, failing test, corrected file, and rerun validation; do not silently edit an accepted frozen source tree or reuse an old smoke receipt.

All 112 non-RGB primary training rows belong on Viper using the new frozen source and the existing ROCm environment/site overlay. The legacy Viper code tree is not current and lacks the PPO launcher. Do not overwrite it. WSL core is smoke/debug only. All 30 RGB rows must share one verified NVIDIA rendering environment. The 5080 is unverified and not assigned work automatically.

Resolve physical storage before local RGB production. The project volume had only about 30 GB free. The 50 GiB guard is deliberately conservative; bypassing it without verifying the actual output/VHD backing volume is not a fix. Check Viper project allocation and user quota separately from filesystem capacity.

There is no verified LLM-generation environment. Torch, Transformers, and Hugging Face Hub were absent from the core venv; Accelerate was not checked. Prepare/verify a separate environment as a bounded setup task, preserving the existing RL environment. Pin the selected environment and exact model revision before generating proposals. Do not claim these missing setup choices were tested by the assistant.

## Acceptance before production

Run one smoke for every core task/method path, both reference task paths, both generated-policy paths, and all six RGB task/input paths. Check saved initial/intermediate/final weights, exact steps, finite outputs, normalization, common-evaluator restoration, actor deletion/selection, zero-command override, and actual camera/constant-image behavior. The smoke receipts enforce training-path checks; their existence does not replace inspection of evaluation/rendering results.

The first full-size production row for each distinct resource-heavy path supplies production memory/timing evidence and counts toward the 142 rows. Preserve its full budget/schedule. Release the remaining rows only after those checks; no scientific sweep is allowed. The production wall-time request is 24 hours, not a prediction. Intermediate snapshots are evaluation-only and cannot resume timed-out optimization.

No old primary run is approved for substitution. The 913-file audit found 907 weighted files and six metrics-only files; no matching verified core counters or RGB weights were located. Old weights can be considered for supplemental diagnostics only with their original provenance, not used to manufacture missing new learning curves.

## Fixed experiment scope

60 core + 10 LLM references + 6 LLM pilots + 36 final LLM runs + 30 RGB = 142 maximum new runs. Keep the seeds, budgets, methods, proposal families, metric definitions, and comparisons in `docs/PROTOCOL.md` and `plan/matrix.json`. Run the checkpoint-only `curves`, `probes`, `shifts`, `pilot`, and `llm` suites. LLM refinement receives pilot validation metrics only, not final-test outcomes. RGB reports fixed-window return, not first-episode return.

Do not tune after seeing favorable or unfavorable results. A bounded LLM generation failure is a recorded failure, not permission for replacement sampling. After this fixed campaign, stop training and produce the paper figures/statistics, contribution and LLM-use disclosures, final text, and a separately reviewed minimal upstream PR.

## Required agent completion report

Report exact package/source hashes, actual interpreter paths/versions, all smoke/evaluation check results, storage/quota resolution, a per-row completed/failed/missing manifest, actual step counts, reused artifacts with proofs, failed infrastructure attempts, and remaining limitations. Do not report “fully wired” or “paper ready” merely because a launcher exists.

All commands are included in `docs/RUNBOOK.md`; full code and configuration listings are in `ALL_CODE.md`.
