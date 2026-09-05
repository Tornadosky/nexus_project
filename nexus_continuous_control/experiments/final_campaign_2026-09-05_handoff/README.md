# NEXUS final campaign — audited handoff, installation still required

**Start with `AGENT_HANDOFF.md`.** The live audit found the actual core sources, GitHub RGB branch head, 913 checkpoint files, WSL interpreter, and a reachable but different Viper deployment. It did not finish remote deployment or run training. The partial remote package remains guarded by `INSTALLING`.

This portable package contains the full launcher/evaluator source, 142 per-run YAMLs, explicit machine profiles, source-freezing/integrity checks, the audit report, and a complete phased runbook. It does not contain the user's 42.9 GB of weights or the raw per-file metadata scan. Those remain on the user machine at the paths in `AUDIT_REPORT.md`.

Install into a **fresh sibling**, not over the partial remote package. Run `freeze_sources.py`, then `verify_installation.py --finish`. It archives its own install guard only after code/config checks pass. GPU/API/rendering/restore and capacity gates remain mandatory. No script is a substitute for those results.

- `AUDIT_REPORT.md`: verified observations and explicit unknowns.
- `AGENT_HANDOFF.md`: exact next actions and stop conditions.
- `docs/RUNBOOK.md`: complete installation, Viper, LLM, RGB, and evaluation commands.
- `docs/PROTOCOL.md`: unchanged scientific design and metric definitions.
- `docs/REVIEW_REMOTE.md`: what the audit corrected and what changed in this package.
- `plan/matrix.csv`, `plan/readiness.csv`: all runs, counts, lanes, and prerequisites.
- `audit/observed_state.json`, `audit/source_contract.json`: transcribed evidence and critical source hashes.
- `ALL_CODE.md`: complete executable code and configuration listings.
- `validation.json`: only tests actually run in the assistant sandbox.

The older `docs/RUNBOOK_PRE_AUDIT.md` is historical context, not the active launch route. `docs/REVIEW.md` records the earlier package review; the remote review supersedes its deployment assumptions.
