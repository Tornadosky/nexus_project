"""Live experiment tracking (Weights & Biases) for continuous-control NEXUS.

Coexist design: W&B is the *live* tracking layer only. The offline
``pickle -> tools/collect_nexus_results.py -> CSV -> tools/phase2_validate_results.py``
pipeline remains the authoritative source of truth for research gates. This package
never replaces that pipeline; it mirrors the same per-update metric reduction so the
W&B dashboards match the CSVs exactly.
"""

from nexus_continuous.tracking.wandb_logger import (
    log_training_run,
    replay_history_to_run,
    resolve_settings,
)

__all__ = ["log_training_run", "replay_history_to_run", "resolve_settings"]
