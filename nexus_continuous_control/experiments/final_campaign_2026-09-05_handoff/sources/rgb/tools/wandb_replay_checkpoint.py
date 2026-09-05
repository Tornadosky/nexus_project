"""Replay saved training checkpoints into Weights & Biases.

The campaign runs in ``runs/loop_fix`` were trained with ``--no-wandb``, but the
pickle checkpoints carry the full stacked metrics history plus the deterministic
eval scalars — exactly what ``tracking.replay_history_to_run`` consumes. This
tool re-creates the live W&B runs after the fact, one run per checkpoint, and
prints the run URLs.

Usage (WSL, repo root, venv active):
    python tools/wandb_replay_checkpoint.py runs/loop_fix/batchA/*.pkl \
        --tag loop_fix --group-suffix loop_fix
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from types import SimpleNamespace

from nexus_continuous.tracking import replay_history_to_run, resolve_settings


def replay_checkpoint(path: Path, extra_tags: list[str], group_suffix: str | None) -> str | None:
    import wandb

    with path.open("rb") as fh:
        payload = pickle.load(fh)

    config = dict(payload.get("config") or {})
    output = SimpleNamespace(
        metrics=payload.get("metrics") or {},
        eval_metrics=payload.get("eval_metrics") or {},
    )
    if not output.metrics:
        print(f"[skip] {path.name}: checkpoint has no metrics history")
        return None

    enabled, settings = resolve_settings(config)
    if not enabled:
        print(f"[skip] {path.name}: tracking disabled by checkpoint config")
        return None
    if group_suffix:
        settings["group"] = f"{settings['group']}_{group_suffix}"

    run = wandb.init(
        project=settings["project"],
        entity=settings["entity"],
        mode=settings["mode"],
        group=settings["group"],
        tags=list(settings["tags"]) + extra_tags,
        job_type="replay",
        name=path.stem,
        config={
            **config,
            "commit_hash": payload.get("commit_hash", "unknown"),
            "seed": int(config.get("SEED", 0)),
            "replayed_from": str(path),
        },
        reinit=True,
    )
    try:
        replay_history_to_run(run, config, output, seed_value=int(config.get("SEED", 0)))
        url = run.url
    finally:
        run.finish()
    print(f"[done] {path.name} -> {url}")
    return url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--tag", action="append", default=[], help="Extra W&B tag (repeatable).")
    parser.add_argument(
        "--group-suffix",
        default=None,
        help="Appended to the default <ENV>_<META> group so replays don't mix with live runs.",
    )
    args = parser.parse_args()

    urls: list[str] = []
    for path in args.checkpoints:
        if not path.exists():
            print(f"[skip] {path}: not found")
            continue
        url = replay_checkpoint(path, args.tag, args.group_suffix)
        if url:
            urls.append(url)

    print(f"\nReplayed {len(urls)} checkpoint(s).")
    for url in urls:
        print(url)


if __name__ == "__main__":
    main()
