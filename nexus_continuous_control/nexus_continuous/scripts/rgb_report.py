"""Aggregate the rgb_distill_nexus per-variant runs into one report artifact.

Reads runs/rgb_nexus_<meta>/summary.json for each meta variant present, and
writes a grouped state-vs-pixel bar chart, a markdown results table, and a
combined.json. This is the figure/table used in the report.

    python -m nexus_continuous.scripts.rgb_report \
        --runs runs --metas nesy,neural,symbolic --out runs/rgb_report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="runs", help="dir holding rgb_nexus_<meta>/ subdirs")
    ap.add_argument("--metas", default="nesy,neural,symbolic")
    ap.add_argument("--out", default="runs/rgb_report")
    args = ap.parse_args(argv)

    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = Path(args.runs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metas = [m for m in args.metas.split(",") if m.strip()]

    found = []
    for m in metas:
        p = runs / f"rgb_nexus_{m}" / "summary.json"
        if p.exists():
            found.append(json.loads(p.read_text()))
        else:
            print(f"[warn] missing {p} -> skipped")
    if not found:
        raise SystemExit("no summaries found")

    # ---- markdown table ----
    lines = [
        "| meta | state success | pixel success | retention | pixel-fallback |",
        "|------|---------------|---------------|-----------|----------------|",
    ]
    for s in found:
        lines.append(
            f"| {s['meta_policy']} "
            f"| {s['state_hierarchy_success_mean']:.3f} ± {s['state_hierarchy_success_std']:.3f} "
            f"| {s['pixel_hierarchy_success_mean']:.3f} ± {s['pixel_hierarchy_success_std']:.3f} "
            f"| {s['retention_fraction']:.2f} "
            f"| {s.get('pixel_fallback_fraction_mean', 0.0):.3f} |"
        )
    table = "\n".join(lines)
    (out / "results_table.md").write_text(table + "\n")
    print(table)

    # ---- grouped bar chart ----
    names = [s["meta_policy"] for s in found]
    st = np.array([s["state_hierarchy_success_mean"] for s in found])
    ste = np.array([s["state_hierarchy_success_std"] for s in found])
    px = np.array([s["pixel_hierarchy_success_mean"] for s in found])
    pxe = np.array([s["pixel_hierarchy_success_std"] for s in found])
    x = np.arange(len(found))
    w = 0.36
    fig = plt.figure(figsize=(2.4 + 1.6 * len(found), 4))
    plt.bar(x - w / 2, st, w, yerr=ste, capsize=5, label="state (privileged)", color="#4C72B0")
    plt.bar(x + w / 2, px, w, yerr=pxe, capsize=5, label="pixel (distilled)", color="#DD8452")
    for xi, (a, b) in enumerate(zip(st, px)):
        plt.text(xi, max(a, b) + 0.03, f"{100 * b / max(a, 1e-6):.0f}% ret.",
                 ha="center", fontsize=9, color="#333")
    plt.xticks(x, names)
    plt.ylabel("closed-loop success rate")
    plt.ylim(0, 1.12)
    plt.title("NEXUS CartpoleBalance: pixel vs state hierarchy (3 seeds)")
    plt.legend(loc="upper right", fontsize=9)
    fig.savefig(out / "comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    (out / "combined.json").write_text(json.dumps(found, indent=2))
    print("\nwrote", (out / "comparison.png").resolve())
    print("wrote", (out / "results_table.md").resolve())


if __name__ == "__main__":
    main()
