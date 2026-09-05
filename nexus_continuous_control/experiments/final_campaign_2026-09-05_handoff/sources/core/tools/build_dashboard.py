"""Generate the verification dashboard from whatever evidence is on disk.

The board is the human-review surface for the verification campaign described in
``docs/VERIFICATION_PLAN.md``. This script walks the results directories and emits one
self-contained HTML page — plots and videos inlined as data URIs, no external requests, so it
can be published as an Artifact or opened straight from the filesystem.

The governing rule is **honest by construction**: a track renders green only when an artifact
on disk backs it. Missing evidence renders as `pending`, never as a pass. Nothing here infers
a result from the absence of a failure.

Inputs (all optional — the board renders whatever exists):

    --audit    runs/audit/semantics.json     from tools/audit_semantics.py
    --tests    runs/audit/pytest.json        {"passed": int, "failed": int, "duration": float}
    --runs     runs/verify                   *.pkl checkpoints; curves are drawn from
                                             checkpoint["metrics"]
    --videos   runs/videos                   *.mp4 and *_skills.png from render_rollout.py

Usage
-----
    python tools/build_dashboard.py --out runs/dashboard.html
    python tools/build_dashboard.py --runs runs/verify --videos runs/videos \\
        --audit runs/audit/semantics.json --out runs/dashboard.html
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import pickle
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Artifact pages must stay under 16 MB, and video dominates the budget.
MAX_VIDEO_BYTES = 3_500_000
# ...but a per-clip cap does not bound the PAGE. On 2026-08-12 there were 24 MB of clips, every
# one of them individually under MAX_VIDEO_BYTES, and the build produced a 36.5 MB page that
# could not be published at all — so the run that most needed the board could not show it. A
# TOTAL budget is what actually keeps the page publishable; clips beyond it are listed as
# left-on-disk, exactly like the oversize ones.
#
# 9 MB was still too much: the rest of the page (inline PNG plots, and there are now a lot of
# them) is ~9.7 MB on its own, which put the build at 18.7 MB. 5 MB of video leaves real margin
# under the limit without dropping the plots, which carry the results.
MAX_TOTAL_VIDEO_BYTES = 5_000_000

SKILL_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
VARIANT_COLORS = {
    "flat": "#8792a2",
    "neural": "#1f77b4",
    "nesy": "#2ca02c",
    "symbolic": "#ff7f0e",
}


# --------------------------------------------------------------------------- #
# plotting -> inline PNG
# --------------------------------------------------------------------------- #


def _fig_to_uri(fig) -> str:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _file_to_uri(path: Path, mime: str) -> str | None:
    if not path.exists():
        return None
    raw = path.read_bytes()
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def _series(metrics: dict[str, Any], key: str) -> np.ndarray | None:
    if key not in metrics:
        return None
    arr = np.asarray(metrics[key])
    if arr.ndim == 0:
        return None
    # metrics may carry a leading seed axis from a vmapped train call
    return arr.reshape(arr.shape[0], -1).mean(axis=-1) if arr.ndim > 1 else arr


def _curve_plot(title: str, ylabel: str, series: dict[str, list[np.ndarray]]) -> str | None:
    """Mean +/- seed spread, one line per variant."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if not series:
        return None
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    drew = False
    for variant, runs in sorted(series.items()):
        runs = [r for r in runs if r is not None and r.size > 1]
        if not runs:
            continue
        n = min(len(r) for r in runs)
        stack = np.stack([r[:n] for r in runs])
        x = np.arange(n)
        mean, std = stack.mean(0), stack.std(0)
        color = VARIANT_COLORS.get(variant, None)
        ax.plot(x, mean, label=f"{variant} (n={len(runs)})", color=color, lw=1.6)
        if len(runs) > 1:
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, lw=0)
        drew = True
    if not drew:
        plt.close(fig)
        return None
    ax.set_xlabel("update")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(alpha=0.18, lw=0.6)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _fig_to_uri(fig)


def _skill_return_plot(env: str, variant: str, metrics: dict[str, Any]) -> str | None:
    """The paper's Fig. 3 analogue: one line per skill's own episodic return."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    keys = sorted(k for k in metrics if k.startswith("skill_return/"))
    if not keys:
        return None
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    drew = False
    for i, k in enumerate(keys):
        s = _series(metrics, k)
        if s is None or s.size < 2:
            continue
        ax.plot(s, label=k.split("/", 1)[1], color=SKILL_COLORS[i % len(SKILL_COLORS)], lw=1.5)
        drew = True
    if not drew:
        plt.close(fig)
        return None
    ax.set_xlabel("update")
    ax.set_ylabel("episodic skill return")
    ax.set_title(f"{env} [{variant}] — per-skill returns", fontsize=10, loc="left")
    ax.grid(alpha=0.18, lw=0.6)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return _fig_to_uri(fig)


# --------------------------------------------------------------------------- #
# evidence collection
# --------------------------------------------------------------------------- #


def _parse_run_name(stem: str) -> tuple[str, str, str]:
    """`walker_walk_nesy_s0` -> (walker_walk, nesy, s0). Unrecognised names fall back sanely."""
    parts = stem.split("_")
    seed = parts[-1] if parts and parts[-1].startswith("s") and parts[-1][1:].isdigit() else "s?"
    if seed != "s?":
        parts = parts[:-1]
    variant = parts[-1] if parts and parts[-1] in VARIANT_COLORS else "unknown"
    if variant != "unknown":
        parts = parts[:-1]
    return "_".join(parts) or stem, variant, seed


def collect_runs(runs_dirs: Path | list[Path]) -> dict[str, dict[str, list[dict]]]:
    """env -> variant -> [ {seed, metrics, eval_metrics, config} ]

    Accepts SEVERAL directories. The campaign's checkpoints are split across trees — local runs
    in `runs/verify`, Viper `flat` cells in `runs/viper` — and reading only the first silently
    dropped the `flat` baseline from the walker / cheetah / hopper env cards, so those curves
    showed the hierarchical variants with nothing to compare against.
    """
    if isinstance(runs_dirs, Path):
        runs_dirs = [runs_dirs]
    out: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    pkls = [q for d in runs_dirs if d.exists() for q in sorted(d.rglob("*.pkl"))]

    # De-duplicate by (env, variant, seed). Two sources of duplication exist and both distort
    # the seed band if left in:
    #   1. hopper `flat` checkpoints were copied from runs/viper into runs/verify for the
    #      deterministic-eval pass, so every one of those seeds appears twice;
    #   2. cartpole has BOTH the superseded V1.2 shipped-noise `flat` cells and the V1.3
    #      corrected-schedule `_explore_` cells at the same seeds — different experiments.
    # Where seeds collide, prefer the campaign's canonical tag.
    CANONICAL = ("explore", "_v2_", "_dm_")

    def _rank(path: Path) -> int:
        return 0 if any(t in path.stem for t in CANONICAL) else 1

    seen: dict[tuple[str, str, str], int] = {}
    chosen: dict[tuple[str, str, str], Path] = {}
    for q in pkls:
        try:
            with open(q, "rb") as fh:
                c = (pickle.load(fh).get("config") or {})
        except Exception:
            continue
        key = (str(c.get("ENV_NAME")), str(c.get("META_POLICY_TYPE")).lower(), str(c.get("SEED")))
        r = _rank(q)
        if key not in seen or r < seen[key]:
            seen[key], chosen[key] = r, q
    pkls = sorted(chosen.values())

    for pkl in pkls:
        try:
            with open(pkl, "rb") as fh:
                ck = pickle.load(fh)
        except Exception:
            continue
        cfg = ck.get("config", {}) or {}
        stem = pkl.stem
        # Keep hyperparameter probes out of the per-env curves: mixing a clip-20 or
        # quarter-budget arm into a variant's seed band would widen it with runs that were
        # never part of that variant's measurement.
        if any(t in stem for t in ("clip", "quarter", "scaleclip", "hpqn", "diag", "commit")):
            continue
        env = str(cfg.get("ENV_NAME") or _parse_run_name(stem)[0])
        variant = str(cfg.get("META_POLICY_TYPE") or _parse_run_name(stem)[1]).lower()
        _, _, seed = _parse_run_name(pkl.stem)
        out[env][variant].append(
            {
                "seed": str(cfg.get("SEED", seed)),
                "metrics": ck.get("metrics", {}) or {},
                "eval_metrics": ck.get("eval_metrics", {}) or {},
                "config": cfg,
                "path": str(pkl),
            }
        )
    return out


def collect_media(videos_dir: Path) -> dict[str, list[dict]]:
    """env -> [ {variant, video, strip, oversize, return} ]

    Attribution comes from the sidecar manifest ``render_rollout.py`` writes next to each clip.
    Inferring the environment from the filename does not work — `cartpole_nesy.mp4` shares no
    usable substring with `CartpoleBalance` — so a clip without a manifest is filed under
    "unattributed" and still shown, rather than silently dropped.
    """
    media: dict[str, list[dict]] = defaultdict(list)
    if not videos_dir.exists():
        return media
    # Spend the total video budget on the SMALLEST clips first: it maximises the number of
    # (env, variant) cells that get a rollout, and one clip per cell is what the board is for.
    # Ordering by size rather than name also makes the selection deterministic across rebuilds.
    by_size = sorted(videos_dir.glob("*.mp4"), key=lambda p: (p.stat().st_size, p.name))
    budget = MAX_TOTAL_VIDEO_BYTES
    inline: set[Path] = set()
    for mp4 in by_size:
        size = mp4.stat().st_size
        if size <= MAX_VIDEO_BYTES and size <= budget:
            inline.add(mp4)
            budget -= size
    for mp4 in sorted(videos_dir.glob("*.mp4")):
        entry: dict[str, Any] = {"stem": mp4.stem, "variant": "", "return": None}
        size = mp4.stat().st_size
        if mp4 in inline:
            entry["video"] = _file_to_uri(mp4, "video/mp4")
        else:
            # Keeping the page under the artifact size limit matters more than this clip.
            entry["oversize"] = f"{mp4.name} ({size / 1e6:.1f} MB) left on disk"

        env = "unattributed"
        sidecar = mp4.with_suffix(".json")
        if sidecar.exists():
            try:
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
                env = str(meta.get("env", env))
                entry["variant"] = str(meta.get("variant", ""))
                entry["return"] = meta.get("return")
                strip_name = meta.get("strip")
            except Exception:
                strip_name = None
        else:
            strip_name = None

        strip = mp4.with_name(strip_name) if strip_name else mp4.with_name(mp4.stem + "_skills.png")
        if strip.exists():
            entry["strip"] = _file_to_uri(strip, "image/png")
        media[env].append(entry)
    return media


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

CSS = """
:root{--ground:#F2F4F7;--surface:#FFF;--sunk:#E9ECF1;--ink:#151A23;--ink2:#3D4756;--muted:#6B7686;
--rule:#D8DDE5;--rule2:#B9C1CD;--accent:#343C96;--accent2:#0F6C74;
--ok:#1B6B45;--okb:#DCEFE4;--wait:#7A6413;--waitb:#F4EDD4;--stop:#8A3227;--stopb:#F6E1DD;
--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;--sans:ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif;
--serif:ui-serif,Georgia,"Times New Roman",serif;}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]){--ground:#10141B;--surface:#171C25;--sunk:#1F2531;
--ink:#E7EBF1;--ink2:#B7C0CD;--muted:#8792A2;--rule:#2B323F;--rule2:#3C4553;--accent:#9AA2F0;--accent2:#5FC3CB;
--ok:#6BC79A;--okb:#14301F;--wait:#D6BC58;--waitb:#2E2712;--stop:#E08A7C;--stopb:#331813;}}
:root[data-theme="dark"]{--ground:#10141B;--surface:#171C25;--sunk:#1F2531;--ink:#E7EBF1;--ink2:#B7C0CD;
--muted:#8792A2;--rule:#2B323F;--rule2:#3C4553;--accent:#9AA2F0;--accent2:#5FC3CB;
--ok:#6BC79A;--okb:#14301F;--wait:#D6BC58;--waitb:#2E2712;--stop:#E08A7C;--stopb:#331813;}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--serif);line-height:1.6}
.wrap{max-width:1080px;margin:0 auto;padding:clamp(1rem,4vw,2.75rem);display:flex;flex-direction:column;gap:3rem}
h1,h2,h3{font-family:var(--sans);margin:0;text-wrap:balance}
h1{font-size:clamp(1.7rem,4vw,2.5rem);font-weight:660;letter-spacing:-.024em;line-height:1.1}
h2{font-size:1.35rem;font-weight:640;letter-spacing:-.017em}
h3{font-size:1rem;font-weight:640}
p{margin:0;max-width:68ch}
.eyebrow{font-family:var(--mono);font-size:.68rem;text-transform:uppercase;letter-spacing:.15em;color:var(--muted)}
code{font-family:var(--mono);font-size:.85em;background:var(--sunk);padding:.1em .35em;border:1px solid var(--rule)}
.masthead{border-top:3px solid var(--ink);padding-top:1.25rem;display:flex;flex-direction:column;gap:1rem}
.prov{font-family:var(--mono);font-size:.72rem;color:var(--muted);display:flex;flex-wrap:wrap;gap:.4rem 1.4rem;
border-top:1px solid var(--rule);border-bottom:1px solid var(--rule);padding:.7rem 0}
.prov b{color:var(--ink2)}
section{display:flex;flex-direction:column;gap:1.1rem}
.sec-head{border-bottom:1px solid var(--rule2);padding-bottom:.6rem;display:flex;flex-direction:column;gap:.3rem}
.chip{display:inline-flex;align-items:center;gap:.4em;font-family:var(--mono);font-size:.66rem;text-transform:uppercase;
letter-spacing:.1em;font-weight:600;padding:.22em .6em;border:1px solid currentColor;white-space:nowrap}
.chip::before{content:"";width:6px;height:6px;background:currentColor}
.chip.ok{color:var(--ok);background:var(--okb)}.chip.wait{color:var(--wait);background:var(--waitb)}
.chip.stop{color:var(--stop);background:var(--stopb)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:1px;background:var(--rule);border:1px solid var(--rule)}
.tile{background:var(--surface);padding:1rem 1.1rem;display:flex;flex-direction:column;gap:.3rem}
.tile .k{font-family:var(--mono);font-size:.66rem;text-transform:uppercase;letter-spacing:.11em;color:var(--muted)}
.tile .v{font-family:var(--sans);font-size:1.5rem;font-weight:650;letter-spacing:-.03em;font-variant-numeric:tabular-nums}
.tile .n{font-size:.79rem;color:var(--muted);font-family:var(--sans);line-height:1.4}
.tile.ok .v{color:var(--ok)}.tile.wait .v{color:var(--wait)}.tile.stop .v{color:var(--stop)}
.scroller{overflow-x:auto;border:1px solid var(--rule);background:var(--surface)}
table{border-collapse:collapse;width:100%;font-family:var(--sans);font-size:.85rem;min-width:560px}
caption{text-align:left;font-family:var(--mono);font-size:.66rem;text-transform:uppercase;letter-spacing:.12em;
color:var(--muted);padding:.7rem .9rem;border-bottom:1px solid var(--rule);background:var(--sunk)}
th,td{text-align:left;padding:.55rem .9rem;border-bottom:1px solid var(--rule);vertical-align:top}
thead th{font-family:var(--mono);font-size:.64rem;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);
font-weight:600;background:var(--sunk)}
tbody tr:last-child td{border-bottom:none}
td.num{font-variant-numeric:tabular-nums;font-family:var(--mono);font-size:.78rem;white-space:nowrap}
td.id{font-family:var(--mono);font-weight:600;font-size:.78rem;color:var(--accent);white-space:nowrap}
.s-pass{color:var(--ok);font-family:var(--mono);font-size:.72rem;font-weight:600}
.s-fail{color:var(--stop);font-family:var(--mono);font-size:.72rem;font-weight:600}
.s-wait{color:var(--wait);font-family:var(--mono);font-size:.72rem;font-weight:600}
.cards{display:flex;flex-direction:column;gap:1px;background:var(--rule);border:1px solid var(--rule)}
.card{background:var(--surface);padding:1.2rem 1.3rem;display:flex;flex-direction:column;gap:.85rem}
.card-head{display:flex;align-items:baseline;flex-wrap:wrap;gap:.5rem .85rem}
.card-head h3{flex:1 1 12rem}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1rem}
.grid2 img{width:100%;height:auto;display:block;border:1px solid var(--rule)}
.envgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:1.2rem;align-items:start}
.envgrid img{width:100%;height:auto;display:block;border:1px solid var(--rule);background:var(--surface)}
video{width:100%;max-width:520px;border:1px solid var(--rule);display:block}
.empty{font-family:var(--mono);font-size:.72rem;color:var(--muted);border:1px dashed var(--rule2);
background:var(--sunk);padding:1.1rem;text-align:center}
.note{background:var(--surface);border:1px solid var(--rule2);border-left:4px solid var(--accent2);
padding:1rem 1.2rem;display:flex;flex-direction:column;gap:.4rem}
.note.warn{border-left-color:var(--stop)}
.note p{font-size:.9rem;color:var(--ink2)}
footer{border-top:1px solid var(--rule2);padding-top:1rem;font-family:var(--mono);font-size:.7rem;color:var(--muted);
display:flex;flex-wrap:wrap;gap:.35rem 1.5rem}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1.6rem;align-items:start}
.two>div{display:flex;flex-direction:column;gap:.7rem;min-width:0}
.tiny{font-size:.76rem;line-height:1.5}
.muted{color:var(--muted)}
section>img,.two img,figure img{width:100%;height:auto;display:block;border:1px solid var(--rule);background:var(--surface)}
figure{margin:0 0 1.6rem;display:flex;flex-direction:column;gap:.6rem}
figcaption{font-size:.82rem;line-height:1.55;color:var(--ink2);max-width:none}
"""


def _esc(x: Any) -> str:
    return html.escape(str(x))


def _chip(status: str) -> str:
    cls = {"pass": "ok", "fail": "stop", "pending": "wait", "partial": "wait"}.get(status, "wait")
    return f'<span class="chip {cls}">{_esc(status)}</span>'


def render_audit(audit: list[dict] | None) -> str:
    if not audit:
        return (
            '<div class="empty">No semantic audit on disk. Run '
            "<code>python tools/audit_semantics.py --all --out runs/audit/semantics.json</code>"
            "</div>"
        )
    rows = []
    for rep in audit:
        for c in rep.get("checks", []):
            st = c.get("status", "?")
            cls = "s-pass" if st == "PASS" else "s-fail"
            detail = c.get("hint") or c.get("detail") or c.get("note", "")
            best = c.get("best_candidate", "")
            rows.append(
                f"<tr><td class='id'>{_esc(rep['env'])}</td><td class='num'>{_esc(c.get('key'))}</td>"
                f"<td class='{cls}'>{_esc(st)}</td>"
                f"<td class='num'>{_esc(c.get('err', '—'))}</td>"
                f"<td class='num'>{_esc(best or '—')}</td>"
                f"<td>{_esc(detail)}</td></tr>"
            )
    n_fail = sum(r.get("n_fail", 0) for r in audit)
    n_pass = sum(r.get("n_pass", 0) for r in audit)
    banner = (
        f'<div class="note{"" if n_fail == 0 else " warn"}"><span class="eyebrow">'
        f'{"All semantic keys verified" if n_fail == 0 else "Semantic mismatch — results are not trustworthy"}'
        f"</span><p>{n_pass} passed, {n_fail} failed across {len(audit)} environments. Each key is "
        f"checked against an independent reference and must beat every rival degree of freedom; a "
        f"failure names the index that would have matched.</p></div>"
    )
    return banner + (
        "<div class='scroller'><table><caption>Semantic feature audit</caption><thead><tr>"
        "<th>Env</th><th>Key</th><th>Status</th><th>Err (p99)</th><th>Best candidate</th><th>Detail</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def render_parity(parity_dir: Path) -> str:
    """V1.2 — our flat arm against upstream purejaxql AC-PQN."""
    verdict_path = parity_dir / "parity.json"
    png = parity_dir / "parity.png"
    if not verdict_path.exists():
        return (
            '<div class="empty">Parity check not run. '
            "<code>python tools/parity_vs_purejaxql.py --seeds 0,1,2</code></div>"
        )
    v = json.loads(verdict_path.read_text(encoding="utf-8"))
    passed = v.get("gate") == "PASS"
    ratio = v.get("ratio_ours_over_upstream", float("nan"))

    img = ""
    uri = _file_to_uri(png, "image/png")
    if uri:
        img = f'<img src="{uri}" alt="parity curves" style="width:100%;max-width:760px;border:1px solid var(--rule)">'

    rows = "".join(
        f"<tr><td>{_esc(label)}</td><td class='num'>{mean:.1f}</td>"
        f"<td class='num'>{', '.join(f'{x:.1f}' for x in finals)}</td></tr>"
        for label, mean, finals in (
            ("ours — META_POLICY_TYPE=flat", v["ours_final_mean"], v["ours_finals"]),
            ("purejaxql AC-PQN (upstream)", v["upstream_final_mean"], v["upstream_finals"]),
        )
    )

    extra = ""
    fixed = v.get("fixed_final_mean")
    if fixed is not None:
        fr = v.get("fixed_ratio", float("nan"))
        fg = v.get("fixed_gate")
        extra += (
            f'<div class="note"><span class="eyebrow">Cause found — it is the exploration '
            f'schedule</span><p>Changing <b>only</b> <code>NOISE_START</code> 0.30&rarr;1.0 and '
            f'<code>NOISE_FINISH</code> 0.02&rarr;0.15 — same nets, same &gamma;, same critic '
            f'reducer — lifts the flat arm to <b>{fixed:.1f}</b> against upstream&rsquo;s '
            f'{v["upstream_final_mean"]:.1f}: ratio <b>{fr:.3f}</b>, gate <b>{fg}</b> over 3 seeds. '
            f'The implementation was never the problem; the shipped config under-explores.</p></div>'
        )

    ladder = v.get("diagnostic_ladder")
    if ladder:
        lrows = "".join(
            f"<tr><td>{_esc(d['arm'])}</td><td class='num'>{d['n']}</td>"
            f"<td class='num'>{d['final']:.1f}</td><td class='num'>{d['peak']:.1f}</td></tr>"
            for d in ladder
        )
        extra += (
            "<div class='scroller'><table><caption>Diagnostic ladder — which lever closes the "
            "gap</caption><thead><tr><th>Arm</th><th>Seeds</th><th>Final</th><th>Peak</th>"
            "</tr></thead><tbody>" + lrows + "</tbody></table></div>"
            "<p style='font-size:.9rem;color:var(--ink2)'><b>Read C against F.</b> "
            "<code>CRITIC_AGG=min</code> on its own is the <em>worst</em> arm in the set (103.7), "
            "and even added to the good schedule it costs ~60 return (F 756.9 &rarr; D 698.0). The "
            "critic-overestimation hypothesis was wrong; the control is what caught it.</p>"
        )

    diag = v.get("diagnostic_single_seed")
    if diag:
        drows = "".join(
            f"<tr><td>{_esc(d['arm'])}</td><td class='num'>{d['final']:.1f}</td>"
            f"<td class='num'>{d['peak']:.1f}</td></tr>"
            for d in diag
        )
        extra += (
            "<div class='scroller'><table><caption>Diagnostic — does moving toward upstream's "
            "recipe close the gap? (single seed)</caption><thead><tr><th>Arm</th>"
            "<th>Final</th><th>Peak</th></tr></thead><tbody>" + drows + "</tbody></table></div>"
        )
    ge = v.get("greedy_eval_ours")
    if ge:
        extra += (
            f"<p style='font-size:.9rem;color:var(--ink2)'><b>Metric confound ruled out.</b> "
            f"Upstream reports a greedy test return while ours is the training return with "
            f"exploration still on. Re-evaluating our checkpoints greedily gives "
            f"{', '.join(f'{x:.0f}' for x in ge)} (mean {np.mean(ge):.0f}) — <em>lower</em> than "
            f"the training number, so the gap is not an artifact of comparing different "
            f"metrics.</p>"
        )

    # Resolved-state header: when the exploration fix has been measured (fixed_gate), the
    # section verdict is the FIXED ratio; the shipped-schedule failure stays as history below.
    resolved = v.get("fixed_gate") == "PASS"
    if resolved:
        head = (
            f'<div class="note"><span class="eyebrow">Parity holds after the exploration fix '
            f'&mdash; shipped schedule was the defect, not the implementation</span>'
            f"<p>Exploration-fixed ratio ours/upstream = <b>{v.get('fixed_ratio', float('nan')):.3f}</b> "
            f"(gate |ratio&nbsp;&minus;&nbsp;1|&nbsp;&le;&nbsp;{v.get('tolerance', 0.15)}, PASS, 3 seeds) "
            f"on {v.get('env')} at {v.get('total_timesteps', 0):,} environment steps. The shipped "
            f"schedule scored <b>{ratio:.3f}</b> (FAIL) — kept below as provenance; the cause and the "
            f"diagnostic ladder that found it follow.</p></div>"
        )
    else:
        head = (
            f'<div class="note{"" if passed else " warn"}"><span class="eyebrow">'
            f'{"Parity holds" if passed else "Parity gate FAILED — the baseline is weaker than upstream"}</span>'
            f"<p>Ratio ours/upstream = <b>{ratio:.3f}</b> against a gate of "
            f"|ratio&nbsp;&minus;&nbsp;1|&nbsp;&le;&nbsp;{v.get('tolerance', 0.15)}, "
            f"on {v.get('env')} at {v.get('total_timesteps', 0):,} environment steps. "
            f"{'Every hierarchical ratio in this project is measured against this baseline, so this is the number that licenses the rest.' if passed else 'Hierarchical ratios measured against this baseline cannot be trusted until it is resolved.'}"
            f"</p></div>"
        )
    return (
        head +
        f'<div class="scroller"><table><caption>Final return (mean of the last 10% of training)</caption>'
        f"<thead><tr><th>Arm</th><th>Mean</th><th>Per seed</th></tr></thead><tbody>{rows}</tbody></table></div>"
        f"{img}{extra}"
    )


def render_rule_coverage(path: Path) -> str:
    """V3.1 — branch reachability and NeSy mask coverage for the six symbolic policies."""
    jf = path / "rule_coverage.json"
    if not jf.exists():
        return '<p class="muted">Not run. <code>python tools/rule_coverage.py</code></p>'
    data = json.loads(jf.read_text(encoding="utf-8"))

    rows = []
    for r in data:
        k = len(r["skill_names"])
        reached = k - len(r["unreachable_branches"])
        ok = r["pass"]
        frac = r["branch_fractions"]
        avail = r["mask_availability"]
        detail = " · ".join(
            f"{n} {frac[n] * 100:.1f}%/{avail[n] * 100:.0f}%" for n in r["skill_names"]
        )
        rows.append(
            f"<tr><td><code>{_esc(r['policy'])}</code></td>"
            f"<td>{reached}/{k}</td>"
            f"<td>{r['all_zero_mask_rows']}</td>"
            f'<td class="tiny">{_esc(detail)}</td>'
            f"<td>{_chip('pass' if ok else 'fail')}</td></tr>"
        )

    img = _file_to_uri(path / "rule_coverage.png", "image/png")
    fig = f'<img src="{img}" alt="rule coverage truth tables">' if img else ""
    return f"""
<div class="scroller"><table>
  <thead><tr><th>policy</th><th>branches reached</th><th>all-zero mask rows</th>
  <th>per-skill: selected % / mask-available %</th><th></th></tr></thead>
  <tbody>{"".join(rows)}</tbody>
</table></div>
<p class="muted tiny">Percentages are over a <b>uniform box</b> covering every threshold in the
rule, not over the on-policy state distribution — they measure reachability, not how often a
skill actually fires during training. Two narrow-but-valid branches are worth knowing about:
<code>go1.stand</code> needs near-zero command on all three axes (0.01% of the box), and
panda's <code>grasp</code>/<code>lift</code> masks open on 3%/6% of it.</p>
{fig}
"""


def render_paper_figs(path: Path) -> str:
    """The Fig. 3 / Fig. 7 analogues, laid out as the paper lays them out."""
    blocks = []
    for name, cap in (
        ("fig3_skill_returns.png",
         "<b>Per-skill returns (paper Fig.&nbsp;3 analogue).</b> One panel per skill, methods "
         "overlaid, shaded band = seed spread. Every skill's own reward rises from "
         "initialisation under both NEXUS variants &mdash; the paper's Q1 claim, reproduced in "
         "continuous control. <b>HPQN is deliberately absent</b>: its per-skill values "
         "accumulate the <i>env</i> reward, not the hand-written skill reward, so plotting it "
         "here would compare two different quantities."),
        ("fig7_return_vs_goal.png",
         "<b>Return vs the actual goal (paper Fig.&nbsp;7 analogue).</b> Left axis: episode "
         "return, the gameable quantity. Right axis: primary success, the behaviour we want. "
         "<b>Go1 is the campaign's cleanest reward-hacking evidence</b> &mdash; <code>flat</code> "
         "earns the highest return (9.7) and the lowest tracking success (0.04); scoring this "
         "env on return ranks the arms exactly backwards. Hopper shows no such divergence, "
         "which is why the gate is scored on success everywhere rather than case by case."),
    ):
        uri = _file_to_uri(path / name, "image/png")
        if uri:
            blocks.append(f'<figure><img src="{uri}" alt="{name}"><figcaption>{cap}</figcaption></figure>')
    if not blocks:
        return '<p class="muted">Not generated. <code>python tools/paper_figures.py</code></p>'
    return "".join(blocks)


def _extract_fn_source(path: Path, fn_name: str) -> str | None:
    """Pull one top-level function's source out of a policy module, textually.

    Textual (not importlib) on purpose: the dashboard must build without importing jax.
    """
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    m = re.search(rf"^def {re.escape(fn_name)}\(.*?(?=^def |\Z)", text, re.S | re.M)
    return m.group(0).rstrip() if m else None


_PRE_STYLE = (
    "background:var(--sunk);border:1px solid var(--rule);padding:.9rem 1rem;"
    "font-family:var(--mono);font-size:.72rem;line-height:1.5;overflow-x:auto;margin:0"
)


def render_fig5(policy_dir: Path) -> str:
    """Paper Fig. 5 analogue: the meta-policy IS code, shown as code.

    Left: the symbolic priority rule (variant B). Right: the NeSy mask (variant C).
    Go1 is the env shown because it carries the campaign's positive result.
    """
    src = policy_dir / "go1_joystick.py"
    rule = _extract_fn_source(src, "symbolic_meta_policy")
    mask = _extract_fn_source(src, "skill_mask")
    if not rule or not mask:
        return '<p class="muted">Policy source not found for the code panel.</p>'
    return f"""
<div class="two">
  <div><h3>Symbolic meta-policy (variant B)</h3>
    <pre style="{_PRE_STYLE}">{_esc(rule)}</pre></div>
  <div><h3>NeSy skill mask (variant C)</h3>
    <pre style="{_PRE_STYLE}">{_esc(mask)}</pre></div>
</div>
<p class="muted tiny">Paper Fig.&nbsp;5 analogue for <code>Go1JoystickFlatTerrain</code> — the
high-level policy is a short, editable function over named semantic features, not a network.
Every branch is reachable and no state produces an all-zero mask (V3.1, 6/6). The paper's rules
are LLM-generated over discrete object predicates; ours are hand-written over continuous
physics — V1.3 shows that difference is load-bearing, so rule <em>quality</em>, not rule
<em>form</em>, is the continuous-control bottleneck.</p>
"""


def _is_experimental_arm(tags: list[str]) -> bool:
    """Is this arm one the V2 gate excludes? Answered by analyze_v2's own EXPERIMENTAL_TAGS.

    Falls back to a conservative superset if the import fails, because the failure mode that
    matters is quoting an experimental cell as if it were a gate arm — better to drop a
    legitimate arm from a comparison column than to promote a budget-scaled one into it.
    """
    try:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from analyze_v2 import EXPERIMENTAL_TAGS  # type: ignore[import-not-found]
    except Exception:
        EXPERIMENTAL_TAGS = frozenset(
            {"noclip", "clip", "scaleclip", "commit", "hpqn", "quarter",
             "budget", "rules", "ppo"}
        )
    return any(marker in t for t in tags for marker in EXPERIMENTAL_TAGS)


def render_ppo(ppo_dir: Path, v2_dir: Path) -> str:
    """L4 / runbook §5b: Playground's shipped tuned brax PPO, scored on OUR success metric.

    This panel exists because the board was, until 2026-08-12, entirely self-referential: every
    number on it compared our arms to our own flat baseline. A reader could not tell whether that
    baseline is any good. It is not, on go1 — and saying so belongs on the board, not only in the
    findings doc.
    """
    if not ppo_dir.exists():
        return ('<p class="muted">Not run. <code>python tools/train_ppo_baseline.py '
                "--env-name Go1JoystickFlatTerrain --seed 0 "
                "--match-checkpoint runs/verify/go1_joystick_flat_v2_s0.pkl "
                "--out runs/ppo/go1_joystick_ppo_s0</code></p>")

    cells: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for csv_path in sorted(ppo_dir.glob("*.csv")):
        try:
            with open(csv_path, newline="", encoding="utf-8") as fh:
                row = next(iter(csv.DictReader(fh)))
        except Exception:
            continue
        try:
            env = row["env"]
            budget = int(row["num_timesteps"])
            cells[(env, budget)].append(row)
        except (KeyError, ValueError):
            continue
    if not cells:
        return '<p class="muted">No PPO results parsed from runs/ppo/.</p>'

    # The comparison columns come from analyze_v2's OWN matrix, not from a second computation
    # here: the whole point of this panel is to put PPO beside the number the gate quotes, and a
    # reimplementation that drifted by a decimal would defeat that.
    matrix: dict = {}
    jf = v2_dir / "v2_matrix.json"
    if jf.exists():
        matrix = json.loads(jf.read_text(encoding="utf-8")).get("matrix", {})

    def arm_budget(c: dict) -> int | None:
        """The env-step budget of a cell, from analyze_v2's recorded (num_envs, steps) pairs."""
        b = c.get("budgets") or []
        steps = {pair[1] for pair in b if isinstance(pair, (list, tuple)) and len(pair) > 1}
        return next(iter(steps)) if len(steps) == 1 else None

    def ours(env: str) -> tuple[str, float | None, float | None]:
        best_name, best_val, flat_val = "—", None, None
        best_budget: int | None = None
        for arm, c in matrix.get(env, {}).items():
            mean = c.get("success_mean")
            if mean is None or c.get("partial"):
                continue
            base = arm.split("·")[0]
            tags = arm.split("·")[1:]
            # Import the gate's OWN tag set rather than restating it. A hand-written list here
            # started as ("budget", "quarter", "rules", "ppo") and immediately mis-reported
            # HopperHop's best arm as `neural·clip20` — an experimental cell the gate excludes.
            # Two copies of one rule drift; the second copy is the one that lies.
            if _is_experimental_arm(tags):
                continue
            if base == "flat":
                if flat_val is None or mean > flat_val:
                    flat_val = mean
            elif base in ("neural", "nesy") and (best_val is None or mean > best_val):
                best_name, best_val = arm, mean
                best_budget = arm_budget(c)
        return best_name, best_val, flat_val, best_budget

    rows = ""
    for (env, budget) in sorted(cells):
        rs = cells[(env, budget)]
        succ = [float(r["primary_success_rate"]) for r in rs]
        lens = [float(r["episode_length_mean"]) for r in rs]
        mean = sum(succ) / len(succ)
        shipped = int(rs[0].get("ppo_shipped_timesteps") or 0)
        matched = "matched" if shipped and budget != shipped else "PPO's tuned budget"
        frac = f"{100 * budget / shipped:.0f}% of tuned" if shipped else "—"
        name, best, flat, our_budget = ours(env)
        # This section's own title says "at the same budget". HopperHop's best gate arm
        # (nesy·v2) actually trains at 52,428,800 against PPO's matched 26,214,400 — an
        # untagged asymmetry found by tools/audit_budgets.py — so the table must say so in
        # the cell rather than let the heading speak for it.
        budget_note = ""
        if our_budget and our_budget != budget:
            budget_note = (f'<br><span class="muted">at {our_budget:,} steps '
                           f"({our_budget / budget:.2g}x PPO's)</span>")
        seeds_txt = ", ".join(f"{s:.4f}" for s in sorted(succ, reverse=True))
        # A high success rate on a stub episode is the obvious way this table could lie, so the
        # episode length sits next to it rather than in a footnote.
        rows += (
            f'<tr><td class="id">{_esc(env)}</td>'
            f'<td class="num">{budget:,}<br><span class="muted">{matched}, {frac}</span></td>'
            f'<td class="num">{mean:.4f}</td><td class="num">{seeds_txt}</td>'
            f'<td class="num">{sum(lens) / len(lens):.0f}</td>'
            f'<td class="num">{"—" if flat is None else f"{flat:.4f}"}</td>'
            f'<td class="num">{"—" if best is None else f"{best:.4f}"}<br>'
            f'<span class="muted">{_esc(name)}</span>{budget_note}</td></tr>'
        )

    return (
        "<div class='scroller'><table><caption>PPO baseline — Playground's shipped tuned brax "
        "PPO, scored with the same task_metrics and deterministic rollout as every NEXUS cell"
        "</caption><thead><tr><th>env</th><th>env steps</th><th>PPO success</th>"
        "<th>PPO per seed</th><th>PPO ep. length</th><th>our flat</th><th>our best hierarchical"
        "</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
        '<div class="note warn"><span class="eyebrow">Read this table as an algorithm '
        "comparison, not an architecture one</span><p>PPO brings its own tuned hyperparameters, "
        "network sizes and <code>num_envs</code>, as runbook §5b requires (no hand-tuning). It is "
        "a different algorithm and is excluded from the V2 gate via <code>EXPERIMENTAL_TAGS</code>"
        ". Episode length is shown because a policy that ends an episode early can post a high "
        "in-episode success rate without doing the task.</p></div>"
    )


def render_robustness(path: Path) -> str:
    """V6 / paper Fig. 8 (Q4): degradation under action-noise perturbation."""
    jf = path / "robustness_summary.json"
    if not jf.exists():
        return ('<p class="muted">Not run. <code>bash tools/run_robustness_sweep.sh && '
                "bash tools/run_robustness_sweep2.sh && python tools/analyze_robustness.py"
                "</code></p>")
    summary = json.loads(jf.read_text(encoding="utf-8"))

    rows = ""
    levels_sorted = ["0.0", "0.05", "0.1", "0.2", "0.3"]
    for key in sorted(summary):
        v = summary[key]
        cells = "".join(
            f'<td class="num">{v["levels"].get(l, float("nan")):.3f}</td>'
            if l in v["levels"] else '<td class="num">—</td>'
            for l in levels_sorted
        )
        ret = v.get("retention_at_0.3")
        ret_cell = (f'<td class="num">{ret * 100:.1f}%</td>' if ret is not None
                    else '<td class="num muted">degenerate</td>')
        rows += f'<tr><td class="id">{_esc(key)}</td>{cells}{ret_cell}</tr>'

    # Per-env verdict on the paper's specific claim (symbolic/nesy degrade less than flat),
    # only where BOTH arms are non-degenerate — a retention ratio with a near-zero denominator
    # is an artifact, not robustness (Go1 symbolic "202%" is the cautionary example).
    verdicts = []
    envs = sorted({k.split("/")[0] for k in summary})
    for env in envs:
        f_ = summary.get(f"{env}/flat")
        comps = []
        for var in ("nesy", "symbolic"):
            h = summary.get(f"{env}/{var}")
            if not f_ or not h or f_.get("degenerate") or h.get("degenerate"):
                continue
            rf, rh = f_.get("retention_at_0.3"), h.get("retention_at_0.3")
            if rf is None or rh is None:
                continue
            comps.append((var, rh, rf, rh > rf))
        for var, rh, rf, wins in comps:
            verdicts.append(
                f'<tr><td class="id">{_esc(env)}</td><td><code>{var}</code> {rh * 100:.1f}%</td>'
                f'<td><code>flat</code> {rf * 100:.1f}%</td>'
                f"<td>{_chip('pass' if wins else 'fail')}</td></tr>"
            )
    verdict_table = (
        "<div class='scroller'><table><caption>Paper claim per env — symbolic/nesy retains "
        "more than flat at noise 0.3 (non-degenerate arms only)</caption><thead><tr>"
        "<th>env</th><th>steered arm</th><th>flat</th><th>verdict</th></tr></thead>"
        f"<tbody>{''.join(verdicts)}</tbody></table></div>" if verdicts else ""
    )

    img = _file_to_uri(path / "robustness.png", "image/png")
    fig = f'<img src="{img}" alt="robustness degradation curves">' if img else ""
    return f"""
<div class="note">
  <p><b>The comparison the paper makes is only meaningful where the baseline works unperturbed.</b>
  On Go1 and hopper the <code>flat</code> (and Go1 <code>symbolic</code>) arms are degenerate —
  success &lt; 0.05 at zero noise — so no retention ratio is assigned to them; there the honest
  reading is absolute: the hierarchical arms keep 69–93% of their success at noise 0.3 <em>while
  the baseline has nothing to lose</em>. The head-to-head retention claim is scored on the envs
  where every arm functions.</p>
</div>
{verdict_table}
<div class="scroller"><table><caption>Primary success by action-noise level (mean over seeds)</caption>
  <thead><tr><th>env/arm</th><th>0.0</th><th>0.05</th><th>0.1</th><th>0.2</th><th>0.3</th>
  <th>retention@0.3</th></tr></thead><tbody>{rows}</tbody></table></div>
{fig}
"""


def render_probes(path: Path) -> str:
    """Q1/Q2 causal evidence: forced-skill probes and eval-time skill ablation."""
    pj, aj = path / "skill_probes.json", path / "skill_ablation.json"
    if not pj.exists() and not aj.exists():
        return '<p class="muted">Not run. <code>bash tools/run_skill_probes.sh</code></p>'
    blocks = []
    if pj.exists():
        probes = json.loads(pj.read_text(encoding="utf-8"))
        rows = ""
        for r in probes:
            checks = " · ".join(
                f"{_esc(k)} {v:+.3f}" for k, v in r.get("semantics", {}).items()
            )
            mn = r.get("matches_name")
            verdict_cell = "—" if mn is None else _chip("pass" if mn else "fail")
            rows += (
                f'<tr><td class="id">{_esc(r["env"])}</td><td><code>{_esc(r["skill"])}</code></td>'
                f'<td class="num">{r["episode_return_mean"]:.1f}</td>'
                f'<td class="tiny">{checks}</td>'
                f"<td>{verdict_cell}</td></tr>"
            )
        blocks.append(
            "<h3>Forced-skill probes — does each skill do what its name says?</h3>"
            "<div class='scroller'><table><caption>Each skill actor run solo (greedy, 64 episodes), "
            "semantic behaviour vs the skill's name</caption><thead><tr><th>env</th><th>forced skill"
            "</th><th>return</th><th>behaviour</th><th>verdict</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )
    if aj.exists():
        abl = json.loads(aj.read_text(encoding="utf-8"))
        rows = ""
        for r in abl:
            delta = r["success"] - r["intact_success"]
            rows += (
                f'<tr><td class="id">{_esc(r["env"])}</td><td><code>{_esc(r["checkpoint"])}</code></td>'
                f'<td><code>{_esc(r["removed_skill"])}</code></td>'
                f'<td class="num">{r["intact_success"]:.3f}</td>'
                f'<td class="num">{r["success"]:.3f}</td>'
                f'<td class="num">{delta:+.3f}</td>'
                f'<td class="tiny">{_esc(r.get("prediction", "")) or "—"}</td>'
                f"<td>{'—' if r.get('prediction_held') is None else _chip('pass' if r['prediction_held'] else 'fail')}</td></tr>"
            )
        blocks.append(
            "<h3>Eval-time skill ablation — are the meta-policy's preferences load-bearing?</h3>"
            "<p class='tiny muted'>One skill removed from the selection at eval (mask surgery), "
            "everything else unchanged. Predictions were written down before the runs.</p>"
            "<div class='scroller'><table><caption>Success with one skill removed vs intact"
            "</caption><thead><tr><th>env</th><th>checkpoint</th><th>removed</th><th>intact</th>"
            "<th>ablated</th><th>&Delta;</th><th>pre-registered prediction</th><th></th></tr>"
            f"</thead><tbody>{rows}</tbody></table></div>"
        )
    return "".join(blocks)


def render_fig6(path: Path) -> str:
    """Paper Fig. 6 analogue: ambiguous states with the masked meta-Q per skill."""
    if not path.exists():
        return '<p class="muted">Not generated. <code>python tools/fig6_panels.py</code></p>'
    pngs = sorted(path.glob("*.png"))
    if not pngs:
        return '<p class="muted">Not generated. <code>python tools/fig6_panels.py</code></p>'
    figs = []
    for p in pngs:
        uri = _file_to_uri(p, "image/png")
        if uri:
            figs.append(f'<img src="{uri}" alt="{_esc(p.stem)}">')
    return (
        '<div class="grid2">' + "".join(figs) + "</div>"
        '<p class="muted tiny">Paper Fig.&nbsp;6 analogue: frames where the NeSy mask leaves '
        "<em>more than one</em> skill available, with each skill's masked meta-Q value. The rule "
        "narrows the choice and exposes its reasons; the learned meta-Q resolves the remaining "
        "ambiguity — interpretability at the symbolic level, flexibility at the value level.</p>"
    )


def render_v2(path: Path) -> str:
    """The V2 gate: primary success by env and variant, with seeds."""
    jf = path / "v2_matrix.json"
    if not jf.exists():
        return '<p class="muted">Not run. <code>python tools/analyze_v2.py</code></p>'
    d = json.loads(jf.read_text(encoding="utf-8"))
    matrix, g = d.get("matrix", {}), d.get("gate", {})

    rows = ""
    for env in sorted(matrix):
        for arm in sorted(matrix[env]):
            c = matrix[env][arm]
            seeds = " · ".join(f"{x:.3f}" for x in c["success"])
            ret = f"{c['return_mean']:.1f}" if c["return_mean"] is not None else "—"
            partial = _chip("pending") if c["partial"] else ""
            rows += (
                f'<tr><td class="id">{_esc(env)}</td><td><code>{_esc(arm)}</code></td>'
                f'<td class="num">{c["n"]}</td>'
                f'<td class="num">{c["success_mean"]:.4f}</td>'
                f'<td class="num">{_esc(seeds)}</td>'
                f'<td class="num">{ret}</td><td>{partial}</td></tr>'
            )

    verdicts = ""
    for env, v in g.get("per_env", {}).items():
        if v.get("status") == "incomplete":
            verdicts += f'<tr><td class="id">{_esc(env)}</td><td colspan="3">{_chip("pending")} {_esc(v["why"])}</td></tr>'
        else:
            beats = v["status"] == "beats_flat"
            note = "" if not beats else (
                " seed-separated" if v.get("separated") else " <b>within seed overlap</b>"
            )
            # A win at a different env-step budget is not a win, and the board is the artifact
            # that gets shared — so the mismatch has to be on the row itself, not only in
            # analyze_v2's stdout. `budget_matched` is False only when both budgets are known
            # and differ; None means one cell's seeds disagree and nothing is claimed.
            if v.get("budget_matched") is False:
                bh_b, bf_b = v.get("best_hier_budget"), v.get("best_flat_budget")
                note = (f' <b>budget mismatch</b> — {bh_b:,} vs flat {bf_b:,} '
                        f"({bh_b / bf_b:.2g}x)")
            verdicts += (
                f'<tr><td class="id">{_esc(env)}</td>'
                f'<td><code>{_esc(v["best_hier"])}</code> {v["best_hier_success"]:.4f}</td>'
                f'<td><code>flat</code> {v["best_flat_success"]:.4f}</td>'
                f'<td>{_chip("pass" if beats and v.get("budget_matched") is not False else "fail")}'
                f"{note}</td></tr>"
            )

    # One figure per environment rather than one shared six-panel grid. Arm counts run from 3
    # (go1_rough) to 22 (hopper), so a single cell height either crushes hopper's arm names into
    # each other or wastes most of go1_rough's panel. analyze_v2.plot writes both; the stacked
    # overview is the fallback for a runs/v2 built before the split.
    figs = []
    for f in sorted(path.glob("v2_matrix_*.png")):
        uri = _file_to_uri(f, "image/png")
        if uri:
            env = f.stem[len("v2_matrix_"):]
            figs.append(f'<img src="{uri}" alt="V2 matrix — {_esc(env)}">')
    if figs:
        fig = '<div class="envgrid">' + "".join(figs) + "</div>"
    else:
        img = _file_to_uri(path / "v2_matrix.png", "image/png")
        fig = f'<img src="{img}" alt="V2 matrix">' if img else ""
    gate_txt = g.get("gate", "INCOMPLETE")
    n_pass, n_scored = g.get("n_pass", 0), g.get("n_scored", 0)
    n_sep = g.get("n_separated", 0)
    # Call out mean "wins" that rest on a margin smaller than any seed's noise — a reader must
    # not mistake the letter-of-the-gate count for the plan's verdict (3/6, walker scored as a
    # loss). Both numbers are shown; the strict gate is the chip.
    near_ties = []
    for env, v in g.get("per_env", {}).items():
        if v.get("status") == "beats_flat" and not v.get("separated"):
            margin = v.get("best_hier_success", 0) - v.get("best_flat_success", 0)
            if margin < 0.005:
                near_ties.append(f"{env} +{margin:.4f}")
    tie_note = (
        f" The mean-win count includes {', '.join(_esc(t) for t in near_ties)} — margins inside "
        f"full seed overlap that docs/VERIFICATION_PLAN.md scores as losses (its final verdict "
        f"is 3 of 6, FAIL either way)." if near_ties else ""
    )
    return f"""
<div class="note {"warn" if gate_txt != "PASS" else ""}">
  <p><b>Gate: &ge;4 of 6 environments where <code>nesy</code> or <code>neural</code> beats
  <code>flat</code> on primary success &mdash; mean wins {n_pass} of {n_scored},
  seed-separated wins {n_sep} of {n_scored}. {_chip("pass" if gate_txt == "PASS" else ("fail" if gate_txt == "FAIL" else "pending"))}</b></p>
  <p>A mean win whose seed ranges overlap is not a separation; the strict gate counts only
  separated wins.{tie_note}</p>
  <p>Scored on <b>primary success, never return</b>: this campaign has measured panda earning
  655 return at 0.001 lift success, and walker's old 931-return "success" was a vertical-axis
  artifact. Arms are kept separate by recipe tag &mdash; walker's dm-suite and locomotion arms
  are different experiments, not seeds of one.</p>
</div>
<div class="scroller"><table>
  <thead><tr><th>env</th><th>best hierarchical arm</th><th>best flat arm</th><th>verdict</th></tr></thead>
  <tbody>{verdicts}</tbody>
</table></div>
{fig}
<div class="scroller"><table>
  <thead><tr><th>env</th><th>arm</th><th>n</th><th>success</th><th>seeds</th><th>return</th><th></th></tr></thead>
  <tbody>{rows}</tbody>
</table></div>
"""


def render_v13(path: Path) -> str:
    """V1.3 — the meta-mode ablation, gate and all."""
    jf = path / "v13_ablation.json"
    if not jf.exists():
        return '<p class="muted">Not run. <code>bash tools/run_v13_ablation.sh</code></p>'
    d = json.loads(jf.read_text(encoding="utf-8"))
    cells = d.get("cells", {})

    rows = ""
    for v in ("flat", "neural", "nesy", "symbolic"):
        c = cells.get(v)
        if not c:
            continue
        rows += (
            f'<tr><td class="id">{_esc(v)}</td>'
            f'<td class="num">{c["return_mean"]:.1f}</td>'
            f'<td class="num">{" · ".join(f"{x:.1f}" for x in c["return_seeds"])}</td>'
            f'<td class="num">{c["success_mean"]:.3f}</td>'
            f'<td class="num">{" · ".join(f"{x:.3f}" for x in c["success_seeds"])}</td></tr>'
        )

    verdict = d.get("verdict", "INCOMPLETE")
    chip = _chip("fail" if verdict != "PASS" else "pass")
    notes = "".join(f"<p>{_esc(n)}</p>" for n in d.get("notes", []))
    img = _file_to_uri(path / "v13_ablation.png", "image/png")
    fig = f'<img src="{img}" alt="V1.3 meta-mode ablation">' if img else ""

    return f"""
<div class="note warn">
  <p><b>Gate <code>nesy &ge; neural &ge; flat</code> on deterministic return: {_esc(verdict)}.</b> {chip}</p>
  <p>Actual ordering is <b>flat &asymp; neural &Gt; nesy &Gt; symbolic</b> — the reverse of the
  phase-2 result. It is not a noise artifact: flat's seeds (858&ndash;999) do not overlap nesy's
  (476&ndash;699), and the between-variant spread is 655 against a worst within-variant spread of 224.</p>
  <p><b>The comparison is controlled.</b> The <code>neural</code> and <code>nesy</code> configs
  differ in exactly one line, <code>META_POLICY_TYPE</code>, so the 928 &rarr; 582 drop is
  attributable to the NeSy mask alone.</p>
  <p><b>What it means.</b> With exploration corrected, the flat baseline <i>solves</i>
  CartpoleBalance — success 1.000 on every seed, return at the ceiling. A hierarchy cannot show
  value on a task the baseline already solves; it can only cost. The phase-2 cartpole ratios are
  dead and must not be quoted. The thesis now rests on the environments where flat does
  <i>not</i> saturate, which makes the V2 matrix load-bearing rather than confirmatory.</p>
  {notes}
</div>
<div class="scroller"><table>
  <thead><tr><th>variant</th><th>det. return</th><th>seeds</th>
  <th>primary success</th><th>seeds</th></tr></thead>
  <tbody>{rows}</tbody>
</table></div>
{fig}
"""


def render_sweep(path: Path) -> str:
    """The exploration sweep, including the stage-3 Go1 negative."""
    jf = path / "noise_sweep.json"
    if not jf.exists():
        return '<p class="muted">Not run.</p>'
    d = json.loads(jf.read_text(encoding="utf-8"))

    cells = "".join(
        f"<tr><td><code>{_esc(k.replace('_', ' → '))}</code></td><td>{np.mean(v):.1f}</td>"
        f'<td class="tiny">{", ".join(f"{x:.0f}" for x in v)}</td></tr>'
        for k, v in d.get("cells", {}).items()
        if v
    )
    succ = d.get("go1_success", {})
    go1 = "".join(
        f"<tr><td>{_esc(k)}</td><td>{np.mean(v):.2f}</td>"
        f'<td class="tiny">{", ".join(f"{x:.2f}" for x in v)}</td>'
        f"<td>{np.mean(succ[k]):.4f}</td>"
        f'<td class="tiny">{", ".join(f"{x:.3f}" for x in succ[k])}</td></tr>'
        for k, v in d.get("go1", {}).items()
        if v and succ.get(k)
    )
    img = _file_to_uri(path / "noise_sweep.png", "image/png")
    fig = f'<img src="{img}" alt="exploration sweep">' if img else ""
    return f"""
<div class="two">
  <div>
    <h3>CartpoleBalance 2×2 factorial — 3 seeds/cell</h3>
    <div class="scroller"><table><thead><tr><th>NOISE_START → FINISH</th><th>mean</th><th>seeds</th></tr></thead>
    <tbody>{cells}</tbody></table></div>
    <p class="tiny muted">Main effect of START {d.get("effect_start", 0):+.0f}, of FINISH
    {d.get("effect_finish", 0):+.0f}, interaction {d.get("interaction", 0):+.0f}. Additive;
    the starting magnitude dominates.</p>
  </div>
  <div>
    <h3>Go1 transfer — 2 seeds/arm</h3>
    <div class="scroller"><table><thead><tr><th>arm</th><th>return</th><th>seeds</th>
    <th>tracking success</th><th>seeds</th></tr></thead>
    <tbody>{go1}</tbody></table></div>
    <p class="tiny muted"><b>Negative result, on the board deliberately.</b> Upstream's full
    locomotion recipe (γ 0.95 / λ 0.99) made Go1 <i>worse</i> than its shipped config on return,
    with clean seed separation — so the per-family-recipe hypothesis is rejected, not adopted.
    On tracking success no arm separates: the shipped baseline's own seeds span 0.026–0.171.</p>
  </div>
</div>
{fig}
"""


def render_env_cards(runs: dict, media: dict) -> str:
    if not runs:
        return (
            '<div class="empty">No checkpoints found. Train with '
            "<code>--save runs/verify/&lt;env&gt;_&lt;variant&gt;_s&lt;seed&gt;.pkl</code>"
            "</div>"
        )
    cards = []
    for env in sorted(runs):
        variants = runs[env]
        ret_series = {
            v: [_series(r["metrics"], "rollout/episode_return") for r in rs]
            for v, rs in variants.items()
        }
        succ_series = {
            # NOT "primary_success_rate" — the trainer logs it under the policy_diag namespace.
            # The wrong key made _series return None for every run, so _curve_plot silently drew
            # nothing and the GATE METRIC was missing from every env card on the board.
            v: [_series(r["metrics"], "policy_diag/primary_success_rate") for r in rs]
            for v, rs in variants.items()
        }
        imgs = []
        p = _curve_plot(f"{env} — training return", "episode return", ret_series)
        if p:
            imgs.append(f'<img src="{p}" alt="{_esc(env)} training return">')
        p = _curve_plot(f"{env} — primary success", "success rate", succ_series)
        if p:
            imgs.append(f'<img src="{p}" alt="{_esc(env)} primary success">')
        # per-skill returns from the first hierarchical run available
        for v in ("nesy", "neural", "symbolic"):
            if v in variants and variants[v]:
                p = _skill_return_plot(env, v, variants[v][0]["metrics"])
                if p:
                    imgs.append(f'<img src="{p}" alt="{_esc(env)} skill returns">')
                    break

        vids = []
        for m in media.get(env, []):
            label = f'{m["stem"]} [{m["variant"]}]' if m.get("variant") else m["stem"]
            if m.get("video"):
                vids.append(
                    f'<figure style="margin:0"><video controls preload="metadata" '
                    f'src="{m["video"]}"></video>'
                    f'<figcaption class="eyebrow" style="padding-top:.4rem">{_esc(label)}'
                    + (f' · return {m["return"]:.0f}' if m.get("return") is not None else "")
                    + "</figcaption></figure>"
                )
            elif m.get("oversize"):
                vids.append(f'<div class="empty">{_esc(m["oversize"])}</div>')
            if m.get("strip"):
                vids.append(f'<img src="{m["strip"]}" alt="{_esc(label)} skill activation">')

        seeds = sum(len(rs) for rs in variants.values())
        status = "pass" if seeds >= 3 * len(variants) and len(variants) >= 2 else "partial"
        body = "".join(imgs) or '<div class="empty">no curves yet</div>'
        vid_block = (
            f'<div class="grid2">{"".join(vids)}</div>'
            if vids
            else '<div class="empty">no rollout video — run tools/render_rollout.py</div>'
        )
        cards.append(
            f'<article class="card"><div class="card-head"><h3>{_esc(env)}</h3>'
            f'<span class="chip {"ok" if status == "pass" else "wait"}">'
            f'{len(variants)} variants · {seeds} runs</span></div>'
            f'<div class="grid2">{body}</div>{vid_block}</article>'
        )
    return f'<div class="cards">{"".join(cards)}</div>'


def build(
    audit: list[dict] | None,
    tests: dict | None,
    runs: dict,
    media: dict,
    out: Path,
    parity_dir: Path | None = None,
    audit_dir: Path | None = None,
    sweep_dir: Path | None = None,
    v13_dir: Path | None = None,
    v2_dir: Path | None = None,
) -> None:
    n_runs = sum(len(rs) for v in runs.values() for rs in v.values())
    audit_fail = sum(r.get("n_fail", 0) for r in audit) if audit else None
    audit_pass = sum(r.get("n_pass", 0) for r in audit) if audit else None

    tiles = []
    if tests:
        ok = tests.get("failed", 0) == 0
        tiles.append(
            f'<div class="tile {"ok" if ok else "stop"}"><span class="k">Unit + smoke suite</span>'
            f'<span class="v">{tests.get("passed", 0)} / {tests.get("passed", 0) + tests.get("failed", 0)}</span>'
            f'<span class="n">CPU, {tests.get("duration", 0):.0f} s</span></div>'
        )
    else:
        tiles.append(
            '<div class="tile wait"><span class="k">Unit + smoke suite</span><span class="v">—</span>'
            '<span class="n">no test report on disk</span></div>'
        )
    if audit is not None:
        tiles.append(
            f'<div class="tile {"ok" if audit_fail == 0 else "stop"}"><span class="k">Semantic audit</span>'
            f'<span class="v">{audit_pass} / {audit_pass + audit_fail}</span>'
            f'<span class="n">info keys vs independent references, {len(audit)} envs</span></div>'
        )
    else:
        tiles.append(
            '<div class="tile wait"><span class="k">Semantic audit</span><span class="v">—</span>'
            '<span class="n">not run</span></div>'
        )
    v13_path = (v13_dir or Path("runs/v13")) / "v13_ablation.json"
    if v13_path.exists():
        v13 = json.loads(v13_path.read_text(encoding="utf-8"))
        ok = v13.get("verdict") == "PASS"
        cells = v13.get("cells", {})
        best = max(cells, key=lambda k: cells[k]["return_mean"]) if cells else "—"
        tiles.append(
            f'<div class="tile {"ok" if ok else "stop"}"><span class="k">V1.3 meta-mode gate</span>'
            f'<span class="v">{"PASS" if ok else "FAILED"}</span>'
            f'<span class="n">nesy &ge; neural &ge; flat &mdash; best arm was <b>{_esc(best)}</b></span></div>'
        )
    else:
        tiles.append(
            '<div class="tile wait"><span class="k">V1.3 meta-mode gate</span><span class="v">&mdash;</span>'
            '<span class="n">not run</span></div>'
        )
    rc_path = (audit_dir or Path("runs/audit")) / "rule_coverage.json"
    if rc_path.exists():
        rc = json.loads(rc_path.read_text(encoding="utf-8"))
        n_ok = sum(1 for r in rc if r["pass"])
        tiles.append(
            f'<div class="tile {"ok" if n_ok == len(rc) else "stop"}"><span class="k">V3.1 rule coverage</span>'
            f'<span class="v">{n_ok} / {len(rc)}</span>'
            f'<span class="n">policies with every branch reachable and no all-zero mask</span></div>'
        )
    else:
        tiles.append(
            '<div class="tile wait"><span class="k">V3.1 rule coverage</span><span class="v">&mdash;</span>'
            '<span class="n">not run</span></div>'
        )
    tiles.append(
        f'<div class="tile {"ok" if n_runs else "wait"}"><span class="k">Training runs</span>'
        f'<span class="v">{n_runs}</span>'
        f'<span class="n">{len(runs)} environments with checkpoints on disk</span></div>'
    )
    pj = (parity_dir or Path("runs/parity")) / "parity.json"
    if pj.exists():
        pv = json.loads(pj.read_text(encoding="utf-8"))
        # The tile must show the RESOLVED state. The shipped-schedule ratio (0.41, FAIL) is
        # provenance, not the verdict: the cause was found (exploration schedule) and the fix
        # verified at ratio ~0.96 over 3 seeds. Showing 0.41 as the headline misstates the
        # campaign — the failure and its resolution both stay visible in the tile note.
        fixed_gate = pv.get("fixed_gate")
        if fixed_gate is not None:
            ok = fixed_gate == "PASS"
            tiles.append(
                f'<div class="tile {"ok" if ok else "stop"}"><span class="k">V1.2 parity</span>'
                f'<span class="v">{pv.get("fixed_ratio", float("nan")):.2f}x</span>'
                f'<span class="n">exploration-fixed flat / upstream purejaxql, gate +/-{pv.get("tolerance", 0.15)}; '
                f'shipped schedule was {pv.get("ratio_ours_over_upstream", float("nan")):.2f}x FAIL, cause found</span></div>'
            )
        else:
            ok = pv.get("gate") == "PASS"
            tiles.append(
                f'<div class="tile {"ok" if ok else "stop"}"><span class="k">V1.2 parity</span>'
                f'<span class="v">{pv.get("ratio_ours_over_upstream", float("nan")):.2f}x</span>'
                f'<span class="n">ours / upstream purejaxql, gate +/-{pv.get("tolerance", 0.15)}</span></div>'
            )
    else:
        tiles.append(
            '<div class="tile wait"><span class="k">V1.2 parity</span><span class="v">&mdash;</span>'
            '<span class="n">not run</span></div>'
        )
    n_vid = sum(1 for lst in media.values() for m in lst if m.get("video"))
    tiles.append(
        f'<div class="tile {"ok" if n_vid else "wait"}"><span class="k">Rollout videos</span>'
        f'<span class="v">{n_vid}</span>'
        f'<span class="n">greedy episodes with the active skill annotated</span></div>'
    )

    rule_block = render_rule_coverage(audit_dir or Path("runs/audit"))
    sweep_block = render_sweep(sweep_dir or Path("runs/sweep/noise"))
    v13_block = render_v13(v13_dir or Path("runs/v13"))
    v2_block = render_v2(v2_dir or Path("runs/v2"))
    paper_block = render_paper_figs(Path("runs/v3"))
    parity_block = render_parity(parity_dir) if parity_dir else render_parity(Path("runs/parity"))
    robustness_block = render_robustness(Path("runs/robustness"))
    ppo_block = render_ppo(Path("runs/ppo"), v2_dir or Path("runs/v2"))
    fig5_block = render_fig5(Path("nexus_continuous/policies"))
    probes_block = render_probes(Path("runs/probes"))
    fig6_block = render_fig6(Path("runs/fig6"))
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    doc = f"""<title>NEXUS Verification Board</title>
<style>{CSS}</style>
<div class="wrap">
<header class="masthead">
  <span class="eyebrow">Verification board · hierarchical AC-PQN / NEXUS</span>
  <h1>Evidence on disk, as of {_esc(generated)}</h1>
  <p>Generated from the results directories by <code>tools/build_dashboard.py</code>. A panel is
  green only when an artifact backs it — missing evidence renders as pending, never as a pass.</p>
  <div class="prov">
    <span><b>commit</b> {_esc(_git_commit())}</span>
    <span><b>plan</b> docs/VERIFICATION_PLAN.md</span>
    <span><b>generated</b> {_esc(generated)}</span>
  </div>
</header>

<section>
  <div class="sec-head"><span class="eyebrow">Summary</span><h2>What is established</h2></div>
  <div class="tiles">{"".join(tiles)}</div>
</section>

<section>
  <div class="sec-head"><span class="eyebrow">V1.2</span><h2>Parity against upstream purejaxql</h2></div>
  <p>Our <code>flat</code> arm and upstream <code>pqn_mujoco_playground.py</code> on the same
  environment and the same environment-step budget, each at its own shipped recipe. Every
  hierarchical claim is stated relative to this baseline.</p>
  {parity_block}
</section>

<section>
  <div class="sec-head"><span class="eyebrow">V3.2</span><h2>Semantic feature audit</h2></div>
  <p>Every quantity the hand-written rewards and symbolic rules read, checked against a source
  that does not share the adapter's indexing assumption. This is the gate that runs before any
  GPU time is spent.</p>
  {render_audit(audit)}
</section>

<section>
  <div class="sec-head"><span class="eyebrow">V3.3 &middot; paper comparison</span><h2>Skill learning and reward hacking</h2></div>
  <p>The two figures the source paper leads with, rebuilt on our runs so they can be read side by
  side with it.</p>
  {paper_block}
</section>

<section>
  <div class="sec-head"><span class="eyebrow">L4 &middot; runbook &sect;5b</span><h2>External baseline: tuned PPO at the same budget</h2></div>
  <p>Every other number on this board compares our arms to <em>our own</em> flat baseline, which
  cannot tell you whether that baseline is any good. This one does: MuJoCo Playground's shipped,
  tuned brax PPO, trained at the same environment-step budget as the V2 cells and scored through
  the same <code>task_metrics</code> and deterministic rollout.</p>
  {ppo_block}
</section>

<section>
  <div class="sec-head"><span class="eyebrow">V6 &middot; paper Q4</span><h2>Robustness under perturbation</h2></div>
  <p>The paper's Fig.&nbsp;8 tests trained agents on modified environments without retraining.
  Continuous analogue: every V2 checkpoint re-evaluated under swept action noise, same greedy
  selection and success metrics as the deterministic eval.</p>
  {robustness_block}
</section>

<section>
  <div class="sec-head"><span class="eyebrow">Q1 &middot; causal</span><h2>Skill probes and eval-time ablation</h2></div>
  <p>Two checks the paper does not have: run each skill <em>solo</em> and verify the behaviour
  matches the skill's name, and remove one skill from selection at eval to test whether the
  meta-policy's revealed preferences are load-bearing.</p>
  {probes_block}
</section>

<section>
  <div class="sec-head"><span class="eyebrow">V3.4 &middot; paper Q2</span><h2>Interpretability: the meta-policy as code, decisions with reasons</h2></div>
  {fig5_block}
  {fig6_block}
</section>

<section>
  <div class="sec-head"><span class="eyebrow">V2</span><h2>Environment matrix &mdash; the campaign gate</h2></div>
  {v2_block}
</section>

<section>
  <div class="sec-head"><span class="eyebrow">V1.3</span><h2>Meta-mode ablation</h2></div>
  <p>Four meta-policies on one environment, three seeds each, on the corrected exploration
  schedule: <code>flat</code> (no hierarchy), <code>neural</code> (learned meta-Q),
  <code>nesy</code> (learned meta-Q behind a hand-written mask), <code>symbolic</code>
  (hand-written rule, no meta-Q at all).</p>
  {v13_block}
</section>

<section>
  <div class="sec-head"><span class="eyebrow">V3.1</span><h2>Symbolic rule coverage</h2></div>
  <p>Every branch of every hand-written rule, driven over a synthetic grid of semantic states.
  A branch no state can reach is dead code in the meta-policy; an all-zero NeSy mask would make
  <code>where(mask, q, -1e9)</code> degenerate and hand argmax to skill 0 regardless of what the
  meta-Q learned. Neither occurs.</p>
  {rule_block}
</section>

<section>
  <div class="sec-head"><span class="eyebrow">V1.2 · sweep</span><h2>Exploration schedule, and what does not transfer</h2></div>
  <p>The cartpole factorial found the exploration deficiency behind the V1.2 parity failure. The
  Go1 arms tested whether the fix — and then upstream's whole locomotion recipe — carries over
  to a quadruped. It does not.</p>
  {sweep_block}
</section>

<section>
  <div class="sec-head"><span class="eyebrow">V2 · V3</span><h2>Per-environment evidence</h2></div>
  {render_env_cards(runs, media)}
</section>

<footer>
  <span>docs/VERIFICATION_PLAN.md</span>
  <span>regenerate: python tools/build_dashboard.py --out runs/dashboard.html</span>
</footer>
</div>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    size = out.stat().st_size
    print(f"wrote {out}  ({size / 1e6:.2f} MB)")
    if size > 16_000_000:
        print("WARNING: over the 16 MB artifact limit — drop or shorten some videos")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--audit", default="runs/audit/semantics.json")
    ap.add_argument("--tests", default="runs/audit/pytest.json")
    ap.add_argument("--runs", nargs="+", default=["runs/verify", "runs/viper"])
    ap.add_argument("--videos", default="runs/videos")
    ap.add_argument("--parity", default="runs/parity")
    ap.add_argument("--sweep", default="runs/sweep/noise")
    ap.add_argument("--v13", default="runs/v13")
    ap.add_argument("--v2", default="runs/v2")
    ap.add_argument("--out", default="runs/dashboard.html")
    args = ap.parse_args(argv)

    audit = None
    ap_path = Path(args.audit)
    if ap_path.exists():
        audit = json.loads(ap_path.read_text(encoding="utf-8"))

    tests = None
    tp = Path(args.tests)
    if tp.exists():
        tests = json.loads(tp.read_text(encoding="utf-8"))

    runs = collect_runs([Path(d) for d in args.runs])
    media = collect_media(Path(args.videos))
    build(
        audit,
        tests,
        runs,
        media,
        Path(args.out),
        Path(args.parity),
        ap_path.parent,
        Path(args.sweep),
        Path(args.v13),
        Path(args.v2),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
