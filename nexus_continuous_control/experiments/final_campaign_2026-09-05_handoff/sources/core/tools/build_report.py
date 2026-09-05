#!/usr/bin/env python3
"""Standalone HTML report — sections, hideable tables, plots, rollout media.

A plain file you open in a browser. Deliberately *not* the verification board:

  * **No gate, no PASS/FAIL chips, no verdict column.** Those belong to
    ``tools/build_dashboard.py``, which scores the campaign. This report shows what was tested and
    what came out, and leaves the adjudication to the reader.
  * **Every table is inside a ``<details>``** so it can be collapsed; "hide all tables" collapses
    the lot at once. Plots are the default reading surface, tables the drill-down.
  * **Plots are embedded as base64, media is referenced relatively.** Inlining the rollout clips
    as well would turn a 7 MB file into a ~45 MB one, so the report expects to sit in ``runs/``
    next to ``videos/`` and ``frames/``. ``--embed-media`` inlines them anyway for a single
    portable file; ``--no-embed-plots`` goes the other way for a small one. The diagnostics
    section is always linked, never embedded -- it alone is ~16 MB of PNG.

Reads only the precomputed analysis JSON/CSV under ``runs/`` — it never walks the run pickles, so
it finishes in seconds rather than the several minutes ``build_dashboard.py`` needs.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

VARIANTS = ["flat", "neural", "symbolic", "nesy", "ppo"]
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml",
        ".mp4": "video/mp4", ".pdf": "application/pdf"}


def esc(x: Any) -> str:
    return html.escape(str(x), quote=True)


def _num(x: Any, fmt: str = ".3f") -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "&mdash;"
    if isinstance(x, (int, float)):
        return f"{x:{fmt}}"
    return esc(x)


class Assets:
    """Turns a path into an ``src``: a data URI when embedding, a relative path otherwise."""

    def __init__(self, root: Path, embed_plots: bool, embed_media: bool) -> None:
        self.root, self.embed_plots, self.embed_media = root, embed_plots, embed_media
        self.embedded_bytes = 0
        self.missing: list[str] = []

    def src(self, path: Path, media: bool = False, link_only: bool = False) -> str | None:
        if not path.exists():
            self.missing.append(str(path))
            return None
        embed = False if link_only else (self.embed_media if media else self.embed_plots)
        if not embed:
            try:
                return path.resolve().relative_to(self.root.resolve()).as_posix()
            except ValueError:
                return path.resolve().as_uri()
        raw = path.read_bytes()
        self.embedded_bytes += len(raw)
        mime = MIME.get(path.suffix.lower(), "application/octet-stream")
        return f"data:{mime};base64," + base64.b64encode(raw).decode()

    def img(self, path: Path, alt: str = "", link_only: bool = False) -> str:
        s = self.src(path, link_only=link_only)
        # `zoom` opts the image into the lightbox. These are dense multi-panel figures; at grid
        # width they are navigational thumbnails, and the full-size view is where they are read.
        return f'<img class="zoom" loading="lazy" src="{s}" alt="{esc(alt)}">' if s else ""


# --------------------------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------------------------
def section(sid: str, eyebrow: str, title: str, tested: str, body: str) -> str:
    """One section: what was tested, then its content. `tested` is prose, not a verdict."""
    return f"""
<section id="{sid}">
  <div class="sec-head"><span class="eyebrow">{eyebrow}</span><h2>{title}</h2></div>
  <div class="tested"><span class="tested-k">What we tested</span><p>{tested}</p></div>
  {body}
</section>"""


def table(caption: str, head: list[str], rows: list[list[str]], open_by_default: bool = False) -> str:
    if not rows:
        return f'<p class="empty">{esc(caption)} &mdash; no data found.</p>'
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return (
        f'<details class="tbl"{" open" if open_by_default else ""}>'
        f"<summary>{esc(caption)} <span class='cnt'>{len(rows)} rows</span></summary>"
        f'<div class="scroller"><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'
        "</details>"
    )


def figures(items: list[tuple[str, str]], layout: str = "grid") -> str:
    """`items` are (img_html, caption). `layout` is grid | wide | thumbs.

    `wide` gives one figure per row: the aggregate figures are eight-panel composites, and at
    two-up grid width their axis labels are smaller than body text. `thumbs` is the opposite --
    a contact sheet for browsing a large set, read via the lightbox.
    """
    items = [(i, c) for i, c in items if i]
    if not items:
        return '<p class="empty">No figures found.</p>'
    cells = "".join(
        f'<figure>{img}{f"<figcaption>{cap}</figcaption>" if cap else ""}</figure>'
        for img, cap in items
    )
    return f'<div class="figs {esc(layout)}">{cells}</div>'


def _read_csv(p: Path) -> list[dict[str, str]]:
    with p.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _f(row: dict, key: str) -> float | None:
    v = row.get(key)
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _variant_key(arm: str) -> tuple[int, str]:
    fam = arm.split("·")[0]
    return (VARIANTS.index(fam) if fam in VARIANTS else len(VARIANTS), arm)


# --------------------------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------------------------
def sec_matrix(v2: Path, a: Assets) -> str:
    jf = v2 / "v2_matrix.json"
    if not jf.exists():
        return section("matrix", "V2", "Environment matrix",
                       "Not generated. Run <code>python tools/analyze_v2.py</code>.", "")
    matrix = json.loads(jf.read_text(encoding="utf-8")).get("matrix", {})

    rows = []
    for env in sorted(matrix):
        for arm in sorted(matrix[env], key=_variant_key):
            c = matrix[env][arm]
            seeds = " &middot; ".join(f"{x:.3f}" for x in c["success"])
            rows.append([
                f'<span class="id">{esc(env)}</span>', f"<code>{esc(arm)}</code>",
                f'<span class="num">{c["n"]}</span>',
                f'<span class="num"><b>{c["success_mean"]:.4f}</b></span>',
                f'<span class="num seeds">{seeds}</span>',
                f'<span class="num">{_num(c.get("return_mean"), ".1f")}</span>',
                "partial" if c["partial"] else "",
            ])

    # One figure per env, each sized to its own arm count -- see analyze_v2.plot.
    figs = [(a.img(p, f"V2 matrix - {p.stem.removeprefix('v2_matrix_')}"), "")
            for p in sorted(v2.glob("v2_matrix_*.png"))]
    if not any(i for i, _ in figs):
        figs = [(a.img(v2 / "v2_matrix.png", "V2 matrix"), "")]

    n_env, n_arm = len(matrix), sum(len(v) for v in matrix.values())
    return section(
        "matrix", "V2", "Environment matrix",
        f"Every variant against a <code>flat</code> baseline across <b>{n_env} environments</b>, "
        f"{n_arm} arms, scored on <b>primary success</b> &mdash; not episode return. Dots are "
        "seeds; hatched bars have fewer seeds than the analysis wants. Arms stay separate by "
        "recipe tag, so <code>walker</code>'s dm-suite and locomotion runs never average together.",
        figures(figs) + table("Per-arm success, with every seed",
                              ["env", "arm", "n", "success", "seeds", "return", ""], rows),
    )


def sec_ablation(v13: Path, a: Assets) -> str:
    """V1.3 - four meta-policies on one environment, read off the eval CSVs at zero noise."""
    by_arm: dict[str, list[tuple[int, float]]] = defaultdict(list)
    env = ""
    for p in sorted(v13.glob("*_s*.csv")):
        m = re.match(r"(.+)_s(\d+)$", p.stem)
        if not m:
            continue
        arm, seed = m.group(1), int(m.group(2))
        for row in _read_csv(p):
            if _f(row, "perturbation") == 0.0:
                env = env or row.get("env", "")
                sr = _f(row, "primary_success_rate")
                if sr is not None:
                    by_arm[arm].append((seed, sr))
                break

    rows = []
    for arm in sorted(by_arm, key=_variant_key):
        pts = sorted(by_arm[arm])
        vals = [v for _, v in pts]
        mean = sum(vals) / len(vals)
        rows.append([
            f"<code>{esc(arm)}</code>", f'<span class="num">{len(vals)}</span>',
            f'<span class="num"><b>{mean:.4f}</b></span>',
            f'<span class="num seeds">{" &middot; ".join(f"{v:.3f}" for v in vals)}</span>',
            f'<span class="num">{min(vals):.3f} &ndash; {max(vals):.3f}</span>',
        ])
    return section(
        "ablation", "V1.3", "Meta-mode ablation",
        f"Four meta-policies on one environment{f' (<code>{esc(env)}</code>)' if env else ''}, "
        "three seeds each, everything else fixed: <code>flat</code> (no hierarchy), "
        "<code>neural</code> (learned meta-Q), <code>nesy</code> (meta-Q behind a hand-written "
        "mask), <code>symbolic</code> (rule only). Seeds listed individually &mdash; the spread "
        "is the result.",
        figures([(a.img(v13 / "v13_ablation.png", "meta-mode ablation"), "")])
        + table("Success at zero perturbation, per seed",
                ["arm", "n", "mean success", "seeds", "range"], rows, open_by_default=True),
    )


def sec_robustness(rob: Path, sweep: Path, a: Assets) -> str:
    """Action-noise sweep: trained checkpoints re-evaluated under perturbation, no retraining."""
    cells: dict[tuple[str, str], dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for p in sorted(rob.glob("*.csv")):
        for row in _read_csv(p):
            env, arm = row.get("env", ""), row.get("meta", "")
            lvl, sr = _f(row, "perturbation"), _f(row, "primary_success_rate")
            if env and arm and lvl is not None and sr is not None:
                cells[(env, arm)][lvl].append(sr)

    levels = sorted({lv for c in cells.values() for lv in c})
    rows = []
    for (env, arm) in sorted(cells, key=lambda k: (k[0], _variant_key(k[1]))):
        c = cells[(env, arm)]
        base = c.get(0.0)
        cs = [f'<span class="id">{esc(env)}</span>', f"<code>{esc(arm)}</code>"]
        for lv in levels:
            vals = c.get(lv)
            cs.append(f'<span class="num">{sum(vals) / len(vals):.3f}</span>' if vals
                      else '<span class="num">&mdash;</span>')
        # Retention says more than any single level: how much of the unperturbed behaviour
        # survives the strongest noise this sweep applied.
        top = c.get(levels[-1]) if levels else None
        ret = None
        if base and top and sum(base):
            ret = (sum(top) / len(top)) / (sum(base) / len(base))
        cs.append(f'<span class="num">{_num(ret, ".2f")}</span>')
        rows.append(cs)

    head = ["env", "arm"] + [f"sigma={lv:g}" for lv in levels] + ["retention"]
    strongest = f"sigma={levels[-1]:g}" if levels else "n/a"
    return section(
        "robustness", "V3.2", "Robustness under action noise",
        "Every checkpoint re-evaluated under swept Gaussian action noise, <b>no retraining</b>. "
        f"Columns are noise levels; <i>retention</i> is success at {strongest} as a fraction of "
        "success at zero.",
        figures([(a.img(sweep / "noise_sweep.png", "noise sweep"), "")])
        + table("Mean success by noise level, averaged over seeds", head, rows),
    )


def sec_probes(probes: Path) -> str:
    pj, aj = probes / "skill_probes.json", probes / "skill_ablation.json"
    body = ""
    if pj.exists():
        rows = []
        for r in json.loads(pj.read_text(encoding="utf-8")):
            sem = r.get("semantics", {})
            spec = r.get("specialty", "")
            rows.append([
                f'<span class="id">{esc(r.get("env", ""))}</span>',
                f'<code>{esc(r.get("skill", ""))}</code>',
                f'<span class="num">{_num(r.get("episode_return_mean"), ".2f")}</span>',
                f"<code>{esc(spec)}</code>",
                f'<span class="num">{_num(sem.get(spec))}</span>',
            ])
        body += table("Each skill run solo, against the metric its name claims",
                      ["env", "skill", "return", "specialty metric", "value"], rows)
    if aj.exists():
        rows = []
        for r in json.loads(aj.read_text(encoding="utf-8")):
            intact, got = r.get("intact_success"), r.get("success")
            delta = (got - intact) if (intact is not None and got is not None) else None
            sign = "+" if (delta is not None and delta > 0) else ""
            rows.append([
                f'<span class="id">{esc(r.get("env", ""))}</span>',
                f'<code>{esc(r.get("checkpoint", ""))}</code>',
                f'<code>{esc(r.get("removed_skill", ""))}</code>',
                f'<span class="num">{_num(intact, ".4f")}</span>',
                f'<span class="num">{_num(got, ".4f")}</span>',
                f'<span class="num">{sign}{_num(delta, ".4f")}</span>',
            ])
        body += table("One skill removed from selection at eval time",
                      ["env", "checkpoint", "removed skill", "intact", "ablated", "delta"], rows)
    return section(
        "probes", "Q1 &middot; causal", "Skill probes and eval-time ablation",
        "Two causal checks. <b>Probes</b> run each skill solo and score it against the metric its "
        "name claims. <b>Ablation</b> deletes one skill from selection at eval time, testing "
        "whether the learned selection is load-bearing.",
        body or '<p class="empty">No probe output found.</p>',
    )


def sec_rules(audit: Path, a: Assets) -> str:
    rj = audit / "rule_coverage.json"
    rows = []
    if rj.exists():
        data = json.loads(rj.read_text(encoding="utf-8"))
        for r in data if isinstance(data, list) else [data]:
            fr = r.get("branch_fractions", {})
            av = r.get("mask_availability", {})
            counts = r.get("branch_counts", {})
            for branch in sorted(fr, key=lambda b: -fr[b]):
                n = counts.get(branch)
                rows.append([
                    f'<code>{esc(r.get("policy", ""))}</code>',
                    f"<code>{esc(branch)}</code>",
                    f'<span class="num">{_num(fr.get(branch), ".5f")}</span>',
                    f'<span class="num">{n:,}</span>' if isinstance(n, int)
                    else '<span class="num">&mdash;</span>',
                    f'<span class="num">{_num(av.get(branch), ".4f")}</span>',
                ])
    return section(
        "rules", "V3.1", "Symbolic rule coverage",
        "Every rule branch driven over a synthetic grid of semantic states, looking for dead code: "
        "branches nothing reaches, and all-zero masks that would collapse "
        "<code>where(mask, q, -1e9)</code> onto skill&nbsp;0. <i>Availability</i> is the fraction "
        "of states leaving that branch selectable.",
        figures([(a.img(audit / "rule_coverage.png", "rule coverage"), "")])
        + table("Branch reachability and mask availability",
                ["policy", "branch", "fraction of states", "count", "availability"], rows),
    )


def sec_parity(parity: Path, a: Assets) -> str:
    pj = parity / "parity.json"
    rows = []
    if pj.exists():
        data = json.loads(pj.read_text(encoding="utf-8"))
        for r in data if isinstance(data, list) else [data]:
            ours, up = r.get("ours_finals", []), r.get("upstream_finals", [])
            rows.append([
                f'<span class="id">{esc(r.get("env", ""))}</span>',
                f'<span class="num">{r.get("total_timesteps", 0):,}</span>',
                f'<span class="num">{_num(r.get("ours_final_mean"), ".1f")}</span>',
                f'<span class="num seeds">{" &middot; ".join(f"{v:.0f}" for v in ours)}</span>',
                f'<span class="num">{_num(r.get("upstream_final_mean"), ".1f")}</span>',
                f'<span class="num seeds">{" &middot; ".join(f"{v:.0f}" for v in up)}</span>',
                f'<span class="num"><b>{_num(r.get("ratio_ours_over_upstream"), ".3f")}</b></span>',
            ])
    return section(
        "parity", "V1.2", "Upstream parity",
        "Our <code>flat</code> baseline against the reference implementation &mdash; same "
        "environment, budget and seeds. A hierarchy that beats a broken baseline proves nothing, "
        "so this ratio conditions every other section.",
        figures([(a.img(parity / "parity.png", "parity"), "")])
        + table("Final return, ours vs upstream",
                ["env", "steps", "ours", "ours seeds", "upstream", "upstream seeds", "ratio"],
                rows, open_by_default=True),
    )


def sec_env(curves: Path, a: Assets) -> str:
    """Per-environment training curves: return, primary success, per-skill reward.

    Rendered ahead of time by ``tools/make_env_curves.py`` -- they live inside the run pickles, so
    drawing them at report-build time would mean unpickling the whole runs tree on every build.
    """
    envs = sorted({p.stem.split("__")[0] for p in curves.glob("*__*.png")})
    blocks = []
    for env in envs:
        items = [
            (a.img(curves / f"{env}__return.png", f"{env} training return"), "Training return"),
            (a.img(curves / f"{env}__success.png", f"{env} primary success"), "Primary success"),
            (a.img(curves / f"{env}__skills.png", f"{env} per-skill returns"), "Per-skill returns"),
        ]
        if not any(i for i, _ in items):
            continue
        blocks.append(f'<h3 class="envh">{esc(env)}</h3>' + figures(items, layout="trio"))
    if not blocks:
        return section(
            "curves", "per-env", "Training curves",
            "Not generated. Run <code>python tools/make_env_curves.py</code>.", "")
    return section(
        "curves", "per-env", "Training curves",
        "Per environment: the gameable quantity (return), the metric the campaign is scored on "
        "(primary success), and each skill's own hand-written reward. Methods overlaid, band = "
        "seed spread.",
        "".join(blocks),
    )


PAPER_GROUPS = [
    ("Headline", "wide", False, [
        ("paper_figs_final/training_return.png",
         "CartpoleBalance training return, 8 seeds, band = s.e.m."),
        ("paper_figs_final/final_performance.png",
         "CartpoleBalance end of training: return beside primary success, 8 seeds."),
        ("v3/fig7_return_vs_goal.png",
         "Return vs the actual goal. Left axis return, right axis primary success &mdash; where "
         "the two diverge, return ranks the arms backwards."),
    ]),
    ("Per-skill reward", "wide", False, [
        ("v3/fig3_skill_returns.png",
         "One panel per skill, methods overlaid, band = seed spread."),
        ("v3/skill_returns.png", "Per-skill returns, earlier render."),
        ("REVIEW/skill_usage_by_env_variant.png", "How often each skill is selected."),
    ]),
    ("Composite overviews", "thumbs", True, [
        ("REVIEW/main_return_curves.png",
         "Every environment and variant on one axis &mdash; superseded by the per-environment "
         "curves above, kept for completeness."),
        ("REVIEW/final_performance_vs_flat.png", "Ratio to flat, all arms on one axis."),
        ("REVIEW/skill_reward_curves_by_env.png", "All skill rewards on one axis."),
        ("REVIEW/mask_availability_vs_selection.png",
         "What the mask permits vs what the meta-Q picks."),
        ("REVIEW/panda_phase_diagnostics.png", "Panda broken out by task phase."),
        ("REVIEW/loss_and_td_diagnostics.png", "Actor, critic and meta losses with TD magnitudes."),
        ("REVIEW/raw_feature_diagnostics.png", "The raw semantic features the rules read."),
    ]),
]


def sec_paper(root: Path, review: Path, a: Assets) -> str:
    """Aggregate figures, grouped so the readable ones lead.

    The composites in the last group put every environment and variant on a single axis behind a
    fifty-entry legend; at any width they are unreadable, and the per-environment curves section
    shows the same quantities faceted. They stay in the report, collapsed, rather than being
    dropped -- but they are not what anyone should be reading first.
    """
    blocks = []
    for label, layout, collapsed, entries in PAPER_GROUPS:
        items = []
        for rel, cap in entries:
            p = (review / rel.removeprefix("REVIEW/")) if rel.startswith("REVIEW/") else (root / rel)
            img = a.img(p, p.stem)
            if img:
                items.append((img, cap))
        if not items:
            continue
        figs = figures(items, layout=layout)
        blocks.append(
            f'<details class="tbl"><summary>{esc(label)} '
            f'<span class="cnt">{len(items)} plots</span></summary>'
            f'<div class="dbody">{figs}</div></details>'
            if collapsed else f'<h3 class="envh">{esc(label)}</h3>{figs}'
        )
    return section(
        "paper", "figures", "Aggregate figures",
        "Campaign-wide views, one per row. Click any figure for full size.",
        "".join(blocks) or '<p class="empty">No figures found.</p>',
    )


def sec_rgb(rgb: Path, a: Assets) -> str:
    """The pixel-observation arm, if it has been run."""
    items = [(a.img(p, p.stem), f"<code>{esc(p.stem)}</code>") for p in sorted(rgb.glob("*.png"))]
    if not any(i for i, _ in items):
        return ""
    return section(
        "rgb", "RGB", "Pixel observations",
        "The same method trained from rendered pixels rather than simulator state, distilled from "
        "a state-space teacher.",
        figures(items, layout="wide"),
    )

def sec_fig6(fig6: Path, a: Assets) -> str:
    items = []
    for p in sorted(fig6.glob("*.png")):
        m = re.match(r"(.+)_s(\d+)_t(\d+)$", p.stem)
        cap = (f"<code>{esc(m.group(1))}</code> seed {m.group(2)}, step {int(m.group(3))}"
               if m else f"<code>{esc(p.stem)}</code>")
        img = a.img(p, p.stem)
        if img:
            items.append((img, cap))
    return section(
        "decisions", "V3.4", "Decision panels",
        "Single moments from a rollout: the semantic state the meta-policy saw, the branches the "
        "rule left available, and the skill it chose.",
        figures(items, layout="wide"),
    )


def sec_media(videos: Path, frames: Path, a: Assets) -> str:
    """Rollout clips grouped by environment, each with its skill-usage breakdown."""
    by_env: dict[str, list[dict]] = defaultdict(list)
    for js in sorted(videos.glob("*.json")):
        if js.name.startswith("_"):  # scratch and stale sidecars
            continue
        try:
            meta = json.loads(js.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        mp4 = videos / (meta.get("video") or f"{js.stem}.mp4")
        if not mp4.exists():
            continue
        meta["_src"] = a.src(mp4, media=True)
        strip = videos / (meta.get("strip") or f"{js.stem}_skills.png")
        meta["_strip"] = a.src(strip, media=True) if strip.exists() else None
        poster = frames / f"{js.stem}.png"
        meta["_poster"] = a.src(poster, media=True) if poster.exists() else None
        meta["_name"] = js.stem
        by_env[meta.get("env", "unknown")].append(meta)

    blocks = []
    total = 0
    for env in sorted(by_env):
        cards = []
        for m in sorted(by_env[env], key=lambda x: (_variant_key(x.get("variant", "")),
                                                    x.get("seed", 0))):
            total += 1
            usage = ""
            names, frac = m.get("skill_names") or [], m.get("skill_usage") or []
            if names and frac:
                bars = "".join(
                    f'<div class="ub"><span class="un">{esc(n)}</span>'
                    f'<span class="ut"><i style="width:{max(0.0, min(1.0, f)) * 100:.1f}%"></i></span>'
                    f'<span class="uv">{f:.3f}</span></div>'
                    for n, f in zip(names, frac)
                )
                usage = f'<div class="usage">{bars}</div>'
            poster = f' poster="{m["_poster"]}"' if m.get("_poster") else ""
            strip = (f'<img class="zoom strip" loading="lazy" src="{m["_strip"]}" '
                     f'alt="skill timeline">' if m.get("_strip") else "")
            ret = m.get("return")
            meta_line = " &middot; ".join(filter(None, [
                f'<code>{esc(m.get("variant", "?"))}</code>',
                f'seed {m.get("seed", "?")}',
                f"{ret:.1f} return" if isinstance(ret, (int, float)) else "",
            ]))
            cards.append(
                f'<figure class="clip"><video controls preload="none"{poster} '
                f'src="{m["_src"]}"></video>'
                f'<figcaption><b>{esc(m["_name"])}</b><br>{meta_line}</figcaption>'
                f"{strip}{usage}</figure>"
            )
        blocks.append(f'<h3 class="envh">{esc(env)} <span class="cnt">{len(by_env[env])} '
                      f'clips</span></h3><div class="figs grid clips">{"".join(cards)}</div>')

    note = ("" if a.embed_media else
            '<p class="note">Clips are linked relatively &mdash; keep this file beside '
            "<code>videos/</code> and <code>frames/</code>, or rebuild with "
            "<code>--embed-media</code>.</p>")
    return section(
        "media", "rollouts", f"Rollout media &mdash; {total} clips",
        "A greedy rollout per checkpoint, with the skill-activation timeline and usage breakdown "
        "for that episode. Clips are bound to a checkpoint by their JSON sidecar, not by filename.",
        note + "".join(blocks) if blocks else '<p class="empty">No rollout media found.</p>',
    )

DIAG_GROUPS = [
    ("aggregate", "Aggregate", "One panel per metric, every env and variant, averaged over seeds."),
    ("by_env", "By environment", "Final skill-usage breakdown per environment."),
    ("by_run", "By run", "No averaging - one line per run, so a divergent seed stays visible."),
]


def sec_diagnostics(review: Path, a: Assets) -> str:
    """Every diagnostic plot in a review directory, linked rather than embedded.

    ~80 files and 16 MB per review directory; inlining them would quadruple the report. Always
    referenced relatively, so this section resolves only while the report sits in the runs tree.
    """
    blocks = []
    for sub, label, blurb in DIAG_GROUPS:
        pngs = sorted((review / sub).glob("*.png"))
        if not pngs:
            continue
        items = [(a.img(p, p.stem, link_only=True), f"<code>{esc(p.stem)}</code>") for p in pngs]
        blocks.append(
            f'<details class="tbl"><summary>{esc(label)} '
            f'<span class="cnt">{len(pngs)} plots</span></summary>'
            f'<div class="dbody"><p class="note">{blurb}</p>'
            f'{figures(items, layout="thumbs")}</div></details>'
        )
    if not blocks:
        return ""
    return section(
        "diagnostics", "per-run", "Diagnostics",
        "The complete plot output of the review pass &mdash; nothing curated. Collapsed by group; "
        "click a thumbnail for full size.",
        "".join(blocks),
    )

CSS = r"""
:root{
  --ground:#F4F6F9; --surface:#FFFFFF; --sunk:#EDF0F5; --ink:#151A23; --ink2:#3D4756;
  --muted:#6B7686; --rule:#DCE1E9; --rule2:#BFC7D3; --accent:#343C96;
  --sans:ui-sans-serif,system-ui,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --serif:ui-serif,Georgia,"Times New Roman",serif;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0F131A; --surface:#161B23; --sunk:#1C222C; --ink:#E7EBF1; --ink2:#B7C0CD;
  --muted:#8792A2; --rule:#2B323F; --rule2:#3C4553; --accent:#9AA2F0;
}}
:root[data-theme="dark"]{
  --ground:#0F131A; --surface:#161B23; --sunk:#1C222C; --ink:#E7EBF1; --ink2:#B7C0CD;
  --muted:#8792A2; --rule:#2B323F; --rule2:#3C4553; --accent:#9AA2F0;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth;scroll-padding-top:5rem}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--serif);
  line-height:1.62;-webkit-text-size-adjust:100%}
.wrap{max-width:1140px;margin:0 auto;padding:clamp(1rem,3.5vw,2.4rem);
  display:flex;flex-direction:column;gap:2.4rem}

header.top{border-bottom:2px solid var(--ink);padding-bottom:1.2rem}
header.top h1{font-size:clamp(1.7rem,4.2vw,2.5rem);line-height:1.15;margin:.2rem 0 .5rem;
  letter-spacing:-.01em}
header.top p{color:var(--ink2);margin:0;max-width:64ch}
.kicker{font-family:var(--sans);font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;
  color:var(--muted)}

nav.toc{position:sticky;top:0;z-index:20;background:var(--ground);border-bottom:1px solid var(--rule);
  margin:0 calc(-1 * clamp(1rem,3.5vw,2.4rem));padding:.55rem clamp(1rem,3.5vw,2.4rem);
  display:flex;gap:.4rem;flex-wrap:wrap;align-items:center}
nav.toc a{font-family:var(--sans);font-size:.78rem;color:var(--ink2);text-decoration:none;
  padding:.24rem .5rem;border-radius:5px;border:1px solid transparent;white-space:nowrap}
nav.toc a:hover{background:var(--sunk);border-color:var(--rule);color:var(--ink)}
nav.toc .spacer{flex:1 1 auto;min-width:.5rem}
button.ctl{font-family:var(--sans);font-size:.74rem;color:var(--ink2);background:var(--surface);
  border:1px solid var(--rule2);border-radius:5px;padding:.3rem .62rem;cursor:pointer}
button.ctl:hover{background:var(--sunk);color:var(--ink)}

section{background:var(--surface);border:1px solid var(--rule);border-radius:10px;
  padding:clamp(1rem,2.6vw,1.7rem);display:flex;flex-direction:column;gap:1.1rem;scroll-margin-top:5rem}
.sec-head{display:flex;align-items:baseline;gap:.7rem;flex-wrap:wrap;
  border-bottom:1px solid var(--rule);padding-bottom:.6rem}
.sec-head h2{margin:0;font-size:clamp(1.15rem,2.6vw,1.55rem);letter-spacing:-.005em}
.eyebrow{font-family:var(--sans);font-size:.66rem;letter-spacing:.13em;text-transform:uppercase;
  color:var(--muted);border:1px solid var(--rule2);border-radius:4px;padding:.14rem .42rem}

.tested{background:var(--sunk);border-left:3px solid var(--accent);border-radius:0 7px 7px 0;
  padding:.75rem .95rem}
.tested p{margin:.25rem 0 0;color:var(--ink2);max-width:78ch}
.tested-k{font-family:var(--sans);font-size:.67rem;letter-spacing:.13em;text-transform:uppercase;
  color:var(--accent);font-weight:600}

details.tbl{border:1px solid var(--rule);border-radius:8px;background:var(--surface);overflow:hidden}
details.tbl>summary{font-family:var(--sans);font-size:.83rem;cursor:pointer;padding:.6rem .85rem;
  background:var(--sunk);color:var(--ink2);user-select:none;display:flex;gap:.5rem;align-items:center}
details.tbl>summary:hover{color:var(--ink)}
details.tbl[open]>summary{border-bottom:1px solid var(--rule)}
.cnt{font-size:.72rem;color:var(--muted);font-family:var(--mono)}
.scroller{overflow-x:auto;max-width:100%}
table{border-collapse:collapse;width:100%;font-family:var(--sans);font-size:.79rem}
th,td{padding:.4rem .6rem;text-align:left;border-bottom:1px solid var(--rule);white-space:nowrap}
th{position:sticky;top:0;background:var(--sunk);color:var(--ink2);font-weight:600;
  font-size:.7rem;letter-spacing:.05em;text-transform:uppercase}
tbody tr:hover{background:var(--sunk)}
.num{font-family:var(--mono);font-variant-numeric:tabular-nums}
.seeds{color:var(--muted);font-size:.74rem}
.id{font-weight:600}
code{font-family:var(--mono);font-size:.86em;background:var(--sunk);padding:.06em .32em;
  border-radius:3px}

.figs{display:grid;gap:1.1rem;align-items:start}
.figs.grid{grid-template-columns:repeat(auto-fit,minmax(330px,1fr))}
.figs.wide{grid-template-columns:1fr;gap:1.6rem}
.figs.thumbs{grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:.7rem}
.figs.trio{grid-template-columns:repeat(auto-fit,minmax(285px,1fr))}
.figs.thumbs figcaption{font-size:.66rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
img.zoom{cursor:zoom-in}
.dbody{padding:.9rem}
figure{margin:0;display:flex;flex-direction:column;gap:.45rem;min-width:0}
figure img,figure video{width:100%;height:auto;display:block;border:1px solid var(--rule);
  border-radius:6px;background:var(--surface)}
figcaption{font-family:var(--sans);font-size:.76rem;color:var(--ink2);line-height:1.45}
img.strip{border-radius:4px}
.clips figure.clip{background:var(--sunk);border:1px solid var(--rule);border-radius:8px;
  padding:.6rem;gap:.5rem}
.envh{font-size:1rem;margin:.6rem 0 0;font-family:var(--sans);display:flex;gap:.6rem;
  align-items:baseline;border-bottom:1px solid var(--rule);padding-bottom:.35rem}
.usage{display:flex;flex-direction:column;gap:.2rem}
.ub{display:grid;grid-template-columns:minmax(72px,1fr) 2.2fr auto;gap:.45rem;align-items:center;
  font-family:var(--sans);font-size:.7rem}
.un{color:var(--ink2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ut{background:var(--rule);border-radius:3px;height:7px;overflow:hidden}
.ut i{display:block;height:100%;background:var(--accent)}
.uv{font-family:var(--mono);color:var(--muted);font-variant-numeric:tabular-nums}

.lb{position:fixed;inset:0;z-index:100;background:rgba(8,10,14,.93);display:flex;
  align-items:center;justify-content:center;overflow:auto;padding:1.2rem;cursor:zoom-out}
.lb[hidden]{display:none}
.lb img{background:#fff;border-radius:6px;box-shadow:0 12px 48px rgba(0,0,0,.5)}
.lb img.fit{max-width:96vw;max-height:92vh;width:auto;height:auto;cursor:zoom-in}
.lb img:not(.fit){max-width:none;width:auto;cursor:zoom-out}
.lb .hint{position:fixed;bottom:.8rem;left:50%;transform:translateX(-50%);font-family:var(--sans);
  font-size:.72rem;color:#C9D2E0;background:rgba(0,0,0,.55);padding:.3rem .7rem;border-radius:5px}
p.empty,p.note{font-family:var(--sans);font-size:.8rem;color:var(--muted);margin:0;
  background:var(--sunk);border:1px dashed var(--rule2);border-radius:6px;padding:.6rem .8rem}
footer{font-family:var(--sans);font-size:.75rem;color:var(--muted);display:flex;gap:1rem;
  flex-wrap:wrap;border-top:1px solid var(--rule);padding-top:1rem}
@media print{nav.toc,button.ctl{display:none}section{break-inside:avoid;border:none}
  details.tbl{border:none} details.tbl>summary{display:none}}
"""

JS = r"""
(function(){
  // Most tables load collapsed -- plots first, numbers on demand -- so the button starts in the
  // "expand" state. Tracking it here rather than reading the DOM keeps the label truthful even
  // after the reader has opened or closed individual tables by hand.
  var open = false;
  var btn = document.getElementById('toggle-tables');
  btn.addEventListener('click', function(){
    open = !open;
    document.querySelectorAll('details.tbl').forEach(function(d){ d.open = open; });
    btn.textContent = open ? 'Hide all tables' : 'Show all tables';
  });
  // Lightbox. The aggregate and diagnostic figures are multi-panel composites; in the page they
  // are thumbnails, and this is where they get read. Click the image again to go 1:1 and scroll.
  var lb = document.getElementById('lb'), lbi = document.getElementById('lbimg');
  function closeLb(){ lb.hidden = true; document.body.style.overflow = ''; }
  document.addEventListener('click', function(e){
    var el = e.target;
    if (el.tagName === 'IMG' && el.classList.contains('zoom')) {
      lbi.src = el.currentSrc || el.src;
      lbi.className = 'fit';
      lb.hidden = false;
      document.body.style.overflow = 'hidden';
      return;
    }
    if (el === lbi) { lbi.classList.toggle('fit'); return; }
    if (!lb.hidden) closeLb();
  });
  document.addEventListener('keydown', function(e){
    if (e.key === 'Escape' && !lb.hidden) closeLb();
  });

  var t = document.getElementById('theme');
  t.addEventListener('click', function(){
    var cur = document.documentElement.getAttribute('data-theme');
    var next = cur === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    t.textContent = next === 'dark' ? 'Light' : 'Dark';
  });
})();
"""

NAV = [("curves", "Training curves"), ("matrix", "Environment matrix"),
       ("ablation", "Meta-mode ablation"),
       ("robustness", "Robustness"), ("probes", "Skill probes"), ("rules", "Rule coverage"),
       ("parity", "Upstream parity"), ("paper", "Paper figures"),
       ("rgb", "Pixel observations"), ("decisions", "Decision panels"),
       ("diagnostics", "Diagnostics"),
       ("media", "Rollout media")]


def build(root: Path, out: Path, embed_plots: bool, embed_media: bool, plots_dir: Path,
          diagnostics: Path | None) -> None:
    a = Assets(out.parent, embed_plots, embed_media)
    secs = [
        sec_env(root / "env_curves", a),
        sec_matrix(root / "v2", a),
        sec_ablation(root / "v13", a),
        sec_robustness(root / "robustness", root / "sweep/noise", a),
        sec_probes(root / "probes"),
        sec_rules(root / "audit", a),
        sec_parity(root / "parity", a),
        sec_paper(root, plots_dir, a),
        sec_rgb(root / "rgb_nexus", a),
        sec_fig6(root / "fig6", a),
        sec_diagnostics(diagnostics, a) if diagnostics else "",
        sec_media(root / "videos", root / "frames", a),
    ]
    nav = "".join(f'<a href="#{i}">{esc(n)}</a>' for i, n in NAV)
    doc = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NEXUS continuous control &mdash; results report</title>
<style>{CSS}</style>
</head><body>
<div class="wrap">
<header class="top">
  <div class="kicker">NEXUS &middot; continuous control</div>
  <h1>Results report</h1>
  <p>Every plot, table and rollout from the campaign. Tables sit behind a toggle so the figures
  read first; click any figure for full size.</p>
</header>
<nav class="toc">{nav}<span class="spacer"></span>
  <button class="ctl" id="toggle-tables">Show all tables</button>
  <button class="ctl" id="theme">Dark</button>
</nav>
{"".join(secs)}
<footer>
  <span>regenerate: <code>python tools/build_report.py</code></span>
  <span>plots {"embedded" if embed_plots else "linked"} &middot;
        media {"embedded" if embed_media else "linked"}</span>
</footer>
</div>
<div class="lb" id="lb" hidden><img id="lbimg" class="fit" alt="">
  <span class="hint">click image to zoom 1:1 &middot; Esc to close</span></div>
<script>{JS}</script>
</body></html>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    mb = out.stat().st_size / 1e6
    print(f"wrote {out}  ({mb:.2f} MB, {a.embedded_bytes / 1e6:.2f} MB embedded)")
    if a.missing:
        print(f"  {len(a.missing)} asset(s) not found, skipped:")
        for m in a.missing[:8]:
            print(f"    {m}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runs", default="runs", help="root holding v2/, v13/, videos/, ...")
    ap.add_argument("--plots", default="runs/finalization_one_seed_review/plots/paper",
                    help="directory holding the aggregate paper figures")
    ap.add_argument("--diagnostics", default="runs/finalization_one_seed_review/plots",
                    help="review plots/ dir to add as a collapsed diagnostics section; these are "
                         "always linked, never embedded. Pass '' to omit the section.")
    ap.add_argument("--out", default="runs/report.html")
    ap.add_argument("--no-embed-plots", action="store_true",
                    help="link plots relatively instead of inlining them (much smaller file)")
    ap.add_argument("--embed-media", action="store_true",
                    help="inline the rollout clips too, for one portable but very large file")
    args = ap.parse_args(argv)

    build(Path(args.runs), Path(args.out), not args.no_embed_plots, args.embed_media,
          Path(args.plots), Path(args.diagnostics) if args.diagnostics else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
