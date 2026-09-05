#!/usr/bin/env python3
"""Build a curated, paper-style W&B Report for a continuous-control NEXUS project.

The default W&B workspace dumps every metric into its own flat panel, which is
unreadable. This builds an ordered, sectioned Report instead:

  1. Training return        -- 3-seed mean +/- std shaded curve (the headline plot)
  2. Final performance      -- bar charts vs flat baseline (error bars across seeds)
  3. Learning stability     -- TD-error curves (banded)
  4. Skill usage            -- per-skill usage curves
  5. Exploration & safety   -- noise/epsilon schedules and NeSy mask violation

Seed aggregation: runs are grouped by their W&B ``group`` field (one group per
variant, one run per seed), so every line plot shows the mean with a shaded
std band -- the standard RL-paper presentation.

Usage:
  python tools/build_wandb_report.py --entity ENT --project PROJ \
      --title "CartpoleBalance: NEXUS vs flat (3 seeds)"
"""

from __future__ import annotations

import argparse

import wandb_workspaces.reports.v2 as wr

# Metric keys (override via CLI if a project uses different names).
RETURN_METRIC = "env/returned_episode_returns"
SUCCESS_METRIC = "policy_diag/primary_success_rate"
TD_METRICS = ["train/critic_abs_td", "train/meta_abs_td"]
NOISE_METRIC = "schedule/noise"
EPS_METRIC = "schedule/meta_epsilon"
MASK_METRIC = "mask/violation_rate"
SKILL_REGEX = "^skill_usage/"

MEAN = "mean"
STD = "stddev"

# Aggregate seeds at the RUNSET level by the config key META_POLICY_TYPE. Grouping
# on the runset (not the panel) is what makes W&B keep each seed's series and draw
# the mean line with a +/- std shaded band; a panel-level groupby only splits the
# mean lines and shows no band. Each variant writes META_POLICY_TYPE to wandb.config.
GROUP_KEY = "config.META_POLICY_TYPE"


def _runset(entity: str, project: str) -> wr.Runset:
    return wr.Runset(entity=entity, project=project, groupby=[GROUP_KEY])


def _line(y, title, *, x="env_step", range_y=None, regex=None, smoothing=0.0, layout=None) -> wr.LinePlot:
    kwargs = dict(
        title=title,
        x=x,
        groupby_aggfunc=MEAN,
        groupby_rangefunc=STD,
        smoothing_factor=smoothing,
        legend_position="north",
    )
    if regex is not None:
        kwargs["metric_regex"] = regex
    else:
        kwargs["y"] = y if isinstance(y, list) else [y]
    if range_y is not None:
        kwargs["range_y"] = range_y
    if layout is not None:
        kwargs["layout"] = layout
    return wr.LinePlot(**kwargs)


def _bar(metric, title, *, range_x=None, layout=None) -> wr.BarPlot:
    kwargs = dict(
        title=title,
        metrics=[metric],
        groupby_aggfunc=MEAN,
        groupby_rangefunc=STD,
        orientation="v",
    )
    if range_x is not None:
        kwargs["range_x"] = range_x
    if layout is not None:
        kwargs["layout"] = layout
    return wr.BarPlot(**kwargs)


def build(entity: str, project: str, title: str) -> wr.Report:
    full = wr.Layout(x=0, y=0, w=24, h=10)

    blocks = [
        wr.H1(text=title),
        wr.P(
            text=(
                "Curated view. Each variant (one W&B group) is trained with 3 seeds; "
                "lines show the seed mean and the shaded band is +/- 1 std. The offline "
                "collect_nexus_results.py -> validator pipeline remains authoritative for gates."
            )
        ),
        wr.H2(text="1. Training return (3 seeds, mean +/- std)"),
        wr.PanelGrid(
            runsets=[_runset(entity, project)],
            panels=[_line(RETURN_METRIC, "Episodic return vs environment steps", layout=full)],
        ),
        wr.H2(text="2. Final performance vs flat baseline"),
        wr.PanelGrid(
            runsets=[_runset(entity, project)],
            panels=[
                _bar(RETURN_METRIC, "Final episodic return"),
                _bar(SUCCESS_METRIC, "Deterministic primary success", range_x=[0, 1]),
            ],
        ),
        wr.H2(text="3. Learning stability (TD error)"),
        wr.PanelGrid(
            runsets=[_runset(entity, project)],
            panels=[_line(m, m) for m in TD_METRICS],
        ),
        wr.H2(text="4. Skill usage"),
        wr.PanelGrid(
            runsets=[_runset(entity, project)],
            panels=[_line(None, "Skill usage probability", regex=SKILL_REGEX, range_y=[0, 1])],
        ),
        wr.H2(text="5. Exploration schedule & NeSy safety"),
        wr.PanelGrid(
            runsets=[_runset(entity, project)],
            panels=[
                _line([NOISE_METRIC, EPS_METRIC], "Exploration: action noise & meta-epsilon"),
                _line(MASK_METRIC, "NeSy mask violation rate (should be 0)", range_y=[0, 1]),
            ],
        ),
    ]
    return wr.Report(project=project, entity=entity, title=title, description="3-seed curated NEXUS view", blocks=blocks, width="fluid")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entity", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--title", default="Continuous-control NEXUS (3 seeds)")
    args = ap.parse_args(argv)

    report = build(args.entity, args.project, args.title)
    report.save()
    print("REPORT_URL:", report.url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
