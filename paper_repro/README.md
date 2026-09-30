# Paper sources for the consistency pass

This folder is everything needed to check `final.tex` against the numbers and to redraw the figures. It does not include training checkpoints.

Run the scripts from `code/` with `numpy` and `matplotlib` installed. The committed PDFs were written by matplotlib 3.10.8; `make_skill_return_figs.py` reproduces them pixel for pixel under that version, and another version shifts the page size by a fraction of a point.

```bash
cd paper_repro/code
python make_skill_return_figs.py
```

Each script writes PDFs into `../figures/`. Paths are relative to this bundle (`code/paths.py`).

## Overleaf

`overleaf/` is the project currently on Overleaf, copied flat, including `final.tex` and `IEEEtran.cls`. Those two files are identical to the copies at the root of this folder.

`final.tex` sets `\graphicspath{{figures/}}`, so a local compile from this folder reads `figures/`. On Overleaf the same PDFs sit next to `final.tex`. Two uploads there do not match the names in the tex:

- `fig1_sweep (1).pdf` is the same file as `figures/fig1_sweep.pdf`.
- `fig_controlled_llm (1).pdf` is not the same file as `figures/fig_controlled_llm.pdf` (19007 bytes vs 14300).

`fig_ladder.pdf` and `fig_skill_returns.pdf` also differ between `overleaf/` and `figures/`. `figures/` is what the scripts in `code/` produce. `overleaf/` is what is uploaded now.

`fig_skill_returns_baselines.pdf` (Flat, HPQN, PPO) and `fig_skill_returns_variants.pdf` (Neural, Symbolic, NeSy) are produced by `make_skill_return_figs.py` and are cited in Appendix B of `final.tex`. Both PDFs are in `overleaf/` and need uploading to the Overleaf project.

`make_sweep_fig.py` does not read a results table. The five seeds in `fig1_sweep.pdf` are written in that file.

## What each figure reads

| Figure | Script | Data |
| --- | --- | --- |
| `fig1_sweep.pdf` | `make_sweep_fig.py` | seed lists in the script |
| `fig_core_sealed.pdf`, `fig_selector.pdf` | `make_core_figs.py` | `data/sealed/` zip: `final_metrics.csv`, `probes.csv` |
| `fig_ladder.pdf` | `make_core_figs.py` | `data/logs/hopper_ladder.csv` |
| `fig_oos.pdf` | `make_core_figs.py` | `data/logs/oos/rt_*_v2_s*.csv` |
| `fig2_core_curves.pdf`, `fig3_reliability.pdf`, `fig5_removal_only.pdf` | `make_legacy_figs.py` | zip `curves.csv`; `data/pack/04_local_gpu_and_viper/robustness/`; `probes/skill_probes.json`, `skill_ablation.json` |
| `fig_llm_bars.pdf`, `fig7_llm_appendix.pdf`, skill-usage PDFs | `make_extra_figs.py` | `data/pack/03_anja_koroveshi_llm_github/analysis_llm/tables/` |
| `fig_all_curves_paper.pdf` | `plot_paper_curves.py` | `data/logs/replot/curves_local.json`, `curves_remote.json`; `curves_return.csv`; zip `curves.csv` |
| `fig8` / six-env curves in `make_extra_figs.py` | `make_extra_figs.py` | `curves_return.csv`, `ppo_baseline/*.progress.json`, zip `curves.csv` |
| `fig_controlled_llm.pdf` | `make_controlled_figs.py` | zip: `llm_v2_metrics.csv`, `final_metrics.csv`, `llm.json` |
| `fig_rgb_change.pdf`, `fig_rgb_controlled.pdf` | `make_rgb_figs.py` | `data/pack/02_rosela_berberi_rgb_melody/state_plus_rgb_eval30/`; zip `rgb_conditions.csv` |
| `fig_skill_returns.pdf`, `fig_skill_returns_baselines.pdf`, `fig_skill_returns_variants.pdf` | `make_skill_return_figs.py` | `data/logs/skill_returns_local.csv`; zip `curves.csv` metric `common_skill_return/` |
| `fig_rgb_fusion.pdf` | `make_rgb_fusion.py` | diagram only, no results file |

The zip is `data/sealed/nexus_matrix_review_2026-09-08_18-28.zip`. The same files are extracted beside it under `data/sealed/nexus_matrix_review/` so tables can be opened without unzipping. `make_controlled_figs.py` also prints the dominant-skill fractions used in the language tables.

## Reductions

Checkpoints were not committed.

- `curves_return.csv` keeps only `env/returned_episode_returns`, the series the curve scripts read. The original file also held `returns/env_reward_mean`, `returns/skill_reward_mean`, and `rollout/episode_return`.
- `skill_returns_local.csv` is `skill_return/*` aligned to `env_step` from the verify NeSy checkpoints: Cartpole and Cheetah `budget4x` (3 seeds), Panda `v2` (9), Walker `dm` (3). Go1 and Hopper skill returns, including flat, HPQN, and PPO, come from `common_skill_return` in the sealed zip. Those lines are the task's hand-written skill rewards scored on each policy, not actors those baselines trained.
- `hopper_ladder.csv` is one row per Hopper checkpoint. `success_tail` is the mean of the last 10% of `policy_diag/primary_success_rate`. `hopper_ladder_failures.txt` is empty.
