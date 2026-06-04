import pickle

import jax.numpy as jnp

from tools.collect_nexus_results import (
    deterministic_eval_to_dataframes,
    discover_runs,
    make_deterministic_baseline_comparison,
)


def _write_run(path, meta_policy_type, episode_returns):
    payload = {
        "config": {
            "ENV_NAME": "CartpoleBalance",
            "POLICY": "cartpole_balance" if meta_policy_type != "flat" else "flat_baseline",
            "META_POLICY_TYPE": meta_policy_type,
            "SEED": 0,
            "EVAL_SEED": 10000,
        },
        "metrics": {
            "env_step": jnp.asarray([1.0]),
            "skill_usage/0_skill": jnp.asarray([1.0]),
        },
        "eval_metrics": {
            "eval_seed": jnp.asarray(10000),
            "num_eval_episodes": jnp.asarray(len(episode_returns)),
        },
        "eval_episode_table": {
            "episode_return": jnp.asarray(episode_returns, dtype=jnp.float32),
            "episode_length": jnp.ones((len(episode_returns),), dtype=jnp.float32) * 10.0,
            "primary_goal_metric": jnp.ones((len(episode_returns),), dtype=jnp.float32),
            "primary_success_rate": jnp.ones((len(episode_returns),), dtype=jnp.float32),
            "cartpole/upright_fraction": jnp.ones((len(episode_returns),), dtype=jnp.float32),
            "cartpole/centered_fraction": jnp.ones((len(episode_returns),), dtype=jnp.float32),
            "cartpole/angle_abs_mean": jnp.zeros((len(episode_returns),), dtype=jnp.float32),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def test_deterministic_eval_collector_emits_required_schema(tmp_path):
    runs = tmp_path / "runs"
    _write_run(runs / "flat_cartpole_balance_seed0.pkl", "flat", [1.0, 2.0])
    _write_run(runs / "cartpole_balance_nesy_seed0.pkl", "nesy", [2.0, 4.0])

    records, failures = discover_runs(runs)
    assert not failures
    episodes, summary, task_success, errors = deterministic_eval_to_dataframes(records)
    assert not errors
    required_cols = {
        "env_name",
        "policy",
        "meta_policy_type",
        "seed",
        "eval_seed",
        "num_eval_episodes",
        "episode_return_mean",
        "episode_return_std",
        "episode_length_mean",
        "primary_goal_metric",
        "primary_success_rate",
        "cartpole/upright_fraction",
        "cartpole/centered_fraction",
        "cartpole/angle_abs_mean",
    }
    assert required_cols.issubset(summary.columns)
    assert len(episodes) == 4
    assert len(task_success) == len(summary)

    ratios = make_deterministic_baseline_comparison(summary)
    nesy_ratio = ratios.loc[ratios["meta_policy_type"] == "nesy", "ratio_to_flat"].iloc[0]
    assert nesy_ratio == 2.0
