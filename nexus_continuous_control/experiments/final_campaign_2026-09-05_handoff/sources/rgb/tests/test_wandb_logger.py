"""Tests for the coexist W&B tracking layer.

These cover the pure reduction/decision logic and the replay path against a fake
run, so they need neither a GPU, jax, nor a live W&B account.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from nexus_continuous.tracking import replay_history_to_run, resolve_settings
from nexus_continuous.tracking.wandb_logger import _series_from_metric, _summary_scalars


class FakeRun:
    def __init__(self) -> None:
        self.rows: list[tuple[int, dict]] = []
        self.summary: dict[str, float] = {}
        self.id = "fake"

    def log(self, row, step=None):
        self.rows.append((step, dict(row)))

    def finish(self):
        pass


def _output(metrics, eval_metrics=None):
    return SimpleNamespace(metrics=metrics, eval_metrics=eval_metrics or {})


# --- resolve_settings ----------------------------------------------------------


def test_enabled_by_default_online():
    enabled, settings = resolve_settings({"ENV_NAME": "CheetahRun", "META_POLICY_TYPE": "nesy"})
    assert enabled is True
    assert settings["mode"] == "online"
    assert settings["group"] == "CheetahRun_nesy"
    assert settings["project"] == "nexus-continuous-control"


def test_cli_disable_wins():
    enabled, _ = resolve_settings({"ENV_NAME": "X"}, cli_disable=True)
    assert enabled is False


def test_config_disabled_flag():
    enabled, _ = resolve_settings({"WANDB": {"enabled": False}})
    assert enabled is False


def test_offline_mode_stays_enabled():
    enabled, settings = resolve_settings({"WANDB": {"mode": "offline"}})
    assert enabled is True
    assert settings["mode"] == "offline"


def test_disabled_mode_disables():
    enabled, _ = resolve_settings({"WANDB": {"mode": "disabled"}})
    assert enabled is False


def test_env_var_mode(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    enabled, _ = resolve_settings({"ENV_NAME": "X"})
    assert enabled is False


def test_config_overrides_project_and_group():
    _, settings = resolve_settings(
        {"WANDB": {"project": "myproj", "group": "g", "tags": ["a"]}, "ENV_NAME": "E"}
    )
    assert settings["project"] == "myproj"
    assert settings["group"] == "g"
    assert settings["tags"] == ["a"]


# --- reduction helpers ---------------------------------------------------------


def test_series_1d_passthrough():
    arr = np.arange(5.0)
    np.testing.assert_array_equal(_series_from_metric(arr), arr)


def test_series_reduces_epoch_minibatch_axes():
    arr = np.ones((4, 2, 3)) * 2.0  # [updates, epochs, minibatches]
    out = _series_from_metric(arr)
    assert out.shape == (4,)
    assert np.allclose(out, 2.0)


def test_series_selects_seed_axis():
    arr = np.stack([np.zeros(3), np.ones(3)])  # [seeds, updates]
    out = _series_from_metric(arr, seed_index=1, num_seeds=2)
    assert np.allclose(out, 1.0)


def test_summary_scalars_seed_selection():
    metrics = {"primary_success_rate": np.array([0.1, 0.9])}  # [seeds]
    out = _summary_scalars(metrics, seed_index=1, num_seeds=2)
    assert out["primary_success_rate"] == pytest.approx(0.9)


# --- replay path ---------------------------------------------------------------


def test_replay_logs_per_update_and_summary():
    metrics = {
        "returns/env_reward_mean": np.array([1.0, 2.0, 3.0]),
        "train/critic_abs_td": np.ones((3, 2, 4)),  # extra axes get reduced
    }
    eval_metrics = {"primary_success_rate": np.array(0.42), "episode_return_mean": np.array(7.0)}
    run = FakeRun()

    replay_history_to_run(
        run,
        {"TOTAL_TIMESTEPS": 30, "NUM_SEEDS": 1},
        _output(metrics, eval_metrics),
    )

    # One log row per update, monotonically stepped, env_step synthesised.
    assert len(run.rows) == 3
    assert [step for step, _ in run.rows] == [0, 1, 2]
    assert run.rows[0][1]["returns/env_reward_mean"] == 1.0
    assert run.rows[-1][1]["env_step"] == pytest.approx(30.0)
    # Eval scalars land in summary, headline key surfaced at top level.
    assert run.summary["eval/primary_success_rate"] == pytest.approx(0.42)
    assert run.summary["primary_success_rate"] == pytest.approx(0.42)


def test_replay_prefers_real_env_step():
    metrics = {
        "env_step": np.array([10.0, 20.0, 40.0]),
        "returns/env_reward_mean": np.array([1.0, 2.0, 3.0]),
    }
    run = FakeRun()
    replay_history_to_run(run, {"NUM_SEEDS": 1}, _output(metrics))
    assert [row["env_step"] for _, row in run.rows] == [10.0, 20.0, 40.0]
    # env_step is not double-logged as a normal metric column.
    assert "env_step" not in {k for _, row in run.rows for k in row if k != "env_step"}
