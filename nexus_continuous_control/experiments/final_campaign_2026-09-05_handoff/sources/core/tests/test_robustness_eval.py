"""CPU unit tests for the robustness-eval helpers (no MuJoCo / GPU needed).

The MuJoCo import in robustness_eval is lazy (inside build_playground_env via the
adapter), so importing the pure helpers here works on CPU. Run with:
    python -m pytest -q tests/test_robustness_eval.py
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tools.robustness_eval import (
    _apply_env_overrides,
    _make_normalizer,
    _panda_overrides,
    _walker_overrides,
)


def test_env_override_parsing_floats_and_strings():
    cfg = _apply_env_overrides({"ENV_NAME": "X"}, ["gravity=-4.9", "task.mode=easy"])
    ov = cfg["ENV_CONFIG_OVERRIDES"]
    assert ov["gravity"] == pytest.approx(-4.9)        # numeric parsed
    assert ov["task.mode"] == "easy"                    # non-numeric kept as str


def test_env_override_requires_key_value():
    with pytest.raises(ValueError):
        _apply_env_overrides({}, ["badformat"])


def test_panda_override_lift_success_logic():
    # cube lifted well above its initial height -> success
    mean = {"panda/cube_height_max_mean": jnp.asarray(0.20)}
    mx = {"panda/cube_height_max_mean": jnp.asarray(0.20)}
    init = {"panda/cube_height_max_mean": jnp.asarray(0.03)}
    out = _panda_overrides(mean, mx, init)
    assert float(out["primary_success_rate"]) == 1.0
    assert float(out["panda/lift_success_rate"]) == 1.0

    # cube never lifted -> failure
    mx2 = {"panda/cube_height_max_mean": jnp.asarray(0.035)}
    out2 = _panda_overrides({"panda/cube_height_max_mean": jnp.asarray(0.035)}, mx2, init)
    assert float(out2["primary_success_rate"]) == 0.0


def test_panda_override_noop_when_not_panda():
    m = {"some_other_metric": jnp.asarray(1.0)}
    assert _panda_overrides(m, m, m) == m


def test_walker_override_net_locomotion():
    # standing AND real forward velocity -> net-walk success
    good = {"walker/forward_velocity_mean": jnp.asarray(0.8),
            "walker/stand_success_rate": jnp.asarray(0.9)}
    out = _walker_overrides(good)
    assert float(out["primary_success_rate"]) == 1.0
    assert float(out["walker/net_walk_success_rate"]) == 1.0

    # swaying in place (zero net velocity) but standing -> NOT success (honest metric)
    sway = {"walker/forward_velocity_mean": jnp.asarray(0.01),
            "walker/stand_success_rate": jnp.asarray(0.9)}
    assert float(_walker_overrides(sway)["primary_success_rate"]) == 0.0


def test_normalizer_applies_stats_to_dict_obs():
    stats = {
        "actor_mean": np.array([1.0, 2.0]), "actor_var": np.array([4.0, 9.0]),
        "critic_mean": np.array([1.0, 2.0]), "critic_var": np.array([4.0, 9.0]),
    }
    norm = _make_normalizer(stats, normalize=True)
    raw = {"raw_actor": jnp.array([[3.0, 5.0]]), "raw_critic": jnp.array([[3.0, 5.0]]),
           "policy_info": {"foo": jnp.array([1.0])}}
    out = norm(raw)
    # (3-1)/sqrt(4) = 1.0 ; (5-2)/sqrt(9) = 1.0
    np.testing.assert_allclose(np.asarray(out["actor"])[0], [1.0, 1.0], atol=1e-5)
    assert "raw_actor" in out and "policy_info" in out  # raw + semantic preserved


def test_normalizer_identity_when_disabled_or_no_stats():
    raw = {"raw_actor": jnp.array([[3.0]]), "raw_critic": jnp.array([[3.0]])}
    assert _make_normalizer(None, normalize=True)(raw) is raw
    assert _make_normalizer({"actor_mean": 0}, normalize=False)(raw) is raw
