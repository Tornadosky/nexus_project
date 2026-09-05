"""Tests for the semantic-feature audit (tools/audit_semantics.py).

The audit's whole value is that it *fails* when a semantic key is wired to the wrong degree
of freedom. A green audit that cannot go red is worse than no audit, so the important test
here is the teeth test: feed the evaluator a deliberately mis-wired key and assert it is
caught and correctly diagnosed.

These tests are synthetic — they drive the decision logic directly rather than stepping MJX,
so they run in milliseconds and need no GPU. The end-to-end MJX path is exercised by running
the tool itself (``python tools/audit_semantics.py --all``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import audit_semantics as aud  # noqa: E402


# --------------------------------------------------------------------------- #
# pure geometry helpers
# --------------------------------------------------------------------------- #


def test_planar_pitch_recovers_a_known_rotation():
    theta = 0.37
    c, s = np.cos(theta), np.sin(theta)
    # world<-body for a rotation about +y; columns are the body axes in world coords
    rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    assert aud._planar_pitch(rot) == pytest.approx(theta, abs=1e-9)


def test_planar_pitch_accepts_flattened_matrices():
    theta = -1.2
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    flat = rot.reshape(1, 9)
    assert aud._planar_pitch(flat)[0] == pytest.approx(theta, abs=1e-9)


def test_wrap_keeps_angle_differences_small_across_the_branch_cut():
    # 3.14 and -3.14 are 0.0032 apart, not 6.28
    assert abs(aud._wrap(np.array(3.14) - np.array(-3.14))) < 0.01


def test_central_diff_matches_an_analytic_derivative():
    dt = 0.01
    t = np.arange(0, 50) * dt
    series = 3.0 * t  # constant velocity 3.0
    got = aud._central_diff(series, dt)
    assert np.allclose(got, 3.0, atol=1e-9)


# --------------------------------------------------------------------------- #
# the teeth test
# --------------------------------------------------------------------------- #


def _synthetic_rollout(reported: np.ndarray):
    """A 3-DOF world where qvel[1] is the true forward velocity.

    ``reported`` is whatever the adapter claims ``x_velocity`` is, so a test can wire it
    correctly or deliberately mis-wire it.
    """
    T, E = 24, 4
    rng = np.random.default_rng(0)
    qvel = rng.normal(size=(T, E, 3))
    dt = 0.02

    # torso_x integrates qvel[1]: the interior central difference recovers it exactly
    torso_x = np.zeros((T, E))
    for t in range(1, T):
        torso_x[t] = torso_x[t - 1] + qvel[t, :, 1] * dt
    # central difference over 2*dt averages consecutive steps, so build the reference the
    # same way the tool will read it
    truth = (torso_x[2:] - torso_x[:-2]) / (2 * dt)

    probe = {"torso_x": torso_x, "qvel": qvel}
    info = {"x_velocity": reported}
    spec = aud.EnvSpec(
        probe=lambda m, d: {},
        checks=[
            aud.Check(
                "x_velocity",
                lambda p, _dt: aud._central_diff(p["torso_x"], _dt),
                candidates="qvel",
                note="synthetic",
            )
        ],
    )
    return {
        "info": info,
        "probe": probe,
        "done": np.zeros((T, E)),
        "dt": dt,
        "spec": spec,
    }, truth


def _run(monkeypatch, reported_index: int):
    """Evaluate the audit with x_velocity wired to qvel[reported_index]."""
    T, E = 24, 4
    placeholder = np.zeros((T, E))
    data, _ = _synthetic_rollout(placeholder)
    # wire the reported value to the requested qvel component, over the interior window
    data["info"]["x_velocity"] = data["probe"]["qvel"][:, :, reported_index]

    monkeypatch.setattr(aud, "rollout", lambda *a, **k: data)
    return aud.audit_env("Synthetic", E, T, 0)


def test_correctly_wired_key_passes(monkeypatch):
    rep = _run(monkeypatch, reported_index=1)  # qvel[1] is the truth
    (check,) = rep["checks"]
    assert check["status"] == "PASS", check
    assert rep["n_fail"] == 0


def test_miswired_key_fails_and_names_the_right_index(monkeypatch):
    """This is the walker bug in miniature: reading qvel[0] when qvel[1] is forward."""
    rep = _run(monkeypatch, reported_index=0)
    (check,) = rep["checks"]
    assert check["status"] == "FAIL", check
    assert check["best_candidate"] == "qvel[1]", check
    assert "qvel[1]" in check["hint"]
    assert rep["n_fail"] == 1


def test_sign_flip_is_reported_as_its_own_failure_mode(monkeypatch):
    T, E = 24, 4
    data, _ = _synthetic_rollout(np.zeros((T, E)))
    data["info"]["x_velocity"] = -data["probe"]["qvel"][:, :, 1]
    monkeypatch.setattr(aud, "rollout", lambda *a, **k: data)

    rep = aud.audit_env("Synthetic", E, T, 0)
    (check,) = rep["checks"]
    assert check["status"] == "FAIL", check
    assert check["best_candidate"] == "qvel[1]", check


def test_missing_key_is_not_silently_a_pass(monkeypatch):
    T, E = 24, 4
    data, _ = _synthetic_rollout(np.zeros((T, E)))
    data["info"] = {}
    monkeypatch.setattr(aud, "rollout", lambda *a, **k: data)

    rep = aud.audit_env("Synthetic", E, T, 0)
    (check,) = rep["checks"]
    assert check["status"] == "MISSING"
    assert rep["n_fail"] == 1


def test_steps_after_episode_end_are_excluded(monkeypatch):
    """Post-termination physics is not what a policy sees, so it must not drive the verdict."""
    T, E = 24, 4
    data, _ = _synthetic_rollout(np.zeros((T, E)))
    data["info"]["x_velocity"] = data["probe"]["qvel"][:, :, 1].copy()
    # corrupt everything after step 10, then mark the episode as ended at step 10
    data["info"]["x_velocity"][11:] = 999.0
    done = np.zeros((T, E))
    done[10:] = 1.0
    data["done"] = done

    monkeypatch.setattr(aud, "rollout", lambda *a, **k: data)
    rep = aud.audit_env("Synthetic", E, T, 0)
    (check,) = rep["checks"]
    assert check["status"] == "PASS", check
