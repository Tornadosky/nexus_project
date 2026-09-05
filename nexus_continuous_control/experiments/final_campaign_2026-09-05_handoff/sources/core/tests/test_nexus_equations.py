"""V1.1 — the paper's equations, tested directly.

The existing suite covers shapes, plumbing and schemas. What it did not cover is whether the
update rule *is the one in the paper*. These three tests close that gap. Each corresponds to a
claim the project makes about being a faithful continuous-action NEXUS:

  1. Eq. 2/3 — the Q(lambda) target, checked against an independent restatement written as
     explicit cases rather than fused arithmetic, including the truncation-vs-termination
     distinction that neither reference implementation makes. Also pinned against the
     `symbolic_options` recursion, which our form must reproduce exactly when truncation is
     not distinguished.
  2. Eq. 3 / Alg. 1 line 11 — the meta head regresses Q_meta(s, skill_taken) toward the
     ENVIRONMENT-reward return, never a skill reward, and symbolic mode trains no meta-Q.
  3. Sec. 3.1 — every skill learns off-policy from the shared trajectory, even skills the
     meta-policy never selects. This is the mechanism behind the paper's Q1 disentanglement
     result, and nothing in the suite guarded it.

Run (CPU):  JAX_PLATFORMS=cpu python -m pytest -q tests/test_nexus_equations.py
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import nexus_continuous.algorithms.hierarchical_ac_pqn_playground as alg  # noqa: E402
from nexus_continuous.algorithms.hierarchical_ac_pqn_playground import run_training  # noqa: E402
from nexus_continuous.envs.playground_adapter import PlaygroundEnvBundle  # noqa: E402
from nexus_continuous.returns import q_lambda_returns  # noqa: E402


# =========================================================================== #
# 1. Eq. 2 / Eq. 3 — the Q(lambda) target
# =========================================================================== #


def _paper_q_lambda(rewards, values, last_value, gamma, lam, dones, terminals):
    """Independent restatement of the paper's Eq. 2, written as explicit cases.

        R_t = r_t                                        if s_t is terminal
        R_t = r_t + gamma * [lam * R_{t+1} + (1-lam) * Q(s_{t+1})]   otherwise

    plus the time-limit refinement (Pardo et al. 2018): on a truncation the episode boundary
    cuts the lambda trace, but the one-step bootstrap is still taken, so the target collapses
    to the plain TD(0) form r + gamma * Q(s_{t+1}).
    """
    T = len(rewards)
    out = [0.0] * T
    carry = float(last_value)  # R_{t+1}; unused at the final step
    for t in reversed(range(T)):
        next_q = float(values[t + 1]) if t + 1 < T else float(last_value)
        if terminals[t]:
            r_t = float(rewards[t])
        elif dones[t]:
            # truncation: bootstrap, but do not propagate the trace across the boundary
            r_t = float(rewards[t]) + gamma * next_q
        else:
            r_t = float(rewards[t]) + gamma * (lam * carry + (1.0 - lam) * next_q)
        out[t] = r_t
        carry = r_t
    return np.asarray(out, dtype=np.float64)


def _symbolic_options_recursion(rewards, values, last_value, gamma, lam, dones):
    """The reference NEXUS recursion, transcribed from hierarchical_pqn_jaxtari.py.

        target_bootstrap = r + gamma * (1 - done) * next_q
        delta            = lambda_returns - next_q
        lambda_returns   = target_bootstrap + gamma * lam * delta
        lambda_returns   = (1 - done) * lambda_returns + done * r

    Note it applies no `(1 - done)` factor to the delta term and instead overrides the whole
    value afterwards. Our implementation folds the cut into the delta and then overrides on
    terminal, which is algebraically the same thing — this test pins that.
    """
    T = len(rewards)
    out = [0.0] * T
    carry = float(last_value)
    for t in reversed(range(T)):
        next_q = float(values[t + 1]) if t + 1 < T else float(last_value)
        d = float(dones[t])
        target_bootstrap = float(rewards[t]) + gamma * (1.0 - d) * next_q
        delta = carry - next_q
        val = target_bootstrap + gamma * lam * delta
        val = (1.0 - d) * val + d * float(rewards[t])
        out[t] = val
        carry = val
    return np.asarray(out, dtype=np.float64)


GAMMA, LAM = 0.97, 0.6
REWARDS = [1.0, -2.0, 0.5, 3.0, -1.5]
VALUES = [10.0, 20.0, -5.0, 7.0, 2.0]
LAST_VALUE = 4.0


def _run_impl(dones, terminals):
    r = jnp.asarray(REWARDS, dtype=jnp.float32)[:, None]
    v = jnp.asarray(VALUES, dtype=jnp.float32)[:, None]
    lv = jnp.asarray([LAST_VALUE], dtype=jnp.float32)
    d = jnp.asarray(dones, dtype=jnp.float32)[:, None]
    t = None if terminals is None else jnp.asarray(terminals, dtype=jnp.float32)[:, None]
    return np.asarray(
        q_lambda_returns(r, d, v, lv, gamma=GAMMA, lambda_=LAM, terminals=t)
    )[:, 0]


def test_q_lambda_matches_the_paper_equation_no_boundaries():
    dones = [0, 0, 0, 0, 0]
    got = _run_impl(dones, dones)
    want = _paper_q_lambda(REWARDS, VALUES, LAST_VALUE, GAMMA, LAM, dones, dones)
    assert np.allclose(got, want, atol=1e-5), f"{got} != {want}"


def test_q_lambda_matches_the_paper_equation_with_a_true_terminal():
    #                     t=0  t=1  t=2  t=3  t=4
    dones = [0, 0, 1, 0, 0]
    terminals = [0, 0, 1, 0, 0]
    got = _run_impl(dones, terminals)
    want = _paper_q_lambda(REWARDS, VALUES, LAST_VALUE, GAMMA, LAM, dones, terminals)
    assert np.allclose(got, want, atol=1e-5), f"{got} != {want}"
    # the paper is explicit: at a terminal the target IS the reward
    assert got[2] == pytest.approx(REWARDS[2], abs=1e-5)


def test_q_lambda_bootstraps_through_a_time_limit_truncation():
    """The refinement neither reference makes, and the one that matters for DM-control."""
    dones = [0, 0, 1, 0, 0]  # episode boundary at t=2 ...
    terminals = [0, 0, 0, 0, 0]  # ... but it is a time limit, not a real termination
    got = _run_impl(dones, terminals)
    want = _paper_q_lambda(REWARDS, VALUES, LAST_VALUE, GAMMA, LAM, dones, terminals)
    assert np.allclose(got, want, atol=1e-5), f"{got} != {want}"

    # It must NOT collapse to the bare reward the way a true terminal does; it bootstraps.
    assert got[2] != pytest.approx(REWARDS[2], abs=1e-3)
    assert got[2] == pytest.approx(REWARDS[2] + GAMMA * VALUES[3], abs=1e-4)

    # And it must differ from treating the truncation as terminal — otherwise the whole
    # truncation feature is a no-op and DM-control targets stay biased at every boundary.
    as_terminal = _run_impl(dones, dones)
    assert not np.allclose(got, as_terminal, atol=1e-3)


def test_q_lambda_reproduces_the_symbolic_options_recursion_when_terminals_default():
    """With terminals omitted, our form must equal the reference implementation exactly."""
    for dones in ([0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 1], [1, 1, 1, 1, 1]):
        got = _run_impl(dones, None)  # terminals=None => defaults to dones
        want = _symbolic_options_recursion(REWARDS, VALUES, LAST_VALUE, GAMMA, LAM, dones)
        assert np.allclose(got, want, atol=1e-5), f"dones={dones}: {got} != {want}"


# =========================================================================== #
# shared fake env for the two integration-level equation tests
# =========================================================================== #

_OBS = 4
_ACT = 1


def _obs_for(batch: int, value: float = 0.0):
    vec = jnp.full((batch, _OBS), value, dtype=jnp.float32)
    semantic = {
        # Drives cartpole's symbolic rule: a large pole angle pins the choice to
        # recover_balance, so exactly one skill is ever selected.
        "cart_position": jnp.zeros((batch,), dtype=jnp.float32),
        "pole_angle": jnp.full((batch,), 1.0, dtype=jnp.float32),
        "cart_velocity": jnp.zeros((batch,), dtype=jnp.float32),
        "pole_angular_velocity": jnp.full((batch,), 5.0, dtype=jnp.float32),
    }
    return {
        "actor": vec,
        "critic": vec,
        "raw_actor": vec,
        "raw_critic": vec,
        "policy_info": semantic,
    }


class _FakeEnv:
    """Deterministic state-mode env with a nonzero reward, so targets are not trivially 0."""

    def reset(self, key, params=None):
        del params
        n = key.shape[0]
        return _obs_for(n), jnp.zeros((n,), dtype=jnp.float32)

    def step(self, key, state, action, params=None):
        del key, params
        n = action.shape[0]
        reward = jnp.full((n,), 1.0, dtype=jnp.float32)
        done = jnp.zeros((n,), dtype=bool)
        return _obs_for(n), state, reward, done, {}


def _fake_bundle(_config):
    return PlaygroundEnvBundle(
        env=_FakeEnv(),
        env_params=None,
        action_low=-jnp.ones((_ACT,), dtype=jnp.float32),
        action_high=jnp.ones((_ACT,), dtype=jnp.float32),
        action_dim=_ACT,
        episode_length=8,
    )


def _config(**overrides):
    cfg = {
        "ALG_NAME": "nexus_ac_pqn",
        "ENV_NAME": "CartpoleBalance",
        "POLICY": "cartpole_balance",
        "TASK_POLICY": "cartpole_balance",
        "META_POLICY_TYPE": "symbolic",
        "TOTAL_TIMESTEPS": 4 * 4 * 2,
        "NUM_ENVS": 4,
        "NUM_STEPS": 4,
        "NUM_EPOCHS": 1,
        "NUM_MINIBATCHES": 2,
        "NUM_SEEDS": 1,
        "SEED": 0,
        "GAMMA": 0.99,
        "LAMBDA": 0.65,
        "SKILL_LAMBDA": 0.65,
        "META_LAMBDA": 0.8,
        "LR": 1e-2,  # large enough that two updates move every parameter measurably
        "LR_START": 1e-2,
        "LR_END": 1e-2,
        "LR_DECAY": 1.0,
        "ANNEAL_LR": False,
        "MAX_GRAD_NORM": 10.0,
        "ACTOR_HIDDEN_SIZES": [16, 16],
        "CRITIC_HIDDEN_SIZES": [16, 16],
        "META_HIDDEN_SIZES": [16, 16],
        "NUM_CRITICS": 2,
        "NORM_TYPE": "layer_norm",
        "ACTIVATION": "relu",
        "NOISE_START": 0.0,
        "NOISE_FINISH": 0.0,
        "NOISE_DECAY": 1.0,
        "META_EPS_START": 0.0,  # no exploration: the meta choice stays pinned
        "META_EPS_FINISH": 0.0,
        "META_EPS_DECAY": 1.0,
        "NORMALIZE_OBS": False,
        "NORMALIZE_REWARD": False,
        "BEHAVIOR_PENALTY_COEFF": 0.0,
        "ACTOR_UPDATE_MODE": "all_states",
        "EVAL_AFTER_TRAIN": False,
        "PRINT_EVERY": 0,
        "SAVE_PATH": None,
    }
    cfg.update(overrides)
    return cfg


def _train(monkeypatch, **overrides):
    monkeypatch.setattr(alg, "build_playground_env", _fake_bundle)
    return run_training(_config(**overrides))


# =========================================================================== #
# 2. Eq. 3 / Alg. 1 line 11 — what the meta head is trained on
# =========================================================================== #


def test_symbolic_mode_trains_no_meta_q(monkeypatch):
    """Alg. 1 line 11 updates Q_meta only for variants A (neural) and C (nesy)."""
    out = _train(monkeypatch, META_POLICY_TYPE="symbolic")
    train_state = out.runner_state[0]
    assert train_state.meta is None, "symbolic mode must not carry a meta-Q at all"
    assert float(np.asarray(out.metrics["loss/meta"]).sum()) == 0.0


# NOTE on the observable used below. The obvious choice, `train/meta_q`, is useless here: it
# is the network's PREDICTION, and Adam's update is invariant to a constant rescaling of the
# loss. Because the fake env holds observations constant, changing a lambda shifts the target
# by (very nearly) a constant factor, so the prediction trajectory comes out bit-identical
# even when the target has more than doubled. `train/meta_abs_td` carries the target itself,
# so it is the observable that can actually distinguish these cases.


def test_meta_target_is_built_from_env_reward_not_skill_reward(monkeypatch):
    """Eq. 3 uses R_env. Changing how skill returns are traced must not touch the meta target.

    SKILL_LAMBDA affects only the skill-reward lambda-returns. If the meta target were
    contaminated by skill rewards, it would move; it must not.
    """
    a = _train(monkeypatch, META_POLICY_TYPE="neural", SKILL_LAMBDA=0.1)
    b = _train(monkeypatch, META_POLICY_TYPE="neural", SKILL_LAMBDA=0.99)

    ta = np.asarray(a.metrics["train/meta_abs_td"]).ravel()
    tb = np.asarray(b.metrics["train/meta_abs_td"]).ravel()
    assert np.allclose(ta, tb, atol=1e-5), (
        "the meta TD error moved when only the SKILL lambda changed — Eq. 3's target is "
        f"reading skill rewards: {ta} vs {tb}"
    )

    # ... while the skill critics DID respond, proving the knob was live and the test is not
    # passing simply because nothing happened.
    ca = np.asarray(a.metrics["train/critic_target"]).ravel()
    cb = np.asarray(b.metrics["train/critic_target"]).ravel()
    assert not np.allclose(ca, cb, atol=1e-6), "SKILL_LAMBDA had no effect; test is vacuous"


def test_meta_lambda_does_reach_the_meta_target(monkeypatch):
    """The mirror of the above: the meta's own lambda must reach the meta target."""
    a = _train(monkeypatch, META_POLICY_TYPE="neural", META_LAMBDA=0.1)
    b = _train(monkeypatch, META_POLICY_TYPE="neural", META_LAMBDA=0.99)
    ta = np.asarray(a.metrics["train/meta_abs_td"]).ravel()
    tb = np.asarray(b.metrics["train/meta_abs_td"]).ravel()
    assert not np.allclose(ta, tb, atol=1e-3), f"META_LAMBDA never reached the target: {ta}"


def test_skill_targets_do_not_read_the_meta_lambda(monkeypatch):
    """Eq. 2 and Eq. 3 are separate traces; the meta knob must not leak into skill targets."""
    a = _train(monkeypatch, META_POLICY_TYPE="neural", META_LAMBDA=0.1)
    b = _train(monkeypatch, META_POLICY_TYPE="neural", META_LAMBDA=0.99)
    ca = np.asarray(a.metrics["train/critic_target"]).ravel()
    cb = np.asarray(b.metrics["train/critic_target"]).ravel()
    assert np.allclose(ca, cb, atol=1e-5), (
        f"skill critic targets moved with META_LAMBDA — the two traces are crossed: {ca} vs {cb}"
    )


# =========================================================================== #
# 3. Sec. 3.1 — all skills learn off-policy from the shared trajectory
# =========================================================================== #


def _leaf_norms(params) -> list[float]:
    return [float(jnp.linalg.norm(x)) for x in jax.tree_util.tree_leaves(params)]


def test_every_skill_critic_learns_even_when_never_selected(monkeypatch):
    """The paper's Q1 mechanism: a skill learns from another skill's actions.

    The fake env pins the symbolic rule to a single skill for the whole run, so only that
    skill is ever executed. Every OTHER skill's critic must still move, because each is
    trained on its own reward over the shared trajectory. If this ever regresses to
    training only the active skill, the disentanglement claim goes with it.
    """
    out = _train(monkeypatch, META_POLICY_TYPE="symbolic")

    # confirm the premise: exactly one skill was ever used
    usage = {
        k: float(np.asarray(v).sum())
        for k, v in out.metrics.items()
        if k.startswith("skill_usage/")
    }
    used = [k for k, v in usage.items() if v > 0]
    assert len(used) == 1, f"test premise broken — expected one active skill, got {usage}"

    critic_params = out.runner_state[0].critic.params
    # leading axis is the skill axis; every skill's slice must differ from its neighbours'
    # initialization-only state. The strongest simple check: no slice is all-zero and the
    # per-skill norms are distinct from a freshly-initialised copy.
    num_skills = jax.tree_util.tree_leaves(critic_params)[0].shape[0]
    assert num_skills >= 2

    fresh = _train(monkeypatch, META_POLICY_TYPE="symbolic", TOTAL_TIMESTEPS=0)
    fresh_params = fresh.runner_state[0].critic.params

    moved = []
    for i in range(num_skills):
        a = jax.tree_util.tree_map(lambda x: x[i], critic_params)  # noqa: B023
        b = jax.tree_util.tree_map(lambda x: x[i], fresh_params)  # noqa: B023
        delta = sum(
            float(jnp.sum(jnp.abs(x - y)))
            for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))
        )
        moved.append(delta)

    assert all(d > 1e-6 for d in moved), (
        "some skill critic did not train. Only skills that were SELECTED are learning, "
        f"which breaks the off-policy sharing the paper's Q1 result depends on: {moved}"
    )
