import jax.numpy as jnp
from nexus_continuous.llm.interpreter import (eval_rule, _term_reward)

def fields():
    return {
        "height": jnp.array([1.2]),
        "forward_velocity": jnp.array([3.0]),
        "pitch": jnp.array([0.05])
    }

def test_eval_true():
    mask = eval_rule("True", fields())

    assert bool(mask[0])

def test_eval_simple_compare():
    mask = eval_rule("height > 1.0", fields())

    assert bool(mask[0])

def test_eval_false():
    mask = eval_rule("height < 0.5", fields())

    assert not bool(mask[0])

def test_eval_and():
    mask = eval_rule("height > 1.0 and forward_velocity > 2.0", fields())

    assert bool(mask[0])

def test_eval_or():
    mask = eval_rule("height < 0.5 or forward_velocity > 2.0", fields())

    assert bool(mask[0])

def test_eval_not():
    mask = eval_rule("not (height < 1.0)", fields())

    assert bool(mask[0])

def test_negative_distance_reward():
    reward = _term_reward(
        {
            "type": "negative_distance",
            "weight": 2.0,
            "lhs": "height",
            "threshold": 1.0,
        },
        fields(),
        jnp.zeros((1, 3)),
    )

    assert reward.shape == (1,)
    assert reward[0] < 0

def test_positive_velocity_reward():
    reward = _term_reward(
        {
            "type": "positive_velocity",
            "weight": 1.0,
            "lhs": "forward_velocity",
        },
        fields(),
        jnp.zeros((1, 3)),
    )

    assert reward[0] > 0

def test_posture_penalty():
    reward = _term_reward(
        {
            "type": "posture_penalty",
            "weight": 1.0,
            "lhs": "pitch",
        },
        fields(),
        jnp.zeros((1, 3)),
    )

    assert reward[0] < 0

def test_action_penalty():
    reward = _term_reward(
        {
            "type": "action_penalty",
            "weight": 1.0,
        },
        fields(),
        jnp.array([[1.0, 2.0, 3.0]]),
    )

    assert reward[0] < 0

def test_binary_bonus():
    reward = _term_reward(
        {
            "type": "binary_bonus",
            "weight": 3.0,
            "lhs": "height",
            "threshold": 1.0,
        },
        fields(),
        jnp.zeros((1, 3)),
    )

    assert reward[0] == 3.0

def test_unknown_reward_returns_zero():
    reward = _term_reward(
        {
            "type": "something_random"
        },
        fields(),
        jnp.zeros((1, 3)),
    )

    assert reward[0] == 0.0

def test_invalid_rule_returns_false():
    mask = eval_rule(
        "this is not valid python",
        fields(),
    )

    assert not bool(mask[0])

def test_missing_field_returns_false():
    mask = eval_rule(
        "unknown_field > 5",
        fields(),
    )

    assert not bool(mask[0])