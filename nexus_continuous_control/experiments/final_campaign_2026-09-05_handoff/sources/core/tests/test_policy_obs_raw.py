import jax.numpy as jnp

from nexus_continuous.envs.playground_adapter import get_policy_obs
from nexus_continuous.policies import cartpole_balance
from nexus_continuous.policies.registry import load_policy_module


POLICIES = [
    "cartpole_balance",
    "cheetah_run",
    "walker_walk",
    "hopper_hop",
    "panda_pick_cube",
    "go1_joystick",
    "flat_baseline",
]


def test_get_policy_obs_prefers_raw_actor():
    actor = jnp.ones((2, 4)) * 100.0
    raw_actor = jnp.arange(8, dtype=jnp.float32).reshape(2, 4)
    policy_info = {"pole_angle": jnp.asarray([0.2, -0.2])}
    obs = {
        "actor": actor,
        "critic": actor + 1.0,
        "raw_actor": raw_actor,
        "raw_critic": raw_actor + 1.0,
        "policy_info": policy_info,
    }
    policy_obs = get_policy_obs(obs)
    assert jnp.array_equal(policy_obs["actor"], raw_actor)
    assert jnp.array_equal(policy_obs["raw_actor"], raw_actor)
    assert policy_obs["policy_info"] is policy_info


def test_policy_info_drives_symbolic_features_before_step_info_exists():
    raw_actor = jnp.zeros((1, 5), dtype=jnp.float32)
    policy_obs = get_policy_obs(
        {
            "actor": raw_actor + 100.0,
            "raw_actor": raw_actor,
            "policy_info": {
                "cart_position": jnp.asarray([0.0]),
                "pole_angle": jnp.asarray([0.31]),
                "cart_velocity": jnp.asarray([0.0]),
                "pole_angular_velocity": jnp.asarray([0.0]),
            },
        }
    )
    diagnostics = cartpole_balance.diagnostics(
        policy_obs,
        policy_obs,
        jnp.zeros((1, 1)),
        jnp.zeros((1,)),
        jnp.zeros((1,), dtype=bool),
    )
    assert jnp.allclose(diagnostics["cartpole/pole_angle"], jnp.asarray([0.31]))


def test_symbolic_policy_ignores_changed_normalized_actor_when_raw_fixed():
    # 0.50 rad is inside the symbolic rule's urgent-angle band (> 0.20).
    raw_actor = jnp.asarray([[0.0, 0.50, 0.0, 0.0]], dtype=jnp.float32)
    obs_a = {"actor": jnp.zeros_like(raw_actor), "raw_actor": raw_actor}
    obs_b = {"actor": jnp.ones_like(raw_actor) * -999.0, "raw_actor": raw_actor}
    skill_a = cartpole_balance.symbolic_meta_policy(obs_a)
    skill_b = cartpole_balance.symbolic_meta_policy(obs_b)
    assert int(skill_a[0]) == int(skill_b[0]) == 0


def test_policy_modules_return_finite_rewards_masks_and_diagnostics_on_raw_obs():
    batch = 4
    raw_obs = jnp.zeros((batch, 32), dtype=jnp.float32)
    obs = {"actor": raw_obs + 100.0, "raw_actor": raw_obs}
    next_obs = {"actor": raw_obs - 100.0, "raw_actor": raw_obs + 0.01}
    action = jnp.zeros((batch, 12), dtype=jnp.float32)
    env_reward = jnp.ones((batch,), dtype=jnp.float32)
    done = jnp.zeros((batch,), dtype=bool)
    for name in POLICIES:
        module = load_policy_module(name)
        rewards = module.skill_rewards(obs, next_obs, action, env_reward, done, {})
        mask = module.skill_mask(obs, {})
        diagnostics = module.diagnostics(obs, next_obs, action, env_reward, done, {})
        assert rewards.shape == (batch, module.NUM_SKILLS)
        assert mask.shape == (batch, module.NUM_SKILLS)
        assert mask.dtype == jnp.bool_
        assert jnp.isfinite(rewards).all()
        assert jnp.any(mask, axis=-1).all()
        for value in diagnostics.values():
            assert value.shape == (batch,)
            assert jnp.isfinite(value).all()
