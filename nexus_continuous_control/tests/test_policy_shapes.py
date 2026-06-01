import jax.numpy as jnp

from nexus_continuous.policies.registry import load_policy_module


POLICIES = [
    "cartpole_balance",
    "cheetah_run",
    "walker_walk",
    "hopper_hop",
    "panda_pick_cube",
    "go1_joystick",
]


def test_policy_shapes():
    batch = 4
    obs = jnp.zeros((batch, 32))
    prev_obs = jnp.zeros((batch, 32))
    action = jnp.zeros((batch, 12))
    env_reward = jnp.zeros((batch,))
    done = jnp.zeros((batch,), dtype=bool)
    for name in POLICIES:
        module = load_policy_module(name)
        rewards = module.skill_rewards(prev_obs, obs, action, env_reward, done, {})
        skill = module.symbolic_meta_policy(obs, {})
        mask = module.skill_mask(obs, {})
        assert rewards.shape == (batch, module.NUM_SKILLS)
        assert skill.shape == (batch,)
        assert mask.shape == (batch, module.NUM_SKILLS)
        assert mask.dtype == jnp.bool_
