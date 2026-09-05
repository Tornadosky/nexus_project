import jax.numpy as jnp

from nexus_continuous.policies import panda_pick_cube


def _obs(tcp, cube, target=(0.4, 0.0, 0.03), gripper=1.0):
    return jnp.asarray([[*tcp, *cube, *target, gripper]], dtype=jnp.float32)


def _metrics(obs):
    return panda_pick_cube.task_metrics(
        obs,
        obs,
        jnp.zeros((1, 4), dtype=jnp.float32),
        jnp.zeros((1,), dtype=jnp.float32),
        jnp.zeros((1,), dtype=bool),
    )


def test_lift_success_false_when_grasp_proxy_true_but_cube_does_not_lift():
    obs = _obs((0.0, 0.0, 0.03), (0.02, 0.0, 0.03), gripper=0.0)
    metrics = _metrics(obs)
    diagnostics = panda_pick_cube.diagnostics(
        obs,
        obs,
        jnp.zeros((1, 4), dtype=jnp.float32),
        jnp.zeros((1,), dtype=jnp.float32),
        jnp.zeros((1,), dtype=bool),
    )
    assert bool(diagnostics["panda/grasp_proxy"][0])
    assert not bool(metrics["panda/lift_success_rate"][0])
    assert not bool(metrics["primary_success_rate"][0])


def test_lift_success_true_when_cube_height_crosses_threshold():
    obs = _obs((0.0, 0.0, 0.09), (0.02, 0.0, 0.09), gripper=0.0)
    metrics = _metrics(obs)
    assert bool(metrics["panda/lift_success_rate"][0])
    assert bool(metrics["primary_success_rate"][0])


def test_place_success_requires_lift_and_target_proximity():
    lifted_far = _obs((0.0, 0.0, 0.09), (0.02, 0.0, 0.09), target=(0.4, 0.0, 0.03))
    lifted_near = _obs((0.0, 0.0, 0.09), (0.02, 0.0, 0.09), target=(0.03, 0.0, 0.09))
    assert not bool(_metrics(lifted_far)["panda/place_success_rate"][0])
    assert bool(_metrics(lifted_near)["panda/place_success_rate"][0])
