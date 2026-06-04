import jax.numpy as jnp

from nexus_continuous.policies import cartpole_balance, cheetah_run, go1_joystick, walker_walk


def test_cartpole_task_metrics_use_wrapped_angle_and_centering():
    obs = jnp.asarray([[0.2, 0.1, 0.0, 0.0], [1.5, 0.1, 0.0, 0.0]], dtype=jnp.float32)
    metrics = cartpole_balance.task_metrics(obs, obs, jnp.zeros((2, 1)), jnp.zeros(2), jnp.zeros(2))
    assert jnp.allclose(metrics["cartpole/upright_fraction"], jnp.asarray([1.0, 1.0]))
    assert jnp.allclose(metrics["cartpole/centered_fraction"], jnp.asarray([1.0, 0.0]))
    assert jnp.allclose(metrics["primary_success_rate"], jnp.asarray([1.0, 0.0]))
    assert jnp.allclose(metrics["primary_goal_metric"], jnp.cos(jnp.asarray([0.1, 0.1])))


def test_cheetah_task_metrics_match_speed_and_posture_thresholds():
    obs = jnp.asarray(
        [[0.0, 0.2, 0.0, 0.0, 2.5], [0.0, 0.8, 0.0, 0.0, 3.0]],
        dtype=jnp.float32,
    )
    metrics = cheetah_run.task_metrics(obs, obs, jnp.zeros((2, 2)), jnp.zeros(2), jnp.zeros(2))
    assert jnp.allclose(metrics["cheetah/speed_success_rate"], jnp.asarray([1.0, 1.0]))
    assert jnp.allclose(metrics["cheetah/posture_stable_rate"], jnp.asarray([1.0, 0.0]))
    assert jnp.allclose(metrics["primary_success_rate"], jnp.asarray([1.0, 0.0]))
    assert jnp.allclose(metrics["primary_goal_metric"], jnp.asarray([2.5, 3.0]))


def test_walker_task_metrics_match_stand_and_walk_thresholds():
    obs = jnp.asarray(
        [[0.9, 0.2, 0.0, 0.6], [0.7, 0.2, 0.0, 1.0]],
        dtype=jnp.float32,
    )
    metrics = walker_walk.task_metrics(obs, obs, jnp.zeros((2, 2)), jnp.zeros(2), jnp.zeros(2))
    assert jnp.allclose(metrics["walker/stand_success_rate"], jnp.asarray([1.0, 0.0]))
    assert jnp.allclose(metrics["walker/walk_success_rate"], jnp.asarray([1.0, 0.0]))
    assert jnp.allclose(metrics["primary_success_rate"], jnp.asarray([1.0, 0.0]))
    assert jnp.allclose(metrics["primary_goal_metric"], jnp.asarray([0.6, 1.0]))


def test_go1_task_metrics_match_tracking_definition():
    obs = jnp.asarray(
        [
            [0.32, 0.1, 0.1, 0.8, 0.0, 0.1, 0.7, 0.0, 0.0],
            [0.18, 0.1, 0.1, 0.8, 0.0, 0.1, 0.7, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    metrics = go1_joystick.task_metrics(obs, obs, jnp.zeros((2, 12)), jnp.zeros(2), jnp.zeros(2))
    assert jnp.allclose(metrics["go1/no_fall_rate"], jnp.asarray([1.0, 0.0]))
    assert jnp.allclose(metrics["go1/tracking_success_rate"], jnp.asarray([1.0, 0.0]))
    assert jnp.allclose(metrics["primary_success_rate"], jnp.asarray([1.0, 0.0]))
    assert metrics["go1/velocity_tracking_error_mean"][0] < 0.2
