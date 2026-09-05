import jax.numpy as jnp

from nexus_continuous.algorithms.hierarchical_ac_pqn_playground import mask_violation_metrics


def test_mask_violation_metrics_are_zero_when_selected_skill_is_available():
    skill_one_hot = jnp.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=jnp.float32)
    mask_float = jnp.asarray([[[1.0, 0.0], [1.0, 1.0]]], dtype=jnp.float32)
    metrics = mask_violation_metrics(skill_one_hot, mask_float, ("a", "b"))
    assert jnp.allclose(metrics["mask/violation_rate"], 0.0)
    assert jnp.allclose(metrics["mask_violation/0_a"], 0.0)
    assert jnp.allclose(metrics["mask_violation/1_b"], 0.0)


def test_mask_violation_metrics_count_unavailable_selected_skills():
    skill_one_hot = jnp.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=jnp.float32)
    mask_float = jnp.asarray([[[0.0, 1.0], [1.0, 1.0]]], dtype=jnp.float32)
    metrics = mask_violation_metrics(skill_one_hot, mask_float, ("a", "b"))
    assert jnp.allclose(metrics["mask/violation_rate"], 0.5)
    assert jnp.allclose(metrics["mask_violation/0_a"], 0.5)
    assert jnp.allclose(metrics["mask_violation/1_b"], 0.0)
