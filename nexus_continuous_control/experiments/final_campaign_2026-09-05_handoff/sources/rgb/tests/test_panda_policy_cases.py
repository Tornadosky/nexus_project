import jax.numpy as jnp

from nexus_continuous.policies import panda_pick_cube


def _obs(tcp, cube, target=(0.4, 0.0, 0.02), gripper=1.0):
    return jnp.asarray([[*tcp, *cube, *target, gripper]], dtype=jnp.float32)


def _case(obs):
    skill = int(panda_pick_cube.symbolic_meta_policy(obs)[0])
    mask = panda_pick_cube.skill_mask(obs)[0]
    return panda_pick_cube.SKILL_NAMES[skill], mask


def test_far_from_cube_not_grasped_reaches():
    skill, mask = _case(_obs((0.0, 0.0, 0.0), (0.2, 0.0, 0.02), gripper=1.0))
    assert skill == "reach_cube"
    assert bool(mask[0])


def test_near_cube_open_not_grasped_grasps():
    skill, mask = _case(_obs((0.0, 0.0, 0.02), (0.02, 0.0, 0.02), gripper=1.0))
    assert skill == "grasp_cube"
    assert bool(mask[1])


def test_near_cube_closed_low_grasped_lifts():
    skill, mask = _case(_obs((0.0, 0.0, 0.02), (0.02, 0.0, 0.02), gripper=0.0))
    assert skill == "lift_cube"
    assert bool(mask[2])


def test_grasped_high_cube_places_or_stabilizes():
    skill, mask = _case(_obs((0.0, 0.0, 0.20), (0.02, 0.0, 0.20), gripper=0.0))
    assert skill == "place_or_stabilize"
    assert bool(mask[3])
