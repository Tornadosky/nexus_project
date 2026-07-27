"""
Environment metadata used for LLM skill generation.
"""

from __future__ import annotations

ENV_REGISTRY = {
    "CartpoleBalance":{
        "fields": (
            "cart_position",
            "pole_angle",
            "cart_velocity",
            "pole_angular_velocity",
        ),
        "task": (
            "Keep the pole upright and centered while minimizing oscillations."
        ),
    },
    "CheetahRun":{
        "fields": (
            "forward_velocity",
            "torso_pitch",
            "pitch",
            "x_velocity",
            "joint_speed",
        ),
        "task": (
            "Run forward as fast as possible while remaining stable."
        ),
    },
    "WalkerWalk":{
        "fields": (
            "torso_height",
            "height",
            "torso_pitch",
            "pitch",
            "x_velocity",
            "forward_velocity",
            "joint_speed"
        ),
        "task": (
            "Walk forward while maintaining balance."
        ),
    },
    "HopperHop":{
        "fields": (
            "torso_pitch",
            "pitch",
            "x_velocity",
            "forward_velocity",
            "joint_speed"
        ),
        "task": (
            "Hop forward efficiently while staying upright."
        ),
    },
    "PandaPickCube":{
        "fields": (
            "tcp_pos",
            "gripper_pos",
            "eef_pos",
            "cube_pos",
            "object_pos",
            "obj_pos",
            "target_pos", 
            "goal_pos"
        ),
        "task": (
            "Reach the cube, grasp it, lift it, and move it to the target position."
        ),
    },
    "Go1JoystickFlatTerrain":{
        "fields": (
            "base_height",
            "height",
            "lin_vel_x",
            "lin_vel_y",
            "x_velocity",
            "y_velocity",
            "yaw_rate",
            "ang_vel_yaw"
        ),
        "task": (
            "Follow joystick commands while maintaining stable quadruped locomotion."
        ),
    },
    "HumanoidWalk":{
        "fields": (
            "base_height",
            "height",
            "x_velocity",
            "y_velocity",
            "forward_velocity",
            "joint_speed",
        ),
        "task": (
            "Walk forward with human-like balance and stability."
        ),
    },
}