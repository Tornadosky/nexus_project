# PandaPickCube Policy

Skills: `reach_cube`, `grasp_cube`, `lift_cube`, `place_or_stabilize`.

The policy first checks semantic `info` keys for TCP, cube, target, gripper, and grasp
state. Fallback raw observation layout is `[tcp_xyz, cube_xyz, target_xyz, gripper]`.
Closed gripper plus small TCP-cube distance is treated as an inferred grasp.

The NeSy mask makes `place_or_stabilize` available only after grasp and sufficient cube
height, avoiding premature place collapse.

Diagnostics: `panda/dist_tcp_cube`, `panda/dist_cube_target`, `panda/cube_height`,
`panda/gripper`, `panda/grasped`.
