# CheetahRun Policy

Skills: `accelerate_forward`, `stabilize_posture`, `energy_efficient_run`.

The policy first checks semantic `info` keys for forward velocity. Fallback raw observation
indices use `1` for torso pitch, the back half of the vector for joint speed, and `-1`
for forward velocity.

Diagnostics: `cheetah/forward_velocity`, `cheetah/torso_pitch`, `cheetah/joint_speed`,
`cheetah/action_norm`.
