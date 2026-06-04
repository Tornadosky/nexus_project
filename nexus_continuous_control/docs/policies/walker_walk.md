# WalkerWalk Policy

Skills: `stand_recover`, `walk_forward`, `stabilize_gait`, `energy_efficient`.

The policy first checks semantic `info` keys for height, pitch, and forward velocity.
Fallback raw observation indices use `0` for height, `1` for pitch, `-1` for forward
velocity, and the back half of the vector for joint speed.

Diagnostics: `walker/height`, `walker/pitch`, `walker/forward_velocity`,
`walker/joint_speed`.
