# HopperHop Policy

Skills: `stand_recover`, `hop_forward`, `stabilize_landing`, `energy_efficient`.

The policy first checks semantic `info` keys for height, pitch, and forward velocity.
Fallback raw observation indices use `0` for height, `1` for pitch, `-1` for forward
velocity, and the back half of the vector for vertical or joint speed.

Hopper rewards include environment reward in `hop_forward` and `energy_efficient` so
at least one skill tracks survival and task return directly.

Diagnostics: `hopper/height`, `hopper/pitch`, `hopper/forward_velocity`,
`hopper/joint_speed`, `hopper/done_fraction`.
