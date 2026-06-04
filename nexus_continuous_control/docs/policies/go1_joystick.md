# Go1JoystickFlatTerrain Policy

Skills: `stand`, `track_velocity`, `turn`, `recover`.

The policy first checks semantic `info` keys for base height, roll, pitch, velocity, yaw,
and command values. Fallback raw observation indices use `0` height, `1` roll, `2` pitch,
`3:5` planar velocity, `5` yaw rate, and `6:9` joystick commands.

Diagnostics: `go1/base_height`, `go1/roll`, `go1/pitch`, `go1/command_xy_norm`,
`go1/command_yaw`.
