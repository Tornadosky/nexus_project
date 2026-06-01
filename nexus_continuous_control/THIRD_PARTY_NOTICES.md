# Third-party notes

This extension is designed to interoperate with:

- `remunds/symbolic_options`, the NEXUS reference codebase.
- `mttga/purejaxql`, especially its MuJoCo Playground Actor-Critic PQN setup and wrappers.
- `google-deepmind/mujoco_playground`, the MuJoCo Playground environment library.

No third-party source file is vendored wholesale in this artifact. The training
code is a new hierarchical implementation that follows the AC-PQN design pattern
and imports PureJAXQL's Playground wrappers at runtime.
