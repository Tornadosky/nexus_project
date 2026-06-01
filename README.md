main deliveries:
- working hierarchical PQN implementation in the NEXUS framework (symbolic_options) in JAX
- ideally working in 4-5 continuous control environments from mujoco MJC/playground
- this includes: reward functions and meta-policy functions (first written by hand / interactively - gotten from LLM)
- later: query LLM for that
- possible extension: use RGB inputs for the skill-agents (might achieve better performance)

resources:
- https://github.com/remunds/symbolic_options
- https://github.com/mttga/purejaxql
- 9847_From_Objects_to_Skills_In.pdf