# Flat AC-PQN Baseline

Skill: `flat_actor`.

This baseline uses the same MuJoCo Playground wrappers, actor/critic code, normalization,
action noise, and logging stack as NEXUS, but has one always-available skill and trains
on environment reward only. It does not use symbolic masks or hand-written skill rewards.

Diagnostics: `flat/env_reward`.
