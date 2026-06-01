# LLM prompt templates for continuous-control NEXUS

Use these after the hand-written policies are working. The output should be JSON
matching `nexus_continuous.llm.schema.NexusSkillSet`; do not execute raw code.

## Skill and reward proposal

You are a reinforcement-learning specialist. I am building a hierarchical NEXUS
agent for MuJoCo Playground environment `<ENV_NAME>`. The low-level actions are
continuous torques/controls. The high-level controller chooses among a short list
of interpretable skills, and each skill actor is trained with its own reward.

Observation/state schema:

```text
<STATE_SCHEMA_WITH_NAMED_FIELDS>
```

Task description:

```text
<TASK_DESCRIPTION>
```

Return 3-5 skills. For each skill provide:

1. `name`, a short snake_case name.
2. `description`, what the skill should learn.
3. `activation_rule`, a simple boolean rule over named state fields.
4. `reward_terms`, a list of terms selected from: negative_distance,
   positive_velocity, target_height, binary_bonus, action_penalty,
   posture_penalty.

Return only JSON matching the schema.

## Meta-policy proposal

Given these skills and rewards:

```json
<SKILL_JSON>
```

Write a priority-ordered symbolic meta-policy and a non-exclusive NeSy mask. Use
simple rules over the named state fields. Return JSON with `symbolic_rules` and
`mask_rules`; do not return executable Python.
