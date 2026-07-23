# LLM Prompts

This file documents the exact prompt templates used by this extension, for
human review/editing. The live copies (the actual strings the code sends)
live in `pipeline.py::build_skill_prompt` and
`refinement_loop.py::build_feedback_prompt` -- keep this file in sync if you
edit those.

## 1. Initial skill-generation prompt (`pipeline.build_skill_prompt`)

**System prompt:**

```
You are an expert reinforcement learning researcher designing interpretable
hierarchical skill policies for continuous control.
```

**User prompt:**

```
You are designing skills for the MuJoCo Playground environment: {env_name}
OBSERVATION SCHEMA: {observation_schema}
TASK DESCRIPTION: {task_description}

Return ONLY valid JSON matching this structure:
{
    "environment": "{env_name}",
    "observation_schema": "...",
    "skills":[
        {
            "name": "string",
            "description": "string",
            "activation_rule": "boolean expression over fields",
            "reward_terms": [
                {
                    "type": "negative_distance | positive_velocity | target_height | binary_bonus | action_penalty | posture_penalty",
                    "weight": 1.0,
                    "lhs": "field_name or null",
                    "rhs": "field_name or null",
                    "threshold": 0.0
                }
            ]
        }
    ],
    "meta_policy_notes": "string"
}

Rules:
- Use ONLY fields from observation schema
- 3-5 skills only
- Skills should form a progression (safe -> locomotion -> optimal performance)
- activation_rule must be a boolean expression (and/or/not, comparisons)
- reward must be stable and dense (avoid sparse-only rewards)
- Do not include markdown fences.
- Do not explain anything.
- Output only the JSON.
```

On invalid JSON, `LLMSkillPipeline.generate_skillset` appends the following
and retries (up to `max_retries` times):

```
IMPORTANT: Previous output was invalid JSON: {last_error}
Return STRICT JSON ONLY.
```

## 2. Refinement / feedback prompt (`refinement_loop.build_feedback_prompt`)

**System prompt:**

```
You are an expert reinforcement learning researcher improving hierarchical
skills for continuous control agents.
```

**User prompt:**

```
We trained a hierarchical RL agent on environment: {env_name}
Previous skillset: {previous_skillset_json}
Training performance metrics: {metrics_summary}

Task: Improve the skill decomposition.

Return a new improved JSON skillset with:
- better reward shaping
- better activation rules
- improved skill progression
- same schema as before

Rules:
- Keep 3-5 skills
- Make skills more distinct and non-overlapping
- Improve weakest-performing skills
- Avoid overly strict activation rules
- Ensure reward signals are dense and learnable

Return ONLY valid JSON.
```

`metrics_summary` is produced by `refinement_loop.summarize_metrics`, which
extracts a fixed set of keys from the raw training metrics dict:

```
returns/env_reward_mean
returns/skill_reward_mean
policy_diag/primary_success_rate
policy_diag/primary_goal_metric
env/returned_episode_returns
```

On invalid JSON, `LLMRefinementLoop.refine_once` appends and retries (up to
`cfg.max_retries` times):

```
IMPORTANT: Previous output was invalid: {last_error}
Return STRICT JSON ONLY, matching the schema exactly.
```

## Fixed reward-term vocabulary (`interpreter._term_reward`)

The LLM is restricted to these six reward-term `type`s; anything else
compiles to a constant zero contribution:

| type                | meaning                                              |
|---------------------|-------------------------------------------------------|
| `negative_distance`  | `-weight * |lhs - rhs|` (or `|lhs - threshold|`, or `|lhs|`) |
| `positive_velocity`  | `weight * lhs`                                        |
| `target_height`      | `weight * clip(lhs / threshold, 0, 1)` if threshold>0  |
| `binary_bonus`       | `weight * (lhs > threshold)`                            |
| `action_penalty`     | `-weight * sum(action ** 2)`                             |
| `posture_penalty`    | `-weight * |lhs|`                                        |

## Fixed activation-rule grammar (`interpreter.eval_rule`)

Restricted AST subset only -- **no `eval`/`exec` of LLM output ever**:
field names, numeric literals, `abs()` / `min()` / `max()`, comparisons
(`>`, `>=`, `<`, `<=`, `==`, `!=`), boolean ops (`and`, `or`, `not`, also
accepts `AND`/`OR`/`NOT`), and `+ - * /`. Anything outside this grammar (or
any parse/eval error) fails **closed** to an all-`False` mask.
