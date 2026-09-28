# LLM Skill Extension for NEXUS

## Overview

This extension adds LLM-generated skills to NEXUS. Instead of relying only on
hand-designed symbolic skills, an LLM proposes a structured skill hierarchy,
which is validated, compiled into executable JAX reward/mask/meta-policy
functions, and trained by the existing NEXUS trainer like a hand-written
policy. The resulting policy is compared against the hand-designed baseline,
and can be iteratively refined against its own training metrics.

The LLM only produces a typed JSON skill spec (`schema.NexusSkillSet`).
`interpreter.py` compiles that spec into JAX functions using a fixed, 
restricted grammar.

---

## Logic / workflow

```
Environment name
        |
        v
envs/env_registry.py            (semantic obs fields + task description)
        |
        v
pipeline.build_skill_prompt      (system + user prompt)
        |
        v
client.LLMClient.generate_json   (mock / hf / openai backend -> JSON)
        |
        v
pipeline.validate_and_build      (JSON -> schema.NexusSkillSet, field-name check)
        |
        v
interpreter.make_policy_module   (NexusSkillSet -> JAX reward/mask/meta-policy fns)
        |
        v
policies.registry.load_policy_module   (routes USE_LLM_SKILLS configs here)
        |
        v
hierarchical_ac_pqn_playground.run_training   (same trainer as hand-written policies)
        |
        v
run_llm_full_suite -> collect_llm_results 
```

### Interpreter safety

- **Activation rule**: boolean expression over field names. Supported: numbers,
  `abs/min/max`, comparisons, `and/or/not`, `AND/OR/NOT`, `&&`, `||`, `!`,
  `+ - * /`. Any parse error fails closed to all-False; if no skill is active
  the mask falls back to skill 0.
- **Reward terms** (fixed vocabulary, anything else contributes 0):
  `negative_distance`, `positive_velocity`, `target_height`, `binary_bonus`,
  `action_penalty`, `posture_penalty`.
- `lhs` / `rhs` resolve to: a field name, a numeric constant (JSON number or
  numeric string), else `None` (term falls back to its default).


## Files

```
nexus_continuous/
  llm/
    __init__.py
    schema.py              NexusSkillSet / SkillSpec / RewardTerm 
    client.py              LLMClient (mock / hf / openai / vertex backends) + MockSkillGenerator
    pipeline.py            Prompt building, JSON -> NexusSkillSet validation, generate and
                           save skillset
    interpreter.py         Safe rule evaluator + reward-term compiler
                           Builds the runnable policy module (skill_rewards/skill_mask/
                           symbolic_meta_policy/task_metrics) the trainer loads
    refinement_loop.py      RefinementConfig + LLMRefinementLoop: propose -> train ->
                           summarize_metrics -> feedback -> revise
    common_fallback.py      Reimplementation of policies.common's obs-parsing
                           helpers, used only if policies.common isn't importable, so
                           interpreter.py stays testable in isolation
    jax_bootstrap.py       ensure_jax(): makes `import jax` resolve to a bundled numpy
                           stub (_jax_stub/) when real jax isn't installed
    mock_training.py       MockTrainer: deterministic, seeded train_fn(skillset) -> metrics stand-in,
                           same call signature as the real trainer

  envs/
    env_registry.py         ENV_REGISTRY: Semantic obs fields + task description
                           used to build prompts. Keys match the canonical Playground env
                           names used on the repository.

 scripts/
    run_llm_full_suite.py    MAIN ENTRY: hand-written baseline + LLM skillset +
                             refinement, per env x backend, writes manifest.json
    collect_llm_results.py   manifest.json -> summary.csv, plots, REPORT.md
    run_llm_refinement.py    refinement loop for one env/backend (also used by the suite)
    run_llm_experiment.py    single env: generate + train one LLM skillset
    run_llm_comparison.py    single env: hand-written vs LLM, N seeds


tests/ (or alongside the files, depending on repo layout)
  test_llm.py                Full suite: schema, client, pipeline, refinement_loop,
                             mock_training, and (skipped if jax unavailable) interpreter +
                             registry integration.
  test_llm_interpreter.py     eval_rule, make_policy_module (rewards/mask/meta-policy
                             shapes, progressive vs strict mask mode).
  test_llm_pipeline.py        Prompt building, reward/skill parsing, validate_and_build,
                             pipeline retry behavior.
  test_client.py              extract_json, mock backend responses.
  test_interpreter.py          eval_rule truth tables, each reward-term type, fail-closed
                             behavior on invalid rules / unknown fields.
```
---

## Reproducibility

### Run the experiment
 
```bash
python -m nexus_continuous.scripts.run_llm_full_suite \
    --envs CartpoleBalance \
    --backends hf \
    --seeds 0 1 2 3 4 \
    --refine-iterations 4 \
    --output results_cartpole_balance_test \
    --override EVAL_AFTER_TRAIN=True \
    --override EVAL_NUM_ENVS=128 \
    --override EVAL_NUM_EPISODES=128
```
 
What it does, in order, for each env:
 
1. **Hand-written baseline**: trains the hand-designed skills once per seed.
2. **LLM skillset**: generates ONE skillset for (env, backend), then trains that
   same skillset on every seed (fair mean/std against the baseline).
3. **Refinement loop** (`--refine-iterations 4`): the LLM proposes a fresh
   skillset, it is trained (seed = first seed), metrics are fed back, the LLM
   revises, repeat for 4 trainings total. Note this loop starts from its own
   proposal, not the skillset from step 2.
`EVAL_AFTER_TRAIN=True` enables the deterministic evaluation that produces
`primary_success_rate` and `primary_goal_metric`.
 
### Get results
 
```bash
python -m nexus_continuous.scripts.collect_llm_results --results results_cartpole_balance_test
``` 
Under `--output <dir>/`:
| File | Content |
|---|---|
| `manifest.json` / `.pkl` | everything, consumed by the collector |
| `<Env>_summary.json` | per-env: all seeds' metrics, both policies, refinement history |
| `<Env>/<backend>/llm_skillset.json` | the generated skillset (step 2) |
| `<Env>/<backend>/refinement/iter_<N>_skillset.json` | skillset used at each iteration |
| `<Env>/<backend>/refinement/history.json`, `final_skillset.json`, `refinement_curve.png` | refinement record |
| `summary.csv` | env, backend, kind, metric, mean, std |
| `<Env>_comparison.png` | bars: hand-written vs LLM vs refined |
| `<Env>_refinement.png` | reward vs refinement iteration |

### Further analysis
```bash
python analyze_llm_results.py --root results_llm --out analysis_output
```
Analyze results by extracting from `<env>/manifest.json` and produces main tables and plots that summarize comaprisons, skill usages and refinement behaviours.
Produces two folders containing results. 
        
        /tables -> Environment comparisons in measurements and generated skills
        
        /plots -> Curves and graphs designed to highlight difference across seed rewards and refinement curves
        
        report_tables.md -> Gathers all tables and renders them 
        
---

All results are available in ```nexus_continuous_control/results_llm```
