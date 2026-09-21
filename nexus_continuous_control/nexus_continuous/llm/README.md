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

Functional end-to-end:
generation -> compile -> train -> compare -> refine

---

## Requirements

Base requirements are the same as the rest of the repo. Based on the
parts of the code used:

| Component | Needs | Notes |
|---|---|---|
| `schema.py`, `interpreter.py` (compile-only) | numpy + a `jax`-like array API | Works with the bundled `_jax_stub/` if real `jax` isn't installed (enough for `mock_training_demo.ipynb`) |
| Real training (`run_llm_experiment.py`, `run_llm_comparison.py`, or any notebook that calls `run_training`) | real `jax`, `flax`, `optax`, `mujoco_playground` | Same as hand-written-policy training |
| `LLMClient(backend="mock")` | nothing extra | Deterministic, seeded, no network. Used for offline demos |
| `LLMClient(backend="hf")` | `transformers`, a local/downloaded model (default `Qwen/Qwen2.5-1.5B-Instruct`) | Loaded in LLMClient |
| `LLMClient(backend="openai")` | `openai` package + `OPENAI_API_KEY` env var | |
| `LLMClient(backend="vertex")` | `gcloud` CLI on `PATH`, authenticated (`gcloud auth login`), with a GCP project (auto-resolved from `gcloud config get-value project`, or pass `LLMConfig(project=...)`) | No Python package needed -- calls the Vertex AI REST API directly over `urllib`. Model defaults to `gemini-2.5-flash`. |
| `tools/llm_skill_gen.py` | `gcloud` CLI, authenticated, with a GCP project set | Separate from LLMClient (talks to Vertex AI directly over `urllib`) |
| Plotting | `matplotlib`, `pandas` (comparison notebook only) | |

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
Performance comparison / refinement feedback
```

### How an LLM skillset reaches the trainer

For a hand-written policy, `cfg["POLICY"]`
is a name string (e.g. `"cartpole_balance"`). For an LLM-generated policy, set:

```python
cfg["USE_LLM_SKILLS"] = True
cfg["LLM_SKILLSET"] = asdict(skillset)      # or the raw JSON dict
cfg["OBS_FIELDS"] = tuple(meta["fields"])   # field names the compiled rules/rewards use
```

`hierarchical_ac_pqn_playground.make_train` resolves the trained policy
(`cfg["POLICY"]`) by forwarding the whole config to
`policies.registry.load_policy_module` whenever `USE_LLM_SKILLS` is set, so it
sees `LLM_SKILLSET`/`OBS_FIELDS` and compiles the skillset via
`interpreter.make_policy_module`. 

### Interpreter safety

`interpreter.py` never calls `eval`/`exec` on LLM output. It applies an evaluation rule
that only allows: named fields, numeric literals,
`abs()`/`min()`/`max()`, comparisons, and/or/not , and `+ - * /`. 
Any parse/eval error, or any field name not
resolvable, fails closed. Rewards are default to `0` and rule masks default to
`False`. `skill_mask` then falls back to skill 0 for any state where nothing
is active, so the mask never leaves an env with zero valid skills.

### Mask modes

`make_policy_module(..., mask_mode=...)`:
- `"strict"` -- the NeSy mask is exactly the compiled activation rule. Faithful
  to what the LLM wrote, but mutually-exclusive rules can collapse the mask to
  one skill everywhere.
- `"progressive"` -- for progression-style skillsets (reach -> grasp ->
  lift), also allows every skill up to (highest currently-active index + 1),
  so the meta-policy can attempt the next step instead of getting stuck. Used
  by `tools/llm_skill_compare.py --mask-mode progressive`.

---

## Files

```
nexus_continuous/
  llm/
    __init__.py
    schema.py              NexusSkillSet / SkillSpec / RewardTerm 
                           (frozen dataclasses LLM's JSON gets deserialized into)
    client.py              LLMClient (mock / hf / openai / vertex backends) + MockSkillGenerator
                           (deterministic seeded stand-in for a real LLM)
    pipeline.py            Prompt building, JSON -> NexusSkillSet validation
                           (validate_and_build), LLMSkillPipeline generate and
                           save skillset
    interpreter.py         Safe rule evaluator + reward-term compiler
                           Builds the runnable policy module (skill_rewards/skill_mask/
                           symbolic_meta_policy/task_metrics) the trainer loads
    refinement_loop.py      RefinementConfig + LLMRefinementLoop: propose -> train ->
                           summarize_metrics -> feedback -> revise, for N iterations
    common_fallback.py      Reimplementation of policies.common's obs-parsing
                           helpers, used only if policies.common isn't importable, so
                           interpreter.py stays testable in isolation
    jax_bootstrap.py       ensure_jax(): makes `import jax` resolve to a bundled numpy
                           stub (_jax_stub/) when real jax isn't installed, so
                           interpreter.py's compile path is exercisable without a
                           real JAX/CUDA stack
    mock_training.py       MockTrainer: deterministic, seeded train_fn(skillset) -> metrics stand-in,
                           same call signature as the real trainer
    plot.py                Plot comparison and refinement

  envs/
    env_registry.py         ENV_REGISTRY: Semantic obs fields + task description
                           used to build prompts. Keys match the canonical Playground env
                           names used on the repository.

  scripts/
    run_llm_experiment.py   CLI + run_llm_experiment(...): generate a skillset for one env
                           and train it. Saves as results/<env>_llm_skills.json and
                           results/<env>_llm_results.pkl.
    run_llm_comparison.py   CLI + compare_policies(...): train hand-written and LLM
                           policies on the same config/env/seeds and summarize
                           returns and env_reward_mean for both.

tools/
  llm_skill_gen.py           Standalone Vertex AI Gemini skill generator (JSON mode),
                           independent of client.LLMClient. Has a --style refined flag
                           that adds guidance fixing the two failure modes seen with raw
                           LLM output (over-tight exclusive masks, weak reward shaping).
  llm_skill_compare.py        Train a saved LLM skillset (from llm_skill_gen.py's --out)
                           head-to-head against the hand-written policy using the SAME
                           trainer/env/config/seed and the hand-written task_metrics.

notebooks/
  llm_skill_generation_training.ipynb   Generate -> compile/sanity-check -> train for
                                        real -> refine (interactive refinement loop).
  llm_vs_handwritten_comparison.ipynb    Multi-seed hand-vs-LLM comparison, single env
                                        then the full suite, plotted + saved to CSV.
  mock_training_demo.ipynb               Same shape of experiment (generate -> compile ->
                                        compare -> refine), fully offline via MockTrainer.

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

## Execution / examples

### 1. Generate one skillset (no training)

```python
from nexus_continuous.llm.client import LLMClient, LLMConfig, MockSkillGenerator
from nexus_continuous.llm.pipeline import generate_skillset
from nexus_continuous.envs.env_registry import ENV_REGISTRY

meta = ENV_REGISTRY["CartpoleBalance"]
client = LLMClient(LLMConfig(backend="mock", seed=0),
                    mock_generator=MockSkillGenerator(meta["fields"], seed=0))
skillset = generate_skillset(
    env_name="CartpoleBalance",
    observation_schema="\n".join(meta["fields"]),
    task_description=meta["task"],
    client=client,
    allowed_fields=set(meta["fields"]),
)
```

Swap `LLMConfig(backend="mock", ...)` for `LLMConfig(backend="hf")`,
`LLMConfig(backend="openai")` or `LLMConfig(backend="vertex")` for a real model.

### 2. Generate + train for real (CLI)

```bash
python -m nexus_continuous.scripts.run_llm_experiment \
    --env CartpoleBalance \
    --config configs/cartpole_balance_nesy.yaml \
    --seed 0 \
    --output results
```

Or from Python / notebook:

```python
from nexus_continuous.scripts.run_llm_experiment import run_llm_experiment

result = run_llm_experiment(
    env_name="CartpoleBalance",
    config="configs/cartpole_balance_nesy.yaml",
    seed=0,
    output="results",
)
result["skillset"], result["metrics"], result["eval_metrics"]
```

`notebooks/llm_skill_generation_training.ipynb` provides the full walkthrough
(generate -> compile/sanity-check -> train -> refine).

### 3. Compare against the hand-written policy (CLI)

```bash
python -m nexus_continuous.scripts.run_llm_comparison \
    --env WalkerWalk \
    --config configs/walker_walk_nesy.yaml \
    --seeds 3
```

Or:

```python
from nexus_continuous.scripts.run_llm_comparison import compare_policies

row = compare_policies("WalkerWalk", "configs/walker_walk_nesy.yaml", num_seeds=3)
row["hand_mean"], row["llm_mean"]
```

`notebooks/llm_vs_handwritten_comparison.ipynb` provides the multi-environment
version (loops over an env->config dict, builds a `pandas.DataFrame`, plots a grouped bar chart, saves `results/llm_vs_handwritten_comparison.csv`).

### 4. Refinement loop

```python
from nexus_continuous.llm.pipeline import LLMSkillPipeline
from nexus_continuous.llm.refinement_loop import LLMRefinementLoop, RefinementConfig

def train_fn(skillset):
    ...  # -> dict with at least "returns/env_reward_mean"

pipeline = LLMSkillPipeline(client)
loop = LLMRefinementLoop(pipeline, client)
cfg = RefinementConfig(
    env_name="CartpoleBalance",
    observation_schema="\n".join(meta["fields"]),
    task_description=meta["task"],
    num_iterations=3,
    allowed_fields=set(meta["fields"]),
)
result = loop.run(cfg, train_fn)
result.final_skillset, result.history, result.stopped_early
```

If a refinement step's LLM call fails after
retries, the loop records `refinement_ok=False` on that iteration and stops
early, returning the last good skillset rather than raising.

### 5. Fully offline demo (no jax/mujoco_playground/GPU/API key)

```
notebooks/mock_training_demo.ipynb
```

Runs generation (mock backend) -> compile+sanity-check -> multi-seed
comparison -> refinement loop, entirely with `mock_training.MockTrainer`, in
well under a second per cell. Useful for CI or for understanding the control
flow before running anything expensive.

### 6. Real LLM generation via Vertex AI (standalone tool)

```bash
python tools/llm_skill_gen.py --env PandaPickCube --style refined \
    --out docs/reports/llm_generated_skills
python tools/llm_skill_compare.py --env PandaPickCube --meta nesy --updates 180
```

`llm_skill_gen.py` authenticates via `gcloud auth print-access-token` and
calls Gemini directly (JSON mode). `--style refined` adds guidance that fixes
the two failure modes observed with raw LLM output.
`llm_skill_compare.py`
then trains the saved JSON against the hand-written baseline with the same
trainer/config/seed and the hand-written `task_metrics`, for a directly
comparable `primary_success_rate`.

---

### 7. Real LLM generation via Vertex AI (in-pipeline, via `LLMClient`)
 
Use `backend="vertex"` directly. No separate tool/JSON-file step:
 
```python
from nexus_continuous.llm.client import LLMClient, LLMConfig
 
client = LLMClient(LLMConfig(backend="vertex"))  # project auto-resolved from gcloud
```

Prefer this over
`llm_skill_gen.py` unless you specifically want the `--style refined` prompt
guidance or the save-JSON-then-compare-later workflow.
 
---

## Testing

```bash
pytest nexus_continuous/llm/test_llm.py -v

# or all of them together:
cd nexus_continuous_control
pytest tests
```