# LLM Skill Discovery Extension for NEXUS Continuous Control

## Overview
This extension adds LLM-generated skills and meta-policies to the NEXUS hierarchical RL framework.

For each environment the following are manually designed:
- Skill reward functions
- Symbolic meta-policies
- Skill masks

This extension follows the generated flow:
1. The LLM proposes interpretable skills.
2. The proposal is converted into executable JAX reward functions.
3. The skills are trained using the existing NEXUS pipeline.
4. Training metrics are fed back to the LLM.
5. Performance is compared against hand-designed policies.

The goal is to evaluate and compare whether LLM-generated hierarchical decompositions can exceed manually designed skill/reward structures.

## Directory Structure
```text
nexus_continous/
    llm/
        __init__.py
        schema.py
        interpreter.py
        client.py
        pipeline.py
        refinement_loop.py
        prompts.md
```
### schema.py
Defines the safe structured representation of an LLM-generated skill set.

**RewardTerm:** Represents a reward component:

```bash
RewardTerm(
    type = "positive_velocity",
    weight = 1.0,
    lhs = "forward_velocity"
)
```
Supported reward types:
- negative_distance
- positive_velocity
- target_height
- binary_bonus
- action_penalty
- posture_penalty

**SkillSpec:** Represents one skill:

```python
SkillSpec(
    name = "walk_forward",
    description = "Move forward steadily",
    activation_rule = "forward_velocity < 2"
)
```
**NexusSkillSet**: Complete LLM-generated policy description:
```python
NexusSkillSet(
    environment = "WalkerWalk",
    skills = [...]
)
```

### interpreter.py
Converts JSON skill descriptions into executable policy modules.

It doesn't execute arbitrary LLM code, but instead creates a workflow:\
``JSON -> Reward terms -> JAX functions -> Training``

**skill_rewards()** compiles reward terms into JAX reward functions.\
**skill_mask** creates NeSy skill masks and supports strict and progressive masking.\
**symbolic_meta_policy()** compiles activation rules into symbolic meta-policies.\
**make_policy_module()** produces a module compatible with the existing NEXUS trainer.

### client.py
Responsible for communication with the LLM through:
- OpenAI support
- Mock fallback mode
- JSON parsing
- Structured generation

Returns a ``dict`` representing a skill set.

### pipeline.py
Coordinates the skill generation process.

**Prompt construction:**
Inject the following into the prompt:
- environment name
- observation schema
- task description

**Validation:** Converts raw LLM output into ``NexusSkillSet`` objects.

**Error Handling**
In case malformed JSON is produced we retry generation and strengthen the prompt constraints.

**Output:** Produces validated skill sets ready for compilation.

### refinement_loop.py
Implements iterative improvement:
1. Generate skill set
2. Train
3. Collect metrics
4. Summarize metrics
5. Feed metrics back to LLM
6. Generate improved skill set

Example of metrics that become feedback to the LLM:
```bash
returns/env_reward_mean
policy_diag/primary_success_rate
primary_goal_metric
```

## System Run
**Step 1:** Set up the environment:
```bash
pip install openai
export OPENAI_API_KEY = ...   # for Linux/macOS
$env: OPENAI_API_KEY = "..." # for Windows PowerShell
```
**Step 2:** Generate a skill set.

```python
from nexus_continuous.llm.pipeline import generate_skillset

skillset = generate_skillset(
    env_name = "WalkerWalk",
    observation_schema = """
    height
    pitch
    forward_velocity
    joint_speed
    """
    task_description = "Walk forward while maintaining balance."
)
```
**Step 4:** Compile skills.
```python
from nexus_continuous.llm.interpreter import make_policy_module
```
Create policy module:
```python 
policy = make_policy_module(...)
``` 
**Step 5:** Train \
Run existing NEXUS trainer using generated policy.

**Running Iterative Refinement**
```python
From nexus_continuous.llm.refinement_loop import(
    LLMRefinementLoop,
    RefinementConfig,
)

loop = LLMRefinementLoop(...)
```
Configure:
```python
cfg = RefinementConfig(
    env_name = "WalkerWalk",
    observation_schema = schema,
    task_description = task,
    num_iteration = 5,
)
```
Execute: ``loop.run(...)``

## Recommended Evaluation
For each environment run 5 seeds for:
- hand-written symbolic
- hand-written NeSy
- LLM-generated symbolic
- LLM-generated NeSy

Metrics:
- success rate
- episode return
- sample efficiency
- skill usage distribution
- training stability



