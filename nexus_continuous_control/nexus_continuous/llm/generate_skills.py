from nexus_continuous.llm.pipeline import generate_skillset
from nexus_continuous.llm.client import LLMClient, LLMConfig
from nexus_continuous.llm.pipeline import LLMSkillPipeline 

client = LLMClient(LLMConfig(backend = "hf"))
pipeline = LLMSkillPipeline(client)

skillset = generate_skillset(
    env_name = "CartpoleBalance",
    observation_schema = "pole_angle, pole_velocity, cart_position, cart_velocity",
    task_description = "Balance cartpole"
)

# with open("cartpole_llm_skills.json", "w") as f:
#     f.write(skillset.model_dump_json(indent = 2))