from nexus_continuous.llm.pipeline import generate_skillset

skillset = generate_skillset(
    env_name = "CartpoleBalance",
    observation_schema = "pole_angle, pole_velocity, cart_position, cart_velocity",
    task_description = "Balance a cartpole"
)

with open("cartpole_llm_skills.json", "w") as f:
    f.write(skillset.model_dump_json(indent = 2))