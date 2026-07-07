from nexus_continuous.llm.pipeline import generate_skillset, save_skillset

def generate_for_environment(cfg):
    skillset = generate_skillset(
        env_name = cfg["ENV_NAME"],
        observation_schema = cfg["OBS_SCHEMA"],
        task_description = cfg["TASK_DESCRIPTION"]
    )

    print(skillset)
    save_skillset(skillset, f"skills/{cfg['ENV_NAME']}_llm.json")
    print("Skills saved!")
    
    return skillset
