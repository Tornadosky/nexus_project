from nexus_continuous.llm.pipeline import generate_skillset

skillset = generate_skillset(
    env_name = "WalkerWalk",
    observation_schema = """
    "torso_height",
    "height",
    "torso_pitch",
    "pitch",
    "x_velocity",
    "forward_velocity",
    "joint_speed"
    """,
    task_description = "Walk forward while staying balanced."
)

print(skillset)