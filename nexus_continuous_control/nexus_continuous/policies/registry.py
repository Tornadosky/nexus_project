"""Registry for environment-specific hand-written NEXUS policies."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any


@dataclass(frozen=True)
class PolicySpec:
    env_aliases: tuple[str, ...]
    module: str
    recommended_env_name: str
    description: str


POLICY_SPECS: dict[str, PolicySpec] = {
    "cartpole_balance": PolicySpec(
        env_aliases=("CartpoleBalance", "cartpole-balance", "cartpole_balance"),
        module="nexus_continuous.policies.cartpole_balance",
        recommended_env_name="CartpoleBalance",
        description="Cart-pole balance with recover/center/damp skills.",
    ),
    "cheetah_run": PolicySpec(
        env_aliases=("CheetahRun", "cheetah-run", "cheetah_run"),
        module="nexus_continuous.policies.cheetah_run",
        recommended_env_name="CheetahRun",
        description="DM Control cheetah run with accelerate/stabilize/efficient skills.",
    ),
    "walker_walk": PolicySpec(
        env_aliases=("WalkerWalk", "walker-walk", "walker_walk"),
        module="nexus_continuous.policies.walker_walk",
        recommended_env_name="WalkerWalk",
        description="DM Control walker walk with stand/walk/stabilize/efficient skills.",
    ),
    "hopper_hop": PolicySpec(
        env_aliases=("HopperHop", "hopper-hop", "hopper_hop"),
        module="nexus_continuous.policies.hopper_hop",
        recommended_env_name="HopperHop",
        description="DM Control hopper hop with stand/hop/stabilize/efficient skills.",
    ),
    "panda_pick_cube": PolicySpec(
        env_aliases=("PandaPickCube", "panda_pick_cube", "panda-pick-cube"),
        module="nexus_continuous.policies.panda_pick_cube",
        recommended_env_name="PandaPickCube",
        description="Panda pick-cube manipulation with reach/grasp/lift/place skills.",
    ),
    "go1_joystick": PolicySpec(
        env_aliases=("Go1JoystickFlatTerrain", "go1_joystick", "go1-joystick"),
        module="nexus_continuous.policies.go1_joystick",
        recommended_env_name="Go1JoystickFlatTerrain",
        description="Unitree Go1 joystick flat-terrain with stand/track/turn/recover skills.",
    ),
    "flat_baseline": PolicySpec(
        env_aliases=("flat", "flat_baseline", "ac_pqn_flat"),
        module="nexus_continuous.policies.flat_baseline",
        recommended_env_name="Any supported Playground environment",
        description="Single-skill flat AC-PQN baseline trained on environment reward.",
    ),
    "ant_walk": PolicySpec(
        env_aliases = ("AntWalk", "ant_walk", "ant-walk"),
        module = "nexus_continuous.policies.ant_walk",
        recommended_env_name = "Ant",
        description = "Ant locomotion with stand/walk/bound/turn/recover skills.",
    ),
    "humanoid_walk": PolicySpec(
        env_aliases = ("HumanoidWalk", "humanoid_walk", "humanoid-walk"),
        module = "nexus_continuous.policies.humanoid_walk",
        recommended_env_name = "Humanoid",
        description = "Humanoid locomotion with stand/walk/bound/turn/recover skills.",
    ),
}


def canonicalize_policy_name(name: str) -> str:
    lowered = name.lower()
    for canonical, spec in POLICY_SPECS.items():
        if lowered == canonical.lower() or any(lowered == alias.lower() for alias in spec.env_aliases):
            return canonical
    raise KeyError(
        f"Unknown policy name/env {name!r}. Available: {', '.join(sorted(POLICY_SPECS))}"
    )


# def load_policy_module(name: str) -> ModuleType:
#     canonical = canonicalize_policy_name(name)
#     return importlib.import_module(POLICY_SPECS[canonical].module)


def list_policies() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "recommended_env_name": spec.recommended_env_name,
            "aliases": spec.env_aliases,
            "description": spec.description,
        }
        for name, spec in POLICY_SPECS.items()
    }


from nexus_continuous.llm.interpreter import make_policy_module

def load_policy_module(name_or_cfg) -> ModuleType:
    """ 
    Loads either:
    - LLM-generated policy
    - Handwritten policy from registry
    """
    if isinstance(name_or_cfg, dict):
        cfg = name_or_cfg
        
        if cfg.get("USE_LLM_SKILLS", False):
            return make_policy_module(skillset = cfg["LLM_SKILLSET"], field_names = tuple(cfg["OBS_FIELDS"]))
        
        name = cfg["POLICY"]
    else:
        name = name_or_cfg
    
    canonical = canonicalize_policy_name(name)
    return importlib.import_module(POLICY_SPECS[canonical].module)