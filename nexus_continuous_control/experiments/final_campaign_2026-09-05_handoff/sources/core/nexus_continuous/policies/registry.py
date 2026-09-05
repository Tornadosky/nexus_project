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
    "cartpole_balance_rules2": PolicySpec(
        # No env alias, same as go1_joystick_rules2: selectable only via an explicit POLICY=,
        # so a plain CartpoleBalance run can never pick up the revised thresholds by accident.
        env_aliases=("cartpole_balance_rules2", "cartpole-balance-rules2"),
        module="nexus_continuous.policies.cartpole_balance_rules2",
        recommended_env_name="CartpoleBalance",
        description="cartpole_balance with mask/rule thresholds set at a working policy's "
                    "visitation quantiles (runbook §5 stage 1).",
    ),
    "go1_joystick": PolicySpec(
        # RoughTerrain is an alias, not a separate policy: it has the identical observation
        # layout (state (48,), privileged_state (123,), action 12) and command semantics as
        # flat terrain, verified by tools/audit_semantics.py. The alias matters for BOTH
        # hierarchical and flat arms — `TASK_POLICY` falls back to `ENV_NAME`, and that
        # lookup is what supplies `task_metrics` (tracking success). Without it the rough
        # cells cannot be scored on the same metric as the flat-terrain ones.
        env_aliases=(
            "Go1JoystickFlatTerrain", "Go1JoystickRoughTerrain",
            "go1_joystick", "go1-joystick",
        ),
        module="nexus_continuous.policies.go1_joystick",
        recommended_env_name="Go1JoystickFlatTerrain",
        description="Unitree Go1 joystick flat/rough terrain with stand/track/turn/recover skills.",
    ),
    "go1_joystick_rules2": PolicySpec(
        # No env alias: this module must only ever be selected explicitly via POLICY=, so a
        # plain Go1 run can never pick up the revised thresholds by accident. `rules2` is
        # already in EXPERIMENTAL_TAGS in tools/analyze_v2.py.
        env_aliases=("go1_joystick_rules2", "go1-joystick-rules2"),
        module="nexus_continuous.policies.go1_joystick_rules2",
        recommended_env_name="Go1JoystickFlatTerrain",
        description="go1_joystick with recover-mask thresholds set at a working policy's "
                    "visitation quantiles (runbook §5 stage 1).",
    ),
    "flat_baseline": PolicySpec(
        env_aliases=("flat", "flat_baseline", "ac_pqn_flat"),
        module="nexus_continuous.policies.flat_baseline",
        recommended_env_name="Any supported Playground environment",
        description="Single-skill flat AC-PQN baseline trained on environment reward.",
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


def load_policy_module(name: str) -> ModuleType:
    canonical = canonicalize_policy_name(name)
    return importlib.import_module(POLICY_SPECS[canonical].module)


def list_policies() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "recommended_env_name": spec.recommended_env_name,
            "aliases": spec.env_aliases,
            "description": spec.description,
        }
        for name, spec in POLICY_SPECS.items()
    }
