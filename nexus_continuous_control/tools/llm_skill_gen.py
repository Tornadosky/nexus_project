"""LLM skill generation for continuous-control NEXUS via Vertex AI Gemini.

Fills the skill-proposal prompt (see nexus_continuous/llm/prompts.md) with a
per-environment state schema + task description, calls a Gemini model on Vertex
AI (JSON mode), and validates the result against
nexus_continuous.llm.schema.NexusSkillSet. No LLM code is executed: only the
JSON skill spec is parsed.

Auth: uses the local gcloud user credentials (``gcloud auth print-access-token``)
and the active gcloud project. Override model/region/project via flags.

Usage:
    python tools/llm_skill_gen.py --env PandaPickCube
    python tools/llm_skill_gen.py --all --out docs/reports/llm_generated_skills
    python tools/llm_skill_gen.py --env HopperHop --samples 3 --temperature 0.7
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from nexus_continuous.llm.schema import NexusSkillSet, RewardTerm, SkillSpec  # noqa: E402

REWARD_TYPES = [
    "negative_distance", "positive_velocity", "target_height",
    "binary_bonus", "action_penalty", "posture_penalty",
]

# Named semantic fields the playground adapter exposes per env + the task text.
ENVS = {
    "CartpoleBalance": {
        "schema": "cart_position (m), pole_angle (rad, 0=upright), cart_velocity (m/s), pole_angular_velocity (rad/s)",
        "task": "Keep the pole balanced upright (|pole_angle| small) while keeping the cart near track center. Continuous action = horizontal force on the cart.",
    },
    "CheetahRun": {
        "schema": "forward_velocity (m/s), torso_pitch (rad), joint_speed (mean |joint vel|)",
        "task": "Make the planar cheetah run forward as fast as possible while staying stable. 6-D joint torque control.",
    },
    "WalkerWalk": {
        "schema": "torso_height (m, ~1.2 standing), torso_pitch (rad, 0=upright), forward_velocity (m/s), joint_speed",
        "task": "Make the planar walker stand up and walk forward at a modest speed without falling. 6-D joint torque control.",
    },
    "HopperHop": {
        "schema": "torso_height (m, >=0.6 standing), torso_pitch (rad, 0=upright), forward_velocity (m/s), joint_speed",
        "task": "Make the one-legged hopper stand up from a random orientation and hop forward. 4-D joint torque control.",
    },
    "PandaPickCube": {
        "schema": "tcp_pos (xyz of gripper), cube_pos (xyz), target_pos (xyz), gripper_open (0=closed..1=open), dist_tcp_cube, dist_cube_target, cube_height (z of cube; table~0.03)",
        "task": "Reach the cube, grasp it, lift it above the table, and move it toward the target. 8-D continuous arm+gripper control.",
    },
    "Go1JoystickFlatTerrain": {
        "schema": "base_height (m), roll (rad), pitch (rad), lin_vel_x, lin_vel_y, yaw_rate, command_x, command_y, command_yaw (commanded velocities)",
        "task": "Quadruped: follow the commanded planar velocity and yaw rate while staying upright (not falling). 12-D joint control.",
    },
}

HANDWRITTEN = {
    "CartpoleBalance": ("recover_balance", "center_cart", "damp_motion"),
    "CheetahRun": ("accelerate_forward", "stabilize_posture", "energy_efficient_run"),
    "WalkerWalk": ("stand_recover", "walk_forward", "stabilize_gait", "energy_efficient"),
    "HopperHop": ("stand_recover", "hop_forward", "stabilize_landing", "energy_efficient"),
    "PandaPickCube": ("reach_cube", "grasp_cube", "lift_cube", "place_or_stabilize"),
    "Go1JoystickFlatTerrain": ("stand", "track_velocity", "turn", "recover"),
}


def _gcloud() -> str:
    return shutil.which("gcloud") or shutil.which("gcloud.cmd") or (
        r"C:\Users\smirn\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
    )


def get_token() -> str:
    out = subprocess.run([_gcloud(), "auth", "print-access-token"], capture_output=True, text=True)
    lines = [l for l in (out.stdout or "").splitlines() if l.strip()]
    if not lines:
        raise RuntimeError("no access token; is gcloud authenticated? stderr=" + (out.stderr or "")[:200])
    return lines[-1].strip()


def get_project() -> str:
    out = subprocess.run([_gcloud(), "config", "get-value", "project"], capture_output=True, text=True)
    lines = [l for l in (out.stdout or "").splitlines() if l.strip() and "(unset)" not in l]
    return lines[-1].strip() if lines else ""


# Extra guidance that fixes the two limiters observed when LLM skills are trained
# (over-tight exclusive masks -> degeneracy; weak reward shaping). See
# docs/reports/llm_skill_generation.md.
_REFINED_GUIDANCE = """
IMPORTANT design guidance (the skills will be both a NeSy availability MASK and
trained reward functions):
- Activation rules are an AVAILABILITY mask, not a hard switch. Make them
  PERMISSIVE and OVERLAPPING: a skill should be available whenever it could
  plausibly help, so the high-level controller can choose among several. Do NOT
  write mutually-exclusive rules that leave only one skill available in the
  common state -- that collapses to a single skill. Prefer generous thresholds
  (e.g. "close to cube" = distance < 0.15, not < 0.05) and allow the
  goal/progress skills (lift, walk, hop, track) to be available early.
- Reward terms must give a DENSE gradient toward the goal at every state, not
  only a sparse bonus at completion. Always include a small action_penalty, and
  weight the primary progress term (positive_velocity / target_height /
  negative_distance to the goal) clearly higher than secondary stabilisers.
"""


def build_prompt(env: str, style: str = "standard") -> str:
    info = ENVS[env]
    extra = _REFINED_GUIDANCE if style == "refined" else ""
    return f"""You are a reinforcement-learning specialist designing a hierarchical NEXUS agent
for the MuJoCo Playground environment {env}. The high-level controller chooses among a
short list of interpretable skills; each skill actor is trained with its own reward.

Observation/state schema (named fields):
{info['schema']}

Task description:
{info['task']}
{extra}
Return 3-5 skills. Output ONLY JSON matching this schema:
{{"environment": str, "observation_schema": str, "meta_policy_notes": str,
  "skills": [{{"name": snake_case str, "description": str, "activation_rule": str (boolean over named fields),
    "reward_terms": [{{"type": one of {REWARD_TYPES}, "weight": float,
       "lhs": field-or-null, "rhs": field-or-null, "threshold": float-or-null, "description": str}}]}}]}}
Use only the named state fields. Return only JSON.
"""


def call_gemini(prompt: str, token: str, project: str, model: str, region: str,
                temperature: float) -> dict:
    uri = (f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}"
           f"/locations/{region}/publishers/google/models/{model}:generateContent")
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"},
    }
    req = urllib.request.Request(
        uri, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        resp = json.loads(r.read())
    return json.loads(resp["candidates"][0]["content"]["parts"][0]["text"])


def parse_skillset(env: str, data: dict) -> NexusSkillSet:
    skills = []
    for s in data.get("skills", []):
        terms = [
            RewardTerm(type=t.get("type"), weight=float(t.get("weight", 1.0)),
                       lhs=t.get("lhs"), rhs=t.get("rhs"), threshold=t.get("threshold"),
                       description=t.get("description", ""))
            for t in s.get("reward_terms", [])
        ]
        skills.append(SkillSpec(name=s["name"], description=s.get("description", ""),
                                activation_rule=s.get("activation_rule", ""), reward_terms=terms))
    return NexusSkillSet(environment=data.get("environment", env),
                         observation_schema=data.get("observation_schema", ""),
                         skills=skills, meta_policy_notes=data.get("meta_policy_notes", ""))


def validate(ss: NexusSkillSet) -> list[str]:
    issues = []
    if not (3 <= len(ss.skills) <= 5):
        issues.append(f"expected 3-5 skills, got {len(ss.skills)}")
    names = [s.name for s in ss.skills]
    if len(set(names)) != len(names):
        issues.append("duplicate skill names")
    for s in ss.skills:
        if not s.reward_terms:
            issues.append(f"skill {s.name}: no reward terms")
        for t in s.reward_terms:
            if t.type not in REWARD_TYPES:
                issues.append(f"skill {s.name}: invalid reward type {t.type!r}")
    return issues


def generate(env: str, *, token: str, project: str, model: str, region: str,
             temperature: float, out_dir: str | None, style: str = "standard",
             suffix: str = "") -> dict:
    data = call_gemini(build_prompt(env, style), token, project, model, region, temperature)
    ss = parse_skillset(env, data)
    issues = validate(ss)
    print(f"=== {env}{suffix}: {len(ss.skills)} skills, schema-valid={not issues} (style={style}, T={temperature}) ===")
    for s in ss.skills:
        types = ",".join(t.type for t in s.reward_terms)
        print(f"  - {s.name}: activate[{s.activation_rule}] rewards[{types}]")
    if issues:
        print("  ISSUES:", issues)
    hw = HANDWRITTEN.get(env)
    if hw:
        print(f"  hand-written: {hw}  (LLM {len(ss.skills)} vs {len(hw)})")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{env}{suffix}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  saved {path}")
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", choices=list(ENVS), default="CartpoleBalance")
    ap.add_argument("--all", action="store_true", help="generate for all envs")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--project", default=None)
    ap.add_argument("--out", default=None, help="directory to save generated JSON")
    ap.add_argument("--style", default="standard", choices=["standard", "refined"],
                    help="refined adds guidance for permissive masks + dense rewards")
    ap.add_argument("--suffix", default="", help="filename suffix, e.g. _refined")
    args = ap.parse_args()

    token = get_token()
    project = args.project or get_project()
    if not project:
        raise SystemExit("no gcloud project set; pass --project")
    envs = list(ENVS) if args.all else [args.env]
    for env in envs:
        for i in range(args.samples):
            try:
                generate(env, token=token, project=project, model=args.model, region=args.region,
                         temperature=args.temperature, out_dir=args.out,
                         style=args.style, suffix=args.suffix)
            except urllib.error.HTTPError as e:
                print(f"[{env}] HTTP {e.code}: {e.read().decode()[:300]}")


if __name__ == "__main__":
    main()
