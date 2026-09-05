"""Independent ground-truth audit of the semantic features the policies read.

Why this exists
---------------
Every hand-written skill reward, symbolic rule and NeSy mask in this repo reads named
quantities out of the adapter's ``info`` dict — ``x_velocity``, ``height``, ``torso_pitch``,
``tcp_pos`` and friends (see ``nexus_continuous/policies/*.py``). Those names are produced by
``_PlaygroundVecWrapper._semantic_state_info`` by *indexing into* ``qpos`` / ``qvel`` / ``xpos``.

An index that points at the wrong degree of freedom is silent: training runs, returns rise,
metrics look plausible, and the conclusion is wrong. That is not hypothetical — WalkerWalk's
root joints are ordered ``rootz(0), rootx(1), rooty(2)``, so ``x_velocity = qvel[0]`` was the
**vertical** velocity. It fed both the walk/efficient skill rewards and
``walker/net_walk_success_rate``, and it invalidated a written-up "no net locomotion" verdict.

How the check works
-------------------
Comparing an adapter value against a geometric reconstruction with an absolute tolerance does
**not** work here, and the first version of this tool got it wrong: in MJX the Cartesian arrays
(``xpos``, ``xmat``, ``site_xpos``) are produced by the forward pass and lag ``qpos``/``qvel`` by
one substep, so every geometry-derived reference carries a systematic offset proportional to
velocity. That offset is not a bug, and gating on it produces false alarms on correct code.

So the gate is **discriminative, not absolute**. For each semantic key we build an independent
reference (finite-differenced world positions, rotation-matrix angles, or the environment's own
sensors) and then ask which *candidate* degree of freedom best explains it. The adapter passes
when the value it reports is the best-matching candidate. Integration lag, substep averaging and
unit scale shift every candidate equally, so they cancel out of the comparison — what survives is
exactly the question worth asking: **is this key wired to the right DOF, with the right sign?**

A failure names the index that would have matched, which is the one-line diagnosis of the
walker bug.

Usage
-----
    JAX_PLATFORMS=cpu python tools/audit_semantics.py --all
    JAX_PLATFORMS=cpu python tools/audit_semantics.py --env WalkerWalk --steps 120 --envs 16
    python tools/audit_semantics.py --all --out runs/audit/semantics.json

Exits non-zero if any check fails, so it works as a gate in front of a training campaign.
Runs on CPU; no GPU and no renderer required.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

# --------------------------------------------------------------------------- #
# geometry helpers — plain numpy, applied after the rollout
# --------------------------------------------------------------------------- #


def _as_mat(xmat: np.ndarray) -> np.ndarray:
    """Accept (..., 9) or (..., 3, 3) rotation matrices and return (..., 3, 3)."""
    if xmat.shape[-1] == 9 and xmat.shape[-2] != 3:
        return xmat.reshape(*xmat.shape[:-1], 3, 3)
    return xmat


def _planar_pitch(xmat: np.ndarray) -> np.ndarray:
    """Rotation angle about +y recovered from the body frame.

    A rotation about +y by theta maps the body x-axis to (cos t, 0, -sin t) in world
    coordinates, and the body x-axis is the first *column* of the world<-body matrix.
    """
    rot = _as_mat(xmat)
    return np.arctan2(-rot[..., 2, 0], rot[..., 0, 0])


def _wrap(angle: np.ndarray) -> np.ndarray:
    """Wrap to (-pi, pi] so 3.14 vs -3.14 does not read as a 6.28 error."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _central_diff(series: np.ndarray, dt: float) -> np.ndarray:
    """d/dt of a [T, ...] series over the interior window [1, T-1).

    Central differences, because MJX's qvel is the velocity after the integrator ran and a
    forward difference is biased against it by exactly one half-step.
    """
    return (series[2:] - series[:-2]) / (2.0 * dt)


def _err(a: np.ndarray, b: np.ndarray, kind: str, mask: np.ndarray) -> float:
    """p99 absolute disagreement between two series over the usable steps."""
    d = _wrap(a - b) if kind == "angle" else (a - b)
    d = np.abs(d)
    valid = mask & np.isfinite(d)
    if not valid.any():
        return float("nan")
    return float(np.percentile(d[valid], 99))


# --------------------------------------------------------------------------- #
# check definitions
# --------------------------------------------------------------------------- #


@dataclass
class Check:
    """One semantic key, an independent reference for it, and the candidates it competes against.

    ``candidates`` is the set of degrees of freedom the adapter *could* plausibly have indexed.
    The key passes when the value it actually reports explains the reference at least as well as
    the best candidate does. ``None`` means there is no meaningful alternative to confuse it
    with (a named site position, say), so an absolute tolerance is used instead.
    """

    key: str
    truth: Callable[[dict[str, np.ndarray], float], np.ndarray]
    kind: str = "linear"  # "linear" | "angle"
    candidates: str | None = None  # "qvel" | "qpos" | None
    # Explicit rival hypotheses, when the interesting alternatives are not "some other index
    # of qpos/qvel" but competing *interpretations* — world frame vs body frame, say. Takes
    # precedence over `candidates`.
    candidate_fn: Callable[[dict[str, np.ndarray], float], dict[str, np.ndarray]] | None = None
    abs_tol: float = 1e-4  # only consulted when there are no candidates at all
    note: str = ""


@dataclass
class EnvSpec:
    probe: Callable[[Any, Any], dict[str, jnp.ndarray]]
    checks: list[Check] = field(default_factory=list)


def _body_id(mj_model: Any, name: str) -> int:
    return int(mj_model.body(name).id)


def _site_id(mj_model: Any, name: str) -> int:
    return int(mj_model.site(name).id)


def _try_body(mj_model: Any, names: Sequence[str]) -> int | None:
    for n in names:
        try:
            return _body_id(mj_model, n)
        except (KeyError, ValueError):
            continue
    return None


def build_specs(env_name: str, mj_model: Any, raw_env: Any) -> EnvSpec:
    """Return the probe + checks for one environment, resolving indices by NAME."""

    lname = env_name.lower()

    # ------------------------------------------------------------------ cartpole
    if "cartpole" in lname:
        cart = _body_id(mj_model, "cart")
        pole = _body_id(mj_model, "pole_1")

        def probe(model, data):
            return {
                "cart_x": data.xpos[cart, 0],
                "pole_mat": data.xmat[pole],
                "qpos": data.qpos,
                "qvel": data.qvel,
            }

        return EnvSpec(
            probe=probe,
            checks=[
                Check(
                    "cart_position",
                    lambda p, dt: p["cart_x"][1:-1],
                    candidates="qpos",
                    note="cart body world x",
                ),
                Check(
                    "pole_angle",
                    lambda p, dt: _planar_pitch(p["pole_mat"])[1:-1],
                    kind="angle",
                    candidates="qpos",
                    note="recovered from the pole's rotation matrix",
                ),
                Check(
                    "cart_velocity",
                    lambda p, dt: _central_diff(p["cart_x"], dt),
                    candidates="qvel",
                    note="finite-difference of the cart's world x",
                ),
                Check(
                    "pole_angular_velocity",
                    lambda p, dt: _central_diff(np.unwrap(_planar_pitch(p["pole_mat"]), axis=0), dt),
                    candidates="qvel",
                    note="finite-difference of the recovered pole angle",
                ),
            ],
        )

    # ----------------------------------------------------- cheetah / walker / hopper
    if "cheetah" in lname or "walker" in lname or "hopper" in lname:
        torso = _body_id(mj_model, "torso")
        foot = _try_body(mj_model, ["foot"]) if "hopper" in lname else None

        def probe(model, data):
            out = {
                "torso_x": data.xpos[torso, 0],
                "torso_z": data.xpos[torso, 2],
                "torso_mat": data.xmat[torso],
                "qpos": data.qpos,
                "qvel": data.qvel,
            }
            if foot is not None:
                out["torso_ic_z"] = data.xipos[torso, 2]
                out["foot_ic_z"] = data.xipos[foot, 2]
            return out

        checks: list[Check] = [
            Check(
                "x_velocity",
                lambda p, dt: _central_diff(p["torso_x"], dt),
                candidates="qvel",
                note="finite-difference of the torso's world x — THE walker-bug check",
            ),
            Check(
                "torso_pitch",
                lambda p, dt: _planar_pitch(p["torso_mat"])[1:-1],
                kind="angle",
                candidates="qpos",
                note="recovered from the torso's rotation matrix",
            ),
        ]
        if "hopper" in lname:
            checks.append(
                Check(
                    "height",
                    lambda p, dt: (p["torso_ic_z"] - p["foot_ic_z"])[1:-1],
                    candidates=None,
                    abs_tol=5e-3,
                    note="torso-minus-foot inertial z, both bodies resolved by name",
                )
            )
        elif "walker" in lname:
            checks.append(
                Check(
                    "height",
                    lambda p, dt: p["torso_z"][1:-1],
                    candidates=None,
                    abs_tol=5e-3,
                    note="torso body world z",
                )
            )
        return EnvSpec(probe=probe, checks=checks)

    # -------------------------------------------------------------------- panda
    if "panda" in lname:
        gripper = _site_id(mj_model, "gripper")
        obj = _try_body(mj_model, ["box", "object", "obj", "cube"])

        def probe(model, data):
            out = {"tcp": data.site_xpos[gripper], "qpos": data.qpos, "qvel": data.qvel}
            if obj is not None:
                out["cube"] = data.xpos[obj]
            return out

        checks = [
            Check(
                "tcp_pos",
                lambda p, dt: p["tcp"][1:-1],
                candidates=None,
                abs_tol=1e-5,
                note="gripper site resolved by name",
            )
        ]
        if obj is not None:
            checks.append(
                Check(
                    "cube_pos",
                    lambda p, dt: p["cube"][1:-1],
                    candidates=None,
                    abs_tol=1e-5,
                    note="object body resolved by name",
                )
            )
        return EnvSpec(probe=probe, checks=checks)

    # ---------------------------------------------------------------------- go1
    if "go1" in lname:
        trunk = _try_body(mj_model, ["trunk", "base", "body"])
        # The go1 env exposes the very sensors its own reward reads. Comparing against them
        # is stronger than any geometry reconstructed here, and they are lag-free.
        get_local_linvel = getattr(raw_env, "get_local_linvel", None)
        get_gyro = getattr(raw_env, "get_gyro", None)

        imu_site = None
        try:
            imu_site = _site_id(mj_model, "imu")
        except (KeyError, ValueError):
            pass

        def probe(model, data):
            out: dict[str, jnp.ndarray] = {"qpos": data.qpos, "qvel": data.qvel}
            if trunk is not None:
                out["trunk_z"] = data.xpos[trunk, 2]
            if imu_site is not None:
                out["imu_mat"] = data.site_xmat[imu_site]
            if get_local_linvel is not None:
                out["local_linvel"] = get_local_linvel(data)
            if get_gyro is not None:
                out["gyro"] = get_gyro(data)
            return out

        def _body_frame_vel(p: dict[str, np.ndarray]) -> np.ndarray:
            """v_body = R^T v_world, the quantity the adapter claims to produce."""
            rot = _as_mat(p["imu_mat"])[1:-1]
            v_world = p["qvel"][1:-1, ..., 0:3]
            return np.einsum("...ij,...i->...j", rot, v_world)

        def _linvel_candidates(p, dt):
            # The go1 tracking reward compares a BODY-frame command against a body-frame
            # velocity. Reading the raw free-joint qvel gives the WORLD frame instead — that
            # was a real bug, and this is the check that would catch it coming back.
            out = {"world qvel[0:3]": p["qvel"][1:-1, ..., 0:3]}
            if "imu_mat" in p:
                out["body-frame R^T v"] = _body_frame_vel(p)
            return out

        # Probes are recorded as [T, E, ...], so a sensor's component axis is the LAST one.
        checks = []
        if trunk is not None:
            checks.append(
                Check(
                    "base_height",
                    lambda p, dt: p["trunk_z"][1:-1],
                    candidates="qpos",
                    note="trunk body world z",
                )
            )
        if get_local_linvel is not None:
            # NOTE on the reference: local_linvel is a velocimeter mounted at the `imu` SITE,
            # which sits ~7 cm off the trunk origin. It therefore reads v + omega x r, so it
            # never matches the origin's velocity exactly and an absolute tolerance against it
            # is wrong by construction. What it *can* do is arbitrate between the world-frame
            # and body-frame readings, which differ by far more than the lever arm.
            checks += [
                Check(
                    "x_velocity",
                    lambda p, dt: p["local_linvel"][1:-1, ..., 0],
                    candidate_fn=lambda p, dt: {
                        k: v[..., 0] for k, v in _linvel_candidates(p, dt).items()
                    },
                    note="env's own get_local_linvel — arbitrates world frame vs body frame",
                ),
                Check(
                    "y_velocity",
                    lambda p, dt: p["local_linvel"][1:-1, ..., 1],
                    candidate_fn=lambda p, dt: {
                        k: v[..., 1] for k, v in _linvel_candidates(p, dt).items()
                    },
                    note="env's own get_local_linvel",
                ),
            ]
        if get_gyro is not None:
            checks.append(
                Check(
                    "yaw_rate",
                    lambda p, dt: p["gyro"][1:-1, ..., 2],
                    candidates="qvel",
                    note="env's own get_gyro (body-frame angular velocity)",
                )
            )
        return EnvSpec(probe=probe, checks=checks)

    return EnvSpec(probe=lambda model, data: {}, checks=[])


# --------------------------------------------------------------------------- #
# rollout
# --------------------------------------------------------------------------- #


def rollout(
    env_name: str, num_envs: int, steps: int, seed: int, action_scale: float = 1.0
) -> dict[str, Any]:
    """Step the raw Playground env and record semantic info alongside geometry probes."""

    from mujoco_playground import registry  # noqa: PLC0415

    from nexus_continuous.envs.playground_adapter import _PlaygroundVecWrapper  # noqa: PLC0415

    env_config = registry.get_default_config(env_name)
    env_config.impl = "jax"
    raw_env = registry.load(env_name, env_config)

    mj_model = getattr(raw_env, "mj_model", None) or getattr(raw_env, "_mj_model")

    # The wrapper is used ONLY as the unit under test: we call its semantic extractor on raw
    # states. Nothing else about it is exercised here.
    wrapper = _PlaygroundVecWrapper(raw_env, env_config, env_name)
    spec = build_specs(env_name, mj_model, raw_env)

    dt = float(getattr(raw_env, "dt", None) or env_config.ctrl_dt)
    action_dim = int(raw_env.action_size)

    reset = jax.jit(jax.vmap(raw_env.reset))
    step = jax.jit(jax.vmap(raw_env.step))
    probe_v = jax.jit(jax.vmap(lambda d: spec.probe(mj_model, d)))

    key = jax.random.PRNGKey(seed)
    key, k_reset = jax.random.split(key)
    state = reset(jax.random.split(k_reset, num_envs))

    infos, probes, dones = [], [], []

    def record(st):
        infos.append(wrapper._semantic_state_info(st))  # noqa: SLF001 — this IS the unit under test
        probes.append(probe_v(st.data))
        dones.append(st.done)

    record(state)
    for _ in range(steps):
        key, k_act = jax.random.split(key)
        action = action_scale * jax.random.uniform(
            k_act, (num_envs, action_dim), minval=-1.0, maxval=1.0
        )
        state = step(state, action)
        record(state)

    def stack(seq: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
        keys = set().union(*(s.keys() for s in seq)) if seq else set()
        return {
            k: np.asarray(jnp.stack([s[k] for s in seq]))
            for k in keys
            if all(k in s for s in seq)
        }

    return {
        "info": stack(infos),
        "probe": stack(probes),
        "done": np.asarray(jnp.stack(dones)),
        "dt": dt,
        "spec": spec,
    }


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #

# How much better a rival candidate must fit before we call the adapter mis-wired. A rival has
# to explain the reference at least this much more tightly; anything inside the band is noise.
_MARGIN = 0.70


def audit_env(
    env_name: str, num_envs: int, steps: int, seed: int, action_scale: float = 1.0
) -> dict:
    data = rollout(env_name, num_envs, steps, seed, action_scale)
    spec: EnvSpec = data["spec"]
    info, probe, dt, done = data["info"], data["probe"], data["dt"], data["done"]

    # Physics after an episode ends is not what a policy ever sees; drop those steps.
    ever_done = np.cumsum(done > 0.5, axis=0) > 0
    usable = ~ever_done[1:-1]  # aligned with the central-difference window

    results = []
    for check in spec.checks:
        if check.key not in info:
            results.append(
                {
                    "key": check.key,
                    "status": "MISSING",
                    "detail": "the adapter never emitted this key for this env",
                    "note": check.note,
                }
            )
            continue

        got = np.asarray(info[check.key])[1:-1]
        want = np.asarray(check.truth(probe, dt))
        if got.shape != want.shape:
            results.append(
                {
                    "key": check.key,
                    "status": "SHAPE",
                    "detail": f"adapter {got.shape} vs reference {want.shape}",
                    "note": check.note,
                }
            )
            continue

        mask = usable
        while mask.ndim < got.ndim:
            mask = mask[..., None]
        mask = np.broadcast_to(mask, got.shape)
        if not mask.any():
            results.append(
                {"key": check.key, "status": "NODATA", "detail": "every step masked out",
                 "note": check.note}
            )
            continue

        e_adapter = _err(got, want, check.kind, mask)
        entry: dict[str, Any] = {
            "key": check.key,
            "err": round(e_adapter, 6),
            "n": int(mask.sum()),
            "note": check.note,
        }

        # Build the rival hypotheses this key competes against.
        errs: dict[str, float] = {}
        if check.candidate_fn is not None:
            for label, cand in check.candidate_fn(probe, dt).items():
                cand = np.asarray(cand)
                if cand.shape != want.shape:
                    continue
                errs[label] = _err(cand, want, check.kind, mask)
                errs[f"-{label}"] = _err(-cand, want, check.kind, mask)
        elif check.candidates is not None:
            source = np.asarray(probe[check.candidates])[1:-1]
            for i in range(source.shape[-1]):
                cand = source[..., i]
                if cand.shape != want.shape:
                    continue
                # a sign-flipped candidate is its own failure mode worth naming
                errs[f"{check.candidates}[{i}]"] = _err(cand, want, check.kind, mask)
                errs[f"-{check.candidates}[{i}]"] = _err(-cand, want, check.kind, mask)
        else:
            # No rival to confuse this key with — fall back to an absolute tolerance, kept
            # loose enough to absorb the one-substep Cartesian lag.
            entry["tolerance"] = check.abs_tol
            entry["status"] = "PASS" if e_adapter <= check.abs_tol else "FAIL"
            results.append(entry)
            continue

        finite = {k: e for k, e in errs.items() if np.isfinite(e)}
        if not finite:
            entry["status"] = "NODATA"
            entry["detail"] = "no finite candidate errors"
            results.append(entry)
            continue

        label = min(finite, key=finite.get)
        best_e = finite[label]
        entry["best_candidate"] = label
        entry["best_candidate_err"] = round(best_e, 6)

        # The adapter passes when nothing explains the reference decisively better.
        if e_adapter <= best_e / _MARGIN or np.isclose(e_adapter, best_e, rtol=1e-3, atol=1e-9):
            entry["status"] = "PASS"
        else:
            entry["status"] = "FAIL"
            entry["hint"] = (
                f"{label} explains the reference to {best_e:.4g} (p99) while the adapter's "
                f"value is off by {e_adapter:.4g} — the key looks wired to the wrong DOF"
            )
        results.append(entry)

    return {
        "env": env_name,
        "dt": dt,
        "steps": steps,
        "num_envs": num_envs,
        "action_scale": action_scale,
        "checks": results,
        "n_pass": sum(1 for r in results if r["status"] == "PASS"),
        "n_fail": sum(1 for r in results if r["status"] != "PASS"),
    }


DEFAULT_ENVS = [
    "CartpoleBalance",
    "CheetahRun",
    "WalkerWalk",
    "HopperHop",
    "PandaPickCube",
    "Go1JoystickFlatTerrain",
]

_GLYPH = {"PASS": "PASS", "FAIL": "FAIL", "MISSING": "MISS", "SHAPE": "SHPE", "NODATA": "----"}


def print_report(reports: list[dict]) -> None:
    width = 98
    print("=" * width)
    print("SEMANTIC FEATURE AUDIT — adapter info keys vs independent references")
    print("=" * width)
    for rep in reports:
        print(f"\n{rep['env']}   dt={rep['dt']:.4f}  steps={rep['steps']}  envs={rep['num_envs']}")
        print("-" * width)
        if not rep["checks"]:
            print("  (no checks defined for this environment)")
            continue
        for c in rep["checks"]:
            line = f"  [{_GLYPH.get(c['status'], c['status'])}] {c['key']:<24}"
            if "err" in c:
                line += f" err {c['err']:<11.5g}"
            if "best_candidate" in c:
                line += f" best {c['best_candidate']} @ {c['best_candidate_err']:.5g}"
            elif "tolerance" in c:
                line += f" tol {c['tolerance']:.5g}"
            if c["status"] not in ("PASS",) and "detail" in c:
                line += f" {c['detail']}"
            if c["status"] == "FAIL":
                line += "  <-- MISMATCH"
            print(line)
            if c.get("note"):
                print(f"         reference: {c['note']}")
            if c.get("hint"):
                print(f"         HINT:      {c['hint']}")
    total_fail = sum(r["n_fail"] for r in reports)
    total_pass = sum(r["n_pass"] for r in reports)
    print("\n" + "=" * width)
    print(f"TOTAL  {total_pass} passed, {total_fail} failed/unchecked")
    print("=" * width)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--env", action="append", default=[], help="env name; repeatable")
    ap.add_argument("--all", action="store_true", help="audit the full 6-env suite")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--action-scale",
        type=float,
        default=1.0,
        help=(
            "magnitude of the random actions. Lag and lever-arm artifacts scale with "
            "velocity while wiring bugs do not, so re-running at e.g. 0.05 separates them."
        ),
    )
    ap.add_argument("--out", type=str, default=None, help="write the JSON report here")
    args = ap.parse_args(argv)

    envs = DEFAULT_ENVS if args.all or not args.env else args.env

    reports = []
    for name in envs:
        try:
            reports.append(audit_env(name, args.envs, args.steps, args.seed, args.action_scale))
        except Exception as exc:  # a broken env must not hide the others' results
            reports.append(
                {
                    "env": name,
                    "dt": float("nan"),
                    "steps": args.steps,
                    "num_envs": args.envs,
                    "action_scale": args.action_scale,
                    "checks": [
                        {"key": "<rollout>", "status": "FAIL", "detail": repr(exc), "note": ""}
                    ],
                    "n_pass": 0,
                    "n_fail": 1,
                }
            )

    print_report(reports)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")

    return 1 if any(r["n_fail"] for r in reports) else 0


if __name__ == "__main__":
    sys.exit(main())
