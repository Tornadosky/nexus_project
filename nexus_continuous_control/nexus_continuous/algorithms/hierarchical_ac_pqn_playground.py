"""Hierarchical Actor-Critic PQN for MuJoCo Playground.

This module is the continuous-control NEXUS implementation requested in the
project deliverables. It keeps the NEXUS hierarchy:

  symbolic/object state -> meta-policy over interpretable skills -> neural skill actor

and replaces discrete skill Q-functions with deterministic actors plus critics,
following the DDPG-style Actor-Critic PQN idea used by purejaxql for continuous
control. Each skill critic is trained on its own hand-written/LLM-generated skill
reward while the learned meta Q-function is trained on the environment reward.

Supported meta-policy variants:
  * neural: learned meta-Q selects the skill.
  * symbolic: fixed hand-written meta-policy selects the skill.
  * nesy: hand-written mask filters valid skills, learned meta-Q chooses among them.
"""

from __future__ import annotations

import time
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from nexus_continuous.envs.playground_adapter import (
    build_playground_env,
    get_actor_obs,
    get_actor_pixels,
    get_critic_obs,
    get_policy_obs,
)
from nexus_continuous.networks import MetaQ, SkillActor, SkillCritic
from nexus_continuous.vision import VisionSkillActor
from nexus_continuous.policies.registry import load_policy_module
from nexus_continuous.returns import q_lambda_returns, smooth_l1_loss
from nexus_continuous.train_state import CounterTrainState, NexusTrainState
from nexus_continuous.utils import flatten_time_env, global_norm, make_minibatches


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    original_action: jnp.ndarray
    skill: jnp.ndarray
    meta_selected_value: jnp.ndarray
    meta_bootstrap_value: jnp.ndarray
    skill_bootstrap_values: jnp.ndarray
    env_reward: jnp.ndarray
    skill_rewards: jnp.ndarray
    obs: Any
    next_obs: Any
    info: Any
    diagnostics: Any
    mask: jnp.ndarray


class NexusTrainOutput(NamedTuple):
    runner_state: Any
    metrics: Any
    eval_metrics: Any
    eval_episode_table: Any
    normalization_stats: Any


def _schedule(start: float, end: float, decay_fraction: float, num_updates: int):
    return optax.linear_schedule(
        init_value=start,
        end_value=end,
        transition_steps=max(1, int(decay_fraction * num_updates)),
    )


def _epsilon_greedy_skill(
    rng: jax.Array,
    greedy_skill: jnp.ndarray,
    epsilon: jnp.ndarray,
    num_skills: int,
    mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    batch = greedy_skill.shape[0]
    rng_sample, rng_uniform = jax.random.split(rng)
    if mask is None:
        random_skill = jax.random.randint(rng_sample, (batch,), 0, num_skills)
    else:
        valid_mask = jnp.where(jnp.any(mask, axis=-1, keepdims=True), mask, jnp.ones_like(mask))
        random_skill = jax.random.categorical(
            rng_sample, jnp.where(valid_mask, 0.0, -1.0e9), axis=-1
        ).astype(jnp.int32)
    explore = jax.random.uniform(rng_uniform, (batch,)) < epsilon
    return jnp.where(explore, random_skill, greedy_skill).astype(jnp.int32)


def _select_rows(x: jnp.ndarray, indices: jnp.ndarray) -> jnp.ndarray:
    return x[jnp.arange(indices.shape[0]), indices]


def _drop_actor_pixels(obs: Any) -> Any:
    """Strip the heavy pixel tensor from an observation dict.

    ``Transition.next_obs`` is never read by target computation, the update loss,
    or metrics — skill/meta bootstrap values are precomputed online at rollout
    time. In RGB mode keeping its ``actor_pixels`` would store a SECOND full
    [T, E, H, W, C] image buffer per rollout for no functional benefit (a likely
    OOM). We therefore drop pixels from the stored ``next_obs`` while keeping the
    rest of the dict intact. In state mode there is no ``actor_pixels`` key, so
    this is a no-op and the state path is byte-identical.
    """

    if isinstance(obs, dict) and "actor_pixels" in obs:
        return {k: v for k, v in obs.items() if k != "actor_pixels"}
    return obs


def masked_meta_bootstrap_value(q_values: jnp.ndarray, mask: jnp.ndarray | None = None) -> jnp.ndarray:
    """Return max_i Q_meta(s, i), applying NeSy masks with -inf-style blocking."""

    if mask is not None:
        q_values = jnp.where(mask, q_values, -1.0e9)
    return jnp.max(q_values, axis=-1)


def mask_violation_metrics(
    skill_one_hot: jnp.ndarray,
    mask_float: jnp.ndarray,
    skill_names: tuple[str, ...],
) -> dict[str, jnp.ndarray]:
    selected_available = jnp.sum(skill_one_hot * mask_float, axis=-1)
    metrics = {"mask/violation_rate": 1.0 - jnp.mean(selected_available)}
    for idx, name in enumerate(skill_names):
        metrics[f"mask_violation/{idx}_{name}"] = jnp.mean(
            skill_one_hot[..., idx] * (1.0 - mask_float[..., idx])
        )
    return metrics


def skill_actor_bootstrap_values(
    critic_params: Any,
    actor_params: Any,
    obs_actor: jnp.ndarray,
    obs_critic: jnp.ndarray,
    actor_apply: Callable[[Any, jnp.ndarray], jnp.ndarray],
    critic_apply: Callable[[Any, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    reduce_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> jnp.ndarray:
    """Return Q_i(s, actor_i(s)) aggregated over critic ensembles as [batch, num_skills].

    ``reduce_fn`` reduces the leading critic-ensemble axis. It defaults to the mean
    (variance reduction). Passing a min-reducer gives TD3-style clipped-double-Q
    pessimism, which curbs value overestimation (config ``CRITIC_AGG: min``).
    """

    if reduce_fn is None:
        reduce_fn = lambda v: jnp.mean(v, axis=0)
    all_actions = actor_apply(actor_params, obs_actor)  # [num_skills, batch, action_dim]

    def one_skill(skill_idx, action_i):
        skill_critic_params = jax.tree_util.tree_map(lambda leaf: leaf[skill_idx], critic_params)
        vals = jax.vmap(lambda p: critic_apply(p, obs_critic, action_i))(skill_critic_params)
        return reduce_fn(vals)

    q_by_skill = jax.vmap(one_skill)(jnp.arange(all_actions.shape[0]), all_actions)
    return jnp.swapaxes(q_by_skill, 0, 1)


def _as_num_steps(config: dict[str, Any]) -> None:
    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = int(
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    if config["MINIBATCH_SIZE"] <= 0:
        raise ValueError("MINIBATCH_SIZE must be positive; reduce NUM_MINIBATCHES")


def make_train(config: dict[str, Any]) -> Callable[[jax.Array], NexusTrainOutput]:
    """Build a compiled train function for one seed.

    The returned function accepts one PRNGKey and returns the final runner state
    and optional metrics. It is intended to be wrapped by `jax.jit` and `jax.vmap`
    over seeds, mirroring the upstream purejaxql style.
    """

    config = dict(config)
    _as_num_steps(config)

    policy_module = load_policy_module(config.get("POLICY", config["ENV_NAME"]))
    task_policy_module = load_policy_module(config.get("TASK_POLICY", config["ENV_NAME"]))
    num_skills = int(getattr(policy_module, "NUM_SKILLS"))
    skill_names = tuple(getattr(policy_module, "SKILL_NAMES"))
    meta_policy_type = config.get("META_POLICY_TYPE", "nesy").lower()
    if meta_policy_type not in ("neural", "symbolic", "nesy", "flat"):
        raise ValueError("META_POLICY_TYPE must be one of: neural, symbolic, nesy, flat")
    if meta_policy_type == "flat" and num_skills != 1:
        raise ValueError("META_POLICY_TYPE=flat requires a policy with exactly one skill")

    env_bundle = build_playground_env(config)
    env = env_bundle.env
    env_params = env_bundle.env_params
    eval_enabled = bool(config.get("EVAL_AFTER_TRAIN", False))
    eval_num_envs = int(config.get("EVAL_NUM_ENVS", 64))
    eval_num_episodes = int(config.get("EVAL_NUM_EPISODES", 128))
    eval_max_steps = int(config.get("EVAL_MAX_STEPS") or env_bundle.episode_length)
    eval_num_batches = max(1, int(np.ceil(eval_num_episodes / eval_num_envs)))
    if eval_enabled:
        eval_config = dict(config)
        eval_config["NORMALIZE_OBS"] = False
        eval_config["NORMALIZE_REWARD"] = False
        # RGB: the eval render context batch must match the eval env count, not NUM_ENVS.
        eval_config["RENDER_NWORLD"] = eval_num_envs
        eval_bundle = build_playground_env(eval_config)
        eval_env = eval_bundle.env
        eval_env_params = eval_bundle.env_params
    else:
        eval_env = None
        eval_env_params = None
    action_low = jnp.asarray(env_bundle.action_low)
    action_high = jnp.asarray(env_bundle.action_high)
    action_dim = env_bundle.action_dim
    action_scale = (action_high - action_low) / 2.0
    action_bias = (action_high + action_low) / 2.0

    lr_schedule = optax.linear_schedule(
        init_value=float(config.get("LR_START", config.get("LR", 3e-4))),
        end_value=float(config.get("LR_END", config.get("LR", 3e-4))),
        transition_steps=max(
            1,
            int(
                config["NUM_UPDATES"]
                * config.get("LR_DECAY", 1.0)
                * config["NUM_MINIBATCHES"]
                * config["NUM_EPOCHS"]
            ),
        ),
    )
    lr = lr_schedule if config.get("ANNEAL_LR", False) else float(config.get("LR", 3e-4))
    noise_schedule = _schedule(
        float(config.get("NOISE_START", 0.30)),
        float(config.get("NOISE_FINISH", 0.02)),
        float(config.get("NOISE_DECAY", 0.8)),
        config["NUM_UPDATES"],
    )
    epsilon_schedule = _schedule(
        float(config.get("META_EPS_START", 1.0)),
        float(config.get("META_EPS_FINISH", 0.02)),
        float(config.get("META_EPS_DECAY", 0.6)),
        config["NUM_UPDATES"],
    )

    # RGB extension: when USE_RGB is set, the skill actors take pixels (+ a
    # proprioception vector) instead of the state vector. Critics, meta-Q, and
    # the symbolic layer stay state-based (privileged-critic design).
    use_rgb = bool(config.get("USE_RGB", False))
    # What the PIXEL actor sees besides the image. Critics/meta always keep the
    # full privileged state; this only restricts the actor's side input.
    #   "none"    -> pixels only (default; the honest "skills from pixels" claim,
    #                matching DrQ / DM-control-from-pixels where the actor gets no
    #                state. The vision env's frame stack already encodes velocity).
    #   "indices" -> only RGB_PROPRIO_INDICES of the state (e.g. robot self-sensing,
    #                NOT privileged world state the camera should infer).
    #   "full"    -> the whole state (discouraged: makes pixels largely redundant).
    rgb_proprio_mode = str(config.get("RGB_PROPRIO", "none")).lower()
    rgb_proprio_indices = config.get("RGB_PROPRIO_INDICES")
    # DrQ-style random-shift image augmentation during the actor update — the
    # single biggest sample-efficiency lever for pixel RL on DM-control.
    rgb_augment = use_rgb and bool(config.get("RGB_AUGMENT", True))
    rgb_aug_pad = int(config.get("RGB_AUG_PAD", 4))

    def _actor_proprio(obs_actor):
        """Restrict the privileged state to what the pixel actor is allowed to see."""
        if rgb_proprio_mode == "full":
            return obs_actor
        if rgb_proprio_mode == "indices" and rgb_proprio_indices is not None:
            idx = jnp.asarray(list(rgb_proprio_indices), dtype=jnp.int32)
            return obs_actor[..., idx]
        return obs_actor[..., :0]  # pixels-only

    def _augment_pixels(pixels, rng):
        """DrQ random shift: replicate-pad by rgb_aug_pad then random-crop back."""
        b, h, w, c = pixels.shape
        padded = jnp.pad(
            pixels, ((0, 0), (rgb_aug_pad, rgb_aug_pad), (rgb_aug_pad, rgb_aug_pad), (0, 0)), mode="edge"
        )
        offsets = jax.random.randint(rng, (b, 2), 0, 2 * rgb_aug_pad + 1)
        crop = lambda img, off: jax.lax.dynamic_slice(img, (off[0], off[1], 0), (h, w, c))
        return jax.vmap(crop)(padded, offsets)
    if use_rgb:
        actor = VisionSkillActor(
            action_dim=action_dim,
            action_scale=action_scale,
            action_bias=action_bias,
            hidden_sizes=tuple(config.get("ACTOR_HIDDEN_SIZES", (256, 256))),
            embedding_dim=int(config.get("RGB_EMBED_DIM", 128)),
        )
    else:
        actor = SkillActor(
            action_dim=action_dim,
            action_scale=action_scale,
            action_bias=action_bias,
            hidden_sizes=tuple(config.get("ACTOR_HIDDEN_SIZES", (256, 256))),
            activation=config.get("ACTIVATION", "relu"),
            norm_type=config.get("NORM_TYPE", "layer_norm"),
            init_scale=float(config.get("ACTOR_INIT_SCALE", 0.01)),
        )
    critic = SkillCritic(
        hidden_sizes=tuple(config.get("CRITIC_HIDDEN_SIZES", (256, 256))),
        activation=config.get("ACTIVATION", "relu"),
        norm_type=config.get("NORM_TYPE", "layer_norm"),
        init_scale=float(config.get("CRITIC_INIT_SCALE", 1.0)),
    )
    meta_q = MetaQ(
        num_skills=num_skills,
        hidden_sizes=tuple(config.get("META_HIDDEN_SIZES", (256, 256))),
        activation=config.get("ACTIVATION", "relu"),
        norm_type=config.get("NORM_TYPE", "layer_norm"),
        init_scale=float(config.get("META_INIT_SCALE", 1.0)),
    )

    tx = optax.chain(
        optax.clip_by_global_norm(float(config.get("MAX_GRAD_NORM", 1.0))),
        optax.radam(learning_rate=lr),
    )

    # Critic-ensemble aggregation. "mean" (default) reduces variance; "min" gives
    # TD3-style clipped-double-Q pessimism to curb value overestimation.
    critic_agg = str(config.get("CRITIC_AGG", "mean")).lower()
    if critic_agg not in ("mean", "min"):
        raise ValueError("CRITIC_AGG must be 'mean' or 'min'")

    def _reduce_critics(vals):  # vals: [num_critics, ...] -> [...]
        return jnp.min(vals, axis=0) if critic_agg == "min" else jnp.mean(vals, axis=0)

    def _actor_apply(actor_params, obs_actor, obs_pixels=None):
        if use_rgb:
            if obs_pixels is None:
                raise ValueError(
                    "USE_RGB is set but the vision actor was called without pixels. "
                    "Every RGB-mode actor call must pass obs_pixels=get_actor_pixels(obs)."
                )
            proprio = _actor_proprio(obs_actor)
            return jax.vmap(lambda p: actor.apply({"params": p}, obs_pixels, proprio))(actor_params)
        return jax.vmap(lambda p: actor.apply({"params": p}, obs_actor))(actor_params)

    def _critic_values_all_skills(critic_params, obs_critic, action):
        """Return mean critic values [batch, num_skills] for an executed action."""

        def one_skill(skill_params):
            vals = jax.vmap(lambda p: critic.apply({"params": p}, obs_critic, action))(skill_params)
            return jnp.mean(vals, axis=0)

        return jnp.swapaxes(jax.vmap(one_skill)(critic_params), 0, 1)

    def _critic_values_all_critics(critic_params, obs_critic, action):
        """Return all critic ensemble values [num_skills, num_critics, batch]."""

        def one_skill(skill_params):
            return jax.vmap(lambda p: critic.apply({"params": p}, obs_critic, action))(skill_params)

        return jax.vmap(one_skill)(critic_params)

    def _meta_values(meta_params, obs_actor):
        return meta_q.apply({"params": meta_params}, obs_actor)

    def _skill_mask(policy_obs):
        mask = policy_module.skill_mask(policy_obs)
        return jnp.asarray(mask, dtype=bool)

    def _symbolic_policy(policy_obs):
        return jnp.asarray(policy_module.symbolic_meta_policy(policy_obs), dtype=jnp.int32)

    def _critic_apply_one(params, obs_critic, action):
        return critic.apply({"params": params}, obs_critic, action)

    def _skill_actor_bootstrap_values(
        critic_params, actor_params, obs_actor, obs_critic, obs_pixels=None
    ):
        # Bind pixels into the actor_apply closure so the bootstrap helper's
        # (params, obs) call signature is unchanged in both modes.
        actor_apply = (lambda ap, oa: _actor_apply(ap, oa, obs_pixels)) if use_rgb else _actor_apply
        return skill_actor_bootstrap_values(
            critic_params,
            actor_params,
            obs_actor,
            obs_critic,
            actor_apply=actor_apply,
            critic_apply=_critic_apply_one,
            reduce_fn=_reduce_critics,
        )

    def _meta_bootstrap_value(meta_params, obs_actor, policy_obs):
        q = _meta_values(meta_params, obs_actor)
        if meta_policy_type == "nesy":
            return masked_meta_bootstrap_value(q, _skill_mask(policy_obs))
        return masked_meta_bootstrap_value(q)

    def _policy_diagnostics(prev_policy_obs, policy_obs, action, env_reward, done, info):
        diagnostics = {}
        if hasattr(policy_module, "diagnostics"):
            diagnostics.update(
                policy_module.diagnostics(prev_policy_obs, policy_obs, action, env_reward, done, info)
            )
        if hasattr(task_policy_module, "task_metrics"):
            diagnostics.update(
                task_policy_module.task_metrics(
                    prev_policy_obs,
                    policy_obs,
                    action,
                    env_reward,
                    done,
                    info,
                )
            )
        return diagnostics

    hold_interval = int(config.get("META_DECISION_INTERVAL", 1))

    def _select_action(
        train_state: NexusTrainState,
        obs: Any,
        rng: jax.Array,
        update_idx: int,
        explore: bool,
        hold_state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
    ):
        obs_actor = get_actor_obs(obs)
        obs_pixels = get_actor_pixels(obs)
        policy_obs = get_policy_obs(obs)
        all_actions = _actor_apply(train_state.actor.params, obs_actor, obs_pixels)  # [N, E, A]
        all_actions_env_major = jnp.swapaxes(all_actions, 0, 1)  # [E, N, A]

        if meta_policy_type == "flat":
            selected_skill = jnp.zeros((obs_actor.shape[0],), dtype=jnp.int32)
            mask = jnp.ones((obs_actor.shape[0], num_skills), dtype=bool)
            meta_values = jnp.zeros((obs_actor.shape[0], num_skills), dtype=obs_actor.dtype)
        elif meta_policy_type == "symbolic":
            selected_skill = _symbolic_policy(policy_obs)
            mask = _skill_mask(policy_obs)
            meta_values = jnp.zeros((obs_actor.shape[0], num_skills), dtype=obs_actor.dtype)
        else:
            meta_values = _meta_values(train_state.meta.params, obs_actor)
            if meta_policy_type == "nesy":
                mask = _skill_mask(policy_obs)
                masked_values = jnp.where(mask, meta_values, -1.0e9)
                greedy_skill = jnp.argmax(masked_values, axis=-1).astype(jnp.int32)
            else:
                mask = jnp.ones_like(meta_values, dtype=bool)
                greedy_skill = jnp.argmax(meta_values, axis=-1).astype(jnp.int32)
            epsilon = epsilon_schedule(update_idx) if explore else jnp.asarray(0.0)
            rng, rng_eps = jax.random.split(rng)
            selected_skill = _epsilon_greedy_skill(rng_eps, greedy_skill, epsilon, num_skills, mask)

        new_hold_state = hold_state
        if hold_interval > 1 and hold_state is not None and meta_policy_type != "flat":
            held_skill, steps_held, prev_done = hold_state
            # Re-decide when the commitment expires, the episode resets, or
            # (nesy) the held skill is no longer allowed by the mask.
            held_valid = jnp.take_along_axis(mask, held_skill[:, None], axis=-1)[:, 0]
            redecide = (steps_held >= hold_interval) | prev_done | ~held_valid
            selected_skill = jnp.where(redecide, selected_skill, held_skill).astype(jnp.int32)
            steps_held = jnp.where(redecide, jnp.ones_like(steps_held), steps_held + 1)
            new_hold_state = (selected_skill, steps_held, prev_done)

        original_action = _select_rows(all_actions_env_major, selected_skill)
        if explore:
            rng, rng_noise = jax.random.split(rng)
            noise_std = noise_schedule(update_idx)
            if config.get("LINSPACE_NOISE", False):
                per_env_noise = jnp.linspace(0.0, noise_std, obs_actor.shape[0])[:, None]
            else:
                per_env_noise = noise_std
            noise = jax.random.normal(rng_noise, original_action.shape) * per_env_noise * action_scale
            action = jnp.clip(original_action + noise, action_low, action_high)
        else:
            action = jnp.clip(original_action, action_low, action_high)
        meta_selected_value = _select_rows(meta_values, selected_skill)
        return original_action, action, selected_skill, meta_selected_value, mask, new_hold_state, rng

    def _normalization_stats_from_state(env_state: Any, obs: Any) -> dict[str, jnp.ndarray]:
        obs_actor = get_actor_obs(obs)
        obs_critic = get_critic_obs(obs)
        if config.get("NORMALIZE_OBS", True):
            return {
                "actor_mean": env_state.actor_mean,
                "actor_var": env_state.actor_var,
                "actor_count": env_state.actor_count,
                "critic_mean": env_state.critic_mean,
                "critic_var": env_state.critic_var,
                "critic_count": env_state.critic_count,
            }
        return {
            "actor_mean": jnp.zeros(obs_actor.shape[1:], dtype=obs_actor.dtype),
            "actor_var": jnp.ones(obs_actor.shape[1:], dtype=obs_actor.dtype),
            "actor_count": jnp.asarray(0.0, dtype=obs_actor.dtype),
            "critic_mean": jnp.zeros(obs_critic.shape[1:], dtype=obs_critic.dtype),
            "critic_var": jnp.ones(obs_critic.shape[1:], dtype=obs_critic.dtype),
            "critic_count": jnp.asarray(0.0, dtype=obs_critic.dtype),
        }

    def _normalized_eval_obs(raw_obs: Any, stats: dict[str, jnp.ndarray]) -> Any:
        if not config.get("NORMALIZE_OBS", True):
            return raw_obs
        raw_actor = raw_obs["raw_actor"] if isinstance(raw_obs, dict) else raw_obs
        raw_critic = raw_obs["raw_critic"] if isinstance(raw_obs, dict) else raw_obs
        obs = {
            "actor": (raw_actor - stats["actor_mean"]) / jnp.sqrt(stats["actor_var"] + 1e-8),
            "critic": (raw_critic - stats["critic_mean"]) / jnp.sqrt(stats["critic_var"] + 1e-8),
            "raw_actor": raw_actor,
            "raw_critic": raw_critic,
        }
        if isinstance(raw_obs, dict) and "policy_info" in raw_obs:
            obs["policy_info"] = raw_obs["policy_info"]
        if isinstance(raw_obs, dict) and "actor_pixels" in raw_obs:
            obs["actor_pixels"] = raw_obs["actor_pixels"]
        return obs

    def _task_metrics(prev_policy_obs, policy_obs, action, env_reward, done, info):
        if hasattr(task_policy_module, "task_metrics"):
            return task_policy_module.task_metrics(
                prev_policy_obs,
                policy_obs,
                action,
                env_reward,
                done,
                info,
            )
        return {}

    def _panda_episode_overrides(
        mean_metrics: dict[str, jnp.ndarray],
        max_metrics: dict[str, jnp.ndarray],
        initial_metrics: dict[str, jnp.ndarray],
    ) -> dict[str, jnp.ndarray]:
        if "panda/cube_height_max_mean" not in mean_metrics:
            return mean_metrics
        out = dict(mean_metrics)
        max_height = max_metrics["panda/cube_height_max_mean"]
        initial_height = initial_metrics["panda/cube_height_max_mean"]
        initial_height = jnp.where(initial_height > 0.01, initial_height, 0.03)
        max_delta = max_height - initial_height
        lift_success = (max_height > initial_height + 0.05) | (max_height > 0.08)
        out["panda/reach_success_rate"] = max_metrics["panda/reach_success_rate"]
        out["panda/closed_near_cube_rate"] = max_metrics["panda/closed_near_cube_rate"]
        out["panda/lift_success_rate"] = lift_success.astype(jnp.float32)
        out["panda/place_success_rate"] = max_metrics["panda/place_success_rate"]
        out["panda/cube_height_max_mean"] = max_height
        out["panda/cube_height_delta_max_mean"] = max_delta
        out["primary_goal_metric"] = max_delta
        out["primary_success_rate"] = lift_success.astype(jnp.float32)
        return out

    def _walker_episode_overrides(
        mean_metrics: dict[str, jnp.ndarray],
    ) -> dict[str, jnp.ndarray]:
        if "walker/forward_velocity_mean" not in mean_metrics:
            return mean_metrics
        out = dict(mean_metrics)
        # Episode-mean forward velocity equals net displacement / time, so this
        # only fires for genuine locomotion. The per-step walk_success counts
        # the forward half of every sway cycle and reads ~0.27 for policies
        # with zero net progress; it is kept as-is for comparability while
        # the primary gate metric switches to the honest episode-level one.
        net_walk = (out["walker/forward_velocity_mean"] > 0.5) & (
            out["walker/stand_success_rate"] > 0.5
        )
        out["walker/net_walk_success_rate"] = net_walk.astype(jnp.float32)
        out["primary_success_rate"] = net_walk.astype(jnp.float32)
        return out

    def _summarize_eval_table(table: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
        summary = {
            "eval_seed": jnp.asarray(int(config.get("EVAL_SEED", 10000)), dtype=jnp.int32),
            "num_eval_episodes": jnp.asarray(eval_num_episodes, dtype=jnp.int32),
            "episode_return_mean": jnp.mean(table["episode_return"]),
            "episode_return_std": jnp.std(table["episode_return"]),
            "episode_length_mean": jnp.mean(table["episode_length"]),
        }
        for key, value in table.items():
            if key in ("eval_episode_index", "episode_return", "episode_length"):
                continue
            summary[key] = jnp.mean(value)
        return summary

    def _run_deterministic_evaluation(
        train_state: NexusTrainState,
        stats: dict[str, jnp.ndarray],
    ) -> tuple[dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
        if not eval_enabled:
            return {}, {}
        assert eval_env is not None

        zero_action = jnp.zeros((eval_num_envs, action_dim), dtype=action_low.dtype)
        zero_reward = jnp.zeros((eval_num_envs,), dtype=action_low.dtype)
        zero_done = jnp.zeros((eval_num_envs,), dtype=bool)

        def _eval_batch(batch_rng):
            reset_rng = jax.random.split(batch_rng, eval_num_envs)
            raw_obs, eval_state = eval_env.reset(reset_rng, eval_env_params)
            obs = _normalized_eval_obs(raw_obs, stats)
            policy_obs = get_policy_obs(obs)
            initial_metrics = _task_metrics(
                policy_obs,
                policy_obs,
                zero_action,
                zero_reward,
                zero_done,
                None,
            )
            zeros_like_metrics = jax.tree_util.tree_map(jnp.zeros_like, initial_metrics)
            neg_inf_metrics = jax.tree_util.tree_map(
                lambda x: jnp.full_like(x, -jnp.inf),
                initial_metrics,
            )
            returns = jnp.zeros((eval_num_envs,), dtype=zero_reward.dtype)
            lengths = jnp.zeros((eval_num_envs,), dtype=zero_reward.dtype)
            done_seen = jnp.zeros((eval_num_envs,), dtype=bool)

            def _eval_step(carry, unused):
                (
                    eval_state,
                    last_obs,
                    rng,
                    done_seen,
                    returns,
                    lengths,
                    sum_metrics,
                    max_metrics,
                    hold_state,
                ) = carry
                rng, rng_action = jax.random.split(rng)
                _original_action, action, _skill, _value, _mask, hold_state, rng_action = (
                    _select_action(
                        train_state,
                        last_obs,
                        rng_action,
                        train_state.actor.n_updates,
                        explore=False,
                        hold_state=hold_state,
                    )
                )
                rng, rng_step = jax.random.split(rng)
                step_rng = jax.random.split(rng_step, eval_num_envs)
                raw_next_obs, next_eval_state, reward, done, info = eval_env.step(
                    step_rng,
                    eval_state,
                    action,
                    eval_env_params,
                )
                next_obs = _normalized_eval_obs(raw_next_obs, stats)
                metrics = _task_metrics(
                    get_policy_obs(last_obs),
                    get_policy_obs(next_obs),
                    action,
                    reward,
                    done,
                    info,
                )
                active = (~done_seen).astype(jnp.float32)
                active_bool = active.astype(bool)
                returns = returns + reward * active
                lengths = lengths + active
                sum_metrics = jax.tree_util.tree_map(
                    lambda acc, value: acc + value * active,
                    sum_metrics,
                    metrics,
                )
                max_metrics = jax.tree_util.tree_map(
                    lambda acc, value: jnp.maximum(acc, jnp.where(active_bool, value, -jnp.inf)),
                    max_metrics,
                    metrics,
                )
                done_seen = done_seen | done
                if hold_state is not None:
                    hold_state = (hold_state[0], hold_state[1], done.astype(bool))
                return (
                    next_eval_state,
                    next_obs,
                    rng,
                    done_seen,
                    returns,
                    lengths,
                    sum_metrics,
                    max_metrics,
                    hold_state,
                ), None

            if hold_interval > 1 and meta_policy_type != "flat":
                eval_init_hold_state = (
                    jnp.zeros((eval_num_envs,), dtype=jnp.int32),
                    jnp.full((eval_num_envs,), hold_interval, dtype=jnp.int32),
                    jnp.ones((eval_num_envs,), dtype=bool),
                )
            else:
                eval_init_hold_state = None
            init_carry = (
                eval_state,
                obs,
                batch_rng,
                done_seen,
                returns,
                lengths,
                zeros_like_metrics,
                neg_inf_metrics,
                eval_init_hold_state,
            )
            final_carry, _ = jax.lax.scan(_eval_step, init_carry, None, eval_max_steps)
            (
                _eval_state,
                _last_obs,
                _rng,
                _done_seen,
                returns,
                lengths,
                sum_metrics,
                max_metrics,
                _eval_hold_state,
            ) = final_carry
            denom = jnp.maximum(lengths, 1.0)
            mean_metrics = jax.tree_util.tree_map(lambda value: value / denom, sum_metrics)
            episode_metrics = _panda_episode_overrides(mean_metrics, max_metrics, initial_metrics)
            episode_metrics = _walker_episode_overrides(episode_metrics)
            table = {
                "episode_return": returns,
                "episode_length": lengths,
            }
            table.update(episode_metrics)
            return table

        eval_rng = jax.random.PRNGKey(int(config.get("EVAL_SEED", 10000)))
        batch_rngs = jax.random.split(eval_rng, eval_num_batches)

        def _batch_scan(_, batch_rng):
            return None, _eval_batch(batch_rng)

        _, batched_table = jax.lax.scan(_batch_scan, None, batch_rngs)
        flat_table = jax.tree_util.tree_map(
            lambda x: x.reshape((-1,) + x.shape[2:])[:eval_num_episodes],
            batched_table,
        )
        flat_table["eval_episode_index"] = jnp.arange(eval_num_episodes, dtype=jnp.int32)
        return _summarize_eval_table(flat_table), flat_table

    def train(rng: jax.Array) -> NexusTrainOutput:
        # Initialize environment.
        rng, rng_reset = jax.random.split(rng)
        reset_rng = jax.random.split(rng_reset, config["NUM_ENVS"])
        obs, env_state = env.reset(reset_rng, env_params)
        obs_actor = get_actor_obs(obs)
        obs_critic = get_critic_obs(obs)
        dummy_action = jnp.zeros((action_dim,), dtype=obs_actor.dtype)
        dummy_actor_obs = jnp.zeros(obs_actor.shape[1:], dtype=obs_actor.dtype)
        dummy_critic_obs = jnp.zeros(obs_critic.shape[1:], dtype=obs_critic.dtype)

        # Initialize one actor per skill and one critic ensemble per skill.
        rng, rng_actor, rng_critic, rng_meta = jax.random.split(rng, 4)
        actor_rngs = jax.random.split(rng_actor, num_skills)
        if use_rgb:
            # The CNN encoder requires a leading batch axis; init with batch=1.
            # The actor's proprio width is the restricted one (e.g. 0 for pixels-only).
            obs_pixels = get_actor_pixels(obs)
            proprio_dim = int(_actor_proprio(obs_actor).shape[-1])
            dummy_pixels = jnp.zeros((1,) + obs_pixels.shape[1:], dtype=obs_pixels.dtype)
            dummy_proprio = jnp.zeros((1, proprio_dim), dtype=obs_actor.dtype)
            actor_params = jax.vmap(
                lambda k: actor.init(k, dummy_pixels, dummy_proprio)["params"]
            )(actor_rngs)
        else:
            actor_params = jax.vmap(lambda k: actor.init(k, dummy_actor_obs)["params"])(actor_rngs)

        num_critics = int(config.get("NUM_CRITICS", 2))
        critic_rngs = jax.random.split(rng_critic, num_skills * num_critics).reshape(
            num_skills, num_critics, 2
        )
        critic_params = jax.vmap(
            lambda ks: jax.vmap(lambda k: critic.init(k, dummy_critic_obs, dummy_action)["params"])(
                ks
            )
        )(critic_rngs)

        meta_state = None
        if meta_policy_type not in ("symbolic", "flat"):
            meta_params = meta_q.init(rng_meta, dummy_actor_obs)["params"]
            meta_state = CounterTrainState.create(apply_fn=meta_q.apply, params=meta_params, tx=tx)

        train_state = NexusTrainState(
            actor=CounterTrainState.create(apply_fn=actor.apply, params=actor_params, tx=tx),
            critic=CounterTrainState.create(apply_fn=critic.apply, params=critic_params, tx=tx),
            meta=meta_state,
        )

        def _env_step(carry, unused):
            train_state, env_state, last_obs, rng, hold_state = carry
            update_idx = train_state.actor.n_updates
            original_action, action, skill, meta_selected_value, mask, hold_state, rng = (
                _select_action(
                    train_state, last_obs, rng, update_idx, explore=True, hold_state=hold_state
                )
            )
            obs_actor = get_actor_obs(last_obs)
            obs_critic = get_critic_obs(last_obs)
            policy_obs = get_policy_obs(last_obs)
            skill_bootstrap = _skill_actor_bootstrap_values(
                train_state.critic.params,
                train_state.actor.params,
                obs_actor,
                obs_critic,
                get_actor_pixels(last_obs),
            )
            if train_state.meta is not None:
                meta_bootstrap = _meta_bootstrap_value(train_state.meta.params, obs_actor, policy_obs)
            else:
                meta_bootstrap = jnp.zeros((obs_actor.shape[0],), dtype=obs_actor.dtype)
            rng, rng_step = jax.random.split(rng)
            step_rng = jax.random.split(rng_step, config["NUM_ENVS"])
            next_obs, next_env_state, env_reward, done, info = env.step(
                step_rng, env_state, action, env_params
            )
            next_policy_obs = get_policy_obs(next_obs)
            skill_rewards = policy_module.skill_rewards(
                policy_obs,
                next_policy_obs,
                action,
                env_reward,
                done,
                info,
            )
            diagnostics = _policy_diagnostics(
                policy_obs,
                next_policy_obs,
                action,
                env_reward,
                done,
                info,
            )
            transition = Transition(
                done=done,
                action=action,
                original_action=original_action,
                skill=skill,
                meta_selected_value=meta_selected_value,
                meta_bootstrap_value=meta_bootstrap,
                skill_bootstrap_values=skill_bootstrap,
                env_reward=env_reward,
                skill_rewards=skill_rewards,
                obs=last_obs,
                # Drop pixels from next_obs: it is never read downstream, and in
                # RGB mode storing it would duplicate the full image buffer.
                next_obs=_drop_actor_pixels(next_obs),
                info=info,
                diagnostics=diagnostics,
                mask=mask,
            )
            if hold_state is not None:
                hold_state = (hold_state[0], hold_state[1], done.astype(bool))
            return (train_state, next_env_state, next_obs, rng, hold_state), transition

        def _update_step(carry, unused):
            train_state, env_state, last_obs, rng, hold_state = carry
            t0 = time.time()
            (train_state, env_state, last_obs, rng, hold_state), traj = jax.lax.scan(
                _env_step,
                (train_state, env_state, last_obs, rng, hold_state),
                None,
                config["NUM_STEPS"],
            )

            # Bootstrap values at the final observation.
            last_obs_actor = get_actor_obs(last_obs)
            last_obs_critic = get_critic_obs(last_obs)
            last_policy_obs = get_policy_obs(last_obs)
            last_skill_bootstrap_values = _skill_actor_bootstrap_values(
                train_state.critic.params,
                train_state.actor.params,
                last_obs_actor,
                last_obs_critic,
                get_actor_pixels(last_obs),
            )
            if train_state.meta is not None:
                last_meta_bootstrap_value = _meta_bootstrap_value(
                    train_state.meta.params,
                    last_obs_actor,
                    last_policy_obs,
                )
            else:
                last_meta_bootstrap_value = jnp.zeros(
                    (last_obs_actor.shape[0],),
                    dtype=last_obs_actor.dtype,
                )

            skill_targets = q_lambda_returns(
                rewards=traj.skill_rewards,
                dones=traj.done,
                values=traj.skill_bootstrap_values,
                last_value=last_skill_bootstrap_values,
                gamma=float(config.get("GAMMA", 0.99)),
                lambda_=float(config.get("SKILL_LAMBDA", config.get("LAMBDA", 0.65))),
            )
            if train_state.meta is not None:
                meta_targets = q_lambda_returns(
                    rewards=traj.env_reward,
                    dones=traj.done,
                    values=traj.meta_bootstrap_value,
                    last_value=last_meta_bootstrap_value,
                    gamma=float(config.get("GAMMA", 0.99)),
                    lambda_=float(config.get("META_LAMBDA", config.get("LAMBDA", 0.65))),
                )
            else:
                meta_targets = jnp.zeros_like(traj.env_reward)

            batch = flatten_time_env((traj, skill_targets, meta_targets))

            def _update_epoch(epoch_state, unused):
                train_state, rng = epoch_state
                rng, rng_perm = jax.random.split(rng)
                minibatches = make_minibatches(batch, rng_perm, config["NUM_MINIBATCHES"])

                def _update_minibatch(mb_carry, mbatch):
                    train_state, rng = mb_carry
                    traj_mb, skill_targets_mb, meta_targets_mb = mbatch
                    obs_actor_mb = get_actor_obs(traj_mb.obs)
                    obs_critic_mb = get_critic_obs(traj_mb.obs)
                    obs_pixels_mb = get_actor_pixels(traj_mb.obs)
                    if rgb_augment:
                        rng, rng_aug = jax.random.split(rng)
                        obs_pixels_mb = _augment_pixels(obs_pixels_mb, rng_aug)

                    def critic_loss_fn(critic_params):
                        values = _critic_values_all_critics(
                            critic_params, obs_critic_mb, traj_mb.action
                        )  # [N, C, B]
                        target = jnp.swapaxes(skill_targets_mb, 0, 1)[:, None, :]  # [N, 1, B]
                        loss_per = smooth_l1_loss(values, target)
                        return jnp.mean(loss_per), {
                            "critic_value": jnp.mean(values),
                            "critic_target": jnp.mean(target),
                            "critic_abs_td": jnp.mean(jnp.abs(values - target)),
                        }

                    (critic_loss, critic_info), critic_grads = jax.value_and_grad(
                        critic_loss_fn, has_aux=True
                    )(train_state.critic.params)
                    critic_grad_norm = global_norm(critic_grads)
                    critic_state = train_state.critic.apply_gradients(grads=critic_grads).replace(
                        grad_steps=train_state.critic.grad_steps + 1
                    )

                    def actor_loss_fn(actor_params):
                        all_actions = _actor_apply(actor_params, obs_actor_mb, obs_pixels_mb)  # [N, B, A]

                        def one_skill_q(skill_idx, action_i):
                            skill_params = jax.tree_util.tree_map(
                                lambda leaf: leaf[skill_idx], critic_state.params
                            )
                            vals = jax.vmap(
                                lambda p: critic.apply({"params": p}, obs_critic_mb, action_i)
                            )(skill_params)
                            return _reduce_critics(vals)

                        q_by_skill = jax.vmap(one_skill_q)(
                            jnp.arange(num_skills), all_actions
                        )  # [N, B]
                        if config.get("ACTOR_UPDATE_MODE", "all_states") == "active_only":
                            active = jax.nn.one_hot(traj_mb.skill, num_skills).T
                            denom = jnp.maximum(jnp.sum(active), 1.0)
                            rl_loss = -jnp.sum(q_by_skill * active) / denom
                        else:
                            rl_loss = -jnp.mean(q_by_skill)

                        # Optional behavior regularizer: prevents early actors from drifting too
                        # far from the action distribution that generated the online batch.
                        coeff = float(config.get("BEHAVIOR_PENALTY_COEFF", 0.0))
                        if coeff > 0:
                            action_diff = all_actions - traj_mb.original_action[None, ...]
                            penalty = jnp.mean(jnp.square(action_diff))
                        else:
                            penalty = jnp.asarray(0.0)
                        return rl_loss + coeff * penalty, {
                            "actor_q": jnp.mean(q_by_skill),
                            "actor_penalty": penalty,
                        }

                    (actor_loss, actor_info), actor_grads = jax.value_and_grad(
                        actor_loss_fn, has_aux=True
                    )(train_state.actor.params)
                    actor_grad_norm = global_norm(actor_grads)
                    actor_state = train_state.actor.apply_gradients(grads=actor_grads).replace(
                        grad_steps=train_state.actor.grad_steps + 1
                    )

                    meta_loss = jnp.asarray(0.0)
                    meta_info = {"meta_q": jnp.asarray(0.0), "meta_abs_td": jnp.asarray(0.0)}
                    meta_grad_norm = jnp.asarray(0.0)
                    meta_state = train_state.meta
                    if train_state.meta is not None:

                        def meta_loss_fn(meta_params):
                            q = _meta_values(meta_params, obs_actor_mb)  # [B, N]
                            selected_q = _select_rows(q, traj_mb.skill)
                            loss = jnp.mean(smooth_l1_loss(selected_q, meta_targets_mb))
                            return loss, {
                                "meta_q": jnp.mean(selected_q),
                                "meta_abs_td": jnp.mean(jnp.abs(selected_q - meta_targets_mb)),
                            }

                        (meta_loss, meta_info), meta_grads = jax.value_and_grad(
                            meta_loss_fn, has_aux=True
                        )(train_state.meta.params)
                        meta_grad_norm = global_norm(meta_grads)
                        meta_state = train_state.meta.apply_gradients(grads=meta_grads).replace(
                            grad_steps=train_state.meta.grad_steps + 1
                        )

                    new_state = NexusTrainState(actor=actor_state, critic=critic_state, meta=meta_state)
                    losses = {
                        "loss/critic": critic_loss,
                        "loss/actor": actor_loss,
                        "loss/meta": meta_loss,
                        "train/critic_loss": critic_loss,
                        "train/actor_loss": actor_loss,
                        "train/meta_loss": meta_loss,
                        "train/actor_q": actor_info["actor_q"],
                        "train/actor_penalty": actor_info["actor_penalty"],
                        "train/critic_value": critic_info["critic_value"],
                        "train/critic_target": critic_info["critic_target"],
                        "train/critic_abs_td": critic_info["critic_abs_td"],
                        "train/meta_q": meta_info["meta_q"],
                        "train/meta_abs_td": meta_info["meta_abs_td"],
                        "train/actor_grad_norm": actor_grad_norm,
                        "train/critic_grad_norm": critic_grad_norm,
                        "train/meta_grad_norm": meta_grad_norm,
                    }
                    return (new_state, rng), losses

                (train_state, rng), losses = jax.lax.scan(
                    _update_minibatch, (train_state, rng), minibatches
                )
                return (train_state, rng), losses

            (train_state, rng), losses = jax.lax.scan(
                _update_epoch, (train_state, rng), None, config["NUM_EPOCHS"]
            )

            timesteps = (train_state.actor.n_updates + 1) * config["NUM_STEPS"] * config["NUM_ENVS"]
            actor_state = train_state.actor.replace(
                n_updates=train_state.actor.n_updates + 1,
                timesteps=timesteps,
            )
            if train_state.meta is not None:
                meta_state = train_state.meta.replace(n_updates=train_state.meta.n_updates + 1)
            else:
                meta_state = None
            train_state = NexusTrainState(actor=actor_state, critic=train_state.critic, meta=meta_state)

            skill_one_hot = jax.nn.one_hot(traj.skill, num_skills)
            skill_counts = jnp.mean(skill_one_hot, axis=(0, 1))
            mask_float = traj.mask.astype(jnp.float32)
            mask_available = jnp.mean(mask_float, axis=(0, 1))
            mask_selected_when_available = jnp.mean(skill_one_hot * mask_float, axis=(0, 1))
            mask_selected_given_available = mask_selected_when_available / jnp.maximum(
                mask_available,
                1e-6,
            )
            violation_metrics = mask_violation_metrics(skill_one_hot, mask_float, skill_names)
            noise_norm = jnp.linalg.norm(traj.action - traj.original_action, axis=-1)
            metrics = {
                "env_step": timesteps,
                "update": train_state.actor.n_updates,
                "noise": noise_schedule(train_state.actor.n_updates),
                "meta_epsilon": epsilon_schedule(train_state.actor.n_updates),
                "schedule/noise": noise_schedule(train_state.actor.n_updates),
                "schedule/meta_epsilon": epsilon_schedule(train_state.actor.n_updates),
                "schedule/skill_epsilon": epsilon_schedule(train_state.actor.n_updates),
                "returns/env_reward_mean": jnp.mean(traj.env_reward),
                "returns/skill_reward_mean": jnp.mean(traj.skill_rewards),
                "episode/done_fraction": jnp.mean(traj.done.astype(jnp.float32)),
                "rollout/done_fraction": jnp.mean(traj.done.astype(jnp.float32)),
                "rollout/episode_return": jnp.mean(traj.env_reward),
                "rollout/episode_length": jnp.asarray(config["NUM_STEPS"], dtype=jnp.float32),
                "action/action_norm_mean": jnp.mean(jnp.linalg.norm(traj.action, axis=-1)),
                "action/original_action_norm_mean": jnp.mean(
                    jnp.linalg.norm(traj.original_action, axis=-1)
                ),
                "action/noise_norm_mean": jnp.mean(noise_norm),
            }
            metrics.update(violation_metrics)
            # PureJAXQL LogVecWrapper exposes returned_episode_returns/lengths in info.
            if isinstance(traj.info, dict):
                for key, value in traj.info.items():
                    if key in (
                        "returned_episode_returns",
                        "returned_episode_lengths",
                        "original_reward",
                        "nonfinite_reward",
                    ):
                        metrics[f"env/{key}"] = jnp.mean(value)
                if "returned_episode_returns" in traj.info:
                    metrics["rollout/episode_return"] = jnp.mean(
                        traj.info["returned_episode_returns"]
                    )
                if "returned_episode_lengths" in traj.info:
                    metrics["rollout/episode_length"] = jnp.mean(
                        traj.info["returned_episode_lengths"]
                    )
            for idx, name in enumerate(skill_names):
                metrics[f"skill_usage/{idx}_{name}"] = skill_counts[idx]
                metrics[f"skill_reward/{idx}_{name}"] = jnp.mean(traj.skill_rewards[..., idx])
                metrics[f"mask_available/{idx}_{name}"] = mask_available[idx]
                metrics[f"mask_selected_when_available/{idx}_{name}"] = (
                    mask_selected_when_available[idx]
                )
                metrics[f"mask_selected_given_available/{idx}_{name}"] = (
                    mask_selected_given_available[idx]
                )
            if isinstance(traj.diagnostics, dict):
                for key, value in traj.diagnostics.items():
                    metrics[f"policy_diag/{key}"] = jnp.mean(value)
            metrics.update({k: jnp.mean(v) for k, v in losses.items()})
            metrics["debug/host_time_token"] = jnp.asarray(t0 - t0)  # stable scalar for pytree shape.

            # Optional host-side logging through callback keeps the train loop jittable.
            if config.get("PRINT_EVERY", 0) > 0:

                def _print_callback(m):
                    step = int(np.asarray(m["env_step"]))
                    if step % int(config["PRINT_EVERY"]) == 0:
                        print(
                            f"step={step} env_r={float(np.asarray(m['returns/env_reward_mean'])):.3f} "
                            f"noise={float(np.asarray(m['noise'])):.3f}"
                        )

                jax.debug.callback(_print_callback, metrics)

            return (train_state, env_state, last_obs, rng, hold_state), metrics

        if hold_interval > 1 and meta_policy_type != "flat":
            init_hold_state = (
                jnp.zeros((config["NUM_ENVS"],), dtype=jnp.int32),
                # steps_held starts at the interval so the first step re-decides.
                jnp.full((config["NUM_ENVS"],), hold_interval, dtype=jnp.int32),
                jnp.ones((config["NUM_ENVS"],), dtype=bool),
            )
        else:
            init_hold_state = None
        runner_state = (train_state, env_state, obs, rng, init_hold_state)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        train_state, env_state, obs, _rng, _hold_state = runner_state
        normalization_stats = _normalization_stats_from_state(env_state, obs)
        eval_metrics, eval_episode_table = _run_deterministic_evaluation(
            train_state,
            normalization_stats,
        )
        return NexusTrainOutput(
            runner_state=runner_state,
            metrics=metrics,
            eval_metrics=eval_metrics,
            eval_episode_table=eval_episode_table,
            normalization_stats=normalization_stats,
        )

    return train


def run_training(config: dict[str, Any]) -> NexusTrainOutput:
    """Convenience entry point for scripts."""

    rng = jax.random.PRNGKey(int(config.get("SEED", 0)))
    num_seeds = int(config.get("NUM_SEEDS", 1))
    rngs = jax.random.split(rng, num_seeds)
    train_fn = make_train(config)
    compiled = jax.jit(jax.vmap(train_fn)) if num_seeds > 1 else jax.jit(train_fn)
    return jax.block_until_ready(compiled(rngs)) if num_seeds > 1 else jax.block_until_ready(compiled(rngs[0]))
