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

import functools
import time
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from nexus_continuous.envs.playground_adapter import (
    build_playground_env,
    get_actor_obs,
    get_critic_obs,
)
from nexus_continuous.networks import MetaQ, SkillActor, SkillCritic
from nexus_continuous.policies.registry import load_policy_module
from nexus_continuous.returns import q_lambda_returns, smooth_l1_loss
from nexus_continuous.train_state import CounterTrainState, NexusTrainState
from nexus_continuous.utils import flatten_time_env, make_minibatches


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    original_action: jnp.ndarray
    skill: jnp.ndarray
    meta_value: jnp.ndarray
    skill_values: jnp.ndarray
    env_reward: jnp.ndarray
    skill_rewards: jnp.ndarray
    obs: Any
    next_obs: Any
    info: Any


class NexusTrainOutput(NamedTuple):
    runner_state: Any
    metrics: Any


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
    num_skills = int(getattr(policy_module, "NUM_SKILLS"))
    skill_names = tuple(getattr(policy_module, "SKILL_NAMES"))
    meta_policy_type = config.get("META_POLICY_TYPE", "nesy").lower()
    if meta_policy_type not in ("neural", "symbolic", "nesy"):
        raise ValueError("META_POLICY_TYPE must be one of: neural, symbolic, nesy")

    env_bundle = build_playground_env(config)
    env = env_bundle.env
    env_params = env_bundle.env_params
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

    def _actor_apply(actor_params, obs_actor):
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

    def _skill_mask(obs):
        mask = policy_module.skill_mask(obs)
        return jnp.asarray(mask, dtype=bool)

    def _symbolic_policy(obs):
        return jnp.asarray(policy_module.symbolic_meta_policy(obs), dtype=jnp.int32)

    def _select_action(train_state: NexusTrainState, obs: Any, rng: jax.Array, update_idx: int):
        obs_actor = get_actor_obs(obs)
        obs_critic = get_critic_obs(obs)
        all_actions = _actor_apply(train_state.actor.params, obs_actor)  # [N, E, A]
        all_actions_env_major = jnp.swapaxes(all_actions, 0, 1)  # [E, N, A]

        if meta_policy_type == "symbolic":
            greedy_skill = _symbolic_policy(obs)
            mask = None
            meta_values = jnp.zeros((obs_actor.shape[0], num_skills), dtype=obs_actor.dtype)
            selected_skill = greedy_skill
        else:
            meta_values = _meta_values(train_state.meta.params, obs_actor)
            if meta_policy_type == "nesy":
                mask = _skill_mask(obs)
                masked_values = jnp.where(mask, meta_values, -1.0e9)
                greedy_skill = jnp.argmax(masked_values, axis=-1).astype(jnp.int32)
            else:
                mask = None
                greedy_skill = jnp.argmax(meta_values, axis=-1).astype(jnp.int32)
            epsilon = epsilon_schedule(update_idx)
            rng, rng_eps = jax.random.split(rng)
            selected_skill = _epsilon_greedy_skill(rng_eps, greedy_skill, epsilon, num_skills, mask)

        original_action = _select_rows(all_actions_env_major, selected_skill)
        rng, rng_noise = jax.random.split(rng)
        noise_std = noise_schedule(update_idx)
        if config.get("LINSPACE_NOISE", False):
            per_env_noise = jnp.linspace(0.0, noise_std, obs_actor.shape[0])[:, None]
        else:
            per_env_noise = noise_std
        noise = jax.random.normal(rng_noise, original_action.shape) * per_env_noise * action_scale
        action = jnp.clip(original_action + noise, action_low, action_high)
        meta_value = _select_rows(meta_values, selected_skill)
        skill_values = _critic_values_all_skills(train_state.critic.params, obs_critic, action)
        return original_action, action, selected_skill, meta_value, skill_values, rng

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
        if meta_policy_type != "symbolic":
            meta_params = meta_q.init(rng_meta, dummy_actor_obs)["params"]
            meta_state = CounterTrainState.create(apply_fn=meta_q.apply, params=meta_params, tx=tx)

        train_state = NexusTrainState(
            actor=CounterTrainState.create(apply_fn=actor.apply, params=actor_params, tx=tx),
            critic=CounterTrainState.create(apply_fn=critic.apply, params=critic_params, tx=tx),
            meta=meta_state,
        )

        def _env_step(carry, unused):
            train_state, env_state, last_obs, rng = carry
            update_idx = train_state.actor.n_updates
            original_action, action, skill, meta_value, skill_values, rng = _select_action(
                train_state, last_obs, rng, update_idx
            )
            rng, rng_step = jax.random.split(rng)
            step_rng = jax.random.split(rng_step, config["NUM_ENVS"])
            next_obs, next_env_state, env_reward, done, info = env.step(
                step_rng, env_state, action, env_params
            )
            skill_rewards = policy_module.skill_rewards(last_obs, next_obs, action, env_reward, done, info)
            transition = Transition(
                done=done,
                action=action,
                original_action=original_action,
                skill=skill,
                meta_value=meta_value,
                skill_values=skill_values,
                env_reward=env_reward,
                skill_rewards=skill_rewards,
                obs=last_obs,
                next_obs=next_obs,
                info=info,
            )
            return (train_state, next_env_state, next_obs, rng), transition

        def _update_step(carry, unused):
            train_state, env_state, last_obs, rng = carry
            t0 = time.time()
            (train_state, env_state, last_obs, rng), traj = jax.lax.scan(
                _env_step,
                (train_state, env_state, last_obs, rng),
                None,
                config["NUM_STEPS"],
            )

            # Bootstrap values at the final observation.
            rng, rng_boot = jax.random.split(rng)
            _, last_action, last_skill, last_meta_value, last_skill_values, rng = _select_action(
                train_state, last_obs, rng_boot, train_state.actor.n_updates
            )
            del last_action, last_skill

            skill_targets = q_lambda_returns(
                rewards=traj.skill_rewards,
                dones=traj.done,
                values=traj.skill_values,
                last_value=last_skill_values,
                gamma=float(config.get("GAMMA", 0.99)),
                lambda_=float(config.get("SKILL_LAMBDA", config.get("LAMBDA", 0.65))),
            )
            if meta_policy_type != "symbolic":
                meta_targets = q_lambda_returns(
                    rewards=traj.env_reward,
                    dones=traj.done,
                    values=traj.meta_value,
                    last_value=last_meta_value,
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

                def _update_minibatch(train_state: NexusTrainState, mbatch):
                    traj_mb, skill_targets_mb, meta_targets_mb = mbatch
                    obs_actor_mb = get_actor_obs(traj_mb.obs)
                    obs_critic_mb = get_critic_obs(traj_mb.obs)

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
                    critic_state = train_state.critic.apply_gradients(grads=critic_grads).replace(
                        grad_steps=train_state.critic.grad_steps + 1
                    )

                    def actor_loss_fn(actor_params):
                        all_actions = _actor_apply(actor_params, obs_actor_mb)  # [N, B, A]

                        def one_skill_q(skill_idx, action_i):
                            skill_params = jax.tree_util.tree_map(
                                lambda leaf: leaf[skill_idx], critic_state.params
                            )
                            vals = jax.vmap(
                                lambda p: critic.apply({"params": p}, obs_critic_mb, action_i)
                            )(skill_params)
                            return jnp.mean(vals, axis=0)

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
                    actor_state = train_state.actor.apply_gradients(grads=actor_grads).replace(
                        grad_steps=train_state.actor.grad_steps + 1
                    )

                    meta_loss = jnp.asarray(0.0)
                    meta_info = {"meta_q": jnp.asarray(0.0), "meta_abs_td": jnp.asarray(0.0)}
                    meta_state = train_state.meta
                    if meta_policy_type != "symbolic":

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
                        meta_state = train_state.meta.apply_gradients(grads=meta_grads).replace(
                            grad_steps=train_state.meta.grad_steps + 1
                        )

                    new_state = NexusTrainState(actor=actor_state, critic=critic_state, meta=meta_state)
                    losses = {
                        "loss/critic": critic_loss,
                        "loss/actor": actor_loss,
                        "loss/meta": meta_loss,
                        "train/actor_q": actor_info["actor_q"],
                        "train/actor_penalty": actor_info["actor_penalty"],
                        "train/critic_value": critic_info["critic_value"],
                        "train/critic_target": critic_info["critic_target"],
                        "train/critic_abs_td": critic_info["critic_abs_td"],
                        "train/meta_q": meta_info["meta_q"],
                        "train/meta_abs_td": meta_info["meta_abs_td"],
                    }
                    return new_state, losses

                train_state, losses = jax.lax.scan(_update_minibatch, train_state, minibatches)
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

            skill_counts = jnp.mean(jax.nn.one_hot(traj.skill, num_skills), axis=(0, 1))
            metrics = {
                "env_step": timesteps,
                "update": train_state.actor.n_updates,
                "noise": noise_schedule(train_state.actor.n_updates),
                "meta_epsilon": epsilon_schedule(train_state.actor.n_updates),
                "returns/env_reward_mean": jnp.mean(traj.env_reward),
                "returns/skill_reward_mean": jnp.mean(traj.skill_rewards),
                "episode/done_fraction": jnp.mean(traj.done.astype(jnp.float32)),
            }
            # PureJAXQL LogVecWrapper exposes returned_episode_returns/lengths in info.
            if isinstance(traj.info, dict):
                for key, value in traj.info.items():
                    if key in (
                        "returned_episode_returns",
                        "returned_episode_lengths",
                        "original_reward",
                    ):
                        metrics[f"env/{key}"] = jnp.mean(value)
            for idx, name in enumerate(skill_names):
                metrics[f"skill_usage/{idx}_{name}"] = skill_counts[idx]
                metrics[f"skill_reward/{idx}_{name}"] = jnp.mean(traj.skill_rewards[..., idx])
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

            return (train_state, env_state, last_obs, rng), metrics

        runner_state = (train_state, env_state, obs, rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return NexusTrainOutput(runner_state=runner_state, metrics=metrics)

    return train


def run_training(config: dict[str, Any]) -> NexusTrainOutput:
    """Convenience entry point for scripts."""

    rng = jax.random.PRNGKey(int(config.get("SEED", 0)))
    num_seeds = int(config.get("NUM_SEEDS", 1))
    rngs = jax.random.split(rng, num_seeds)
    train_fn = make_train(config)
    compiled = jax.jit(jax.vmap(train_fn)) if num_seeds > 1 else jax.jit(train_fn)
    return jax.block_until_ready(compiled(rngs)) if num_seeds > 1 else jax.block_until_ready(compiled(rngs[0]))
