"""Train-state containers used by hierarchical AC-PQN."""

from __future__ import annotations

from typing import Any, NamedTuple

from flax.training.train_state import TrainState


class CounterTrainState(TrainState):
    """Flax TrainState with lightweight counters."""

    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


class NexusTrainState(NamedTuple):
    """All learned components for continuous NEXUS."""

    actor: CounterTrainState
    critic: CounterTrainState
    meta: CounterTrainState | None


Params = Any
