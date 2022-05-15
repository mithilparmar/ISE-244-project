import abc

from typing import (
    NamedTuple,
    Text,
    Mapping,
    Optional,
)
import numpy as np

Action = int


class TimeStep(NamedTuple):
    """Environment timestep"""

    observation: Optional[np.ndarray]
    reward: Optional[float]
    done: Optional[bool]
    first: Optional[bool]  # first step of an episode


class Agent(abc.ABC):
    """Agent interface."""

    agent_name: str  # agent name

    @abc.abstractmethod
    def step(self, timestep: TimeStep) -> Action:
        """Selects action given timestep and potentially learns."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Resets the agent's episodic state such as frame stack and action repeat.

        This method should be called at the beginning of every episode.
        """

    @property
    @abc.abstractmethod
    def statistics(self) -> Mapping[Text, float]:
        """Returns current agent statistics as a dictionary."""
