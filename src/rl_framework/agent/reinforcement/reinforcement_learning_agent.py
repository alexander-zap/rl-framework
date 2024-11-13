from abc import ABC, abstractmethod
from typing import List

from rl_framework.agent.base_agent import Agent
from rl_framework.environment import Environment
from rl_framework.util.saving_and_loading import Connector


class RLAgent(Agent, ABC):
    @abstractmethod
    def train(
        self,
        total_timesteps: int,
        connector: Connector,
        training_environments: List[Environment] = None,
        *args,
        **kwargs,
    ):
        raise NotImplementedError
