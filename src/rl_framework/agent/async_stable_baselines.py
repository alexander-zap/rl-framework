from dataclasses import dataclass
from typing import Dict

from rl_framework.agent import StableBaselinesAgent, StableBaselinesAlgorithm
from async_gym_agents.agents.async_agent import get_injected_agent
from async_gym_agents.envs.multi_env import IndexableMultiEnv


@dataclass
class FakeEnum:
    name: str
    value: any


class AsyncStableBaselinesAgent(StableBaselinesAgent):
    def __init__(
        self, algorithm: StableBaselinesAlgorithm = StableBaselinesAlgorithm.PPO, algorithm_parameters: Dict = None
    ):
        super().__init__(FakeEnum(algorithm.name, get_injected_agent(algorithm.value)), algorithm_parameters)

    def to_vectorized_env(self, env_fns):
        return IndexableMultiEnv(env_fns)
