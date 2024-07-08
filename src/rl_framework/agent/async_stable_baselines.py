from typing import Dict, Type

import stable_baselines3
from async_gym_agents.agents.async_agent import get_injected_agent
from async_gym_agents.envs.multi_env import IndexableMultiEnv
from stable_baselines3.common.base_class import BaseAlgorithm

from rl_framework.agent.stable_baselines import StableBaselinesAgent


class AsyncStableBaselinesAgent(StableBaselinesAgent):
    def __init__(self, algorithm_class: Type[BaseAlgorithm] = stable_baselines3.PPO, algorithm_parameters: Dict = None):
        super().__init__(get_injected_agent(algorithm_class), algorithm_parameters)

    def to_vectorized_env(self, env_fns):
        return IndexableMultiEnv(env_fns)
