from rl_framework.agent import Agent
from rl_framework.environment import Environment
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import A2C, DDPG, DQN, HER, PPO, SAC, TD3
from typing import List, Text, Optional, Dict
from functools import partial
from enum import Enum


class StableBaselinesAlgorithm(Enum):
    A2C = A2C
    DDPG = DDPG
    DQN = DQN
    HER = HER
    PPO = PPO
    SAC = SAC
    TD3 = TD3


class StableBaselinesAgent(Agent):
    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    def __init__(
        self,
        rl_algorithm: StableBaselinesAlgorithm = StableBaselinesAlgorithm.PPO,
        rl_algorithm_parameters: Dict = None,
        pretrained_model: Optional[BaseAlgorithm] = None,
    ):
        """
        Initialize an agent which will trained on one of Stable-Baselines3 algorithms.

        Args:
            rl_algorithm (StableBaselinesAlgorithm): Enum with values being SB3 RL Algorithm classes (Types).
                Specifies the algorithm for RL training.
                Defaults to PPO.
            rl_algorithm_parameters (Dict): Parameters / keyword arguments for the specified SB3 RL Algorithm class.
                See https://stable-baselines3.readthedocs.io/en/master/modules/base.html for details on common params.
                See individual docs (e.g., https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
                for algorithm-specific params.
            pretrained_model (BaseAlgorithm): Pretrained SB3 model.
                This variable is mainly used for loading previously saved models.
        """

        self.model = pretrained_model

        if rl_algorithm_parameters is None:
            rl_algorithm_parameters = {
                "policy": "MlpPolicy",
                "learning_rate": 0.001,
            }

        self._model_builder = partial(
            rl_algorithm.value,
            **rl_algorithm_parameters
        )

    def train(self, environments: List[Environment], total_timesteps: int = 100000):
        """
        Train the instantiated agent on the environment.

        This training is done by using the agent-on-environment training method provided by Stable-baselines3.

        The model is changed in place, therefore the updated model can be accessed in the `.model` attribute
        after the agent has been trained.

        Args:
            environments (List[Environment]): List of environments on which the agent should be trained on.
                Providing multiple environments enables parallel training of an agent.
            total_timesteps (int): Amount of individual steps the agent should take before terminating the training.
        """

        environment_iterator = iter(environments)
        training_env = make_vec_env(
            lambda: next(environment_iterator), n_envs=len(environments)
        )

        self.model = self._model_builder(env=training_env)

        self.model.learn(total_timesteps=total_timesteps)

    def save(self, file_path: Text):
        """
        Save the model of the agent to a zipped file.

        Args:
            file_path (Text): Path where the model should be saved to.
        """
        self.model.save(file_path)
