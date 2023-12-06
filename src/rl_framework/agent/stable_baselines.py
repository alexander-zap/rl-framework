from enum import Enum
from functools import partial
from pathlib import Path
from typing import Dict, List

from stable_baselines3 import A2C, DDPG, DQN, HER, PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env

from rl_framework.agent import Agent
from rl_framework.environment import Environment


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
    def algorithm(self):
        return self._algorithm

    @algorithm.setter
    def algorithm(self, value):
        self._algorithm = value

    def __init__(
        self,
        algorithm: StableBaselinesAlgorithm = StableBaselinesAlgorithm.PPO,
        algorithm_parameters: Dict = None,
    ):
        """
        Initialize an agent which will trained on one of Stable-Baselines3 algorithms.

        Args:
            algorithm (StableBaselinesAlgorithm): Enum with values being SB3 RL Algorithm classes.
                Specifies the algorithm for RL training.
                Defaults to PPO.
            algorithm_parameters (Dict): Parameters / keyword arguments for the specified SB3 RL Algorithm class.
                See https://stable-baselines3.readthedocs.io/en/master/modules/base.html for details on common params.
                See individual docs (e.g., https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
                for algorithm-specific params.
        """

        self.algorithm_class = algorithm.value

        if algorithm_parameters is None:
            algorithm_parameters = {"policy": "MlpPolicy"}

        self.algorithm = None
        self._algorithm_builder = partial(self.algorithm_class, **algorithm_parameters)

    def train(self, training_environments: List[Environment], total_timesteps: int = 100000, *args, **kwargs):
        """
        Train the instantiated agent on the environment.

        This training is done by using the agent-on-environment training method provided by Stable-baselines3.

        The model is changed in place, therefore the updated model can be accessed in the `.model` attribute
        after the agent has been trained.

        Args:
            training_environments (List[Environment): Environment on which the agent should be trained on.
                If n_environments is set above 1, multiple environments enables parallel training of an agent.
            total_timesteps (int): Amount of individual steps the agent should take before terminating the training.
        """

        environment_iterator = iter(training_environments)
        training_env = make_vec_env(
            lambda: next(environment_iterator),
            n_envs=len(training_environments),
        )

        self.algorithm = self._algorithm_builder(env=training_env)

        self.algorithm.learn(total_timesteps=total_timesteps)

    def choose_action(self, observation: object, *args, **kwargs):
        """
        Chooses action which the agent will perform next, according to the observed environment.

        Args:
            observation (object): Observation of the environment

        Returns: action (int): Action to take according to policy.

        """

        # SB3 model expects multiple observations as input and will output an array of actions as output
        (
            action,
            _,
        ) = self.algorithm.predict(
            [observation],
            deterministic=True,
        )
        return action[0]

    def save_to_file(self, file_path: Path, *args, **kwargs) -> None:
        """Save the agent to a folder (for later loading).

        Args:
            file_path (Path): The file where the agent should be saved to (SB3 expects a file name ending with .zip).
        """
        self.algorithm.save(file_path)

    def load_from_file(self, file_path: Path, algorithm_parameters: Dict, *args, **kwargs) -> None:
        """Load the agent in-place from an agent-save folder.

        Args:
            file_path (Path): The model filename (file ending with .zip).
            algorithm_parameters: Parameters to be set for the loaded algorithm.
        """
        algorithm = self.algorithm_class.load(
            file_path,
            custom_objects=algorithm_parameters,
            print_system_info=True,
        )
        self.algorithm = algorithm
