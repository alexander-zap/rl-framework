from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from rl_framework.agent import Agent
from rl_framework.environment import Environment
from rl_framework.util import Connector

from .custom_algorithms import QLearning


class CustomAlgorithm(Enum):
    Q_LEARNING = QLearning


class CustomAgent(Agent):
    @property
    def algorithm(self):
        return self._algorithm

    @algorithm.setter
    def algorithm(self, value):
        self._algorithm = value

    def __init__(
        self,
        algorithm: CustomAlgorithm = CustomAlgorithm.Q_LEARNING,
        algorithm_parameters: Dict = None,
    ):
        """
        Initialize an agent which will trained on one of custom implemented algorithms.

        Args:
            algorithm (CustomAlgorithm): Enum with values being custom implemented Algorithm classes (Types).
                Specifies the algorithm for RL training.
                Defaults to Q-Learning.
            algorithm_parameters (Dict): Parameters / keyword arguments for the specified Algorithm class.
        """

        algorithm_class = algorithm.value

        if algorithm_parameters is None:
            algorithm_parameters = {}

        self.algorithm = algorithm_class(**algorithm_parameters)

    def train(
        self,
        training_environments: List[Environment],
        total_timesteps: int,
        connector: Optional[Connector] = None,
        *args,
        **kwargs,
    ):
        """
        Train the instantiated agent on the environment.

        This training is done by using the agent-on-environment training method provided by the custom algorithm.

        Args:
            training_environments (List[Environment): Environment on which the agent should be trained on.
                If n_environments is set above 1, multiple environments enables parallel training of an agent.
            total_timesteps (int): Amount of individual steps the agent should take before terminating the training.
            connector (Connector): Connector for executing callbacks (e.g., logging metrics and saving checkpoints)
                on training time. Logging is executed by calling the connector.log method.
                Calls need to be declared manually in the code.
        """

        self.algorithm.train(
            training_environments=training_environments,
            total_timesteps=total_timesteps,
            connector=connector,
            *args,
            **kwargs,
        )

    def choose_action(self, observation: object, *args, **kwargs):
        """
        Chooses action which the agent will perform next, according to the observed environment.

        Args:
            observation (object): Observation of the environment

        Returns: action (int): Action to take according to policy.

        """

        return self.algorithm.choose_action(observation=observation, *args, **kwargs)

    def save_to_file(self, file_path: Path, *args, **kwargs):
        """Save the agent to a file at a certain path (to be loadable again later).

        Args:
            file_path: File where the agent should be saved to.
        """
        self.algorithm.save_to_file(file_path=file_path)

    def load_from_file(self, file_path: Path, algorithm_parameters: Optional[Dict] = None, *args, **kwargs):
        """Load the agent from a previously save agent-file.

        Args:
            file_path: File where the agent-file to be loaded is located at.
            algorithm_parameters (Optional[Dict]): Parameters which should overwrite the algorithm after loading.
        """
        self.algorithm.load_from_file(file_path, algorithm_parameters=algorithm_parameters)
