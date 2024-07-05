from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

from rl_framework.environment import Environment
from rl_framework.util.saving_and_loading import Connector


class Agent(ABC):
    @property
    @abstractmethod
    def algorithm(self):
        return NotImplementedError

    @abstractmethod
    def __init__(self, algorithm, algorithm_parameters: Dict, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def train(
        self, training_environments: List[Environment], total_timesteps: int, connector: Connector, *args, **kwargs
    ):
        raise NotImplementedError

    @abstractmethod
    def choose_action(self, observation: object, deterministic: bool, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def save_to_file(self, file_path: Path, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_from_file(self, file_path: Path, algorithm_parameters: Optional[Dict], *args, **kwargs) -> None:
        raise NotImplementedError

    def upload(
        self,
        connector: Connector,
        evaluation_environment: Environment,
        deterministic_evaluation: bool = False,
    ) -> None:
        """
        Evaluate and upload the decision-making agent (and its .algorithm attribute) to the connector.
            Additional option: Generate a video of the agent interacting with the environment.

        Args:
            connector: Connector for uploading.
            evaluation_environment: Environment used for final evaluation and clip creation before upload.
            deterministic_evaluation (bool): Whether the action chosen by the agent in the evaluation
                should be determined in a deterministic or stochastic way.
        """
        connector.upload(
            agent=self, evaluation_environment=evaluation_environment, deterministic_evaluation=deterministic_evaluation
        )

    def download(
        self,
        connector: Connector,
        algorithm_parameters: Optional[Dict] = None,
    ):
        """
        Download a previously saved decision-making agent from the connector and replace the `self` agent instance
            in-place with the newly downloaded saved-agent.

        NOTE: Agent and Algorithm class need to be the same as the saved agent.

        Args:
            connector: Connector for downloading.
            algorithm_parameters (Optional[Dict]): Parameters to be set for the downloaded agent.
        """

        # Get the model from the Hub, download and cache the model on your local disk
        agent_file_path = connector.download()
        self.load_from_file(agent_file_path, algorithm_parameters)
