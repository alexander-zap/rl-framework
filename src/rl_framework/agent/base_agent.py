from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

from rl_framework.agent.custom_algorithms import Algorithm
from rl_framework.environment import Environment
from rl_framework.util.saving_and_loading import Connector, DownloadConfig, UploadConfig


class Agent(ABC):
    @property
    @abstractmethod
    def algorithm(self):
        return NotImplementedError

    @abstractmethod
    def __init__(self, algorithm: Algorithm, algorithm_parameters: Dict, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def train(self, training_environments: List[Environment], total_timesteps: int, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def choose_action(self, observation: object, *args, **kwargs):
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
        connector_config: UploadConfig,
        evaluation_environment: Environment,
        video_length: int = 0,
    ) -> None:
        """
        Evaluate and upload the decision-making agent (and its .algorithm attribute) to the connector.
            Additional option: Generate a video of the agent interacting with the environment.

        Args:
            connector: Connector for uploading.
            connector_config: Configuration data for connector.
            evaluation_environment: Environment used for final evaluation and clip creation before upload.
            video_length (int): Length of video in frames (which should be generated and uploaded to the connector).
                No video is uploaded if length is 0 or negative. Set to 0 by default.
        """
        connector.upload(
            agent=self,
            evaluation_environment=evaluation_environment,
            config=connector_config,
            video_length=video_length,
        )

    def download(
        self,
        connector: Connector,
        connector_config: DownloadConfig,
        algorithm_parameters: Optional[Dict] = None,
    ):
        """
        Download a previously saved decision-making agent from the connector and replace the `self` agent instance
            in-place with the newly downloaded saved-agent.

        NOTE: Agent and Algorithm class need to be the same as the saved agent.

        Args:
            connector: Connector for downloading.
            connector_config: Configuration data for connector.
            algorithm_parameters (Optional[Dict]): Parameters to be set for the downloaded agent.
        """

        # Get the model from the Hub, download and cache the model on your local disk
        agent_file_path = connector.download(connector_config)
        self.load_from_file(agent_file_path, algorithm_parameters)
