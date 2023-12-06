from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

from rl_framework.agent.custom_algorithms import Algorithm
from rl_framework.environment import Environment
from rl_framework.util.saving_and_loading import (
    Connector,
    DownloadConfig,
    UploadConfig,
    download,
    upload,
)


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
    ) -> None:
        """Evaluate, generate a video and upload the agent to the connector.

        Args:
            connector: Connector for uploading.
            connector_config: Configuration data for connector.
            evaluation_environment: Environment used for final evaluation and clip creation before upload.
        """
        upload(
            connector=connector,
            connector_config=connector_config,
            agent=self,
            evaluation_environment=evaluation_environment,
        )

    def download(
        self,
        connector: Connector,
        connector_config: DownloadConfig,
        algorithm_parameters: Optional[Dict] = None,
    ):
        """Download a reinforcement learning model from the connector and update the agent in-place.

        Args:
            connector: Connector for downloading.
            connector_config: Configuration data for connector.
            algorithm_parameters (Optional[Dict]): Parameters to be set for the downloaded algorithm.
        """

        download(
            connector=connector,
            connector_config=connector_config,
            agent=self,
            algorithm_parameters=algorithm_parameters,
        )
