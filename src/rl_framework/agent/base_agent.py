from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Text

from rl_framework.agent.custom_algorithms import Algorithm
from rl_framework.environment import Environment


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

    @abstractmethod
    def upload(
        self,
        repository_id: Text,
        evaluation_environment: Environment,
        environment_name: Text,
        file_name: Text,
        model_architecture: Text,
        commit_message: Text,
        n_eval_episodes: int,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def download(
        self, repository_id: Text, file_name: Text, algorithm_parameters: Optional[Dict]
    ) -> None:
        raise NotImplementedError
