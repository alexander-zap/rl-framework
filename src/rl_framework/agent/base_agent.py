from abc import ABC, abstractmethod
from typing import Dict, Optional, Text

from rl_framework.environment import Environment


class Agent(ABC):
    @property
    @abstractmethod
    def algorithm(self):
        return NotImplementedError

    @abstractmethod
    def __init__(self, algorithm, algorithm_parameters, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def train(self, training_environments, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def choose_action(self, observation, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def upload_to_huggingface_hub(
        self,
        repository_id: Text,
        evaluation_environment: Environment,
        environment_name: Text,
        model_name: Text,
        model_architecture: Text,
        commit_message: Text,
        n_eval_episodes: int,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def download_from_huggingface_hub(
        self, repository_id: Text, filename: Text, algorithm_parameters: Optional[Dict] = None
    ) -> None:
        raise NotImplementedError
