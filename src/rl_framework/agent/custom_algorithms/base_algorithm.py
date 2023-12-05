from abc import ABC, abstractmethod


class Algorithm(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def train(self, training_environments, total_timesteps, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def choose_action(self, observation, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def save_to_file(self, file_path, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def load_from_file(self, file_path, *args, **kwargs):
        raise NotImplementedError
