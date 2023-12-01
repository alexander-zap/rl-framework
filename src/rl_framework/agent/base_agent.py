from abc import ABC, abstractmethod


class Agent(ABC):
    @property
    @abstractmethod
    def algorithm(self):
        return NotImplementedError

    @abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def train(self, training_environments, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def choose_action(self, observation, *args, **kwargs):
        raise NotImplementedError
