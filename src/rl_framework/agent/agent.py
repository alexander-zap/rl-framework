from abc import ABC, abstractmethod


class Agent(ABC):
    @property
    def model(self):
        raise NotImplementedError

    @abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def save(self, *args, **kwargs):
        raise NotImplementedError
