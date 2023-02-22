from abc import abstractmethod


class Environment:
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass
