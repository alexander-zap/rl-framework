import gym
from abc import abstractmethod


class Environment(gym.Env):
    """
    Environment base class, from which every Environment should be inherited.

    This Environment provides the interface in the OpenAI Gym format (actually it is even inherited from gym.Env).
    This is a standardized environment interface and should be used for every new environment created.

    For more guidance, on how to create new custom environments, see following description:
    https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
    """

    @property
    def action_space(self):
        raise NotImplementedError

    @property
    def observation_space(self):
        raise NotImplementedError

    @abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def step(self, action):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def render(self, mode="human"):
        raise NotImplementedError
