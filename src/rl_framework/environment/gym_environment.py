import gym
from rl_framework.environment import Environment


class GymEnvironmentWrapper(Environment):
    """
    This is a wrapper for Gym environments.
    """

    def __init__(self, environment_name):
        self.raw_environment = gym.make(environment_name)

    def step(self, action):
        pass
