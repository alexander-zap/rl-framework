import gym
from rl_framework.environment import Environment


class GymEnvironmentWrapper(Environment):
    """
    This is a wrapper for Gym environments.
    """

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @action_space.setter
    def action_space(self, value):
        self._action_space = value

    @observation_space.setter
    def observation_space(self, value):
        self._observation_space = value

    def __init__(self, environment_name):
        self._gym_environment: gym.Env = gym.make(environment_name)
        self._action_space = self._gym_environment.action_space
        self._observation_space = self._gym_environment.observation_space

    def step(self, action):
        return self._gym_environment.step(action)

    def reset(self):
        return self._gym_environment.reset()

    def render(self, mode="human"):
        return self._gym_environment.render(mode=mode)
