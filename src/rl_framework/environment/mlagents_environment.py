"""
This is a wrapper Environment class for MLAgents environments (environments built in Unity).
"""

from typing import Tuple, Text
from rl_framework.environment import Environment


class MLAgentsEnvironmentWrapper(Environment):
    """
    """

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def reward_range(self):
        return self._reward_range

    @action_space.setter
    def action_space(self, value):
        self._action_space = value

    @observation_space.setter
    def observation_space(self, value):
        self._observation_space = value

    @reward_range.setter
    def reward_range(self, value):
        self._reward_range = value

    def __init__(self, environment_name: Text):
        """
         TODO
        """
        self._unity_environment = None
        self._action_space = self._unity_environment.action_space
        self._observation_space = self._unity_environment.observation_space
        self._reward_range = self._unity_environment.reward_range

    def step(self, action: object) -> Tuple[object, float, bool, dict]:
        """
        """
        return self._unity_environment.step(action)

    def reset(self) -> object:
        """
        """
        return self._unity_environment.reset()

    def render(self, mode="human") -> None:
        """
        """
        return self._unity_environment.render(mode=mode)
