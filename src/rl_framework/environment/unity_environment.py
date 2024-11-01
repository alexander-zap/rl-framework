from typing import Tuple, Text
from rl_framework.environment import Environment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment


class UnityGameEnvironment(Environment):
    """
        This is a wrapper Environment class for Unity environments (environments built in the Unity game engine).
        Documentation about the interface can be found in the under following link:
        https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Python-Gym-API.md
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

    @property
    def render_mode(self):
        return self._render_mode

    @action_space.setter
    def action_space(self, value):
        self._action_space = value

    @observation_space.setter
    def observation_space(self, value):
        self._observation_space = value

    @reward_range.setter
    def reward_range(self, value):
        self._reward_range = value

    @render_mode.setter
    def render_mode(self, value):
        self._render_mode = value

    def __init__(self, path: Text, render_mode: Text = None, *args, **kwargs):
        """
        Initialize the wrapping attributes of a Unity environment instance.

        Args:
            path (Text): Path to the built Unity game environment.
                See here for mor e information on building Unity environments:
                https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Learning-Environment-Executable.md
            render_mode (Text): Mode for environment .render method (see .render-method for possible modes)
        """
        unity_env = UnityEnvironment(file_name=path)
        self._unity_environment = UnityToGymWrapper(unity_env)
        self._action_space = self._unity_environment.action_space
        self._observation_space = self._unity_environment.observation_space
        self._reward_range = self._unity_environment.reward_range
        self._render_mode = render_mode

    def step(self, action: object, *args, **kwargs) -> Tuple[object, float, bool, dict]:
        return self._unity_environment.step(action)

    def reset(self, *args, **kwargs) -> object:
        return self._unity_environment.reset()

    def render(self, *args, **kwargs) -> None:
        return self._unity_environment.render()
