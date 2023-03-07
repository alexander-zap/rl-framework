"""
This is a wrapper Environment class for Gym environments.
"""

from typing import Tuple, Text
import gym
from rl_framework.environment import Environment


class GymEnvironmentWrapper(Environment):
    """
    Original Gym documentation:

    The main OpenAI Gym class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.

    The main API methods that users of this class need to know are:

        step
        reset
        render
        close
        seed

    And set the following attributes:

        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards

    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.

    The methods are accessed publicly as "step", "reset", etc...
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
        Initialize the wrapping attributes of a Gym environment instance.

        Args:
            environment_name (Text): Name of the environment, as registered in Gym.
        """
        self._gym_environment: gym.Env = gym.make(environment_name)
        self._action_space = self._gym_environment.action_space
        self._observation_space = self._gym_environment.observation_space
        self._reward_range = self._gym_environment.reward_range

    def step(self, action: object) -> Tuple[object, float, bool, dict]:
        """
        Original Gym documentation:

        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        return self._gym_environment.step(action)

    def reset(self) -> object:
        """
        Original Gym documentation:

        Resets the environment to an initial state and returns an initial
        observation.

        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.

        Returns:
            observation (object): the initial observation.
        """
        return self._gym_environment.reset()

    def render(self, mode="human") -> None:
        """
        Original Gym documentation:

        Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return a numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with

        Example:

        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}

            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        return self._gym_environment.render(mode=mode)
