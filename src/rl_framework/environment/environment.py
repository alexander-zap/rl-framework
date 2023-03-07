import gym
from abc import ABC, abstractmethod
from typing import Tuple


class Environment(ABC, gym.Env):
    """
    Environment base class, from which every Environment should be inherited.

    It encapsulates an environment with arbitrary behind-the-scenes dynamics.

    This Environment provides the interface in the Gym format (actually it is even inherited from gym.Env).
    This is a standardized environment interface and should be used for every new environment created.
    For more guidance, on how to create new custom environments, see following description:
    https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
    """

    @property
    def action_space(self):
        """
        The Space object corresponding to valid actions
        """
        raise NotImplementedError

    @property
    def observation_space(self):
        """
        The Space object corresponding to valid observations
        """
        raise NotImplementedError

    @property
    def reward_range(self):
        """
        A tuple corresponding to the min and max possible rewards
        """
        raise NotImplementedError

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """Initialize the environment."""
        raise NotImplementedError

    @abstractmethod
    def step(self, action) -> Tuple[object, float, bool, dict]:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            Tuple consisting of following elements:
                observation (object): agent's observation of the current environment
                reward (float) : amount of reward returned after previous action
                done (bool): whether the episode has ended, in which case further step() calls will return undefined results
                info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> object:
        """Resets the environment to an initial state and returns an initial observation.

        Returns:
            observation (object): the initial observation.
        """
        raise NotImplementedError

    @abstractmethod
    def render(self, mode="human"):
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
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
        raise NotImplementedError
