from abc import ABC, abstractmethod
from typing import Any, Optional, SupportsFloat, Tuple

import gymnasium as gym


class Environment(ABC, gym.Env):
    """
    Environment base class, from which every Environment should be inherited.

    It encapsulates an environment with arbitrary behind-the-scenes dynamics.

    This Environment provides the interface in the Gym format (actually it is even inherited from gym.Env).
    This is a standardized environment interface and should be used for every new environment created.
    For more guidance, on how to create new custom environments, see following description:
    https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
    https://www.gymlibrary.dev/content/environment_creation/
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

    @property
    def render_mode(self):
        """
        A flag specifying which kind of rendering should be performed by the `.render` method
        """
        raise NotImplementedError

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """Initialize the environment."""
        raise NotImplementedError

    @abstractmethod
    def step(self, action) -> Tuple[object, SupportsFloat, bool, bool, dict]:
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
                terminated (bool): Whether the agent reaches the terminal state (as defined under the MDP of the task)
                    which can be positive or negative.
                    If true, the user needs to call .reset()
                truncated (bool): Whether the truncation condition outside the scope of the MDP is satisfied.
                    Typically, this is a timelimit, but could also be used to indicate an agent physically going out of
                    bounds. Can be used to end the episode prematurely before a terminal state is reached.
                    If true, the user needs to call .reset()
                info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        raise NotImplementedError

    @abstractmethod
    def reset(
        self,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Tuple[object, dict[str, Any]]:
        """Resets the environment to an initial state and returns the initial observation.

        This method can reset the environment's random number generator(s) if ``seed`` is an integer or
        if the environment has not yet initialized a random number generator.
        If the environment already has a random number generator and :meth:`reset` is called with ``seed=None``,
        the RNG should not be reset. Moreover, :meth:`reset` should (in the typical use case) be called with an
        integer seed right after initialization and then never again.

        Args:
            seed (optional int): The seed that is used to initialize the environment's PRNG.
                If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed,
                a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
                However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will *not* be
                reset. If you pass an integer, the PRNG will be reset even if it already exists.
                Usually, you want to pass an integer *right after the environment has been initialized and then never
                again*. Please refer to the minimal example above to see this paradigm in action.
            return_info (bool): If true, return additional information along with initial observation.
                This info should be analogous to the info returned in :meth:`step`
            options (optional dict): Additional information to specify how the environment is reset (optional,
                depending on the specific environment)


        Returns:
            observation (object): Observation of the initial state. This will be an element of :attr:`observation_space`
                (typically a numpy array) and is analogous to the observation returned by :meth:`step`.
            info (optional dictionary): This will *only* be returned if ``return_info=True`` is passed.
                It contains auxiliary information complementing ``observation``. This dictionary should be analogous to
                the ``info`` returned by :meth:`step`.
        """
        raise NotImplementedError

    @abstractmethod
    def render(self):
        """Renders the environment.

        NOTE: Rendering mode is configured in self.render_mode since v26, so it should be set before calling .render()

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

        Example:

        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}

            def render(self):
                if self.render_mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif self.render_mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render() # just raise an exception
        """
        raise NotImplementedError
