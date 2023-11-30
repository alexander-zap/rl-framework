"""
This is a wrapper Environment class for Gym environments.
"""

from typing import Any, Optional, SupportsFloat, Text, Tuple

import gymnasium as gym

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
        render_mode: A flag specifying which kind of rendering should be performed by the `.render` method

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

    def __init__(self, environment_name: Text, render_mode: Text = None, *args, **kwargs):
        """
        Initialize the wrapping attributes of a Gym environment instance.

        Args:
            environment_name (Text): Name of the environment, as registered in Gym.
            render_mode (Text): Mode for environment .render method (see .render-method for possible modes)
        """
        self._gym_environment: gym.Env = gym.make(environment_name, render_mode=render_mode, *args, **kwargs)
        self._action_space = self._gym_environment.action_space
        self._observation_space = self._gym_environment.observation_space
        self._reward_range = self._gym_environment.reward_range
        self._render_mode = self._gym_environment.render_mode

    def step(self, action: object) -> Tuple[object, SupportsFloat, bool, bool, dict]:
        """Original Gym documentation:

        Run one timestep of the environment's dynamics using the agent actions.

         When the end of an episode is reached (``terminated or truncated``), it is necessary to call :meth:`reset` to
         reset this environment's state for the next episode.

         .. versionchanged:: 0.26

             The Step API was changed removing ``done`` in favor of ``terminated`` and ``truncated`` to make it clearer
             to users when the environment had terminated or truncated which is critical for reinforcement learning
             bootstrapping algorithms.

         Args:
             action (ActType): an action provided by the agent to update the environment state.

         Returns:
             observation (ObsType): An element of the environment's :attr:`observation_space` as the next observation
                due to the agent actions.
                An example is a numpy array containing the positions and velocities of the pole in CartPole.
             reward (SupportsFloat): The reward as a result of taking the action.
             terminated (bool): Whether the agent reaches the terminal state (as defined under the MDP of the task)
                 which can be positive or negative. An example is reaching the goal state or moving into the lava from
                 the Sutton and Barton, Gridworld. If true, the user needs to call :meth:`reset`.
             truncated (bool): Whether the truncation condition outside the scope of the MDP is satisfied.
                 Typically, this is a timelimit, but could also be used to indicate an agent physically going out of
                 bounds. Can be used to end the episode prematurely before a terminal state is reached.
                 If true, the user needs to call :meth:`reset`.
             info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                 This might, for instance, contain: metrics that describe the agent's performance state, variables that
                 are hidden from observations, or individual reward terms that are combined to produce the total reward.
                 In OpenAI Gym <v26, it contains "TimeLimit.truncated" to distinguish truncation and termination,
                 however this is deprecated in favour of returning terminated and truncated variables.
             done (bool): (Deprecated) A boolean value for if the episode has ended, in which case further :meth:`step`
                calls will return undefined results. This was removed in OpenAI Gym v26 in favor of terminated and
                truncated attributes. A done signal may be emitted for different reasons: Maybe the task underlying
                the environment was solved successfully, a certain timelimit was exceeded,
                or the physics simulation has entered an invalid state.
        """
        return self._gym_environment.step(action)

    def reset(self, seed: Optional[int] = None, *args, **kwargs) -> Tuple[object, dict[str, Any]]:
        """Original Gym documentation:

        Resets the environment to an initial internal state, returning an initial observation and info.

        This method generates a new starting state often with some randomness to ensure that the agent explores the
        state space and learns a generalised policy about the environment. This randomness can be controlled
        with the ``seed`` parameter otherwise if the environment already has a random number generator and
        :meth:`reset` is called with ``seed=None``, the RNG is not reset.

        Therefore, :meth:`reset` should (in the typical use case) be called with a seed right after initialization and
        then never again.

        For Custom environments, the first line of :meth:`reset` should be ``super().reset(seed=seed)`` which implements
        the seeding correctly.

        .. versionchanged:: v0.25

            The ``return_info`` parameter was removed and now info is expected to be returned.

        Args:
            seed (optional int): The seed that is used to initialize the environment's PRNG (`np_random`).
                If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed,
                a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
                However, if the environment already has a PRNG and ``seed=None`` is passed,
                the PRNG will *not* be reset. If you pass an integer, the PRNG will be reset even if it already exists.
                Usually, you want to pass an integer *right after the environment has been initialized and then never
                again*. Please refer to the minimal example above to see this paradigm in action.
            options (optional dict): Additional information to specify how the environment is reset (optional,
                depending on the specific environment)

        Returns:
            observation (ObsType): Observation of the initial state.
                This will be an element of :attr:`observation_space` (typically a numpy array) and is analogous to the
                observation returned by :meth:`step`.
            info (dictionary):  This dictionary contains auxiliary information complementing ``observation``.
                It should be analogous to the ``info`` returned by :meth:`step`.
        """
        return self._gym_environment.reset(seed=seed)

    def render(self) -> None:
        """Original Gym documentation:

        Compute the render frames as specified by :attr:`render_mode` during the initialization of the environment.

        The environment's :attr:`metadata` render modes (`env.metadata["render_modes"]`) should contain the possible
        ways to implement the render modes. In addition, list versions for most render modes is achieved through
        `gymnasium.make` which automatically applies a wrapper to collect rendered frames.

        Note:
            As the :attr:`render_mode` is known during ``__init__``, the objects used to render the environment state
            should be initialised in ``__init__``.

        By convention, if the :attr:`render_mode` is:

        - None (default): no render is computed.
        - "human": The environment is continuously rendered in the current display or terminal,
            usually for human consumption.
            This rendering should occur during :meth:`step` and :meth:`render` doesn't need to be called.
            Returns ``None``.
        - "rgb_array": Return a single frame representing the current state of the environment.
            A frame is a ``np.ndarray`` with shape ``(x, y, 3)`` representing RGB values for an x-by-y pixel image.
        - "ansi": Return a strings (``str``) or ``StringIO.StringIO`` containing a terminal-style text representation
            for each time step. The text can include newlines and ANSI escape sequences (e.g. for colors).
        - "rgb_array_list" and "ansi_list": List based version of render modes are possible (except Human) through the
            wrapper, :py:class:`gymnasium.wrappers.RenderCollection` that is automatically applied during
            ``gymnasium.make(..., render_mode="rgb_array_list")``.
            The frames collected are popped after :meth:`render` is called or :meth:`reset`.

        Note:
            Make sure that your class's :attr:`metadata` ``"render_modes"`` key includes the list of supported modes.

        .. versionchanged:: 0.25.0

            The render function was changed to no longer accept parameters, rather these parameters should be specified
            in the environment initialised, i.e., ``gymnasium.make("CartPole-v1", render_mode="human")``
        """
        return self._gym_environment.render()
