import logging
from typing import Optional, SupportsFloat, Text, Tuple

import numpy as np
from dm_env import StepType, specs
from dm_env_rpc.v1 import connection as dm_env_rpc_connection
from dm_env_rpc.v1 import dm_env_adaptor, dm_env_rpc_pb2
from gymnasium.spaces import Box, Discrete, Space

from rl_framework.environment import Environment


class RemoteEnvironment(Environment):
    """
    Wrapper implementation of remote environments,
    Implements central client-side management for the open communication interface between this wrapper Environment
    class and remotely running environments.
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

    def __init__(self, url: Text, port: int, *args, **kwargs):
        """
        Initialize the remote environment connection.
        """

        def convert_to_space(spec: specs.Array) -> Space:
            if isinstance(spec, specs.DiscreteArray):
                return Discrete(n=spec.num_values)
            # UInt-check required for detecting pixel-based RGB observations (these should be embedded in the Box space)
            if (
                isinstance(spec, specs.BoundedArray)
                and np.issubdtype(spec.dtype, np.integer)
                and not isinstance(spec.dtype, np.dtypes.UInt8DType)
            ):
                return Discrete(n=spec.maximum - spec.minimum + 1, start=spec.minimum)
            else:
                if isinstance(spec, specs.BoundedArray):
                    _min = spec.minimum
                    _max = spec.maximum

                    if np.isscalar(_min) and np.isscalar(_max):
                        # same min and max for every element
                        return Box(low=_min, high=_max, shape=spec.shape, dtype=spec.dtype)
                    else:
                        # different min and max for every element
                        return Box(
                            low=_min + np.zeros(spec.shape),
                            high=_max + np.zeros(spec.shape),
                            shape=spec.shape,
                            dtype=spec.dtype,
                        )
                elif isinstance(spec, specs.Array):
                    return Box(-np.inf, np.inf, shape=spec.shape)
                else:
                    logging.error(
                        f"Unable to transform dm_env.spec {type(spec)} to Gym space."
                        f"Support for this dm_env.spec type can be added at the location of the raised ValueError."
                    )
                    raise ValueError

        self.connection = dm_env_rpc_connection.create_secure_channel_and_connect(f"{url}:{port}")
        self.remote_environment, self.world_name = dm_env_adaptor.create_and_join_world(
            self.connection, create_world_settings={}, join_world_settings={}
        )

        action_spec = self.remote_environment.action_spec()["action"]
        observation_spec = self.remote_environment.observation_spec()["observation"]
        reward_spec = self.remote_environment.reward_spec()

        # Set local environment attributes for remote wrapper
        self._action_space = convert_to_space(action_spec)
        self._observation_space = convert_to_space(observation_spec)
        self._reward_range = (
            (reward_spec.minimum, reward_spec.maximum)
            if isinstance(reward_spec, specs.BoundedArray)
            else (-float("inf"), float("inf"))
        )

        # TODO: Render remote environments. Currently not possible.
        self._render_mode = None

    def step(self, action, *args, **kwargs) -> Tuple[object, SupportsFloat, bool, bool, dict]:
        """
        Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        """
        timestep = self.remote_environment.step({"action": action})

        observation = timestep.observation["observation"]
        reward = timestep.reward
        discount_factor = timestep.discount
        step_type = timestep.step_type

        terminated = False
        truncated = False

        if step_type is StepType.LAST:
            # NOTE: Assumption that discount_factor is 0.0 only for termination steps
            #   See https://github.com/google-deepmind/dm_env/blob/master/dm_env/_environment.py#L228
            if discount_factor == 0.0:
                terminated = True
            else:
                truncated = True

        return observation, reward, terminated, truncated, {}

    def reset(self, seed: Optional[int] = None, *args, **kwargs):
        """
        Resets the environment to an initial state.
        Returns the initial observation.
        """
        timestep = self.remote_environment.reset()

        observation = timestep.observation["observation"]

        return observation, {}

    def render(self):
        """
        Renders the environment.
        """
        # TODO: Render remote environments. Currently not possible.

    # FIXME: Does not trigger in main-file, investigate.
    def __del__(self):
        self.connection.send(dm_env_rpc_pb2.DestroyWorldRequest(world_name=self.world_name))
        self.connection.close()
        self.remote_environment.__del__()
