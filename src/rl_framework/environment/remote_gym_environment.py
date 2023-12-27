from concurrent import futures
from typing import SupportsFloat, Tuple, Optional

import grpc
from dm_env import StepType
from dm_env_rpc.v1 import connection as dm_env_rpc_connection
from dm_env_rpc.v1 import dm_env_adaptor
from dm_env_rpc.v1 import dm_env_rpc_pb2
from dm_env_rpc.v1 import dm_env_rpc_pb2_grpc
from dm_env_rpc.v1 import spec_manager
from dm_env_rpc.v1 import tensor_spec_utils
from google.rpc import code_pb2
from google.rpc import status_pb2
from gymnasium.spaces import Discrete

from rl_framework.environment import Environment, GymEnvironmentWrapper


# TODO: Think about using class decorators instead


class RemoteGymEnvironment(Environment):
    """
    Base class implementation of remote environments, from which every remote environment should be inherited.
    Implements central environment-side management for the open communication interface between the Environment class
        and remotely running games.
    This class is not abstract and the children classes should therefore execute __super__ calls for inherited methods
        additionally (next to own implementation of environment details).
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

    def __init__(self, *args, **kwargs):
        """
        Initialize the environment.
        """

        def start_server():
            """Starts the Catch gRPC server."""
            server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
            servicer = RemoteEnvironmentService()
            dm_env_rpc_pb2_grpc.add_EnvironmentServicer_to_server(servicer, server)

            port = server.add_secure_port('localhost:0', grpc.local_server_credentials())
            server.start()
            return server, port

        self.server, self.port = start_server()

        self.connection = dm_env_rpc_connection.create_secure_channel_and_connect(f'localhost:{self.port}')
        self.remote_env, self.world_name = dm_env_adaptor.create_and_join_world(self.connection,
                                                                                create_world_settings={},
                                                                                join_world_settings={})

        # FIXME: Hardcoded; instead, take from environment
        self._action_space = Discrete(6)
        self._observation_space = Discrete(500)
        self._reward_range = (-float('inf'), float('inf'))
        # TODO: Render remote environments. Currently not possible.
        self._render_mode = None

    def step(self, action, *args, **kwargs) -> Tuple[object, SupportsFloat, bool, bool, dict]:
        """
        Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        """
        timestep = self.remote_env.step({"action": action})

        observation = timestep.observation["observation"]
        reward = timestep.reward
        terminated = True if timestep.step_type is StepType.LAST else False
        truncated = False

        return observation, reward, terminated, truncated, {}

    def reset(self, seed: Optional[int] = None, *args, **kwargs):
        """
        Resets the environment to an initial state.
        Returns the initial observation.
        """
        timestep = self.remote_env.reset()

        observation = timestep.observation["observation"]

        return observation, {}

    def render(self):
        """
        Renders the environment.
        """
        pass

    def __del__(self):
        self.connection.send(dm_env_rpc_pb2.DestroyWorldRequest(world_name=self.world_name))
        self.server.stop(None)


class RemoteEnvironmentService(dm_env_rpc_pb2_grpc.EnvironmentServicer):
    """Runs the environment as a gRPC EnvironmentServicer."""

    def __init__(self):
        # FIXME: Hardcoded
        self.env = GymEnvironmentWrapper("Taxi-v3", render_mode="rgb_array")

    def Process(self, request_iterator, context):
        """Processes incoming EnvironmentRequests.

        For each EnvironmentRequest the internal message is extracted and handled.
        The response for that message is then placed in a EnvironmentResponse which
        is returned to the client.

        An error status will be returned if an unknown message type is received or
        if the message is invalid for the current world state.


        Args:
          request_iterator: Message iterator provided by gRPC.
          context: Context provided by gRPC.

        Yields:
          EnvironmentResponse: Response for each incoming EnvironmentRequest.
        """

        observation_spec = {
            1:
                dm_env_rpc_pb2.TensorSpec(
                    name="observation",
                    # FIXME: Hardcoded (Discrete space equals [] shape)
                    shape=[],
                    dtype=dm_env_rpc_pb2.INT32),
            2:
                dm_env_rpc_pb2.TensorSpec(
                    name="reward",
                    dtype=dm_env_rpc_pb2.FLOAT)
        }
        tensor_spec_utils.set_bounds(
            observation_spec[2],
            minimum=self.env.reward_range[0],
            maximum=self.env.reward_range[1],
        )
        action_spec = {
            1:
                dm_env_rpc_pb2.TensorSpec(
                    name="action",
                    # FIXME: Hardcoded (Discrete space equals [] shape)
                    shape=[],
                    dtype=dm_env_rpc_pb2.INT8)
        }

        action_manager = spec_manager.SpecManager(action_spec)
        observation_manager = spec_manager.SpecManager(observation_spec)
        environment_should_be_reset = False

        for request in request_iterator:
            environment_response = dm_env_rpc_pb2.EnvironmentResponse()

            try:
                message_type = request.WhichOneof('payload')
                internal_request = getattr(request, message_type)

                if message_type == 'create_world':
                    environment_should_be_reset = True
                    response = dm_env_rpc_pb2.CreateWorldResponse(world_name="world")

                elif message_type == 'join_world':
                    environment_should_be_reset = True
                    response = dm_env_rpc_pb2.JoinWorldResponse()
                    for uid, action_space in action_spec.items():
                        response.specs.actions[uid].CopyFrom(action_space)
                    for uid, observation_space in observation_spec.items():
                        response.specs.observations[uid].CopyFrom(observation_space)

                elif message_type == 'step':
                    if environment_should_be_reset:
                        observation, info = self.env.reset()
                        reward = 0.0
                        terminated = False
                        truncated = False
                        environment_should_be_reset = False
                    else:
                        unpacked_actions = action_manager.unpack(internal_request.actions)
                        action = unpacked_actions.get("action")
                        observation, reward, terminated, truncated, info = self.env.step(action)

                    response = dm_env_rpc_pb2.StepResponse()
                    packed_observations = observation_manager.pack({
                        "observation": observation,
                        "reward": reward
                    })

                    for requested_observation in internal_request.requested_observations:
                        response.observations[requested_observation].CopyFrom(
                            packed_observations[requested_observation])
                    if terminated or truncated:
                        response.state = dm_env_rpc_pb2.EnvironmentStateType.TERMINATED
                        environment_should_be_reset = True
                    else:
                        response.state = dm_env_rpc_pb2.EnvironmentStateType.RUNNING

                elif message_type == 'reset':
                    environment_should_be_reset = True
                    response = dm_env_rpc_pb2.ResetResponse()
                    for uid, action_space in action_spec.items():
                        response.specs.actions[uid].CopyFrom(action_space)
                    for uid, observation_space in observation_spec.items():
                        response.specs.observations[uid].CopyFrom(observation_space)

                elif message_type == 'reset_world':
                    environment_should_be_reset = True
                    response = dm_env_rpc_pb2.ResetWorldResponse()

                elif message_type == 'leave_world':
                    response = dm_env_rpc_pb2.LeaveWorldResponse()

                elif message_type == 'destroy_world':
                    self.env = None
                    response = dm_env_rpc_pb2.DestroyWorldResponse()

                else:
                    raise RuntimeError('Unhandled message: {}'.format(message_type))

                getattr(environment_response, message_type).CopyFrom(response)

            except Exception as e:  # pylint: disable=broad-except
                environment_response.error.CopyFrom(
                    status_pb2.Status(code=code_pb2.INTERNAL, message=str(e)))

            yield environment_response
