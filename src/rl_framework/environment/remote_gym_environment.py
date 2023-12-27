from concurrent import futures
from typing import Optional, SupportsFloat, Tuple

import grpc
from dm_env import StepType
from dm_env_rpc.v1 import connection as dm_env_rpc_connection
from dm_env_rpc.v1 import (
    dm_env_adaptor,
    dm_env_rpc_pb2,
    dm_env_rpc_pb2_grpc,
    spec_manager,
    tensor_spec_utils,
)
from google.rpc import code_pb2, status_pb2

from rl_framework.environment import Environment


def remote_environment(cls: type):
    """
    Decorator implementation of remote environments, with which every environment can be transformed to a remote one.
    Implements central environment-side management for the open communication interface between the Environment class
        and remotely running games.

    Args:
        cls: Environment class which should be extended with remote environment functionality.

    Returns:
        Decorated environment class.

    """

    class RemoteEnvironment(cls, Environment):
        def __init__(self, *args, **kwargs):
            """
            Initialize the environment.
            """

            def start_server(local_environment: Environment):
                """Starts the Catch gRPC server."""
                server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
                servicer = RemoteEnvironmentService(environment=local_environment)
                dm_env_rpc_pb2_grpc.add_EnvironmentServicer_to_server(servicer, server)

                port = server.add_secure_port("localhost:0", grpc.local_server_credentials())
                server.start()
                return server, port

            # Instantiate environment locally
            environment = cls(*args, **kwargs)

            # Set local environment attributes for remote wrapper
            self._action_space = environment.action_space
            self._observation_space = environment.observation_space
            self._reward_range = environment.reward_range
            # TODO: Render remote environments. Currently not possible.
            self._render_mode = None

            # Pass locally instantiated environment to gRPC server.
            self.server, self.port = start_server(local_environment=environment)

            self.connection = dm_env_rpc_connection.create_secure_channel_and_connect(f"localhost:{self.port}")
            self.remote_environment, self.world_name = dm_env_adaptor.create_and_join_world(
                self.connection, create_world_settings={}, join_world_settings={}
            )

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

        # FIXME: Does not trigger in main-file, investigate.
        def __del__(self):
            self.remote_environment.__del__()
            self.connection.send(dm_env_rpc_pb2.DestroyWorldRequest(world_name=self.world_name))
            super(self.connection).close()
            self.connection.close()
            self.server.stop(None)

    return RemoteEnvironment


class RemoteEnvironmentService(dm_env_rpc_pb2_grpc.EnvironmentServicer):
    """Runs the environment as a gRPC EnvironmentServicer."""

    def __init__(self, environment: Environment):
        self.environment = environment

        self.observation_spec = {
            1: dm_env_rpc_pb2.TensorSpec(
                name="observation",
                # FIXME: Hardcoded (Discrete space equals [] shape; variable dtype)
                shape=[],
                dtype=dm_env_rpc_pb2.INT32,
            ),
            2: dm_env_rpc_pb2.TensorSpec(name="reward", dtype=dm_env_rpc_pb2.FLOAT),
        }
        tensor_spec_utils.set_bounds(
            self.observation_spec[2],
            minimum=self.environment.reward_range[0],
            maximum=self.environment.reward_range[1],
        )

        self.action_spec = {
            1: dm_env_rpc_pb2.TensorSpec(
                name="action",
                # FIXME: Hardcoded (Discrete space equals [] shape; variable dtype)
                shape=[],
                dtype=dm_env_rpc_pb2.INT8,
            )
        }

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

        action_manager = spec_manager.SpecManager(self.action_spec)
        observation_manager = spec_manager.SpecManager(self.observation_spec)
        environment_should_be_reset = False

        for request in request_iterator:
            environment_response = dm_env_rpc_pb2.EnvironmentResponse()

            try:
                message_type = request.WhichOneof("payload")
                internal_request = getattr(request, message_type)

                if message_type == "create_world":
                    environment_should_be_reset = True
                    response = dm_env_rpc_pb2.CreateWorldResponse(world_name="world")

                elif message_type == "join_world":
                    environment_should_be_reset = True
                    response = dm_env_rpc_pb2.JoinWorldResponse()
                    for uid, action_space in self.action_spec.items():
                        response.specs.actions[uid].CopyFrom(action_space)
                    for uid, observation_space in self.observation_spec.items():
                        response.specs.observations[uid].CopyFrom(observation_space)

                elif message_type == "step":
                    if environment_should_be_reset:
                        observation, info = self.environment.reset()
                        reward = 0.0
                        terminated = False
                        truncated = False
                        environment_should_be_reset = False
                    else:
                        unpacked_actions = action_manager.unpack(internal_request.actions)
                        action = unpacked_actions.get("action")
                        observation, reward, terminated, truncated, info = self.environment.step(action)

                    response = dm_env_rpc_pb2.StepResponse()
                    packed_observations = observation_manager.pack({"observation": observation, "reward": reward})

                    for requested_observation in internal_request.requested_observations:
                        response.observations[requested_observation].CopyFrom(
                            packed_observations[requested_observation]
                        )
                    if terminated or truncated:
                        response.state = dm_env_rpc_pb2.EnvironmentStateType.TERMINATED
                        environment_should_be_reset = True
                    else:
                        response.state = dm_env_rpc_pb2.EnvironmentStateType.RUNNING

                elif message_type == "reset":
                    environment_should_be_reset = True
                    response = dm_env_rpc_pb2.ResetResponse()
                    for uid, action_space in self.action_spec.items():
                        response.specs.actions[uid].CopyFrom(action_space)
                    for uid, observation_space in self.observation_spec.items():
                        response.specs.observations[uid].CopyFrom(observation_space)

                elif message_type == "reset_world":
                    environment_should_be_reset = True
                    response = dm_env_rpc_pb2.ResetWorldResponse()

                elif message_type == "leave_world":
                    response = dm_env_rpc_pb2.LeaveWorldResponse()

                elif message_type == "destroy_world":
                    response = dm_env_rpc_pb2.DestroyWorldResponse()

                else:
                    raise RuntimeError("Unhandled message: {}".format(message_type))

                getattr(environment_response, message_type).CopyFrom(response)

            except Exception as e:  # pylint: disable=broad-except
                environment_response.error.CopyFrom(status_pb2.Status(code=code_pb2.INTERNAL, message=str(e)))

            yield environment_response
