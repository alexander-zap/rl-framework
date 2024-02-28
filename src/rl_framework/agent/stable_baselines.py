import tempfile
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from rl_framework.agent import Agent
from rl_framework.environment import Environment
from rl_framework.util import Connector


class StableBaselinesAlgorithm(Enum):
    A2C = A2C
    DDPG = DDPG
    DQN = DQN
    PPO = PPO
    SAC = SAC
    TD3 = TD3


class StableBaselinesAgent(Agent):
    @property
    def algorithm(self) -> Optional[BaseAlgorithm]:
        return self._algorithm

    @algorithm.setter
    def algorithm(self, value: Optional[BaseAlgorithm]):
        self._algorithm = value

    def __init__(
        self,
        algorithm: StableBaselinesAlgorithm = StableBaselinesAlgorithm.PPO,
        algorithm_parameters: Dict = None,
    ):
        """
        Initialize an agent which will trained on one of Stable-Baselines3 algorithms.

        Args:
            algorithm (StableBaselinesAlgorithm): Enum with values being SB3 RL Algorithm classes.
                Specifies the algorithm for RL training.
                Defaults to PPO.
            algorithm_parameters (Dict): Parameters / keyword arguments for the specified SB3 RL Algorithm class.
                See https://stable-baselines3.readthedocs.io/en/master/modules/base.html for details on common params.
                See individual docs (e.g., https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
                for algorithm-specific params.
        """
        self.algorithm_class = algorithm.value

        algorithm_parameters = self._add_required_default_parameters(algorithm_parameters)

        self.algorithm: Optional[BaseAlgorithm] = None
        self.algorithm_builder: Callable[..., BaseAlgorithm] = partial(self.algorithm_class, **algorithm_parameters)

    def train(
        self,
        training_environments: List[Environment],
        total_timesteps: int = 100000,
        logging_connector: Optional[Connector] = None,
        *args,
        **kwargs,
    ):
        """
        Train the instantiated agent on the environment.

        This training is done by using the agent-on-environment training method provided by Stable-baselines3.

        The model is changed in place, therefore the updated model can be accessed in the `.model` attribute
        after the agent has been trained.

        Args:
            training_environments (List[Environment]): Environment on which the agent should be trained on.
                If n_environments is set above 1, multiple environments enables parallel training of an agent.
            total_timesteps (int): Amount of individual steps the agent should take before terminating the training.
            logging_connector (Connector): Connector for logging metrics on training time.
                Logging is executed by calling the connector.log method. Calls need to be declared manually in the code.
        """

        class LoggingCallback(BaseCallback):
            """
            A custom callback that derives from ``BaseCallback``.

            :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
            """

            def __init__(self, verbose=0):
                super().__init__(verbose)
                self.episode_reward = 0

            def _on_step(self) -> bool:
                """
                This method will be called by the model after each call to `env.step()`.

                For child callback (of an `EventCallback`), this will be called
                when the event is triggered.

                :return: If the callback returns False, training is aborted early.
                """
                # Only calculate for first environment (at index 0)
                self.episode_reward += self.locals["rewards"][0]
                done = self.locals["dones"][0]
                if done:
                    logging_connector.log_value(self.num_timesteps, self.episode_reward, "Episode reward")
                    self.episode_reward = 0

                return True

        environment_iterator = iter(training_environments)
        training_env = make_vec_env(
            lambda: next(environment_iterator),
            n_envs=len(training_environments),
        )

        self.algorithm = self.algorithm_builder(env=training_env)

        self.algorithm.learn(total_timesteps=total_timesteps, callback=LoggingCallback())

    def choose_action(self, observation: object, *args, **kwargs):
        """
        Chooses action which the agent will perform next, according to the observed environment.

        Args:
            observation (object): Observation of the environment

        Returns: action (int): Action to take according to policy.

        """

        # SB3 model expects multiple observations as input and will output an array of actions as output
        (
            action,
            _,
        ) = self.algorithm.predict(
            [observation],
            deterministic=True,
        )
        return action[0]

    def save_to_file(self, file_path: Path, *args, **kwargs) -> None:
        """Save the agent to a folder (for later loading).

        Args:
            file_path (Path): The file where the agent should be saved to (SB3 expects a file name ending with .zip).
        """
        self.algorithm.save(file_path)

    def load_from_file(self, file_path: Path, algorithm_parameters: Dict = None, *args, **kwargs) -> None:
        """Load the agent in-place from an agent-save folder.

        Args:
            file_path (Path): The model filename (file ending with .zip).
            algorithm_parameters: Parameters to be set for the loaded algorithm.
        """
        algorithm_parameters = self._add_required_default_parameters(algorithm_parameters)
        self.algorithm_builder = partial(
            self.algorithm_class.load, path=file_path, custom_objects=algorithm_parameters, print_system_info=True
        )

    @staticmethod
    def _add_required_default_parameters(algorithm_parameters: Optional[Dict]):
        """
        Add missing required parameters to `algorithm_parameters`.
        Required parameters currently are:
            - "policy": needs to be set for every BaseRLAlgorithm. Set to "MlpPolicy" if not provided.
            - "tensorboard_log": needs to be set for logging callbacks. Set to newly created temp dir if not provided.

        Args:
            algorithm_parameters (Optional[Dict]): Parameters passed by user (in .__init__ or .load_from_file).

        Returns:
            algorithm_parameters (Dict): Parameter dictionary with filled up default parameter entries

        """
        if algorithm_parameters is None:
            algorithm_parameters = {"policy": "MlpPolicy"}

        # Existing tensorboard log paths can be used (e.g., for continuing training of downloaded agents).
        # If not provided, tensorboard will be logged to newly created temp dir.
        if "tensorboard_log" not in algorithm_parameters:
            tensorboard_log_path = tempfile.mkdtemp()
            algorithm_parameters.update({"tensorboard_log": tensorboard_log_path})

        return algorithm_parameters
