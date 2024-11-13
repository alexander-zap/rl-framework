import copy
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Type

import gymnasium
import numpy as np
from imitation.algorithms import bc
from imitation.algorithms.base import DemonstrationAlgorithm
from imitation.data.rollout import flatten_trajectories
from imitation.data.types import TrajectoryWithRew

from rl_framework.agent.imitation.imitation_learning_agent import ILAgent
from rl_framework.util import Connector


class ImitationAgent(ILAgent):
    @property
    def algorithm(self) -> DemonstrationAlgorithm:
        return self._algorithm

    @algorithm.setter
    def algorithm(self, value: DemonstrationAlgorithm):
        self._algorithm = value

    def __init__(
        self,
        algorithm_class: Type[DemonstrationAlgorithm] = bc.BC,
        algorithm_parameters: Dict = None,
    ):
        """
        Initialize an agent which will trained on one of imitation algorithms.

        Args:
            algorithm_class (Type[BaseAlgorithm]): SB3 RL algorithm class. Specifies the algorithm for RL training.
                Defaults to PPO.
            algorithm_parameters (Dict): Parameters / keyword arguments for the specified imitation algorithm class.
                See https://imitation.readthedocs.io/en/latest/_api/imitation.algorithms.base.html for details on
                    common params.
                See individual docs (e.g., https://imitation.readthedocs.io/en/latest/algorithms/bc.html)
                for algorithm-specific params.
        """
        self.algorithm_class: Type[DemonstrationAlgorithm] = algorithm_class
        self.algorithm_parameters = algorithm_parameters if algorithm_parameters else {}
        self.algorithm = None

    def train(
        self,
        total_timesteps: int,
        connector: Connector,
        episode_sequences: List[List[Tuple[object, object, object, float, bool, bool, dict]]] = None,
        training_environments: List[gymnasium.Env] = None,
        *args,
        **kwargs,
    ):
        """
        Train the instantiated agent on a list of trajectories.

        This training is done by using imitation learning policies, provided by the imitation library.

        The model is changed in place, therefore the updated model can be accessed in the `.model` attribute
        after the agent has been trained.

        Args:
            total_timesteps (int): Amount of (recorded) timesteps to train the agent on.
            episode_sequences (List): List of episode sequences on which the agent should be trained on.
            training_environments (List): List of environments
                Required for interaction or attribute extraction (e.g., action/observation space) for some algorithms
            connector (Connector): Connector for executing callbacks (e.g., logging metrics and saving checkpoints)
                on training time. Calls need to be declared manually in the code.
        """

        def convert_sequences_to_transitions(sequences):
            trajectories = []
            for episode_sequence in sequences:
                observations, actions, next_observations, rewards, terminations, truncations, infos = (
                    np.array(x) for x in list(zip(*episode_sequence))
                )
                all_observations = np.vstack([copy.deepcopy(observations), copy.deepcopy(next_observations[-1])])
                episode_trajectory = TrajectoryWithRew(
                    obs=all_observations, acts=actions, rews=rewards, infos=infos, terminal=terminations[-1]
                )
                trajectories.append(episode_trajectory)

            transitions = flatten_trajectories(trajectories)
            return transitions

        if not episode_sequences:
            raise ValueError("No transitions have been provided to the train-method.")

        if not self.algorithm:
            if self.algorithm_class == bc.BC:
                assert len(training_environments) > 0, "Behavioral Cloning requires training environment to be passed."
                env = training_environments[0]
                self.algorithm_parameters.update(
                    {
                        "observation_space": env.observation_space,
                        "action_space": env.action_space,
                        "rng": np.random.default_rng(0),
                    }
                )

            self.algorithm: DemonstrationAlgorithm = self.algorithm_class(
                demonstrations=None, **self.algorithm_parameters
            )

        transitions = convert_sequences_to_transitions(episode_sequences)

        self.algorithm.set_demonstrations(transitions)
        self.algorithm.train(n_epochs=1)

    def choose_action(self, observation: object, deterministic: bool = False, *args, **kwargs):
        """
        Chooses action which the agent will perform next, according to the observed environment.

        Args:
            observation (object): Observation of the environment
            deterministic (bool): Whether the action should be determined in a deterministic or stochastic way.

        Returns: action (int): Action to take according to policy.

        """

        if not self.algorithm:
            raise ValueError("Cannot predict action for uninitialized agent. Start a training first to initialize.")

        # SB3 model expects multiple observations as input and will output an array of actions as output
        (
            action,
            _,
        ) = self.algorithm.policy.predict(
            [observation],
            deterministic=deterministic,
        )
        return action[0]

    def save_to_file(self, file_path: Path, *args, **kwargs) -> None:
        """Save the agent to a folder (for later loading).

        Args:
            file_path (Path): The file where the agent should be saved to (SB3 expects a file name ending with .zip).
        """
        with open(file_path, "wb") as f:
            pickle.dump(self.algorithm, f)

    def load_from_file(self, file_path: Path, algorithm_parameters: Dict = None, *args, **kwargs) -> None:
        """Load the agent in-place from an agent-save folder.

        Args:
            file_path (Path): The file path the agent has been saved to before.
            algorithm_parameters: Parameters to be set for the loaded algorithm.
        """

        with open(file_path, "rb") as f:
            self.algorithm: DemonstrationAlgorithm = pickle.load(f)

            if self.algorithm_class == bc.BC:
                self.algorithm._bc_logger = bc.BCLogger(self.algorithm.logger)

        if algorithm_parameters:
            self.algorithm_parameters.update(**algorithm_parameters)
