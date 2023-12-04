import logging
import pickle
import random
from typing import List, Text

import numpy as np
from tqdm import tqdm

from rl_framework.agent.custom_algorithms.base_algorithm import Algorithm
from rl_framework.environment import Environment


class QLearning(Algorithm):
    @property
    def q_table(self):
        return self._q_table

    @q_table.setter
    def q_table(self, value):
        self._q_table = value

    def __init__(
        self,
        n_actions: int,
        n_observations: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        randomize_q_table: bool = True,
    ):
        """
        Initialize an Q-Learning agent which will be trained.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.n_actions = n_actions

        if randomize_q_table:
            self.q_table = np.random.random_sample((n_observations, n_actions)) * 0.1
        else:
            self.q_table = np.full((n_observations, n_actions), 0.0)

    def _update_q_table(
        self,
        prev_observation: object,
        prev_action: int,
        observation: object,
        reward: float,
    ):
        """
        Update _q_table based on previous observation, previous action, new observation and received reward

        Args:
            prev_observation (object): Previous observation (St)
            prev_action (in): Previous action (at)
            observation (object): New observation (St+1) after executing action at in state St
            reward (float): Reward for executing action at in state St

        """
        q_old = self._q_table[prev_observation, prev_action]
        q_new = (1 - self.alpha) * q_old + self.alpha * (reward + self.gamma * np.max(self._q_table[observation]))
        self._q_table[prev_observation, prev_action] = q_new

    def _update_epsilon(self, n_episodes: int):
        """
        Gradually reduce epsilon after every done episode

        Args:
            n_episodes (int): Number of episodes (information required to reduce epsilon steadily.

        """
        self.epsilon = self.epsilon - 2 / n_episodes if self.epsilon > self.epsilon_min else self.epsilon_min

    def choose_action(self, observation: object, *args, **kwargs) -> int:
        """
        Chooses action which the agent will perform next, according to the observed environment.

        Args:
            observation (object): Observation of the environment

        Returns: action (int): Action to take according to policy.

        """

        return np.argmax(self._q_table[observation])

    def train(
        self,
        training_environments: List[Environment],
        n_episodes: int = 10000,
        *args,
        **kwargs,
    ):
        """
        Train the instantiated agent on the environment.

        This training is done by using the Q-Learning method.

        The Q-table is changed in place, therefore the updated Q-table can be accessed in the `.q_table` attribute
        after the agent has been trained.

        Args:
            training_environments (List[Environment]): List of environments on which the agent should be trained on.
                # NOTE: This class only supports training on one environment
            n_episodes (int): Number of episodes the agent should train for before terminating the training.
        """

        # TODO: Exploration-exploitation strategy is currently hard-coded as epsilon-greedy.
        #   Pass exploration-exploitation strategy from outside
        def choose_action_according_to_exploration_exploitation_strategy(obs):
            greedy_action = self.choose_action(obs)
            # Choose random action with probability epsilon
            if random.random() < self.epsilon:
                return random.randrange(self.n_actions)
            # Greedy action is chosen with probability (1 - epsilon)
            else:
                return greedy_action

        if len(training_environments) > 1:
            logging.info(
                f"Reinforcement Learning algorithm {self.__class__.__qualname__} does not support "
                f"training on multiple environments in parallel. Continuing with one environment as "
                f"training environment."
            )

        training_environment = training_environments[0]

        for _ in tqdm(range(n_episodes)):
            episode_reward = 0
            prev_observation, _ = training_environment.reset()
            prev_action = choose_action_according_to_exploration_exploitation_strategy(prev_observation)

            while True:
                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = training_environment.step(prev_action)
                done = terminated or truncated
                action = choose_action_according_to_exploration_exploitation_strategy(observation)
                episode_reward += reward
                self._update_q_table(prev_observation, prev_action, observation, reward)

                prev_observation = observation
                prev_action = action

                if done:
                    self._update_epsilon(n_episodes)
                    break

    def save(self, file_path: Text, *args, **kwargs):
        """
        Save the action-prediction model (Q-Table) of the agent to pickle file.

        Args:
            file_path (Text): Path where the model should be saved to.
        """
        with open(file_path, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, file_path: Text, *args, **kwargs):
        """
        Load the action-prediction model (Q-Table) from a previously created (by the .save function) pickle file.

         Args:
            file_path (Text): Path where the model has been previously saved to.
        """
        with open(file_path, "rb") as f:
            self.q_table = pickle.load(f)
