from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pettingzoo
from tqdm import tqdm

from rl_framework.util.saving_and_loading import Connector


class Agent(ABC):
    @property
    @abstractmethod
    def algorithm(self):
        return NotImplementedError

    @abstractmethod
    def __init__(self, algorithm, algorithm_parameters: Dict, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def train(
        self,
        training_environments: List[Union[gym.Env, pettingzoo.ParallelEnv]],
        total_timesteps: int,
        connector: Connector,
        *args,
        **kwargs,
    ):
        raise NotImplementedError

    def evaluate(
        self,
        evaluation_environment: Union[gym.Env, pettingzoo.ParallelEnv],
        n_eval_episodes: int,
        seeds: Optional[List[int]] = None,
        deterministic: bool = False,
    ) -> Tuple:
        """
        Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.

        Args:
            evaluation_environment (gym.Env or pettingzoo.ParallelEnv): The evaluation environment.
            n_eval_episodes (int): Number of episode to evaluate the agent.
            seeds (Optional[List[int]]): List of seeds for evaluations.
                No seed is used if not provided or fewer seeds are provided then n_eval_episodes.
            deterministic (bool): Whether the agents' actions should be determined in a deterministic or stochastic way.
        """

        if seeds is None:
            seeds = []
        episode_rewards = []
        for episode in tqdm(range(n_eval_episodes)):
            seed = seeds[episode] if episode < len(seeds) else None
            episode_reward = 0

            if isinstance(evaluation_environment, pettingzoo.ParallelEnv):
                prev_observations, _ = evaluation_environment.reset(seed=seed)
                prev_actions = {
                    agent: self.choose_action(prev_observations[agent], deterministic=deterministic)
                    for agent in evaluation_environment.agents
                }

                while True:
                    (
                        observations,
                        rewards,
                        terminations,
                        truncations,
                        infos,
                    ) = evaluation_environment.step(prev_actions)
                    done = any(terminations.values()) or any(truncations.values())
                    # next action to be executed (based on new observation)
                    actions = {
                        agent: self.choose_action(observations[agent], deterministic=deterministic)
                        for agent in evaluation_environment.agents
                    }
                    episode_reward += sum(rewards.values())
                    prev_actions = actions

                    if done:
                        episode_rewards.append(episode_reward)
                        break

            else:
                prev_observation, _ = evaluation_environment.reset(seed=seed)
                prev_action = self.choose_action(prev_observation, deterministic=deterministic)

                while True:
                    (
                        observation,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = evaluation_environment.step(prev_action)
                    done = terminated or truncated
                    # next action to be executed (based on new observation)
                    action = self.choose_action(observation, deterministic=deterministic)
                    episode_reward += reward
                    prev_action = action

                    if done:
                        episode_rewards.append(episode_reward)
                        break

            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            return mean_reward, std_reward

    @abstractmethod
    def choose_action(self, observation: object, deterministic: bool, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def save_to_file(self, file_path: Path, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_from_file(self, file_path: Path, algorithm_parameters: Optional[Dict], *args, **kwargs) -> None:
        raise NotImplementedError

    def upload(
        self,
        connector: Connector,
        evaluation_environment: gym.Env,
        variable_values_to_log: Dict,
    ) -> None:
        """
        Evaluate and upload the decision-making agent (and its .algorithm attribute) to the connector.
            Additional option: Generate a video of the agent interacting with the environment.

        Args:
            connector: Connector for uploading.
            evaluation_environment: Environment used for final evaluation and clip creation before upload.
            variable_values_to_log (Dict): Variable name and values to be uploaded and logged, e.g. evaluation metrics.
        """
        connector.upload(
            agent=self, evaluation_environment=evaluation_environment, variable_values_to_log=variable_values_to_log
        )

    def download(
        self,
        connector: Connector,
        algorithm_parameters: Optional[Dict] = None,
    ):
        """
        Download a previously saved decision-making agent from the connector and replace the `self` agent instance
            in-place with the newly downloaded saved-agent.

        NOTE: Agent and Algorithm class need to be the same as the saved agent.

        Args:
            connector: Connector for downloading.
            algorithm_parameters (Optional[Dict]): Parameters to be set for the downloaded agent.
        """

        # Get the model from the Hub, download and cache the model on your local disk
        agent_file_path = connector.download()
        self.load_from_file(agent_file_path, algorithm_parameters)
