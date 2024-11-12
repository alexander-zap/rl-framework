from typing import List, Tuple

import gymnasium as gym
import imitation.algorithms.bc
import numpy as np
from clearml import Task
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from rl_framework.agent.imitation import ImitationAgent
from rl_framework.util import (
    ClearMLConnector,
    ClearMLDownloadConfig,
    ClearMLUploadConfig,
)


def sample_expert_sequences(
    environment, timesteps
) -> List[List[Tuple[object, object, object, float, bool, bool, dict]]]:
    def download_expert_policy(venv):
        policy = load_policy(
            "ppo-huggingface",
            organization="HumanCompatibleAI",
            env_name="seals-CartPole-v0",
            venv=venv,
        )
        return policy

    environment_return_functions = [lambda: RolloutInfoWrapper(Monitor(environment))]
    vectorized_environment = DummyVecEnv(env_fns=environment_return_functions)
    expert_policy = download_expert_policy(vectorized_environment)

    rollouts = rollout.rollout(
        expert_policy,
        vectorized_environment,
        rollout.make_sample_until(min_timesteps=timesteps),
        rng=np.random.default_rng(0),
    )

    episode_sequences = []
    for trajectory in rollouts:
        obs = trajectory.obs[:-1]
        acts = trajectory.acts
        rews = trajectory.rews
        next_obs = trajectory.obs[1:]
        terminations = np.zeros(len(trajectory.acts), dtype=bool)
        truncations = np.zeros(len(trajectory.acts), dtype=bool)
        terminations[-1] = trajectory.terminal
        truncations[-1] = not trajectory.terminal
        infos = np.array([{}] * len(trajectory)) if trajectory.infos is None else trajectory.infos
        episode_sequence = list(zip(*[obs, acts, next_obs, rews, terminations, truncations, infos]))
        episode_sequences.append(episode_sequence)

    return episode_sequences


DOWNLOAD_EXISTING_AGENT = False
N_EXPERT_SAMPLING_TIMESTEPS = 20000
N_EVALUATION_EPISODES = 10

if __name__ == "__main__":
    environment = gym.make("seals:seals/CartPole-v0", render_mode="rgb_array")

    # Create connector
    task = Task.init(project_name="synthetic-player")
    upload_connector_config = ClearMLUploadConfig(
        file_name="agent.pkl",
        video_length=0,
    )
    download_connector_config = ClearMLDownloadConfig(
        model_id="", file_name="agent.pkl", download=DOWNLOAD_EXISTING_AGENT
    )
    connector = ClearMLConnector(
        task=task, upload_config=upload_connector_config, download_config=download_connector_config
    )

    # Create new agent
    agent = ImitationAgent(algorithm_class=imitation.algorithms.bc.BC)

    if DOWNLOAD_EXISTING_AGENT:
        # Download existing agent from repository
        agent.download(connector=connector)

    # Train agent
    sequences = sample_expert_sequences(environment, N_EXPERT_SAMPLING_TIMESTEPS)
    agent.train(episode_sequences=sequences, training_environments=[environment], connector=connector)

    # Evaluate the model
    mean_reward, std_reward = agent.evaluate(evaluation_environment=environment, n_eval_episodes=N_EVALUATION_EPISODES)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    # Upload the model
    agent.upload(
        connector=connector,
        evaluation_environment=environment,
        variable_values_to_log={"mean_reward": mean_reward, "std_reward": std_reward},
    )
