import logging
import sys

from rl_framework.agent import CustomAgent, CustomAlgorithm
from rl_framework.environment.gym_environment import GymEnvironmentWrapper
from rl_framework.environment.remote_gym_environment import remote_environment
from rl_framework.util import (
    HuggingFaceConnector,
    HuggingFaceDownloadConfig,
    HuggingFaceUploadConfig,
    evaluate_agent,
)

# Create logging handler to output logs to stdout
root = logging.getLogger()
root.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)

# TODO: Use configs instead of manual setting of variables in scripts

ENV_ID = "Taxi-v3"
# FIXME: This should be set by config and should be used for automatic setting of algorithm
MODEL_ARCHITECTURE = "QLearning"
PARALLEL_ENVIRONMENTS = 32

DOWNLOAD_EXISTING_AGENT = False
REPO_ID = f"zap-thamm/{MODEL_ARCHITECTURE}-{ENV_ID}"
COMMIT_MESSAGE = f"Upload of a new agent trained with {MODEL_ARCHITECTURE} on {ENV_ID}"

N_TRAINING_TIMESTEPS = 100000
N_EVALUATION_EPISODES = 100

if __name__ == "__main__":
    # Create environment(s); multiple environments for parallel training
    environments = [
        remote_environment(GymEnvironmentWrapper)(ENV_ID, render_mode="rgb_array") for _ in range(PARALLEL_ENVIRONMENTS)
    ]

    # Print some environment information (observation and action space)
    print("_____OBSERVATION SPACE_____ \n")
    print("Observation Space Shape", environments[0].observation_space.shape)
    print("Observation Space N", environments[0].observation_space.n)
    print("Sample observation", environments[0].observation_space.sample())

    print("\n _____ACTION SPACE_____ \n")
    print("Action Space Shape", environments[0].action_space.shape)
    print("Action Space N", environments[0].action_space.n)
    print("Action Space Sample", environments[0].action_space.sample())

    print("\n _____REWARD RANGE_____ \n")
    print("Reward Range Interval", environments[0].reward_range)

    seeds = None

    connector = HuggingFaceConnector()
    upload_connector_config = HuggingFaceUploadConfig(
        repository_id=REPO_ID,
        environment_name=ENV_ID,
        file_name="algorithm.pkl",
        model_architecture=MODEL_ARCHITECTURE,
        commit_message=COMMIT_MESSAGE,
        n_eval_episodes=50,
    )
    download_connector_config = HuggingFaceDownloadConfig(repository_id=REPO_ID, file_name="algorithm.pkl")

    # Create new agent
    agent = CustomAgent(
        algorithm=CustomAlgorithm.Q_LEARNING,
        algorithm_parameters={
            "alpha": 0.1,
            "gamma": 0.99,
            "epsilon": 1.0,
            "epsilon_min": 0.05,
            "n_actions": environments[0].action_space.n,
            "n_observations": environments[0].observation_space.n,
            "randomize_q_table": False,
        },
    )

    if DOWNLOAD_EXISTING_AGENT:
        # Download existing agent from repository
        agent.download(connector=connector, connector_config=download_connector_config)

    else:
        # Train agent
        agent.train(training_environments=environments, total_timesteps=N_TRAINING_TIMESTEPS)

    # Evaluate the model
    mean_reward, std_reward = evaluate_agent(
        agent=agent, evaluation_environment=environments[0], n_eval_episodes=N_EVALUATION_EPISODES, seeds=seeds
    )
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    # TODO: Find another way to log hyperparams
    model_dictionary = {
        "env_id": ENV_ID,
        "max_steps": 99,
        "n_training_timesteps": N_TRAINING_TIMESTEPS,
        "n_eval_episodes": N_EVALUATION_EPISODES,
        "eval_seed": seeds,
        "learning_rate": 0.1,
        "gamma": 0.99,
        "max_epsilon": 1.0,
        "min_epsilon": 0.05,
    }

    # Upload the model
    agent.upload(
        connector=connector,
        connector_config=upload_connector_config,
        evaluation_environment=environments[0],
    )
