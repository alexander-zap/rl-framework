import logging
import sys

from rl_framework.agent import StableBaselinesAgent, StableBaselinesAlgorithm
from rl_framework.environment.gym_environment import GymEnvironmentWrapper
from rl_framework.environment.remote_environment import RemoteEnvironment
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

ENV_ID = "CarRacing-v2"
REMOTE_ENVIRONMENT = False
PORT = 56729

# FIXME: This should be set by config and should be used for automatic setting of algorithm
MODEL_ARCHITECTURE = "Remote-DQN"
PARALLEL_ENVIRONMENTS = 32

DOWNLOAD_EXISTING_AGENT = False
REPO_ID = f"zap-thamm/{MODEL_ARCHITECTURE}-{ENV_ID}"
COMMIT_MESSAGE = f"Upload of a new agent trained with {MODEL_ARCHITECTURE} on {ENV_ID}"

N_TRAINING_TIMESTEPS = 100000
N_EVALUATION_EPISODES = 100

if __name__ == "__main__":
    # Create environment(s); multiple environments for parallel training
    if REMOTE_ENVIRONMENT:
        environments = [RemoteEnvironment(url="localhost", port=PORT)]
    else:
        environments = [GymEnvironmentWrapper(ENV_ID, render_mode="rgb_array") for _ in range(PARALLEL_ENVIRONMENTS)]

    # Print some environment information (observation and action space)
    print("_____OBSERVATION SPACE_____ \n")
    print("Observation Space Shape", environments[0].observation_space.shape)
    print("Sample observation", environments[0].observation_space.sample())

    print("\n _____ACTION SPACE_____ \n")
    print("Action Space Shape", environments[0].action_space.shape)
    print("Action Space Sample", environments[0].action_space.sample())

    print("\n _____REWARD RANGE_____ \n")
    print("Reward Range Interval", environments[0].reward_range)

    seeds = None

    connector = HuggingFaceConnector()
    upload_connector_config = HuggingFaceUploadConfig(
        repository_id=REPO_ID,
        environment_name=ENV_ID,
        file_name="algorithm.zip",
        model_architecture=MODEL_ARCHITECTURE,
        commit_message=COMMIT_MESSAGE,
        n_eval_episodes=50,
    )
    download_connector_config = HuggingFaceDownloadConfig(repository_id=REPO_ID, file_name="algorithm.zip")

    # Create new agent
    agent = StableBaselinesAgent(
        algorithm=StableBaselinesAlgorithm.PPO,
        algorithm_parameters={
            "policy": "MlpPolicy",
            "learning_rate": 0.001,
            # "n_steps": 1024,
            "batch_size": 64,
            # "n_epochs": 4,
            "gamma": 0.999,
            # "gae_lambda": 0.98,
            # "ent_coef": 0.01,
            "verbose": 1,
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

    # Upload the model
    agent.upload(
        connector=connector,
        connector_config=upload_connector_config,
        evaluation_environment=environments[0],
    )
