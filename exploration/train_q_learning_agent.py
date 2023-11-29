from rl_framework.agent import CustomAgent, CustomAlgorithm
from rl_framework.environment.gym_environment import GymEnvironmentWrapper
from rl_framework.util.util import evaluate_agent

ENV_ID = "Taxi-v3"
MODEL_ARCHITECTURE = "QLearning"
PARALLEL_ENVIRONMENTS = 32

DOWNLOAD_EXISTING_AGENT = False
MODEL_NAME = f"{MODEL_ARCHITECTURE}-{ENV_ID}"
REPO_ID = f"zap-thamm/{MODEL_NAME}"
COMMIT_MESSAGE = f"Upload of a new agent trained with {MODEL_ARCHITECTURE} on {ENV_ID}"

N_TRAINING_EPISODES = 10000
N_EVALUATION_EPISODES = 100

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
MAX_EPSILON = 1.0
MIN_EPSILON = 0.05

if __name__ == "__main__":
    # Create environment
    environment = GymEnvironmentWrapper(ENV_ID, render_mode="rgb_array")

    # Print some environment information (observation and action space)
    print("_____OBSERVATION SPACE_____ \n")
    print("Observation Space Shape", environment.observation_space.shape)
    print("Sample observation", environment.observation_space.sample())

    print("\n _____ACTION SPACE_____ \n")
    print("Action Space Shape", environment.action_space.n)
    print("Action Space Sample", environment.action_space.sample())

    print("\n _____REWARD RANGE_____ \n")
    print("Reward Range Interval", environment.reward_range)

    seeds = None

    # Create new agent
    algorithm_parameters = {
        "alpha": LEARNING_RATE,
        "gamma": DISCOUNT_FACTOR,
        "epsilon": MAX_EPSILON,
        "epsilon_min": MIN_EPSILON,
        "n_actions": environment.action_space.n,
        "n_observations": environment.observation_space.n,
        "randomize_q_table": False,
    }
    agent = CustomAgent(algorithm=CustomAlgorithm.Q_LEARNING, algorithm_parameters=algorithm_parameters)

    if DOWNLOAD_EXISTING_AGENT:
        agent.download_from_huggingface_hub(repository_id=REPO_ID, filename="q-learning.pkl")

    else:
        # Train agent
        agent.train(training_environments=[environment], n_episodes=N_TRAINING_EPISODES)

    mean_reward, std_reward = evaluate_agent(agent=agent,
                                             evaluation_environment=environment,
                                             n_eval_episodes=N_EVALUATION_EPISODES,
                                             seeds=seeds)
    print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    model_dictionary = {
        "env_id": ENV_ID,
        "max_steps": 99,
        "n_training_episodes": N_TRAINING_EPISODES,
        "n_eval_episodes": N_EVALUATION_EPISODES,
        "eval_seed": seeds,
        "learning_rate": LEARNING_RATE,
        "gamma": DISCOUNT_FACTOR,
        "max_epsilon": MAX_EPSILON,
        "min_epsilon": MIN_EPSILON,
        "qtable": agent.algorithm.q_table
    }

    agent.upload_to_huggingface_hub(repository_id=REPO_ID,
                                    environment=environment,
                                    environment_name=ENV_ID,
                                    evaluation_seeds=seeds,
                                    model_dictionary=model_dictionary)
