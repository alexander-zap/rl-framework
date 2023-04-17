from rl_framework.agent.q_learning import QLearningAgent
from rl_framework.environment.gym_environment import GymEnvironmentWrapper
from rl_framework.util.util import evaluate_agent

ENV_ID = "Taxi-v3"
MODEL_ARCHITECTURE = "QLearning"
PARALLEL_ENVIRONMENTS = 32

DOWNLOAD_EXISTING_AGENT = True
MODEL_NAME = f"{MODEL_ARCHITECTURE}-{ENV_ID}"
REPO_ID = f"zap-thamm/{MODEL_NAME}"
COMMIT_MESSAGE = f"Upload of a new agent trained with {MODEL_ARCHITECTURE} on {ENV_ID}"

N_TRAINING_EPISODES = 100000
N_EVALUATION_EPISODES = 100

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
MAX_EPSILON = 1.0
MIN_EPSILON = 0.05

if __name__ == "__main__":
    # Create environment
    environment = GymEnvironmentWrapper(ENV_ID)

    # Print some environment information (observation and action space)
    print("_____OBSERVATION SPACE_____ \n")
    print("Observation Space Shape", environment.observation_space.shape)
    print("Sample observation", environment.observation_space.sample())

    print("\n _____ACTION SPACE_____ \n")
    print("Action Space Shape", environment.action_space.n)
    print("Action Space Sample", environment.action_space.sample())

    print("\n _____REWARD RANGE_____ \n")
    print("Reward Range Interval", environment.reward_range)

    seeds = [
        16,
        54,
        165,
        177,
        191,
        191,
        120,
        80,
        149,
        178,
        48,
        38,
        6,
        125,
        174,
        73,
        50,
        172,
        100,
        148,
        146,
        6,
        25,
        40,
        68,
        148,
        49,
        167,
        9,
        97,
        164,
        176,
        61,
        7,
        54,
        55,
        161,
        131,
        184,
        51,
        170,
        12,
        120,
        113,
        95,
        126,
        51,
        98,
        36,
        135,
        54,
        82,
        45,
        95,
        89,
        59,
        95,
        124,
        9,
        113,
        58,
        85,
        51,
        134,
        121,
        169,
        105,
        21,
        30,
        11,
        50,
        65,
        12,
        43,
        82,
        145,
        152,
        97,
        106,
        55,
        31,
        85,
        38,
        112,
        102,
        168,
        123,
        97,
        21,
        83,
        158,
        26,
        80,
        63,
        5,
        81,
        32,
        11,
        28,
        148,
    ]

    # Create new agent
    agent = QLearningAgent(
        alpha=LEARNING_RATE,
        gamma=DISCOUNT_FACTOR,
        epsilon=MAX_EPSILON,
        epsilon_min=MIN_EPSILON,
        n_actions=environment.action_space.n,
        n_observations=environment.observation_space.n,
        randomize_q_table=False,
    )

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

    # model_dictionary = {
    #     "env_id": ENV_ID,
    #     "max_steps": 99,
    #     "n_training_episodes": N_TRAINING_EPISODES,
    #     "n_eval_episodes": N_EVALUATION_EPISODES,
    #     "eval_seed": seeds,
    #     "learning_rate": LEARNING_RATE,
    #     "gamma": DISCOUNT_FACTOR,
    #     "max_epsilon": MAX_EPSILON,
    #     "min_epsilon": MIN_EPSILON,
    #     "qtable": agent.q_table
    # }
    #
    # agent.upload_to_huggingface_hub(repository_id=REPO_ID,
    #                                 environment=environment,
    #                                 environment_name=ENV_ID,
    #                                 evaluation_seeds=seeds,
    #                                 model_dictionary=model_dictionary)
