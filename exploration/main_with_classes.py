import gym
from rl_framework.algorithm.stable_baselines import StableBaselinesAgent
from rl_framework.environment.gym_environment import GymEnvironmentWrapper
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

ENV_ID = "LunarLander-v2"
MODEL_ARCHITECTURE = "PPO"
parallel_environments = 4


def evaluate(agent_to_evaluate, evaluation_environment):
    model = agent_to_evaluate.model
    mean_reward, std_reward = evaluate_policy(model, evaluation_environment, n_eval_episodes=10, deterministic=True)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")


def upload_to_huggingface_hub(agent_to_upload, evaluation_environment):
    repo_id = f"zap-thamm/{MODEL_ARCHITECTURE}-{ENV_ID}"
    commit_message = f"Upload of a new agent trained with {MODEL_ARCHITECTURE} on {ENV_ID}"

    # Create a Stable-baselines3 vector environment (required for HuggingFace upload function)
    vec_env = DummyVecEnv([lambda: evaluation_environment])

    model = agent_to_upload.model

    # package_to_hub(
    #     model=model,
    #     model_name=model_name,
    #     model_architecture=MODEL_ARCHITECTURE,
    #     env_id=ENV_ID,
    #     eval_env=vec_env,
    #     repo_id=repo_id,
    #     commit_message=commit_message,
    # )


if __name__ == "__main__":
    # Create environment(s); multiple environments for parallel training
    environments = [GymEnvironmentWrapper(ENV_ID) for _ in range(parallel_environments)]

    # Create agent
    agent = StableBaselinesAgent(environments)
    agent.train()

    # Optional: Save the model
    agent.save(file_path=f"{MODEL_ARCHITECTURE}-{ENV_ID}")

    # Evaluate the model
    evaluate(agent_to_evaluate=agent, evaluation_environment=environments[0].raw_environment)

    # Upload the model
    upload_to_huggingface_hub(agent, environments[0].raw_environment)
