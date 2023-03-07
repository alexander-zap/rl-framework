from huggingface_sb3 import package_to_hub, load_from_hub
from rl_framework.agent import Agent
from rl_framework.environment import Environment
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Text, Tuple


def evaluate(
    agent_to_evaluate: Agent,
    evaluation_environment: Environment,
    n_evaluation_episodes: int = 10,
    deterministic_action: bool = True,
) -> Tuple[float, float]:
    """
    Evaluate the performance (average reward and standard deviation of the reward) by running the policy of a trained
    agent on an environment.

    Args:
        agent_to_evaluate (Agent): Agent whose policy should be evaluated.
        evaluation_environment (Environment): Environment, on which the agent's policy should be evaluated
        n_evaluation_episodes (int): Number of episode to evaluate the agent
        deterministic_action (bool): Whether to use deterministic or stochastic actions

    Returns:
        Tuple of:
            average reward (float)
            standard deviation of reward (float)
    """
    mean_reward, std_reward = evaluate_policy(
        agent_to_evaluate.model,
        evaluation_environment,
        n_eval_episodes=n_evaluation_episodes,
        deterministic=deterministic_action,
    )
    return mean_reward, std_reward


def upload_to_huggingface_hub(
    agent_to_upload: Agent,
    evaluation_environment: Environment,
    model_name: Text,
    model_architecture: Text,
    environment_name: Text,
    repository_id: Text,
    commit_message: Text,
    gym_environment: bool = True,
):
    if gym_environment:
        # Create a Stable-baselines3 vector environment (required for HuggingFace upload function)
        vec_env = DummyVecEnv([lambda: evaluation_environment])

        model = agent_to_upload.model

        package_to_hub(
            model=model,
            model_name=model_name,
            model_architecture=model_architecture,
            env_id=environment_name,
            eval_env=vec_env,
            repo_id=repository_id,
            commit_message=commit_message,
        )
    else:
        # TODO
        pass


def download_from_huggingface_hub(repository_id: Text):
    # TODO
    pass
