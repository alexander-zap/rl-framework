from huggingface_sb3 import package_to_hub, load_from_hub
from rl_framework.agent.stable_baselines import StableBaselinesAgent, StableBaselinesAlgorithm
from rl_framework.environment import Environment
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Text, Tuple


def evaluate(
    agent_to_evaluate: StableBaselinesAgent,
    evaluation_environment: Environment,
    n_evaluation_episodes: int = 10,
    deterministic_action: bool = True,
) -> Tuple[float, float]:
    """
    Evaluate the performance (average reward and standard deviation of the reward) by running the policy of a trained
    agent on an environment.

    Args:
        agent_to_evaluate (StableBaselinesAgent): Agent whose policy should be evaluated.
            SB3 model is stored in `.model` attribute.
        evaluation_environment (Environment): Environment, on which the agent's policy should be evaluated
        n_evaluation_episodes (int): Number of episode to evaluate the agent
        deterministic_action (bool): Whether to use deterministic or stochastic actions

    Returns:
        Tuple of:
            average reward (float)
            standard deviation of reward (float)
    """
    model = agent_to_evaluate.model
    mean_reward, std_reward = evaluate_policy(
        model,
        evaluation_environment,
        n_eval_episodes=n_evaluation_episodes,
        deterministic=deterministic_action,
    )
    return mean_reward, std_reward


def upload_to_huggingface_hub(
    agent_to_upload: StableBaselinesAgent,
    evaluation_environment: Environment,
    model_name: Text,
    model_architecture: Text,
    environment_name: Text,
    repository_id: Text,
    commit_message: Text,
    gym_environment: bool = True,
) -> None:
    """

    Args:
        agent_to_upload (StableBaselinesAgent): Agent class with the SB3 model stored in attribute `.model`.
        evaluation_environment (Environment): Environment used for final evaluation and clip creation before upload.
        model_name (Text): Name of the model (uploaded model .zip will be named accordingly).
        model_architecture (Text): Name of the used model architecture (only used for model card and metadata).
        environment_name (Text): Name of the environment (only used for model card and metadata).
        repository_id (Text): Id of the model repository from the Hugging Face Hub.
        commit_message (Text): Commit message for the HuggingFace repository commit.
        gym_environment (bool): Flag whether the used environment is a Gym environment.
            Only Gym environments are supported for detailed model card creation with `package_to_hub`.
            Alternatively we will use `push_to_hub` and create a minimal model card.

    NOTE: If after running the package_to_hub function, and it gives an issue of rebasing, please run the following code
        `cd <path_to_repo> && git add . && git commit -m "Add message" && git pull`
        And don't forget to do a `git push` at the end to push the change to the hub.

    """
    if gym_environment:
        # Create a Stable-baselines3 vector environment (required for HuggingFace upload function)
        vectorized_evaluation_environment = DummyVecEnv(
            [lambda: evaluation_environment]
        )

        model = agent_to_upload.model
        package_to_hub(
            model=model,
            model_name=model_name,
            model_architecture=model_architecture,
            env_id=environment_name,
            eval_env=vectorized_evaluation_environment,
            repo_id=repository_id,
            commit_message=commit_message,
        )
    else:
        # TODO
        pass


def download_from_huggingface_hub(
    rl_algorithm: StableBaselinesAlgorithm, repository_id: Text, filename: Text
):
    """
    Download a reinforcement learning model from the HuggingFace Hub and return an Agent.

    Args:
        rl_algorithm (StableBaselinesAlgorithm): Enum with values being SB3 RL Algorithm classes (Types).
                Specifies the SB3 RL Algorithm which the model to be downloaded was based on.
        repository_id (Text): Repository ID of the reinforcement learning model we want to download.
        filename (Text): The model filename (file ending with .zip) located in the hugging face repository.

    Returns:

    """

    # When the model was trained on Python 3.8 the pickle protocol is 5
    # But Python 3.6, 3.7 use protocol 4
    # In order to get compatibility we need to:
    # 1. Install pickle5
    # 2. Create a custom empty object we pass as parameter to PPO.load()
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }

    rl_algorithm_class = rl_algorithm.value
    checkpoint = load_from_hub(repository_id, filename)
    model = rl_algorithm_class.load(
        checkpoint, custom_objects=custom_objects, print_system_info=True
    )
    agent = StableBaselinesAgent(
        rl_algorithm=rl_algorithm,
        pretrained_model=model
    )
    return agent
