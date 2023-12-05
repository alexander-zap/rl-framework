import datetime
import json
import logging
from pathlib import Path
from typing import Dict, Text

import stable_baselines3
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.repocard import metadata_eval_result, metadata_save

from .evaluating import evaluate_agent
from .video_recording import record_video


def upload_to_huggingface_hub(
    agent,
    evaluation_environment,
    repository_id: Text,
    environment_name: Text,
    file_name: Text,
    model_architecture: Text,
    commit_message: Text,
    n_eval_episodes: int,
):
    """
    Evaluate, Generate a video and Upload a model to Hugging Face Hub.
    This method does the complete pipeline:
    - It evaluates the model
    - It generates the model card
    - It generates a replay video of the agent
    - It pushes everything to the Hub

    Args:
        agent (Agent): Agent (and its .algorithm attribute) to be uploaded.
        evaluation_environment (Environment): Environment used for final evaluation and clip creation before upload.
        repository_id (Text): Id of the model repository from the Hugging Face Hub.
        environment_name (Text): Name of the environment (only used for the model card).
        file_name (Text): Name of the model (uploaded model .pkl will be named accordingly).
        model_architecture (Text): Name of the used model architecture (only used for model card and metadata).
        commit_message (Text): Commit message for the HuggingFace repository commit.
        n_eval_episodes (int): Number of episodes for agent evaluation to compute evaluation metrics

    NOTE: If after running the package_to_hub function, and it gives an issue of rebasing, please run the
        following code: `cd <path_to_repo> && git add . && git commit -m "Add message" && git pull`
        And don't forget to do a `git push` at the end to push the change to the hub.
    """

    # TODO: Improve readability through sub-functions
    logging.info(
        "This function will save your agent, evaluate its performance, generate a video of your agent, "
        "create a HuggingFace model card and push everything to the HuggingFace hub. "
    )

    _, repo_name = repository_id.split("/")

    api = HfApi()

    # Step 1: Create the repo
    repo_url = api.create_repo(
        repo_id=repository_id,
        exist_ok=True,
    )

    # Step 2: Download files
    repo_local_path = Path(snapshot_download(repo_id=repository_id))

    # Step 3: Save the model
    agent.save_to_file(repo_local_path / file_name)

    # Step 4: Evaluate the model and build JSON with evaluation metrics
    mean_reward, std_reward = evaluate_agent(
        agent=agent,
        evaluation_environment=evaluation_environment,
        n_eval_episodes=n_eval_episodes,
    )

    evaluate_data = {
        "env_id": environment_name,
        "mean_reward": mean_reward,
        "n_eval_episodes": n_eval_episodes,
        "eval_datetime": datetime.datetime.now().isoformat(),
    }

    # Write a JSON file called "results.json" that will contain the
    # evaluation results
    with open(repo_local_path / "results.json", "w") as outfile:
        json.dump(evaluate_data, outfile)

    # Step 5: Create a system info file
    with open(repo_local_path / "system.json", "w") as outfile:
        env_info, _ = stable_baselines3.get_system_info()
        json.dump(env_info, outfile)

    # Step 6: Create the model card (README.md)

    # TODO: Format with correct agent class and agent algorithm in code part
    model_card = f"""
# Custom implemented {model_architecture} agent playing on *{environment_name}*

This is a trained model of an agent playing on the environment *{environment_name}*.
The agent was trained with a {model_architecture} algorithm and evaluated for {n_eval_episodes} episodes.
See further agent and evaluation metadata in the according README section.

## Usage

```python

# Create new agent instance
agent = CustomAgent(
    algorithm=CustomAlgorithm.Q_LEARNING,   # Replace with the algorithm of the agent to be downloaded
    algorithm_parameters={{
        ...
    }},
)

# Download existing agent from HF Hub
repository_id = {repository_id}
file_name = {file_name}
agent.download(repository_id=repository_id, filename=file_name)

```
"""

    readme_path = repo_local_path / "README.md"
    if readme_path.exists():
        with readme_path.open("r", encoding="utf8") as f:
            readme = f.read()
    else:
        readme = model_card

    with readme_path.open("w", encoding="utf-8") as f:
        f.write(readme)

    metadata = {
        "tags": [
            environment_name,
            "reinforcement-learning",
            "rl-framework"
        ],
    }

    # Add metrics
    metadata_eval = metadata_eval_result(
        model_pretty_name=repo_name,
        task_pretty_name="reinforcement-learning",
        task_id="reinforcement-learning",
        metrics_pretty_name="mean_reward",
        metrics_id="mean_reward",
        metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
        dataset_pretty_name=environment_name,
        dataset_id=environment_name,
    )

    # Merges both dictionaries
    metadata = {**metadata, **metadata_eval}

    # Save our metrics to Readme metadata
    metadata_save(readme_path, metadata)

    # Step 6: Record a video
    video_path = repo_local_path / "replay.mp4"
    record_video(
        agent=agent,
        evaluation_environment=evaluation_environment,
        file_path=video_path,
        fps=1,
        video_length=1000,
        sb3_replay=False,
    )

    logging.info(f"Pushing repo {repository_id} to the Hugging Face Hub")

    # Step 7. Push everything to the Hub
    api.upload_folder(
        repo_id=repository_id,
        folder_path=repo_local_path,
        path_in_repo=".",
        commit_message=commit_message,
    )

    logging.info(f"Your model is pushed to the Hub. You can view your model here: {repo_url}")


def download_from_huggingface_hub(agent, repository_id: Text, file_name: Text, algorithm_parameters: Dict):
    """
    Download a reinforcement learning agent from the HuggingFace Hub and update the passed agent in-place.

    Args:
        agent (Agent): Agent instance to be replaced with the newly downloaded saved-agent.
            NOTE: Agent and Algorithm class need to be the same as the saved agent.
        repository_id (Text): Repository ID from the HF Hub of the RL agent we want to download.
        file_name (Text): The saved-agent filename located in the hugging face repository.
        algorithm_parameters (Optional[Dict]): Parameters to be set for the downloaded agent-algorithm.

    """
    # Get the model from the Hub, download and cache the model on your local disk
    agent_file_path = hf_hub_download(repository_id, file_name)
    agent.load_from_file(Path(agent_file_path), algorithm_parameters)
