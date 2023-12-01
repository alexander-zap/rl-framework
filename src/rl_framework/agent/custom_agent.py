import datetime
import json
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Text

import imageio
import numpy as np
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.repocard import metadata_eval_result, metadata_save

from rl_framework.agent import Agent
from rl_framework.environment import Environment
from rl_framework.util import evaluate_agent

from .custom_algorithms import QLearning


class CustomAlgorithm(Enum):
    Q_LEARNING = QLearning


class CustomAgent(Agent):
    @property
    def algorithm(self):
        return self._algorithm

    @algorithm.setter
    def algorithm(self, value):
        self._algorithm = value

    def __init__(
        self,
        algorithm: CustomAlgorithm = CustomAlgorithm.Q_LEARNING,
        algorithm_parameters: Dict = None,
    ):
        """
        Initialize an agent which will trained on one of custom implemented algorithms.

        Args:
            algorithm (CustomAlgorithm): Enum with values being custom implemented Algorithm classes (Types).
                Specifies the algorithm for RL training.
                Defaults to Q-Learning.
            algorithm_parameters (Dict): Parameters / keyword arguments for the specified Algorithm class.
        """

        algorithm_class = algorithm.value

        if algorithm_parameters is None:
            algorithm_parameters = {}

        self.algorithm = algorithm_class(**algorithm_parameters)

    def train(
        self,
        training_environments: List[Environment],
        total_timesteps: int = 100000,
        *args,
        **kwargs,
    ):
        """
        Train the instantiated agent on the environment.

        This training is done by using the agent-on-environment training method provided by the custom algorithm.

        Args:
            training_environments (List[Environment): Environment on which the agent should be trained on.
                If n_environments is set above 1, multiple environments enables parallel training of an agent.
            total_timesteps (int): Amount of individual steps the agent should take before terminating the training.
        """

        self.algorithm.train(
            training_environments=training_environments,
            total_timesteps=total_timesteps,
            *args,
            **kwargs,
        )

    def choose_action(self, observation: object, *args, **kwargs):
        """
        Chooses action which the agent will perform next, according to the observed environment.

        Args:
            observation (object): Observation of the environment

        Returns: action (int): Action to take according to policy.

        """

        return self.algorithm.choose_action(observation=observation, *args, **kwargs)

    # TODO: Implement upload/download as adapters; support ClearML

    def upload_to_huggingface_hub(
        self,
        repository_id: Text,
        evaluation_environment: Environment,
        environment_name: Text,
        model_file_name: Text,
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
            repository_id (Text): Id of the model repository from the Hugging Face Hub.
            evaluation_environment (Environment): Environment used for final evaluation and clip creation before upload.
            environment_name (Text): Name of the environment (only used for the model card).
            model_file_name (Text): Name of the model (uploaded model .pkl will be named accordingly).
            model_architecture (Text): Name of the used model architecture (only used for model card and metadata).
            commit_message (Text): Commit message for the HuggingFace repository commit.
            n_eval_episodes (int): Number of episodes for agent evaluation to compute evaluation metrics
        """

        def record_video(env: Environment, out_directory: Path, fps: int = 1):
            """
            Generate a replay video of the agent.
                env (Environment): Environment used for final evaluation and clip creation before upload.
                agent (QLearningAgent): Agent to record video for.
                out_directory (Path): Path where video should be saved to.
                fps: How many frame per seconds to record the video replay.
            """

            images = []
            done = False
            state, _ = env.reset()
            img = env.render()
            images.append(img)
            while not done:
                # Take the action (index) that have the maximum expected future reward given that state
                action = self.choose_action(state)
                state, reward, terminated, truncated, info = env.step(
                    action
                )  # We directly put next_state = state for recording logic
                done = terminated or truncated
                img = env.render()
                images.append(img)
            imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)

        _, repo_name = repository_id.split("/")

        eval_env = evaluation_environment
        api = HfApi()

        # Step 1: Create the repo
        repo_url = api.create_repo(
            repo_id=repository_id,
            exist_ok=True,
        )

        # Step 2: Download files
        repo_local_path = Path(snapshot_download(repo_id=repository_id))

        # Step 3: Save the model
        self.algorithm.save(repo_local_path / model_file_name)

        # Step 4: Evaluate the model and build JSON with evaluation metrics
        mean_reward, std_reward = evaluate_agent(
            agent=self,
            evaluation_environment=eval_env,
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

        # Step 5: Create the model card
        metadata = {"tags": [environment_name, "reinforcement-learning", "custom-implementation"]}

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

        model_card = f"""
        # Custom implemented agent playing *{environment_name}*
        This is a trained model of an agent playing *{environment_name}* .
        The agent was trained with a {model_architecture} algorithm.

        ## Usage

        ```python

        model = load_from_hub(repo_id="{repository_id}")

        # Don't forget to check if you need to add additional attributes
        env = gym.make(model["env_id"])
        ```
        """

        readme_path = repo_local_path / "README.md"
        print(readme_path.exists())
        if readme_path.exists():
            with readme_path.open("r", encoding="utf8") as f:
                readme = f.read()
        else:
            readme = model_card

        with readme_path.open("w", encoding="utf-8") as f:
            f.write(readme)

        # Save our metrics to Readme metadata
        metadata_save(readme_path, metadata)

        # Step 6: Record a video
        video_path = repo_local_path / "replay.mp4"
        record_video(env=evaluation_environment, out_directory=video_path, fps=1)

        # Step 7. Push everything to the Hub
        api.upload_folder(
            repo_id=repository_id,
            folder_path=repo_local_path,
            path_in_repo=".",
            commit_message=commit_message,
        )

        print("Your model is pushed to the Hub. You can view your model here: ", repo_url)

    def download_from_huggingface_hub(
        self, repository_id: Text, filename: Text, algorithm_parameters: Optional[Dict] = None
    ):
        """
        Download a reinforcement learning model from the HuggingFace Hub and update the .algorithm attribute in-place.

        Args:
            repository_id (Text): Model repository ID from the HF Hub of the RL model we want to download.
            filename (Text): The model filename located in the hugging face repository.
            algorithm_parameters (Optional[Dict]): Parameters to be set for the downloaded algorithm.

        """

        # Get the model from the Hub, download and cache the model on your local disk
        pickle_model = hf_hub_download(repo_id=repository_id, filename=filename)
        self.algorithm.load(pickle_model)
        # TODO: Find a way to load algorithm_parameters into algorithm
