from rl_framework.agent import Agent
from rl_framework.environment import Environment
from rl_framework.util import evaluate_agent
from typing import Text, List, Optional, Dict
import numpy as np
import random
from tqdm import tqdm
import logging
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.repocard import metadata_eval_result, metadata_save
from pathlib import Path
import datetime
import json
import imageio
import pickle
from huggingface_hub import hf_hub_download


class QLearningAgent(Agent):
    @property
    def q_table(self):
        return self._q_table

    @q_table.setter
    def q_table(self, value):
        self._q_table = value

    def __init__(self, n_actions: int, n_observations: int, alpha: float = 0.1, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_min: float = 0.05, randomize_q_table: bool = True):
        """
        Initialize an Q-Learning agent which will be trained.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.n_actions = n_actions

        if randomize_q_table:
            self.q_table = np.random.random_sample((n_observations, n_actions)) * 0.1
        else:
            self.q_table = np.full((n_observations, n_actions), 0.0)

    def _update_q_table(self, prev_observation: object, prev_action: int, observation: object, reward: float):
        """
        Update _q_table based on previous observation, previous action, new observation and received reward

        Args:
            prev_observation (object): Previous observation (St)
            prev_action (in): Previous action (at)
            observation (object): New observation (St+1) after executing action at in state St
            reward (float): Reward for executing action at in state St

        """
        q_old = self._q_table[prev_observation, prev_action]
        q_new = (1 - self.alpha) * q_old + self.alpha * (reward + self.gamma * np.max(self._q_table[observation]))
        self._q_table[prev_observation, prev_action] = q_new

    def _update_epsilon(self, n_episodes: int):
        """
        Gradually reduce epsilon after every done episode

        Args:
            n_episodes (int): Number of episodes (information required to reduce epsilon steadily.

        """
        self.epsilon = self.epsilon - 2 / n_episodes if self.epsilon > self.epsilon_min else self.epsilon_min

    def choose_action(self, observation: object, *args, **kwargs) -> int:
        """
        Chooses action which the agent will perform next, according to the observed environment.

        Args:
            observation (object): Observation of the environment

        Returns: action (int): Action to take according to policy.

        """

        return np.argmax(self._q_table[observation])

    # TODO: Exploration-exploitation strategy is currently hard-coded as epsilon-greedy.
    #   Pass exploration-exploitation strategy from outside
    def train(self, training_environments: List[Environment], n_episodes: int = 10000, *args, **kwargs):
        """
        Train the instantiated agent on the environment.

        This training is done by using the Q-Learning method.

        The Q-table is changed in place, therefore the updated Q-table can be accessed in the `.q_table` attribute
        after the agent has been trained.

        Args:
            training_environments (List[Environment]): List of environments on which the agent should be trained on.
                # NOTE: This class only supports training on one environment
            n_episodes (int): Number of episodes the agent should train for before terminating the training.
        """

        def choose_action_according_to_exploration_exploitation_strategy(obs):
            greedy_action = self.choose_action(obs)
            # Choose random action with probability epsilon
            if random.random() < self.epsilon:
                return random.randrange(self.n_actions)
            # Greedy action is chosen with probability (1 - epsilon)
            else:
                return greedy_action

        if len(training_environments) > 1:
            logging.info(
                f"Reinforcement Learning algorithm {self.__class__.__qualname__} does not support "
                f"training on multiple environments in parallel. Continuing with one environment as "
                f"training environment.")

        training_environment = training_environments[0]

        for _ in tqdm(range(n_episodes)):
            episode_reward = 0
            prev_observation, _ = training_environment.reset()
            prev_action = choose_action_according_to_exploration_exploitation_strategy(prev_observation)

            while True:
                observation, reward, terminated, truncated, info = training_environment.step(prev_action)
                done = terminated or truncated
                action = choose_action_according_to_exploration_exploitation_strategy(observation)
                episode_reward += reward
                self._update_q_table(prev_observation, prev_action, observation, reward)

                prev_observation = observation
                prev_action = action

                if done:
                    self._update_epsilon(n_episodes)
                    break

    def save(self, file_path: Text):
        """
        Save the model of the agent to a zipped file.

        Args:
            file_path (Text): Path where the model should be saved to.
        """
        raise NotImplementedError

    def upload_to_huggingface_hub(self,
                                  repository_id: Text,
                                  environment: Environment,
                                  environment_name: Text,
                                  model_dictionary: Dict,
                                  evaluation_seeds: Optional[List[int]] = None,
                                  video_fps: int = 1):
        """
        Evaluate, Generate a video and Upload a model to Hugging Face Hub.
        This method does the complete pipeline:
        - It evaluates the model
        - It generates the model card
        - It generates a replay video of the agent
        - It pushes everything to the Hub

        Args:
            agent_to_upload (StableBaselinesAgent): Agent class with the SB3 model stored in attribute `.model`.
            repository_id (Text): Id of the model repository from the Hugging Face Hub.
            environment (Environment): Environment used for final evaluation and clip creation before upload.
            environment_name (Text): Name of the environment (for the model card).
            model_dictionary (Dict): The model dictionary that contains the model and the hyperparameters.
            evaluation_seeds (Optional[List[int]]): List of seeds for evaluations.
            video_fps (int): How many frame per seconds to record the video replay.
        """

        def record_video(env: Environment, agent: QLearningAgent, out_directory: Path, fps: int = 1):
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
                action = agent.choose_action(state)
                state, reward, terminated, truncated, info = env.step(action)  # We directly put next_state = state for recording logic
                done = terminated or truncated
                img = env.render()
                images.append(img)
            imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)

        _, repo_name = repository_id.split("/")

        eval_env = environment
        api = HfApi()

        # Step 1: Create the repo
        repo_url = api.create_repo(
            repo_id=repository_id,
            exist_ok=True,
        )

        # Step 2: Download files
        repo_local_path = Path(snapshot_download(repo_id=repository_id))

        # Step 3: Save the model
        # Pickle the model
        with open(repo_local_path / "q-learning.pkl", "wb") as f:
            pickle.dump(model_dictionary, f)

        # Step 4: Evaluate the model and build JSON with evaluation metrics
        mean_reward, std_reward = evaluate_agent(
            agent=self, evaluation_environment=eval_env, n_eval_episodes=model_dictionary["n_eval_episodes"],
            seeds=evaluation_seeds
        )

        evaluate_data = {
            "env_id": model_dictionary["env_id"],
            "mean_reward": mean_reward,
            "n_eval_episodes": model_dictionary["n_eval_episodes"],
            "eval_datetime": datetime.datetime.now().isoformat(),
        }

        # Write a JSON file called "results.json" that will contain the
        # evaluation results
        with open(repo_local_path / "results.json", "w") as outfile:
            json.dump(evaluate_data, outfile)

        # Step 5: Create the model card
        env_id = model_dictionary["env_id"]

        metadata = {"tags": [env_id, "q-learning", "reinforcement-learning", "custom-implementation"]}

        # Add metrics
        metadata_eval = metadata_eval_result(
            model_pretty_name=repo_name,
            task_pretty_name="reinforcement-learning",
            task_id="reinforcement-learning",
            metrics_pretty_name="mean_reward",
            metrics_id="mean_reward",
            metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
            dataset_pretty_name=env_id,
            dataset_id=env_id,
        )

        # Merges both dictionaries
        metadata = {**metadata, **metadata_eval}

        model_card = f"""
        # **Q-Learning** Agent playing **{environment_name}**
        This is a trained model of a **Q-Learning** agent playing **{environment_name}** .

        ## Usage

        ```python

        model = load_from_hub(repo_id="{repository_id}", filename="q-learning.pkl")

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
        # video_path = repo_local_path / "replay.mp4"
        # record_video(env=environment, agent=agent_to_upload, out_directory=video_path, fps=video_fps)

        # Step 7. Push everything to the Hub
        api.upload_folder(
            repo_id=repository_id,
            folder_path=repo_local_path,
            path_in_repo=".",
        )

        print("Your model is pushed to the Hub. You can view your model here: ", repo_url)

    def download_from_huggingface_hub(self, repository_id: Text, filename: Text):
        """
        Download a reinforcement learning model from the HuggingFace Hub and return a QLearningAgent.

        Args:
            repository_id (Text): Repository ID of the reinforcement learning model we want to download.
            filename (Text): The model filename (file ending with .zip) located in the hugging face repository.

        Returns:
            QLearningAgent

        """

        """
        Download a model from Hugging Face Hub.
        :param repo_id: id of the model repository from the Hugging Face Hub
        :param filename: name of the model zip file from the repository
        """
        # Get the model from the Hub, download and cache the model on your local disk
        pickle_model = hf_hub_download(repo_id=repository_id, filename=filename)

        with open(pickle_model, "rb") as f:
            downloaded_model_file = pickle.load(f)

        q_table = downloaded_model_file["qtable"]

        self.q_table = q_table
