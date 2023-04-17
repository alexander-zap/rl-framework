from rl_framework.agent import Agent
from rl_framework.environment import Environment
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, DDPG, DQN, HER, PPO, SAC, TD3
from typing import List, Text, Optional, Dict
from huggingface_sb3 import package_to_hub, load_from_hub
from functools import partial
from enum import Enum


class StableBaselinesAlgorithm(Enum):
    A2C = A2C
    DDPG = DDPG
    DQN = DQN
    HER = HER
    PPO = PPO
    SAC = SAC
    TD3 = TD3


class StableBaselinesAgent(Agent):
    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    def __init__(
        self,
        rl_algorithm: StableBaselinesAlgorithm = StableBaselinesAlgorithm.PPO,
        rl_algorithm_parameters: Dict = None,
        pretrained_model: Optional[BaseAlgorithm] = None,
    ):
        """
        Initialize an agent which will trained on one of Stable-Baselines3 algorithms.

        Args:
            rl_algorithm (StableBaselinesAlgorithm): Enum with values being SB3 RL Algorithm classes (Types).
                Specifies the algorithm for RL training.
                Defaults to PPO.
            rl_algorithm_parameters (Dict): Parameters / keyword arguments for the specified SB3 RL Algorithm class.
                See https://stable-baselines3.readthedocs.io/en/master/modules/base.html for details on common params.
                See individual docs (e.g., https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
                for algorithm-specific params.
            pretrained_model (BaseAlgorithm): Pretrained SB3 model.
                This variable is mainly used for loading previously saved models.
        """

        self.sb3_algorithm = rl_algorithm.value
        self.model = pretrained_model

        if rl_algorithm_parameters is None:
            rl_algorithm_parameters = {
                "policy": "MlpPolicy",
                "learning_rate": 0.001,
            }

        self._model_builder = partial(
            self.sb3_algorithm,
            **rl_algorithm_parameters
        )

    def train(self, training_environments: List[Environment], total_timesteps: int = 100000, *args, **kwargs):
        """
        Train the instantiated agent on the environment.

        This training is done by using the agent-on-environment training method provided by Stable-baselines3.

        The model is changed in place, therefore the updated model can be accessed in the `.model` attribute
        after the agent has been trained.

        Args:
            training_environments (List[Environment): Environment on which the agent should be trained on.
                If n_environments is set above 1, multiple environments enables parallel training of an agent.
            total_timesteps (int): Amount of individual steps the agent should take before terminating the training.
        """

        environment_iterator = iter(training_environments)
        training_env = make_vec_env(
            lambda: next(environment_iterator), n_envs=len(training_environments)
        )

        self.model = self._model_builder(env=training_env)

        self.model.learn(total_timesteps=total_timesteps)

    def choose_action(self, observation: object, *args, **kwargs):
        """
        Chooses action which the agent will perform next, according to the observed environment.

        Args:
            observation (object): Observation of the environment

        Returns: action (int): Action to take according to policy.

        """

        action, _ = self.model.predict(observation)
        return action

    def save(self, file_path: Text):
        """
        Save the model of the agent to a zipped file.

        Args:
            file_path (Text): Path where the model should be saved to.
        """
        self.model.save(file_path)

    def upload_to_huggingface_hub(
        self,
        evaluation_environment: Environment,
        model_name: Text,
        model_architecture: Text,
        environment_name: Text,
        repository_id: Text,
        commit_message: Text
    ) -> None:
        """

        Args:
            evaluation_environment (Environment): Environment used for final evaluation and clip creation before upload.
            model_name (Text): Name of the model (uploaded model .zip will be named accordingly).
            model_architecture (Text): Name of the used model architecture (only used for model card and metadata).
            environment_name (Text): Name of the environment (only used for model card and metadata).
            repository_id (Text): Id of the model repository from the Hugging Face Hub.
            commit_message (Text): Commit message for the HuggingFace repository commit.

        NOTE: If after running the package_to_hub function, and it gives an issue of rebasing, please run the following code
            `cd <path_to_repo> && git add . && git commit -m "Add message" && git pull`
            And don't forget to do a `git push` at the end to push the change to the hub.

        """
        # Create a Stable-baselines3 vector environment (required for HuggingFace upload function)
        vectorized_evaluation_environment = DummyVecEnv(
            [lambda: evaluation_environment]
        )

        model = self.model
        package_to_hub(
            model=model,
            model_name=model_name,
            model_architecture=model_architecture,
            env_id=environment_name,
            eval_env=vectorized_evaluation_environment,
            repo_id=repository_id,
            commit_message=commit_message,
        )

    def download_from_huggingface_hub(
        self, repository_id: Text, filename: Text
    ):
        """
        Download a reinforcement learning model from the HuggingFace Hub and update the agent policy in-place.

        Args:
            repository_id (Text): Repository ID of the reinforcement learning model we want to download.
            filename (Text): The model filename (file ending with .zip) located in the hugging face repository.

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

        checkpoint = load_from_hub(repository_id, filename)
        model = self.sb3_algorithm.load(
            checkpoint, custom_objects=custom_objects, print_system_info=True
        )
        self.model = model
