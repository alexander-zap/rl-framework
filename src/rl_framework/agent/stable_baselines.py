from rl_framework.agent import Agent
from rl_framework.environment import Environment
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from typing import Callable, List, Text


class StableBaselinesAgent(Agent):
    def __init__(self, environments: List[Environment]):
        """
        Initialize an agent which will trained on one of Stable-Baselines3 algorithms.

        Args:
            environments (List[Environment]): List of environments on which the agent should be trained on.
                Providing multiple environments enables parallel training of an agent.
        """

        environment_iterator = iter(environments)
        training_env = make_vec_env(lambda: next(environment_iterator), n_envs=len(environments))

        # TODO: Do not hardcode the algorithm here.
        self.model = PPO(
            policy="MlpPolicy",
            env=training_env,
            n_steps=1024,
            batch_size=128,
            n_epochs=5,
            gamma=0.998,
            gae_lambda=0.99,
            ent_coef=0.01,
            verbose=1,
        )

    def train(self):
        """
        Train the instantiated agent on the environment.

        This training is done by using the agent-on-environment training method provided by Stable-baselines3.

        The model is changed in place, therefore the updated model can be accessed in the `.model` attribute
        after the agent has been trained.
        """
        self.model.learn(total_timesteps=1000)

    def save(self, file_path: Text):
        """
        Save the model of the agent to a zipped file.

        Args:
            file_path (Text): Path where the model should be saved to.
        """
        self.model.save(file_path)
