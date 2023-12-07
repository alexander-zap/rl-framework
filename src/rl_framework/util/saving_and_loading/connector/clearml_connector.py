import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from rl_framework.util import evaluate_agent

from .base_connector import Connector, DownloadConfig, UploadConfig


@dataclass
class ClearMLUploadConfig(UploadConfig):
    """
            n_eval_episodes (int): Number of episodes for agent evaluation to compute evaluation metrics
    """

    n_eval_episodes: int


@dataclass
class ClearMLDownloadConfig(DownloadConfig):
    pass


class ClearMLConnector(Connector):
    def __init__(self, task):
        """
        Initialize the connector and pass a ClearML Task object for tracking parameters/artifacts/results.

        Args:
            task (Task): Active task object to track parameters/artifacts/results in the experiment run(s).
                See https://clear.ml/docs/latest/docs/clearml_sdk/task_sdk/ on how to use tasks for your purposes.
        """
        self.task = task

    def upload(self, connector_config: ClearMLUploadConfig, agent, evaluation_environment, *args, **kwargs) -> None:
        """Evaluate the agent on the evaluation environment and generate a video.
         Afterward, upload the artifacts and the agent itself to a ClearML task.

        Args:
            connector_config: Connector configuration data for uploading to HuggingFace.
                See above for the documented dataclass attributes.
            agent (Agent): Agent (and its .algorithm attribute) to be uploaded.
            evaluation_environment (Environment): Environment used for final evaluation and clip creation before upload.
        """

        # Save agent to temporary path and upload folder to ClearML
        with tempfile.TemporaryDirectory() as temp_path:
            agent_save_path = Path(os.path.join(temp_path, "agent"))
            agent.save_to_file(agent_save_path)
            while not os.path.exists(agent_save_path):
                time.sleep(1)
            self.task.upload_artifact(name="agent", artifact_object=temp_path)

        # Evaluate the agent and build a JSON with evaluation metrics
        mean_reward, std_reward = evaluate_agent(
            agent=agent,
            evaluation_environment=evaluation_environment,
            n_eval_episodes=100,
        )
        experiment_result = {
            "mean_reward": round(mean_reward, 2),
            "std_reward": round(std_reward, 2),
        }
        self.task.upload_artifact(name="experiment_result", artifact_object=experiment_result)

    def download(self, connector_config: ClearMLDownloadConfig, *args, **kwargs) -> Path:
        raise NotImplementedError
