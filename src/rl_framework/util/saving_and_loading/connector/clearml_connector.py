import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import SupportsFloat, Text

import stable_baselines3
from clearml import Task

from rl_framework.util import evaluate_agent
from rl_framework.util.video_recording import record_video

from .base_connector import Connector, DownloadConfig, UploadConfig


@dataclass
class ClearMLUploadConfig(UploadConfig):
    """
    file_name (str): File name of agent to be saved to
    n_eval_episodes (int): Number of episodes for agent evaluation to compute evaluation metrics
    """

    file_name: str
    n_eval_episodes: int


@dataclass
class ClearMLDownloadConfig(DownloadConfig):
    """
    task_id (str): Id of the existing ClearML task to download the agent from
    file_name (str): File name of previously saved agent
    """

    task_id: str
    file_name: str


class ClearMLConnector(Connector):
    def __init__(self, task: Task):
        """
        Initialize the connector and pass a ClearML Task object for tracking parameters/artifacts/results.

        Args:
            task (Task): Active task object to track parameters/artifacts/results in the experiment run(s).
                See https://clear.ml/docs/latest/docs/clearml_sdk/task_sdk/ on how to use tasks for your purposes.
        """
        super().__init__()
        self.task = task

    def log_value(self, timestep: int, value_scalar: SupportsFloat, value_name: Text) -> None:
        """
        Log scalar value to create a sequence of values over time steps.
        Will appear in the "Scalar" section of the ClearML experiment page.

        Args:
            timestep: Time step which the scalar value corresponds to (x-value)
            value_scalar: Scalar value which should be logged (y-value)
            value_name: Name of scalar value (e.g., "avg. sum of reward")
        """
        super().__init__()
        self.task.get_logger().report_scalar(
            title=value_name, series=value_name, value=value_scalar, iteration=timestep
        )

    def upload(
        self, connector_config: ClearMLUploadConfig, agent, evaluation_environment, generate_video, *args, **kwargs
    ) -> None:
        """Evaluate the agent on the evaluation environment and generate a video.
         Afterward, upload the artifacts and the agent itself to a ClearML task.

        Args:
            connector_config: Connector configuration data for uploading to HuggingFace.
                See above for the documented dataclass attributes.
            agent (Agent): Agent (and its .algorithm attribute) to be uploaded.
            evaluation_environment (Environment): Environment used for final evaluation and clip creation before upload.
            generate_video (bool): Flag whether a video should be generated and uploaded to the connector.
        """
        file_name = connector_config.file_name
        n_eval_episodes = connector_config.n_eval_episodes

        assert file_name, n_eval_episodes

        logging.info(
            "This function will evaluate the performance of your agent and log the model as well as the experiment "
            "results as artifacts to ClearML. Also, a video of the agent's performance on the evaluation environment "
            "will be generated and uploaded to the 'Debug Sample' section of the ClearML experiment."
        )

        # Step 1: Save agent to temporary path and upload .zip file to ClearML
        with tempfile.TemporaryDirectory() as temp_path:
            logging.debug(f"Saving agent to .zip file at {temp_path} and uploading artifact ...")
            agent_save_path = Path(os.path.join(temp_path, file_name))
            agent.save_to_file(agent_save_path)
            while not os.path.exists(agent_save_path):
                time.sleep(1)
            self.task.upload_artifact(name="agent", artifact_object=temp_path)

        # Step 2: Evaluate the agent and upload a dictionary with evaluation metrics
        logging.debug("Evaluating agent and uploading experiment results ...")
        mean_reward, std_reward = evaluate_agent(
            agent=agent,
            evaluation_environment=evaluation_environment,
            n_eval_episodes=n_eval_episodes,
        )
        experiment_result = {
            "mean_reward": round(mean_reward, 2),
            "std_reward": round(std_reward, 2),
        }
        self.task.upload_artifact(name="experiment_result", artifact_object=experiment_result)

        # Step 3: Create a system info dictionary and upload it
        logging.debug("Uploading system meta information ...")
        system_info, _ = stable_baselines3.get_system_info()
        self.task.upload_artifact(name="system_info", artifact_object=system_info)

        # Step 4: Record a video and log local video file
        if generate_video:
            temp_path = tempfile.mkdtemp()
            logging.debug(f"Recording video to {temp_path} and uploading as debug sample ...")
            video_path = Path(temp_path) / "replay.mp4"
            record_video(
                agent=agent,
                evaluation_environment=evaluation_environment,
                file_path=video_path,
                fps=1,
                video_length=1000,
            )
            self.task.get_logger().report_media(
                "video ", "agent-in-environment recording", iteration=1, local_path=video_path
            )

        # TODO: Save README.md

    def download(self, connector_config: ClearMLDownloadConfig, *args, **kwargs) -> Path:
        task_id = connector_config.task_id
        file_name = connector_config.file_name

        assert task_id, file_name

        # Get previous task of same project
        project_name = self.task.get_project_name()

        logging.debug(f"Downloading agent from Task with project name {project_name} and task id {task_id} ...")
        # Download previously uploaded agent
        preprocess_task = Task.get_task(task_id=task_id, project_name=project_name)
        file_path = preprocess_task.artifacts["agent"].get_local_copy()
        return Path(file_path) / file_name
