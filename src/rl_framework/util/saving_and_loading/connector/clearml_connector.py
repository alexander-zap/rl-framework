import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, SupportsFloat, Text

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
    video_length (int): Length of video in frames (which should be generated and uploaded to the connector).
        No video is uploaded if length is 0 or negative.
    """

    file_name: str
    n_eval_episodes: int
    video_length: int


@dataclass
class ClearMLDownloadConfig(DownloadConfig):
    """
    task_id (str): Id of the existing ClearML task to download the agent from
    file_name (str): File name of previously saved agent
    """

    task_id: str
    file_name: str


class ClearMLConnector(Connector):
    def __init__(self, upload_config: ClearMLUploadConfig, download_config: ClearMLDownloadConfig, task: Task):
        """
        Initialize the connector and pass a ClearML Task object for tracking parameters/artifacts/results.

        Args:
            task (Task): Active task object to track parameters/artifacts/results in the experiment run(s).
                See https://clear.ml/docs/latest/docs/clearml_sdk/task_sdk/ on how to use tasks for your purposes.
        """
        super().__init__(upload_config, download_config)
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
        super().log_value(timestep, value_scalar, value_name)
        self.task.get_logger().report_scalar(
            title=value_name, series=value_name, value=value_scalar, iteration=timestep
        )

    def upload(self, agent, evaluation_environment, checkpoint_id: Optional[int] = None, *args, **kwargs) -> None:
        """Evaluate the agent on the evaluation environment and generate a video.
         Afterward, upload the artifacts and the agent itself to a ClearML task.

        Args:
            agent (Agent): Agent (and its .algorithm attribute) to be uploaded.
            evaluation_environment (Environment): Environment used for final evaluation and clip creation before upload.
            checkpoint_id (int): If specified, we do not performa a final upload with evaluating and generating but
                instead upload only a model checkpoint to ClearML.
        """
        file_name = self.upload_config.file_name
        n_eval_episodes = self.upload_config.n_eval_episodes
        video_length = self.upload_config.video_length

        assert file_name, n_eval_episodes

        # Step 1: Save agent to temporary path and upload .zip file to ClearML
        with tempfile.TemporaryDirectory() as temp_path:
            logging.debug(f"Saving agent to .zip file at {temp_path} and uploading artifact ...")

            file_literal, file_ending = str.split(file_name, ".")
            checkpoint_suffix = f"-{checkpoint_id}" if checkpoint_id else ""

            agent_save_path = Path(os.path.join(temp_path, f"{file_literal}{checkpoint_suffix}.{file_ending}"))
            agent.save_to_file(agent_save_path)
            while not os.path.exists(agent_save_path):
                time.sleep(1)

            self.task.upload_artifact(name=f"{file_literal}{checkpoint_suffix}", artifact_object=temp_path)

        if not checkpoint_id:
            logging.info(
                "This function will evaluate the performance of your agent and log the model as well as the experiment "
                "results as artifacts to ClearML. Also, a video of the agent's performance on the evaluation "
                "environment will be generated and uploaded to the 'Debug Sample' section of the ClearML experiment."
            )

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
            if video_length > 0:
                temp_path = tempfile.mkdtemp()
                logging.debug(f"Recording video to {temp_path} and uploading as debug sample ...")
                video_path = Path(temp_path) / "replay.mp4"
                record_video(
                    agent=agent,
                    evaluation_environment=evaluation_environment,
                    file_path=video_path,
                    fps=1,
                    video_length=video_length,
                )
                self.task.get_logger().report_media(
                    "video ", "agent-in-environment recording", iteration=1, local_path=video_path
                )

            # TODO: Save README.md

    def download(self, *args, **kwargs) -> Path:
        task_id = self.download_config.task_id
        file_name = self.download_config.file_name

        assert task_id, file_name

        # Get previous task of same project
        project_name = self.task.get_project_name()

        # Download previously uploaded agent
        preprocess_task = Task.get_task(task_id=task_id, project_name=project_name)
        file_literal = str.split(file_name, ".")[0]
        logging.debug(f"Downloading [Agent {file_literal}] from [Task {task_id}] of [Project {project_name}] ...")
        file_path = preprocess_task.artifacts[file_literal].get_local_copy()
        return Path(file_path) / file_name
