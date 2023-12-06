import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .base_connector import Connector


@dataclass
class ClearMLUploadConfig:
    pass


@dataclass
class ClearMLDownloadConfig:
    pass


# FIXME: Implement ClearMLConnector correctly (according to Connector-interface)
class ClearMLConnector(Connector):
    def __init__(self, task):
        self.task = task

    def upload(self, agent, *args, **kwargs) -> None:
        # Save agent to temporary path and upload folder to ClearML
        with tempfile.TemporaryDirectory() as temp_path:
            agent_save_path = os.path.join(temp_path, "agent")
            agent.save_to_file(Path(agent_save_path))
            while not os.path.exists(agent_save_path):
                time.sleep(1)
            self.task.upload_artifact(name="agent", artifact_object=temp_path)

    def download(self, *args, **kwargs) -> Path:
        raise NotImplementedError
