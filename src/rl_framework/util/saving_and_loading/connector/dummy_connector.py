from pathlib import Path

from .base_connector import Connector


class DummyConnector(Connector):
    def __init__(self):
        super().__init__(None, None)

    def upload(self, agent, evaluation_environment, checkpoint_id=None, *args, **kwargs) -> None:
        pass

    def download(self, *args, **kwargs) -> Path:
        pass
