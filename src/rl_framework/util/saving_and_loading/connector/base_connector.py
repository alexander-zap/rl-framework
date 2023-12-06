from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass
class UploadConfig(ABC):
    def get_config_dict(self) -> Dict:
        return vars(self)


@dataclass
class DownloadConfig(ABC):
    def get_config_dict(self) -> Dict:
        return vars(self)


class Connector(ABC):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def upload(self, config: UploadConfig, agent, evaluation_environment, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def download(self, config: DownloadConfig, *args, **kwargs) -> Path:
        raise NotImplementedError
