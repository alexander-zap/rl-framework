from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, SupportsFloat, Text, Tuple


@dataclass
class UploadConfig(ABC):
    def get_config_dict(self) -> Dict:
        return vars(self)


@dataclass
class DownloadConfig(ABC):
    def get_config_dict(self) -> Dict:
        return vars(self)


class Connector(ABC):
    def __init__(self):
        """Initialize connector for uploading or downloading agents from the HuggingFace Hub.

        All attributes which are relevant for uploading or downloading are set in the config object.
        This is a conscious design decision to enable generic exchange of connectors and prevent the requirements
        of parameter passing for upload/download method calls.

        For repeated use of the same connector instance, change the attributes of the config object which is passed
        to the upload/download method.

        self attributes:
            logging_history: Dictionary mapping each logged value name to a list of logged values, e.g.:
            {
                "Episode reward": [(50.6, 10), (90.5, 20), (150.3, 30), (200.0, 40)],
                "Epsilon": [(1.0, 10), (0.74, 20), (0.46, 30), (0.15, 10)]
            }
            Elements of each list are tuples of timestep-value-points.
        """
        self.logging_history: Dict[Text, List[Tuple]] = defaultdict(list)

    def log_value(self, timestep: int, value_scalar: SupportsFloat, value_name: Text) -> None:
        """
        Log scalar value to create a sequence of values over time steps.
        Can be used afterward for visualization (e.g., plotting of value over time).

        Args:
            timestep: Time step which the scalar value corresponds to (x-value)
            value_scalar: Scalar value which should be logged (y-value)
            value_name: Name of scalar value (e.g., "avg. sum of reward")
        """
        self.logging_history[value_name].append((timestep, value_scalar))

    @abstractmethod
    def upload(self, connector_config: UploadConfig, agent, evaluation_environment, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def download(self, connector_config: DownloadConfig, *args, **kwargs) -> Path:
        raise NotImplementedError
