from typing import Dict, Optional


def upload(agent, evaluation_environment, connector, connector_config):
    """Evaluate, generate a video and upload a model.

    Args:
        agent: Agent (and its .algorithm attribute) to be uploaded.
        evaluation_environment: Environment used for final evaluation and clip creation before upload.
        connector: Connector for uploading.
        connector_config: Configuration data for connector.
    """
    connector.upload(agent=agent, evaluation_environment=evaluation_environment, connector_config=connector_config)


def download(agent, algorithm_parameters: Optional[Dict], connector, connector_config):
    """
    Download a reinforcement learning agent and update the passed agent in-place.

    Args:
        agent (Agent): Agent instance to be replaced with the newly downloaded saved-agent.
            NOTE: Agent and Algorithm class need to be the same as the saved agent.
        algorithm_parameters (Optional[Dict]): Parameters to be set for the downloaded agent-algorithm.
        connector: Connector for downloading.
        connector_config: Configuration data for connector.
    """
    # Get the model from the Hub, download and cache the model on your local disk
    agent_file_path = connector.download(connector_config)
    agent.load_from_file(agent_file_path, algorithm_parameters)
