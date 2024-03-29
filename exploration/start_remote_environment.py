import logging
import sys

from remote_gym import start_as_remote_environment

from rl_framework.environment.gym_environment import GymEnvironmentWrapper

ENV_ID = "CarRacing-v2"

# Create logging handler to output logs to stdout
root = logging.getLogger()
root.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)

if __name__ == "__main__":
    server = start_as_remote_environment(
        url="localhost", port=56789, local_environment=GymEnvironmentWrapper(ENV_ID, render_mode="rgb_array")
    )

    try:
        server.wait_for_termination()
    except Exception as e:
        server.stop(None)
        logging.exception(e)
