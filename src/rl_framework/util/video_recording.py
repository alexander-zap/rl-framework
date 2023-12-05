import logging
import os
import tempfile
from pathlib import Path

import imageio
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecVideoRecorder


# TODO: use video_length for simple replay and use video_length *correctly* for SB3-replay
def record_video(
    agent, evaluation_environment, file_path: Path, fps: int = 1, video_length=1000, sb3_replay: bool = False
):
    """
    Generate a replay video of the agent.

    Args:
        agent (Agent): Agent to record video for.
        evaluation_environment (Environment): Environment used for evaluation and clip creation.
        file_path (Path): Path where video should be saved to.
        fps (int): How many frame per seconds to record the video replay.
        video_length (int): How many time steps of the agent should be recorded
        sb3_replay (bool): Determines recording mode
            If True: Use SB3's VecVideoRecorder and FFMPEG to record video
            If False: Use simple recording method (saving RGB outputs from environment render)
    """

    if not sb3_replay:
        images = []
        done = False
        observation, _ = evaluation_environment.reset()
        img = evaluation_environment.render()
        images.append(img)
        while not done:
            action = agent.choose_action(observation)
            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = evaluation_environment.step(action)
            done = terminated or truncated
            img = evaluation_environment.render()
            images.append(img)
        imageio.mimsave(file_path, [np.array(img) for i, img in enumerate(images)], fps=fps)
    else:
        # Create a Stable-baselines3 vector environment (VecEnv)
        if not isinstance(evaluation_environment, VecEnv):
            evaluation_environment = DummyVecEnv([lambda: evaluation_environment])

        # This is another temporary directory for video outputs (replay video file will be copied to repository path).
        # Copying is required, since SB3 creates other files which we don't want in the repo.
        with tempfile.TemporaryDirectory() as tmpdirname:
            env = VecVideoRecorder(
                evaluation_environment,
                tmpdirname,
                record_video_trigger=lambda x: x == 0,
                video_length=video_length,
                name_prefix="",
            )

            obs = env.reset()

            try:
                for _ in range(video_length):
                    action = agent.algorithm.choose_action(obs)
                    obs, _, _, _ = env.step(np.array([action]))

                # Save the video
                env.close()

                # Convert the video with x264 codec
                inp = env.video_recorder.path
                out = file_path
                os.system(f"ffmpeg -y -i {inp} -vcodec h264 {out}")

            except KeyboardInterrupt:
                pass
            except Exception as e:
                logging.error(str(e))
