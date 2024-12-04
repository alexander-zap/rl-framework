import copy
from itertools import tee
from typing import Generator, Iterable, List, Sequence, Sized, Tuple

import d3rlpy
import imitation
import numpy as np
from imitation.data import serialize

GenericEpisode = List[Tuple[object, object, object, float, bool, bool, dict]]


class EpisodeSequence(Iterable[GenericEpisode], Sized):
    """
    Class to load, transform and iterate over episodes, optimized for memory efficiency.
        - Using HuggingFace "load_from_disk" for loading
        - Using generators for underlying data management
        - Format changing transformations also return generators


    Each episode consists of a sequence, which has the following format:
        [
            (obs_t0, action_t0, next_obs_t0, reward_t0, terminated_t0, truncated_t0, info_t0),
            (obs_t1, action_t1, next_obs_t1, reward_t1, terminated_t1, truncated_t1, info_t1),
            ...
        ]
        Interpretation: Transition from obs to next_obs with action, receiving reward.
            Additional information returned about transition to next_obs: terminated, truncated and info.
    """

    def __init__(self):
        def empty() -> Generator:
            yield from ()

        self._episode_generator: Generator[GenericEpisode, None, None] = empty()
        self._len = 0

    def __len__(self):
        return self._len

    def __iter__(self):
        self._episode_generator, episode_generator_copy = tee(self._episode_generator)
        return episode_generator_copy

    @staticmethod
    def from_episode_generator(episode_generator: Generator[GenericEpisode, None, None]) -> "EpisodeSequence":
        """
        Initialize an EpisodeSequence based on a provided generator (of GenericEpisode objects).

        Args:
            episode_generator: Custom episode generator generating GenericEpisodes every time __next__() is called.

        Returns:
            episode_sequence: Representation of episode sequence (this class).
        """
        episode_generator, episode_generator_copy = tee(episode_generator)
        len_episodes = 0
        for _ in episode_generator_copy:
            len_episodes += 1

        episode_sequence = EpisodeSequence()
        episode_sequence._episode_generator = episode_generator
        episode_sequence._len = len_episodes
        return episode_sequence

    @staticmethod
    def from_episodes(episodes: Sequence[GenericEpisode]) -> "EpisodeSequence":
        """
        Initialize an EpisodeSequence based on a sequence of GenericEpisode objects.

        Args:
            episodes (Sequence[GenericEpisode]): Episodes in generic format.

        Returns:
            episode_sequence: Representation of episode sequence (this class).
        """

        def generate_episodes(generic_episodes: Sequence[GenericEpisode]) -> Generator[GenericEpisode, None, None]:
            for episode in generic_episodes:
                yield episode

        episode_sequence = EpisodeSequence()
        episode_sequence._episode_generator = generate_episodes(episodes)
        episode_sequence._len = len(episodes)
        return episode_sequence

    @staticmethod
    def from_dataset(file_path: str) -> "EpisodeSequence":
        """
        Initialize an EpisodeSequence based on provided huggingface dataset path.

        Episode sequences are loaded from a provided file path in the agent section of the config.
        Files of recorded episode sequences are generated by saving Trajectory objects (`imitation` library).
        https://imitation.readthedocs.io/en/latest/main-concepts/trajectories.html#storing-loading-trajectories

        Args:
            file_path (str): Path to huggingface dataset recording of episodes.

        Returns:
            episode_sequence: Representation of episode sequence (this class).
        """

        def generate_episodes(
            imitation_trajectories: Sequence[imitation.data.types.TrajectoryWithRew],
        ) -> Generator[GenericEpisode, None, None]:
            for trajectory in imitation_trajectories:
                obs = trajectory.obs[:-1]
                acts = trajectory.acts
                rews = trajectory.rews
                next_obs = trajectory.obs[1:]
                terminations = np.zeros(len(trajectory.acts), dtype=bool)
                truncations = np.zeros(len(trajectory.acts), dtype=bool)
                terminations[-1] = trajectory.terminal
                truncations[-1] = not trajectory.terminal
                infos = np.array([{}] * len(trajectory)) if trajectory.infos is None else trajectory.infos
                episode: GenericEpisode = list(zip(*[obs, acts, next_obs, rews, terminations, truncations, infos]))
                yield episode

        episode_sequence = EpisodeSequence()
        trajectories = serialize.load_with_rewards(file_path)
        episode_sequence._episode_generator = generate_episodes(trajectories)
        episode_sequence._len = len(trajectories)
        return episode_sequence

    def save(self, file_path):
        """
        Save episode sequence into a file, saved as HuggingFace dataset.

        Args:
            file_path: File path and file name to save episode sequence to.
        """
        trajectories: Sequence[imitation.data.types.TrajectoryWithRew] = [
            trajectory for trajectory in self.to_imitation_episodes()
        ]
        serialize.save(file_path, trajectories)

    def to_imitation_episodes(self) -> Generator[imitation.data.types.TrajectoryWithRew, None, None]:
        self._episode_generator, episode_generator_copy = tee(self._episode_generator)

        for generic_episode in episode_generator_copy:
            observations, actions, next_observations, rewards, terminations, truncations, infos = (
                np.array(x) for x in list(zip(*generic_episode))
            )
            all_observations = np.vstack([copy.deepcopy(observations), copy.deepcopy(next_observations[-1])])
            episode_trajectory = imitation.data.types.TrajectoryWithRew(
                obs=all_observations, acts=actions, rews=rewards, infos=infos, terminal=terminations[-1]
            )
            yield episode_trajectory

    def to_d3rlpy_episodes(self) -> Generator[d3rlpy.dataset.components.Episode, None, None]:
        self._episode_generator, episode_generator_copy = tee(self._episode_generator)

        for generic_episode in episode_generator_copy:
            observations, actions, next_observations, rewards, terminations, truncations, infos = (
                np.array(x) for x in list(zip(*generic_episode))
            )
            episode = d3rlpy.dataset.components.Episode(
                observations=observations,
                actions=actions,
                rewards=rewards,
                terminated=terminations[-1],
            )
            yield episode

    def to_generic_episodes(self) -> Generator[GenericEpisode, None, None]:
        self._episode_generator, episode_generator_copy = tee(self._episode_generator)
        return episode_generator_copy
