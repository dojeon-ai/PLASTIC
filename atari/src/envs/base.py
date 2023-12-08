from abc import *
from collections import namedtuple


EnvStep = namedtuple("EnvStep",
    ["observation", "reward", "done", "env_info"])
EnvInfo = namedtuple("EnvInfo", ["game_score", "traj_done"])  # Define in env file.
EnvSpaces = namedtuple("EnvSpaces", ["observation", "action"])


class BaseEnv():
    """
    The learning task, e.g. an MDP containing a transition function T(state,
    action)-->state'.  Has a defined observation space and action space.
    """
    @classmethod
    def get_name(cls):
        return cls.name

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def spaces(self):
        return EnvSpaces(
            observation=self.observation_space,
            action=self.action_space,
        )

    @property
    def horizon(self):
        """Episode horizon of the environment, if it has one."""
        raise NotImplementedError

    def close(self):
        """Any clean up operation."""
        pass