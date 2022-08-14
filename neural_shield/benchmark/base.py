from abc import ABC, abstractmethod

import numpy as np
from gym import Env


class BenchBase(ABC):
    def __init__(self, env: Env, env_id: str):
        self.env = env.unwrapped
        # env_id will be used for create envrionment when it is not serializable
        self.env_id = env_id
        self._expose()

    def _expose(self):
        self.__dict__.update(self.env.__dict__.copy())

        if not hasattr(self, 'step'):
            self.step = self.env.step
        if not hasattr(self, 'render'):
            self.render = self.env.render

    def reset(self, state=None):
        """Support the reset to any given state.
        Returns:
            np.ndarray: observation
        """

        if state is None:
            return self.env.reset()
        else:
            self.set_state(state)
            return self.get_obs()

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def set_state(self, state):
        pass

    @abstractmethod
    def get_obs(self) -> np.ndarray:
        pass
