import gym
import numpy as np

from neural_shield.benchmark.base import BenchBase


class Pendulum(BenchBase):
    """
        benchmark link:
        https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
    """

    def __init__(self):
        env_id = "Pendulum-v0"
        env = gym.make(env_id)
        super(Pendulum, self).__init__(env, env_id)

    def reset(self, state=None):
        """
        Returns:
            np.ndarray: observation
        """

        if state is None:
            return self.env.reset()
        else:
            self.env.state = state
            return self.env._get_obs()

    def get_state(self):
        return self.env.state

    def set_state(self, state):
        self.env.state = state

    def get_obs(self) -> np.ndarray:
        return self.env._get_obs()
