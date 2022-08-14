import gym
import numpy as np

from neural_shield.benchmark.base import BenchBase


class Hopper(BenchBase):
    """
        benchmark link:
        https://github.com/openai/gym/blob/master/gym/envs/mujoco/hopper_v3.py
    """

    def __init__(self):
        env_id = "Hopper-v3"
        env = gym.make(env_id)
        super(Hopper, self).__init__(env, env_id)

    def get_state(self):
        return self.env.sim.get_state()

    def set_state(self, state):
        self.env.sim.set_state(state)

    def get_obs(self) -> np.ndarray:
        obs = self.env._get_obs()
        return obs
