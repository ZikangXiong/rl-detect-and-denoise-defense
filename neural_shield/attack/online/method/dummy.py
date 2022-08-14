import numpy as np

from neural_shield.attack.online.base import AttackBase


class DummyAttack(AttackBase):
    def __init__(self):
        super(DummyAttack, self).__init__(0, None, 0, 0, 1000, None, "obs", 0)

    def attack_obs(self, obs: np.ndarray):
        self.attack_count += 1
        return False, obs

    def attack_action(self, obs: np.ndarray, action: np.ndarray):
        raise NotImplementedError()
