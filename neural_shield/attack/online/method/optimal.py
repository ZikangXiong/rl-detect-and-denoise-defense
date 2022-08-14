import numpy as np

from neural_shield.attack.common.em_optimal_attack import EMOptimalAttackPolicy
from neural_shield.attack.online.base import AttackBase, VulnerableFuncBase


class OptimalAttack(AttackBase):
    """
    https://openreview.net/pdf?id=sCZbhBvqQaU
    """

    def __init__(self,
                 attack_policy: EMOptimalAttackPolicy,
                 vun_func: VulnerableFuncBase,
                 attack_time_limit: int,
                 epsilon: float,
                 initial_safe_step: int = 0):

        self.attack_policy = attack_policy

        super(OptimalAttack, self).__init__(epsilon, None, None, None,
                                            attack_time_limit, vun_func,
                                            "obs", initial_safe_step)

    def attack_obs(self, obs: np.ndarray):
        if self.attack_count < self.attack_time_limit:
            vulnerable = self.vun_func(obs)
        else:
            vulnerable = False

        if vulnerable and self.attack_count < self.attack_time_limit:
            pert_obs = self.attack_policy.attack(obs)
            self.attack_count += 1

            return vulnerable, pert_obs
        else:
            return vulnerable, obs

    def attack_action(self, obs: np.ndarray, action: np.ndarray):
        raise NotImplementedError("Do not support attack action")
