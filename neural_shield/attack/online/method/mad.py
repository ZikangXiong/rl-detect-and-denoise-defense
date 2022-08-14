import numpy as np

from neural_shield.attack.common.network import ControllerNet
from neural_shield.attack.common.pgd import mad_pgd
from neural_shield.attack.online.base import AttackBase, VulnerableFuncBase


class MADAttack(AttackBase):
    """
        https://arxiv.org/pdf/1702.02284.pdf
        Maximize the output action's distance to the unattacked action
    """

    def __init__(self,
                 controller_net: ControllerNet,
                 vun_func: VulnerableFuncBase,
                 attack_type: str,
                 attack_time_limit: int,
                 epsilon: float,
                 attack_norm: int = None,
                 attack_lr: float = None,
                 pgd_max_iter: int = None,
                 initial_safe_step: int = 0):

        self.controller_net = controller_net

        super(MADAttack, self).__init__(epsilon, attack_norm, attack_lr, pgd_max_iter,
                                        attack_time_limit, vun_func, attack_type, initial_safe_step)

    def attack_obs(self, obs: np.ndarray):
        if self.attack_count < self.attack_time_limit:
            vulnerable = self.vun_func(obs)
        else:
            vulnerable = False

        if vulnerable and self.attack_count < self.attack_time_limit:
            pert_obs = mad_pgd(obs,
                               self.controller_net,
                               self.attack_norm,
                               self.epsilon,
                               self.attack_lr,
                               self.pgd_max_iter)
            # pert_obs = obs + self.epsilon * np.random.uniform(-1, 1, size=obs.shape)
            self.attack_count += 1

            return vulnerable, pert_obs
        else:
            return vulnerable, obs

    def attack_action(self, obs: np.ndarray, action: np.ndarray):
        raise NotImplementedError("MAD is an attack working on observation space")
