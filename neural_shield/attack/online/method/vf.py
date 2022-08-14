import numpy as np

from neural_shield.attack.common.network import ValueNet
from neural_shield.attack.common.pgd import value_func_pgd
from neural_shield.attack.online.base import AttackBase, VulnerableFuncBase


class ValueFuncAttack(AttackBase):
    """
    https://arxiv.org/pdf/1712.03632.pdf
    https://arxiv.org/pdf/1705.06452.pdf
    """

    def __init__(self,
                 value_net: ValueNet,
                 vun_func: VulnerableFuncBase,
                 attack_type: str,
                 attack_time_limit: int,
                 epsilon: float,
                 attack_lr: float = None,
                 pgd_max_iter: int = None,
                 initial_safe_step: int = 0):

        self.value_net = value_net

        super(ValueFuncAttack, self).__init__(epsilon, None, attack_lr, pgd_max_iter,
                                              attack_time_limit, vun_func, attack_type, initial_safe_step)

    def attack_obs(self, obs: np.ndarray):
        if self.attack_count < self.attack_time_limit:
            vulnerable = self.vun_func(obs)
        else:
            vulnerable = False

        if vulnerable and self.attack_count < self.attack_time_limit:
            pert_obs = value_func_pgd(obs,
                                      self.value_net,
                                      self.epsilon,
                                      self.attack_lr,
                                      self.pgd_max_iter)
            self.attack_count += 1

            return vulnerable, pert_obs
        else:
            return vulnerable, obs

    def attack_action(self, obs: np.ndarray, action: np.ndarray):
        raise NotImplementedError()


def learn_q_func(env_id, algo):
    pass
