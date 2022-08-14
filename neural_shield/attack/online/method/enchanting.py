from typing import Tuple

import numpy as np

from neural_shield.attack.common.adversarial_planner import AdversarialPlanner
from neural_shield.attack.common.network import ControllerNet
from neural_shield.attack.common.pgd import enchanting_pgd
from neural_shield.attack.online.base import AttackBase, VulnerableFuncBase


class EnchantingAttack(AttackBase):
    def __init__(self,
                 adversarial_planner: AdversarialPlanner,
                 controller_net: ControllerNet,
                 vun_func: VulnerableFuncBase,
                 attack_type: str,
                 attack_time_limit: int,
                 epsilon: float,
                 attack_norm: int = None,
                 attack_lr: float = None,
                 pgd_max_iter: int = None,
                 initial_safe_step: int = 0):

        self.adversarial_planner = adversarial_planner
        self.env = adversarial_planner.env
        self.controller_net = controller_net
        self.exact_enchanting = False

        super(EnchantingAttack, self).__init__(
            epsilon, attack_norm, attack_lr, pgd_max_iter,
            attack_time_limit, vun_func, attack_type, initial_safe_step)

    def enable_exact_enchanting(self):
        self.exact_enchanting = True

    def disable_exact_enchanting(self):
        self.exact_enchanting = False

    def attack_obs(self, obs: np.ndarray) -> Tuple[bool, np.ndarray]:
        if self.attack_count < self.attack_time_limit:
            vulnerable = self.vun_func(obs)
        else:
            vulnerable = False

        if vulnerable and self.attack_count < self.attack_time_limit:
            target_action = self.adversarial_planner.act(obs)
            pert_obs = enchanting_pgd(obs,
                                      target_action,
                                      self.controller_net,
                                      self.epsilon,
                                      self.attack_norm,
                                      self.attack_lr,
                                      self.pgd_max_iter,
                                      self.env)
            self.attack_count += 1

            return vulnerable, pert_obs
        else:
            return vulnerable, obs

    def attack_action(self, obs: np.ndarray, action: np.ndarray) -> Tuple[bool, np.ndarray]:
        if self.attack_count < self.attack_time_limit:
            vulnerable = self.vun_func(obs)
        else:
            vulnerable = False

        if vulnerable and self.attack_count < self.attack_time_limit:
            target_action = self.adversarial_planner.act(obs)
            self.attack_count += 1

            if self.exact_enchanting:
                return vulnerable, target_action
            else:
                return vulnerable, target_action.clip(action - self.epsilon, action + self.epsilon)
        else:
            return vulnerable, action
