from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class VulnerableFuncBase(ABC):
    """
    The attack will only be triggered when the agent is vulnerable.
    This function decides whether the agent is vulnerable.
    See https://arxiv.org/abs/1703.06748
    """
    @abstractmethod
    def __call__(self, obs: np.ndarray) -> bool:
        """
        Args:
            obs (np.ndarray): observation

        Returns:
            bool: returning True means vulnerable
        """
        pass


class AttackBase(ABC):
    def __init__(self,
                 epsilon: float,
                 attack_norm: int,
                 attack_lr: float,
                 pgd_max_iter: int,
                 attack_time_limit: int,
                 vun_func: VulnerableFuncBase,
                 attack_type: str,
                 initial_safe_step: int):
        self.epsilon = epsilon
        self.vun_func = vun_func
        self.attack_norm = attack_norm
        self.pgd_max_iter = pgd_max_iter
        self.attack_lr = attack_lr
        self.attack_time_limit = attack_time_limit
        self.attack_count = 0
        self.attack_type = attack_type
        self.initial_safe_step = initial_safe_step

    def reset(self):
        self.attack_count = 0
        if hasattr(self, "controller_net"):
            self.controller_net.reset()
        if hasattr(self, "value_net"):
            self.value_net.reset()

    def set_attack_type(self, attack_type: str):
        assert attack_type in ("obs", "action")
        self.attack_type = attack_type

    def get_attack_type(self) -> str:
        return self.attack_type

    @abstractmethod
    def attack_obs(self, obs: np.ndarray) -> Tuple[bool, np.ndarray]:
        """Output perturbed observation

        Args:
            obs (np.ndarray): observation

        Returns:
            bool: attack happens or not
            np.ndarray: the perturbed observation
        """
        pass

    @abstractmethod
    def attack_action(self, obs: np.ndarray, action: np.ndarray) -> Tuple[bool, np.ndarray]:
        """Output perturbed action

        Args:
            obs (np.ndarray): observation
            action (np.ndarray): action

        Returns:
            bool: attack happens or not
            np.ndarray: the perturbed action
        """
        pass
