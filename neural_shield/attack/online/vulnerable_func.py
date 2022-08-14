import numpy as np
import torch as th

from neural_shield.attack.common.network import ValueNet
from neural_shield.attack.online.base import VulnerableFuncBase


class VFBasedVunFunc(VulnerableFuncBase):
    def __init__(self, vf: ValueNet, threshold):
        self.vf = vf
        self.threshold = threshold

    def __call__(self, obs: np.ndarray) -> bool:
        device = self.vf.policy.device
        obs_tensor = th.tensor(obs, dtype=th.float32, device=device)
        value = self.vf.forward(obs_tensor)
        value = value.detach().cpu().numpy()

        # print(value)

        return (value < self.threshold).flatten().item()
