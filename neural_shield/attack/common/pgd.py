import numpy as np
import torch as th

from neural_shield.attack.common.network import ControllerNet, ValueNet
from neural_shield.attack.common.utils import normalize_action


def grad_attack(loss_fn, x: th.Tensor, epsilon: float, lr: float, max_iter: int):
    upper = x + epsilon
    lower = x - epsilon

    x.stateful = True
    if max_iter < 0:
        # FGSM
        loss = loss_fn(x)
        x.stateful = False
        grad = th.autograd.grad(loss, x)[0]
        new_x = x - lr * grad.sign()
        new_x = th.clamp(new_x, min=lower, max=upper)
        x.data = new_x.data
    else:
        for _ in range(max_iter):
            # import ipdb
            # ipdb.set_trace()
            loss = loss_fn(x)
            x.stateful = False
            grad = th.autograd.grad(loss, x)[0]
            new_x = x - lr * grad
            new_x = th.clamp(new_x, min=lower, max=upper)
            x.data = new_x.data

    return x


def enchanting_pgd(obs: np.ndarray,
                   target_action: np.ndarray,
                   controller_net: ControllerNet,
                   epsilon: float,
                   norm: float,
                   lr: float,
                   max_iter: int,
                   env,
                   detector=None):

    device = controller_net.policy.device
    obs_tensor = th.tensor(obs,
                           dtype=th.float32,
                           device=device,
                           requires_grad=True)

    if controller_net.squashed_action:
        target_action = normalize_action(env, target_action, clip_first=True)
    target_action = th.tensor(target_action,
                              dtype=th.float32,
                              device=device)

    for p in controller_net.policy.parameters():
        p.requires_grad = False

    def loss(x: th.Tensor) -> th.Tensor:
        pred_action = controller_net.forward(x)
        return th.norm(target_action - pred_action, p=norm)

    obs_tensor = grad_attack(loss, obs_tensor, epsilon, lr, max_iter)
    new_obs = obs_tensor.detach().cpu().numpy()

    return new_obs


def mad_pgd(obs: np.ndarray,
            controller_net: ControllerNet,
            norm: int,
            epsilon: float,
            lr: float,
            max_iter: int,
            detector=None):
    device = controller_net.policy.device
    obs_tensor = th.tensor(obs,
                           dtype=th.float32,
                           device=device,
                           requires_grad=True)

    for p in controller_net.policy.parameters():
        p.requires_grad = False

    with th.no_grad():
        distribution_mean = controller_net.forward(obs_tensor, deterministic=True)

    def loss(x: th.Tensor) -> th.Tensor:
        new_distribution_mean = controller_net.forward(x, deterministic=True)
        # At first step, the gradient of f(x) - f(x) is always 0. Thus, added 1e-5.
        diff = th.norm(new_distribution_mean + 1e-5 - distribution_mean, p=norm)

        return -diff

    obs_tensor = grad_attack(loss, obs_tensor, epsilon, lr, max_iter)
    new_obs = obs_tensor.detach().cpu().numpy()

    return new_obs


def value_func_pgd(obs: np.ndarray,
                   value_net: ValueNet,
                   epsilon: float,
                   lr: float,
                   max_iter: int,
                   detector=False):
    device = value_net.policy.device
    obs_tensor = th.tensor(obs,
                           dtype=th.float32,
                           device=device,
                           requires_grad=True)

    for p in value_net.policy.parameters():
        p.requires_grad = False

    def loss(x: th.Tensor) -> th.Tensor:
        loss = th.mean(value_net.forward(x))
        return loss

    obs_tensor = grad_attack(loss, obs_tensor, epsilon, lr, max_iter)
    new_obs = obs_tensor.detach().cpu().numpy()

    return new_obs
