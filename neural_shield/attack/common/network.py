import torch as th

from neural_shield.config import default_device
from neural_shield.controller.pretrain import load_model
from neural_shield.defense.defense import Defense


class ControllerNet:
    def __init__(self, env_id, algo, load_defense=False):
        self.model = load_model(env_id, algo, device=default_device)
        self.policy = self.model.policy

        if not load_defense:
            self.defense = None
        else:
            self.defense = Defense(env_id, algo)
            self.reset()

    def reset(self):
        if self.defense is not None:
            self.defense.reset()

    def forward(self, x: th.Tensor, deterministic=False) -> th.Tensor:
        if hasattr(x, "stateful"):
            stateful = x.stateful
        else:
            stateful = False

        if len(x.shape) == 1:
            x = x.view([1, -1])
        if self.defense is not None:
            # pgd can call this function iteratively
            x = self.defense.repairer.forward(x, stateful)
        if hasattr(self.policy, "critic"):
            return self.policy.forward(x, deterministic=deterministic)
        elif hasattr(self.policy, "value_net"):
            action, _, _ = self.policy.forward(x, deterministic=deterministic)
            return action
        else:
            import ipdb
            ipdb.set_trace()
            return 0  # noqa

    @property
    def squashed_action(self) -> bool:
        return self.policy.squash_output


class ValueNet:
    def __init__(self, env_id, algo, load_defense=False):
        self.model = load_model(env_id, algo, device=default_device)
        self.policy = self.model.policy

        if not load_defense:
            self.defense = None
        else:
            self.defense = Defense(env_id, algo)
            self.reset()

    def reset(self):
        if self.defense is not None:
            self.defense.reset()

    def forward(self, x: th.Tensor) -> th.Tensor:
        if hasattr(x, "stateful"):
            stateful = x.stateful
        else:
            stateful = False

        if len(x.shape) == 1:
            x = x.view([1, -1])

        if hasattr(self.policy, "critic"):
            if self.defense is not None:
                x = self.defense.repairer.forward(x, stateful)

            # actor-critic policy (TD3, DDPG, SAC)
            action = self.policy.actor(x)
            q_star = self.policy.critic(x.detach(), action)[0]
            return q_star
        elif hasattr(self.policy, "value_net"):
            # Policy with value net (PPO, A2C)
            _, value, _ = self.policy.forward(x)
            return value
        else:
            import ipdb
            ipdb.set_trace()
            return 0  # noqa
