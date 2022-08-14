import os
import pickle
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import ray

from neural_shield.benchmark import get_env
from neural_shield.benchmark.base import BenchBase
from neural_shield.config import attack_config, n_cpu


class AdversarialPlanner(ABC):
    def __init__(self, env: BenchBase, max_horizon: int):
        self.env = env
        self.max_horizon = max_horizon

    def plan(self, initial_state: np.ndarray, length=np.inf) -> List:
        actions = []
        obs = self.env.reset(state=initial_state)

        plan_horizon = min(length, self.max_horizon)
        for _ in range(plan_horizon):
            action = self.act(obs)
            actions.append(action)
            obs, _, done, _ = self.env.step(action)
            if done:
                break

        return actions

    @abstractmethod
    def act(self, obs: np.ndarray) -> np.ndarray:
        pass

    def save(self, path: str):
        dirname = os.path.dirname(path)
        os.makedirs(dirname, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(self, f)

        print(f"planner was saved to {path}")

    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)


class CEMPlanner(AdversarialPlanner):
    """CEM planner parallelized by ray
    ref: https://github.com/ADGEfficiency/cem
    """

    def __init__(self, env_id):
        env = get_env(env_id)
        self.config = attack_config["cem_planner"][env_id]
        max_horizon = self.config["max_horizon"]

        super(CEMPlanner, self).__init__(env, max_horizon)
        self.elite_frac = self.config["elite_frac"]
        self.population_num = self.config["population_num"]
        self.exploration_noise = self.config["exploration_noise"]
        self.verbose = self.config["verbose"]

        self.obs_len = self.env.observation_space.shape[0]
        self.action_len = self.env.action_space.shape[0]
        theta_shape = (self.obs_len + 1) * self.action_len
        self.means = np.random.uniform(-1, 1, size=theta_shape)
        self.stds = np.ones(theta_shape)

        self.rollout_workers = [RolloutWorker.remote(
            env.env_id) for _ in range(self.population_num)]

        self.history = {'epoch': [],
                        'avg_rew': [],
                        'std_rew': [],
                        'avg_elites': [],
                        'std_elites': []}
        self.epoch = 0

        self.elite_w = None
        self.elite_b = None

        if not ray.is_initialized():
            log_to_driver = self.verbose > 1
            ray.init(num_cpus=n_cpu, log_to_driver=log_to_driver)

    def evolve(self):
        for _ in range(self.config["n_gen"]):
            mode = self.config["mode"]
            assert mode in ["min", "max"]
            self.epoch += 1

            thetas = self.sample_population(batch_size=self.population_num, explore=True)

            reward_sums = self.evaluate_population(thetas, None)
            elite_num = int(self.elite_frac * self.population_num)
            if mode == "min":
                # minimize reward
                elite_indx = reward_sums.argsort()[:elite_num]
            elif mode == "max":
                # maximize reward
                elite_indx = reward_sums.argsort()[:elite_num:-1]
            elites = thetas[elite_indx]

            weights = reward_sums[elite_indx]
            weights = (weights - weights.min()) / (weights.max() - weights.min())
            self.means = np.average(elites, weights=weights, axis=0)
            self.stds = elites.std(axis=0)

            self.summary(reward_sums, elite_indx)

        print("final generation reward mean:", self.history["avg_rew"][-1])
        print("final generation reward std:", self.history["std_rew"][-1])

    def sample_population(self, batch_size, explore):
        thetas = np.random.normal(
            loc=self.means,
            scale=np.abs(self.stds),
            size=(batch_size, len(self.means)))

        if explore:
            # avoid local minimum
            exploration_noise = self.exploration_noise * np.random.uniform(-1, 1, size=thetas.shape)
        else:
            exploration_noise = 0

        return thetas + exploration_noise

    def evaluate_population(self, thetas, initial_state) -> np.ndarray:
        reward_ids = []

        for i in range(self.population_num):
            reward_id = self.rollout_workers[i].rollout.remote(
                thetas[i], self.max_horizon, initial_state, self.obs_len, self.action_len)

            reward_ids.append(reward_id)

        rewards = np.array([ray.get(_id) for _id in reward_ids])
        return rewards

    def summary(self, rewards, elite_idx):
        self.history['epoch'].append(self.epoch)
        self.history['avg_rew'].append(np.mean(rewards))
        self.history['std_rew'].append(np.std(rewards))
        self.history['avg_elites'].append(np.mean(rewards[elite_idx]))
        self.history['std_elites'].append(np.std(rewards[elite_idx]))

        if self.verbose > 0:
            print(
                'epoch {} - pop mean: {:2.1f} pop std: {:2.1f} -'
                'elite mean: {:2.1f} elite std: {:2.1f} '.format(
                    self.epoch,
                    self.history['avg_rew'][-1],
                    self.history['std_rew'][-1],
                    self.history['avg_elites'][-1],
                    self.history['std_elites'][-1]
                )
            )

    def act(self, obs: np.ndarray) -> np.ndarray:
        if self.elite_w is None:
            self.elite_w = self.means[:self.obs_len*self.action_len].reshape(
                [self.obs_len, self.action_len])
            self.elite_b = self.means[self.obs_len*self.action_len:]
        action = obs @ self.elite_w + self.elite_b
        action = action.clip(self.env.action_space.low, self.env.action_space.high)

        return action

    def save(self, path: str):
        temp = self.rollout_workers, self.env
        self.rollout_workers, self.env = [], self.env.env_id
        super(CEMPlanner, self).save(path)
        self.rollout_workers, self.env = temp

    @classmethod
    def load(cls, path: str):
        planner = super(CEMPlanner, cls).load(path)
        planner.env = get_env(planner.env)  # the env was stroed as env_id
        return planner


@ray.remote
class RolloutWorker:
    def __init__(self, env_id: str):
        self.env = get_env(env_id)

    def rollout(self, theta, max_horizon, initial_state, obs_len, action_len):
        w = theta[:obs_len*action_len].reshape([obs_len, action_len])
        b = theta[obs_len*action_len:]

        obs = self.env.reset(state=initial_state)

        reward_sum = 0
        for _ in range(max_horizon):
            action = obs @ w + b
            action = action.clip(self.env.action_space.low, self.env.action_space.high)
            obs, reward, done, _ = self.env.step(action)
            reward_sum += reward
            if done:
                break

        return reward_sum
