import os
import pickle

import numpy as np
import ray
import torch as th

from neural_shield.benchmark import get_env
from neural_shield.config import attack_config, n_cpu
from neural_shield.controller.pretrain import load_model

th.set_num_threads(1)


class EMOptimalAttackPolicy:
    def __init__(self, env_id, algo):
        config = attack_config["optimal_attack_policy"][env_id]
        self.elite_frac = config["elite_frac"]
        self.population_num = config["population_num"]
        self.exploration_noise = config["exploration_noise"]
        self.verbose = config["verbose"]
        self.max_horizon = config["max_horizon"]
        self.epsilon = attack_config["attack_constraint"][env_id]["obs"]["epsilon"]

        env = get_env(env_id)
        self.observation_space = env.observation_space
        self.obs_len = self.observation_space.shape[0]
        theta_shape = (self.obs_len + 1) * self.obs_len
        self.means = np.random.uniform(-1, 1, size=theta_shape)
        self.stds = np.ones(theta_shape)

        self.rollout_workers = [RolloutWorker.remote(env_id, algo)
                                for _ in range(self.population_num)]

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

            if self.verbose > 1:
                print("Using {n_cpu} cores")
            ray.init(num_cpus=n_cpu, log_to_driver=log_to_driver)

    def evolve(self, n_gen: int,
               initial_state: np.ndarray = None,
               mode="min"):
        if self.verbose > 1:
            print(f"evolve {n_gen} generation under {mode} mode")

        for _ in range(n_gen):
            assert mode in ["min", "max"]
            self.epoch += 1

            thetas = self.sample_population(batch_size=self.population_num, explore=True)

            reward_sums = self.evaluate_population(thetas, initial_state)
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
        # this can be performance bottomneck as the dimension increases
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
                thetas[i], self.epsilon, self.max_horizon, initial_state, self.obs_len)

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

    def attack(self, obs: np.ndarray) -> np.ndarray:
        if self.elite_w is None:
            self.elite_w = self.means[:self.obs_len*self.obs_len].reshape(
                [self.obs_len, self.obs_len])
            self.elite_b = self.means[self.obs_len*self.obs_len:]

        noise = obs @ self.elite_w + self.elite_b
        noise = noise.clip(-self.epsilon, self.epsilon)

        pert_obs = obs + noise
        pert_obs = pert_obs.clip(self.observation_space.low,
                                 self.observation_space.high)

        return pert_obs

    def save(self, path: str):
        temp = self.rollout_workers
        self.rollout_workers = []

        dirname = os.path.dirname(path)
        os.makedirs(dirname, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(self, f)

        self.rollout_workers = temp

        print(f"Optimal attack policy was saved to {path}")

    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)


@ray.remote
class RolloutWorker:
    def __init__(self, env_id: str, algo: str):
        self.env = get_env(env_id)
        self.victim_pi = load_model(env_id, algo, "cpu")
        self.initial_safe_step = attack_config["attack_constraint"][env_id]["obs"].get(
            "initial_safe_step", 0)

    def rollout(self, theta, epsilon, max_horizon, initial_state, obs_len):
        w = theta[:obs_len*obs_len].reshape([obs_len, obs_len])
        b = theta[obs_len*obs_len:]

        obs = self.env.reset(state=initial_state)

        reward_sum = 0
        for i in range(max_horizon):
            if i > self.initial_safe_step:
                noise = obs @ w + b
                noise = noise.clip(-epsilon, epsilon)
                obs += noise
                obs = obs.clip(self.env.observation_space.low, self.env.observation_space.high)

            action, _ = self.victim_pi.predict(obs)

            obs, reward, done, _ = self.env.step(action)
            reward_sum += reward

            if done:
                break

        return reward_sum
