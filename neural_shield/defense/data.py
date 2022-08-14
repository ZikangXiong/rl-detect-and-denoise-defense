import os

import numpy as np
import ray
import torch as th
from torch.utils.data import Dataset

from neural_shield.benchmark import get_env
from neural_shield.config import (default_device, defense_config, get_path,
                                  n_cpu)
from neural_shield.controller.pretrain import load_model


@ray.remote
class DataCollectionWorker:
    def __init__(self, env_id, algo):
        self.env = get_env(env_id)
        self.policy = load_model(env_id, algo)

        self.trajs = []
        self.action_sequences = []

    def rollout(self, n_epoch, n_step):
        for _ in range(n_epoch):
            traj = []
            actions = []
            obs = self.env.reset()
            traj.append(obs)

            terminate_early = False
            for _ in range(n_step):
                a, _ = self.policy.predict(obs)
                obs, rew, done, _ = self.env.step(a)

                traj.append(obs)
                actions.append(a)

                if done:
                    terminate_early = True
                    break

            if not terminate_early:
                self.trajs.append(traj)
                self.action_sequences.append(actions)

        return self.trajs, self.action_sequences

    def get_history(self):
        return self.trajs, self.action_sequences


def collect_data(env_id: str, algo: str):
    if not ray.is_initialized:
        ray.init(num_cpus=n_cpu)

    workers = [DataCollectionWorker.remote(env_id, algo) for _ in range(n_cpu)]
    config = defense_config["data_collection"][env_id]
    n_epoch = config['n_epoch']
    n_step = config['n_step']

    n_epoch_each_worker = max(n_epoch // n_cpu, 1)

    res_list = []
    for w in workers:
        res_id = w.rollout.remote(n_epoch_each_worker, n_step)
        res_list.append(res_id)

    trajs = []
    actions = []
    for res_id in res_list:
        t, a = ray.get(res_id)
        trajs.append(t)
        actions.append(a)

    trajs = np.concatenate(trajs)
    actions = np.concatenate(actions)

    pathes = get_path("data_collection", env_id, algo)
    traj_path = pathes["trajs"]
    action_path = pathes["actions"]

    os.makedirs(os.path.dirname(traj_path), exist_ok=True)
    np.save(f"{traj_path}", trajs)
    np.save(f"{action_path}", actions)

    return trajs, actions


class DetectorDataset(Dataset):
    def __init__(self, data_array: np.ndarray):
        self.data_array = data_array
        self.running_mean = data_array.mean(axis=(0, 1))
        self.running_std = data_array.std(axis=(0, 1))

    def __getitem__(self, index):
        return th.tensor(self.data_array[index],
                         dtype=th.float32,
                         device=default_device)

    def __len__(self):
        return len(self.data_array)


class RepairerDataset(Dataset):
    def __init__(self, normal_data_array: np.ndarray,
                 mad_data_array: np.ndarray,
                 vf_data_array: np.ndarray):
        self.normal_data_array = normal_data_array
        self.running_mean = normal_data_array.mean(axis=(0, 1))
        self.running_std = normal_data_array.std(axis=(0, 1))

        self.mad_data_array = mad_data_array
        self.vf_data_array = vf_data_array

    def __getitem__(self, index):
        normal_data = th.tensor(self.normal_data_array[index],
                                dtype=th.float32,
                                device=default_device)
        adv_data = [th.tensor(self.mad_data_array[index],
                              dtype=th.float32,
                              device=default_device),
                    th.tensor(self.vf_data_array[index],
                              dtype=th.float32,
                              device=default_device)]
        return normal_data, adv_data

    def __len__(self):
        return len(self.normal_data_array)
