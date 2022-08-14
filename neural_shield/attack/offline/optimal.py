import sys

import numpy as np

from neural_shield.attack.common.em_optimal_attack import EMOptimalAttackPolicy
from neural_shield.attack.common.utils import sizeof_fmt
from neural_shield.config import attack_config, get_path


def offline_optimal(env_id, algo):
    # load the offline trajectory dataset
    traj_path = get_path("data_collection", env_id, algo)["trajs"]
    traj_dataset = np.load(traj_path)
    print("trajectory dataset size is", sizeof_fmt(sys.getsizeof(traj_dataset)))

    config = attack_config["attack_constraint"][env_id]["obs"]
    print(config)
    traj_dataset_shape = traj_dataset.shape
    traj_dataset = traj_dataset.reshape([-1, traj_dataset_shape[-1]])

    save_path = get_path("optimal_attack_policy", env_id, algo)["model"]
    attack_policy = EMOptimalAttackPolicy.load(save_path)

    adv_dataset = attack_policy.attack(traj_dataset)
    adv_dataset = adv_dataset.reshape(traj_dataset_shape)

    # save attacked dataset
    dataset_path = get_path("offline_attack", env_id, algo)["optimal_trajs"]
    np.save(dataset_path, adv_dataset)
