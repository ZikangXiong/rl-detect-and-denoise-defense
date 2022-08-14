import sys

import numpy as np

from neural_shield.attack.common.network import ControllerNet
from neural_shield.attack.common.pgd import mad_pgd
from neural_shield.attack.common.utils import batch_forward, sizeof_fmt
from neural_shield.config import attack_config, get_path, max_batch_size


def offline_mad(env_id, algo):
    # load the offline trajectory dataset
    traj_path = get_path("data_collection", env_id, algo)["trajs"]
    traj_dataset = np.load(traj_path)
    print("trajectory dataset size is", sizeof_fmt(sys.getsizeof(traj_dataset)))

    # grad attack
    controller_net = ControllerNet(env_id, algo)
    config = attack_config["attack_constraint"][env_id]["obs"]
    print(config)
    traj_dataset_shape = traj_dataset.shape
    traj_dataset = traj_dataset.reshape([-1, traj_dataset_shape[-1]])

    adv_dataset_batch = batch_forward(mad_pgd, (traj_dataset, ), max_batch_size,
                                      dict(controller_net=controller_net,
                                           norm=config["attack_norm"],
                                           epsilon=config["epsilon"],
                                           lr=config["attack_lr"],
                                           max_iter=config["pgd_max_iter"]))
    adv_dataset = np.concatenate(adv_dataset_batch)

    traj_dataset = adv_dataset.reshape(traj_dataset_shape)

    # save attacked dataset
    dataset_path = get_path("offline_attack", env_id, algo)["mad_trajs"]
    np.save(dataset_path, traj_dataset)
