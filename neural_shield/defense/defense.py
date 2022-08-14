import os
from copy import deepcopy

import numpy as np
import pandas as pd
import torch as th
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from neural_shield.config import (attack_config, default_device,
                                  defense_config, get_path)
from neural_shield.controller.pretrain import load_model
from neural_shield.defense.data import DetectorDataset, RepairerDataset
from neural_shield.defense.model import Detector, GRURepairer


class Defense:
    def __init__(self, env_id, algo, perfect_repairer=False, without_detector=False):
        env_config = defense_config["model"][env_id]
        self.detector = Detector(env_config, algo)
        self.repairer = GRURepairer(env_config)
        self.perfect_repairer = perfect_repairer
        self.without_detector = without_detector

        detector_path = get_path("detector", env_id, algo)["model"]
        repairer_path = get_path("repairer", env_id, algo)["model"]
        self.detector.load_state_dict(th.load(detector_path))
        self.repairer.load_state_dict(th.load(repairer_path))

    def reset(self):
        self.detector.reset()
        self.repairer.reset()

    def detect_and_repair(self, obs):
        alter = self.without_detector or self.detector.detect(obs)
        repaired_obs = self.repairer.repair(obs)

        return alter, repaired_obs


def train_gru_vae_detector(env_id, algo):
    env_config = defense_config["model"][env_id]
    detector_config = env_config["detector"]

    trajs_data_path = get_path("data_collection", env_id, algo)["trajs"]
    data_array = np.load(trajs_data_path)
    training_data_array = data_array[:-detector_config["test_data_size"]]
    test_data_array = data_array[-detector_config["test_data_size"]:]
    test_tensor = th.tensor(test_data_array, dtype=th.float32, device=default_device)
    _dataset = DetectorDataset(training_data_array)
    data_loader = DataLoader(_dataset, batch_size=detector_config["batch_size"])

    detector = Detector(env_config, algo, _dataset.running_mean, _dataset.running_std)
    detector_optimizer = Adam(detector.parameters(), lr=detector_config["lr"])

    loss_dict = {"recon_loss": [], "reg_loss": []}

    for i in tqdm(range(detector_config["n_ep"])):
        for traj in data_loader:
            detector_optimizer.zero_grad()
            losses = detector.loss(traj, detector_config["beta"])
            losses["loss"].backward()
            detector_optimizer.step()

        test_losses = detector.loss(test_tensor)
        loss_dict["recon_loss"].append(test_losses["recon_loss"].detach().cpu().numpy().item())
        loss_dict["reg_loss"].append(test_losses["reg_loss"].detach().cpu().numpy().item())

    pathes = get_path("detector", env_id, algo)
    model_path = pathes["model"]
    loss_plot_path = pathes["loss_plot"]

    log_path = os.path.dirname(model_path)
    os.makedirs(log_path, exist_ok=True)
    th.save(detector.state_dict(), model_path)

    ax = pd.DataFrame(loss_dict).plot(y=["recon_loss", "reg_loss"], use_index=True)
    ax.get_figure().savefig(loss_plot_path)


def test_on_normal_data(env_id, algo):
    env_config = defense_config["model"][env_id]
    trajs_data_path = get_path("data_collection", env_id, algo)["trajs"]
    detector_config = env_config["detector"]

    data_array = np.load(trajs_data_path)
    detector = Detector(env_config, algo)

    detector_path = get_path("detector", env_id, algo)["model"]
    detector.load_state_dict(th.load(detector_path))

    test_data_array = data_array[-detector_config["test_data_size"]:]
    test_tensor = th.tensor(test_data_array, dtype=th.float32, device=default_device)
    losses = detector.loss(test_tensor)

    var_loss = losses["reg_loss"].detach().cpu().numpy()
    recon_loss = losses["recon_loss"].detach().cpu().numpy()

    return pd.DataFrame({"reg_loss": [var_loss],  "recon_loss": [recon_loss]})


def robustness_regularizer(normal_traj, repairer, policy, epsilon):
    rr_trajs = deepcopy(normal_traj)
    rr_trajs.requires_grad = True

    robust_reg = 0
    for rr_traj in rr_trajs:
        rep_traj = repairer.forward(rr_traj.detach())
        pred_action_0 = policy.policy.forward(rep_traj.view(-1, rep_traj.shape[-1]))
        if isinstance(pred_action_0, tuple):
            pred_action_0 = pred_action_0[0]

        rep_traj = repairer.forward(rr_traj + 1e-5)
        pred_action_1 = policy.policy.forward(rep_traj.view(-1, rep_traj.shape[-1]))
        if isinstance(pred_action_1, tuple):
            pred_action_1 = pred_action_1[0]

        loss = th.sum(th.abs(pred_action_1 - pred_action_0))

        grad_sign = th.sign(th.autograd.grad(loss, rr_traj)[0])
        robust_adv_traj = rr_traj.detach() + grad_sign * epsilon
        rep_traj = repairer.forward(robust_adv_traj)
        pred_action_3 = policy.policy.forward(rep_traj.view(-1, rep_traj.shape[-1]))
        if isinstance(pred_action_3, tuple):
            pred_action_3 = pred_action_3[0]

        robust_reg += th.mean(th.abs(pred_action_3 - pred_action_0))

    return robust_reg / len(rr_trajs)


def train_gru_vae_repairer(env_id, algo):
    policy = load_model(env_id, algo, device=default_device)
    epsilon = attack_config["attack_constraint"][env_id]["obs"]["epsilon"]

    env_config = defense_config["model"][env_id]
    repairer_config = env_config["repairer"]

    normal_trajs_data_path = get_path("data_collection", env_id, algo)["trajs"]
    mad_trajs_data_path = get_path("offline_attack", env_id, algo)["mad_trajs"]
    vf_trajs_data_path = get_path("offline_attack", env_id, algo)["value_func_trajs"]

    loss_dict = {"normal": [], "mad": [], "vf": []}

    normal_data_array = np.load(normal_trajs_data_path)
    mad_data_array = np.load(mad_trajs_data_path)
    vf_data_array = np.load(vf_trajs_data_path)

    normal_train_data_array = normal_data_array[:-repairer_config["test_data_size"]]
    mad_train_data_array = mad_data_array[:-repairer_config["test_data_size"]]
    vf_train_data_array = vf_data_array[:-repairer_config["test_data_size"]]

    normal_test_data_array = normal_data_array[-repairer_config["test_data_size"]:]
    mad_test_data_array = mad_data_array[-repairer_config["test_data_size"]:]
    vf_test_data_array = vf_data_array[-repairer_config["test_data_size"]:]

    dataset = RepairerDataset(normal_train_data_array,
                              mad_train_data_array,
                              vf_train_data_array)
    data_loader = DataLoader(
        dataset, batch_size=repairer_config["batch_size"],
        shuffle=False)

    repairer = GRURepairer(env_config, dataset.running_mean, dataset.running_std)
    repairer_optimizer = Adam(repairer.parameters(), lr=repairer_config["lr"])

    for i in tqdm(range(repairer_config["n_ep"])):
        for normal_traj, adv_trajs in data_loader:
            repairer_optimizer.zero_grad()
            loss_sum = 0
            losses = repairer.loss(normal_traj, normal_traj, repairer_config["beta"])
            loss_sum += losses["loss"]

            for i, adv_traj in enumerate(adv_trajs):
                losses = repairer.loss(adv_traj, normal_traj, repairer_config["beta"])
                loss_sum += losses["loss"]

                if i == 1:
                    break

                rep_traj = repairer.forward(adv_traj)
                pred_action = policy.policy.forward(rep_traj.view(-1, rep_traj.shape[-1]))
                true_action = policy.policy.forward(normal_traj.view(-1, normal_traj.shape[-1]))
                if isinstance(pred_action, tuple):
                    pred_action = pred_action[0]
                    true_action = true_action[0]
                robust_reg = th.mean(th.abs(pred_action - true_action))
                loss_sum += robust_reg

            # robust regularizer
            loss_sum += robustness_regularizer(normal_traj, repairer, policy, epsilon)

            loss_sum.backward()
            repairer_optimizer.step()

        # test
        normal_traj = th.tensor(normal_test_data_array, dtype=th.float32, device=default_device)
        recon_loss = repairer.loss(normal_traj, normal_traj, repairer_config["beta"])["recon_loss"]
        loss_dict["normal"].append(recon_loss.detach().cpu().numpy().item())

        adv_traj = th.tensor(mad_test_data_array, dtype=th.float32, device=default_device)
        recon_loss = repairer.loss(adv_traj, normal_traj, repairer_config["beta"])["recon_loss"]
        loss_dict["mad"].append(recon_loss.detach().cpu().numpy().item())

        adv_traj = th.tensor(vf_test_data_array, dtype=th.float32, device=default_device)
        recon_loss = repairer.loss(adv_traj, normal_traj, repairer_config["beta"])["recon_loss"]
        loss_dict["vf"].append(recon_loss.detach().cpu().numpy().item())

    pathes = get_path("repairer", env_id, algo)
    model_path = pathes["model"]
    loss_plot_path = pathes["loss_plot"]

    log_path = os.path.dirname(model_path)
    os.makedirs(log_path, exist_ok=True)
    th.save(repairer.state_dict(), model_path)

    ax = pd.DataFrame(loss_dict).plot(
        y=["normal", "mad", "vf"],
        use_index=True)
    ax.get_figure().savefig(loss_plot_path)


def test_gru_detector(env_id, algo):
    print(test_on_normal_data(env_id, algo))
