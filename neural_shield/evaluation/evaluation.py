import copy

import gym
import numpy as np
import pandas as pd

from neural_shield.attack.common.adversarial_planner import CEMPlanner
from neural_shield.attack.common.em_optimal_attack import EMOptimalAttackPolicy
from neural_shield.attack.common.network import ControllerNet, ValueNet
from neural_shield.attack.online.method.dummy import DummyAttack
from neural_shield.attack.online.method.enchanting import EnchantingAttack
from neural_shield.attack.online.method.mad import MADAttack
from neural_shield.attack.online.method.optimal import OptimalAttack
from neural_shield.attack.online.method.vf import ValueFuncAttack
from neural_shield.attack.online.vulnerable_func import VFBasedVunFunc
from neural_shield.benchmark import get_env
from neural_shield.config import attack_config, get_path, simulation_config
from neural_shield.controller.pretrain import load_model
from neural_shield.defense.data import collect_data
from neural_shield.defense.defense import (Defense, test_gru_detector,
                                           train_gru_vae_detector,
                                           train_gru_vae_repairer)


def simulation(env, model, n_step, n_rollout, adversary=None, defense=None, render=False):
    info_dict = {
        "ep_len": [],
        "ep_reward": [],
        "attack_triggered": [],
        "defense_tp": [],
        "defense_fp": [],
        "defense_tn": [],
        "defense_fn": [],
        "repair_error": [],
        "attack_tensity": []
    }

    for _ in range(n_rollout):
        obs = env.reset()

        if adversary is not None:
            adversary.reset()
            assert adversary.attack_count == 0

        if defense is not None:
            defense.reset()

        if render:
            env.render()

        reward_sum = 0
        attack_count = 0
        defense_tp = 0
        defense_fp = 0
        defense_tn = 0
        defense_fn = 0
        repair_error = 0
        attack_tensity = 0

        for i in range(n_step):
            true_obs = obs  # perfect repairer
            if adversary is not None and adversary.get_attack_type() == "obs":
                # attack before predicting action
                if i < adversary.initial_safe_step:
                    vulnerable = False
                else:
                    vulnerable, obs = adversary.attack_obs(obs)
                    # if vulnerable:
                    #     dev = true_obs - obs
                    #     obs += dev + np.random.random(size=dev.shape) * dev
                    attack_tensity += np.mean(np.abs(obs - true_obs))

                if vulnerable:
                    attack_count += 1

                alter = False
                if defense is not None:
                    # only support detect observation for now
                    alter, repaired_obs = defense.detect_and_repair(obs)
                    if alter:
                        if defense.perfect_repairer:
                            repaired_obs = true_obs

                        obs = repaired_obs.clip(obs-adversary.epsilon,
                                                obs+adversary.epsilon)
                        repair_error += np.mean(np.abs(obs - true_obs))

                if alter:
                    if vulnerable:
                        defense_tp += 1
                    else:
                        defense_fp += 1
                else:
                    if vulnerable:
                        defense_fn += 1
                    else:
                        defense_tn += 1

            if hasattr(model, 'predict'):
                # sb3 model
                action, _ = model.predict(obs)
            elif hasattr(model, 'act'):
                # cem planner model
                action = model.act(obs)

            if adversary is not None and adversary.get_attack_type() == "action":
                # attack after action is predicted
                _, action = adversary.attack_action(obs, action)

            action = action.clip(env.action_space.low, env.action_space.high)
            obs, reward, done, _ = env.step(action)

            reward_sum += reward
            if render:
                env.render()
            if done:
                break

        info_dict["ep_len"].append(i)
        info_dict["ep_reward"].append(reward_sum)
        if adversary is not None:
            info_dict["attack_triggered"].append(attack_count)
            info_dict["attack_tensity"].append(attack_tensity / (adversary.attack_count + 1e-5))
            if defense is not None and adversary.get_attack_type() == "obs":
                info_dict["defense_tp"].append(defense_tp)
                info_dict["defense_fp"].append(defense_fp)
                info_dict["defense_tn"].append(defense_tn)
                info_dict["defense_fn"].append(defense_fn)
                info_dict["repair_error"].append(repair_error / (defense_tp + defense_fp + 1e-5))

    for k in [k for k in info_dict]:
        if len(info_dict[k]) == 0:
            info_dict.pop(k)

    return pd.DataFrame(info_dict)


def inspect_pretrain(env_id, algo):
    model = load_model(env_id, algo)
    env = gym.make(env_id)

    res = simulation(env, model, **simulation_config[env_id])
    return res


def prepare_cem_planner(env_id):
    planner = CEMPlanner(env_id)
    planner.evolve()
    save_path = get_path("cem_planner", env_id, None)["model"]
    planner.save(save_path)


def inspect_cem_planner(env_id):
    env = get_env(env_id)
    save_path = get_path("cem_planner", env_id, None)["model"]
    cem_planner = CEMPlanner.load(save_path)
    res = simulation(env=env, model=cem_planner, adversary=None, **simulation_config[env_id])

    return res


def prepare_optimal_attack(env_id, algo):
    config = attack_config["optimal_attack_policy"][env_id]
    attack_policy = EMOptimalAttackPolicy(env_id, algo)
    attack_policy.evolve(config["n_gen"], config["initial_state"], config["mode"])
    save_path = get_path("optimal_attack_policy", env_id, algo)["model"]
    attack_policy.save(save_path)


def enchanting_attack(
        env_id, algo, attack_type, exact_enchanting=False, defense=None, attack_repairer=False):
    if exact_enchanting:
        assert attack_type == "action", "exact enchanting requires attack on action"

    env = get_env(env_id)
    cem_planner_path = get_path("cem_planner", env_id, None)["model"]
    adversarial_planner = CEMPlanner.load(cem_planner_path)

    config = attack_config["attack_constraint"][env_id]
    vulnerable_threshold = attack_config["vulnerable_threshold"][env_id][algo]
    attack_kwargs = copy.deepcopy(config[attack_type])

    controller_net = ControllerNet(env_id, algo, attack_repairer)
    value_net = ValueNet(env_id, algo)
    vun_func = VFBasedVunFunc(value_net, vulnerable_threshold)

    adversary = EnchantingAttack(adversarial_planner=adversarial_planner,
                                 controller_net=controller_net,
                                 vun_func=vun_func, attack_type=attack_type,
                                 **attack_kwargs)
    if exact_enchanting:
        adversary.enable_exact_enchanting()

    res = simulation(env=env, model=value_net.model, adversary=adversary,
                     defense=defense, **simulation_config[env_id])

    return res


def optimal_attack(env_id, algo, attack_type, defense=None):
    assert attack_type == "obs", "Optimal attack only supports to attack observation"
    env = get_env(env_id)

    config = attack_config["attack_constraint"][env_id]
    vulnerable_threshold = attack_config["vulnerable_threshold"][env_id][algo]
    attack_kwargs = config[attack_type]

    save_path = get_path("optimal_attack_policy", env_id, algo)["model"]
    attack_policy = EMOptimalAttackPolicy.load(save_path)

    value_net = ValueNet(env_id, algo)
    vun_func = VFBasedVunFunc(value_net, vulnerable_threshold)

    attack_time_limit = attack_kwargs.get("attack_time_limit")
    epsilon = attack_kwargs.get("epsilon")
    initial_safe_step = attack_kwargs.get("initial_safe_step", 0)

    adversary = OptimalAttack(attack_policy=attack_policy,
                              vun_func=vun_func,
                              attack_time_limit=attack_time_limit,
                              epsilon=epsilon,
                              initial_safe_step=initial_safe_step)

    res = simulation(env=env, model=value_net.model, adversary=adversary, defense=defense,
                     **simulation_config[env_id])

    return res


def mad_attack(env_id, algo, attack_type, defense=None, attack_repairer=False):
    assert attack_type == "obs", "MAD only supports to attack observation"

    env = get_env(env_id)

    config = attack_config["attack_constraint"][env_id]
    vulnerable_threshold = attack_config["vulnerable_threshold"][env_id][algo]
    attack_kwargs = config[attack_type]

    controller_net = ControllerNet(env_id, algo, attack_repairer)
    value_net = ValueNet(env_id, algo)
    vun_func = VFBasedVunFunc(value_net, vulnerable_threshold)

    adversary = MADAttack(controller_net=controller_net,
                          vun_func=vun_func, attack_type=attack_type, **attack_kwargs)
    res = simulation(env=env, model=controller_net.model, adversary=adversary, defense=defense,
                     **simulation_config[env_id])

    return res


def value_func_attack(env_id, algo, attack_type, defense=None, attack_repairer=False):
    assert attack_type == "obs", "Value function attack only supports to attack observation for now"
    env = get_env(env_id)

    config = attack_config["attack_constraint"][env_id]
    vulnerable_threshold = attack_config["vulnerable_threshold"][env_id][algo]
    attack_kwargs = copy.deepcopy(config[attack_type])
    attack_kwargs.pop("attack_norm")

    value_net = ValueNet(env_id, algo, attack_repairer)
    vun_func = VFBasedVunFunc(ValueNet(env_id, algo), vulnerable_threshold)

    adversary = ValueFuncAttack(value_net=value_net,
                                vun_func=vun_func, attack_type=attack_type, **attack_kwargs)
    res = simulation(env=env, model=value_net.model, adversary=adversary,
                     defense=defense, **simulation_config[env_id])

    return res


def dummy_attack(env_id, algo,  attack_type, defense=None):
    env = get_env(env_id)

    adversary = DummyAttack()
    controller_net = ControllerNet(env_id, algo)
    res = simulation(env=env, model=controller_net.model, adversary=adversary,
                     defense=defense, **simulation_config[env_id])

    return res


def defend(
        env_id, algo, attack_algo, perfect_repairer=False, without_detector=False,
        attack_repairer=False):
    defense = Defense(env_id, algo, perfect_repairer, without_detector)

    if attack_algo == "enchanting":
        res = enchanting_attack(env_id, algo, attack_type="obs",
                                defense=defense, attack_repairer=attack_repairer)
    elif attack_algo == "mad":
        res = mad_attack(env_id, algo, attack_type="obs", defense=defense)
    elif attack_algo == "value_func":
        res = value_func_attack(env_id, algo, attack_type="obs",
                                defense=defense, attack_repairer=attack_repairer)
    elif attack_algo == "optimal":
        res = optimal_attack(env_id, algo, attack_type="obs",
                             defense=defense)
    elif attack_algo == "normal":
        res = dummy_attack(env_id, algo, attack_type="obs",
                           defense=defense)

    return res


def collect_normal_transitions(env_id, algo):
    collect_data(env_id, algo)


def train_detector(env_id, algo):
    train_gru_vae_detector(env_id, algo)


def train_repairer(env_id, algo):
    train_gru_vae_repairer(env_id, algo)


def test_detector(env_id, algo):
    test_gru_detector(env_id, algo)
