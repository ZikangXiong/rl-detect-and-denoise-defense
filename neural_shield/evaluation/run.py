from neural_shield.attack.offline.enchanting import offline_enchanting
from neural_shield.attack.offline.mad import offline_mad
from neural_shield.attack.offline.optimal import offline_optimal
from neural_shield.attack.offline.vf import offline_value_func
from neural_shield.evaluation.evaluation import (collect_normal_transitions,
                                                 defend, enchanting_attack,
                                                 inspect_cem_planner,
                                                 inspect_pretrain, mad_attack,
                                                 optimal_attack,
                                                 prepare_cem_planner,
                                                 prepare_optimal_attack,
                                                 test_detector, train_detector,
                                                 train_repairer,
                                                 value_func_attack)


def prepare(task, env_id, algo):
    print(f"====={env_id}-{algo}: {task}=====")

    if task == "train_cem_planner":
        prepare_cem_planner(env_id)
    elif task == "train_optimal_attack":
        prepare_optimal_attack(env_id, algo)
    elif task == "inspect_cem_planner":
        res = inspect_cem_planner(env_id)
        print(res.describe())
    elif task == "inspect_pretrain":
        res = inspect_pretrain(env_id, algo)
        print(res.describe())


def online_attack(task, env_id, algo, attack_type):
    print(f"====={env_id}-{algo}: {task} -  attack_{attack_type}=====")

    if task == "exact_enchanting":
        res = enchanting_attack(env_id, algo, attack_type, exact_enchanting=True)
    elif task == "enchanting":
        res = enchanting_attack(env_id, algo, attack_type)
    elif task == "mad":
        res = mad_attack(env_id, algo, attack_type)
    elif task == "value_func":
        res = value_func_attack(env_id, algo, attack_type)
    elif task == "optimal":
        res = optimal_attack(env_id, algo, attack_type)

    print(res.describe())


def offline_attack(task, env_id, algo):
    print(f"====={env_id}-{algo}: {task} -  offline_attack=====")
    if task == "mad":
        offline_mad(env_id, algo)
    elif task == "value_func":
        offline_value_func(env_id, algo)
    elif task == "enchanting":
        offline_enchanting(env_id, algo)
    elif task == "optimal":
        offline_optimal(env_id, algo)


def defense(task, env_id, algo):
    print(f"====={env_id}-{algo}: {task}=====")
    if task == "collect_normal_transitions":
        collect_normal_transitions(env_id, algo)
    elif task == "train_detector":
        train_detector(env_id, algo)
    elif task == "test_detector":
        test_detector(env_id, algo)
    elif task == "train_repairer":
        train_repairer(env_id, algo)
    elif task == "defend_enchanting_with_perfect_repairer":
        print(defend(env_id, algo, "enchanting", True, False).describe())
    elif task == "defend_mad_with_perfect_repairer":
        print(defend(env_id, algo, "mad", True, False).describe())
    elif task == "defend_value_func_with_perfect_repairer":
        print(defend(env_id, algo, "value_func", True, False).describe())
    elif task == "defend_optimal_with_perfect_repairer":
        print(defend(env_id, algo, "optimal", True, False).describe())
    elif task == "defend_normal_without_detector":
        print(defend(env_id, algo, "normal", False, True).describe())
    elif task == "defend_mad_without_detector":
        print(defend(env_id, algo, "mad", False, True).describe())
    elif task == "defend_value_func_without_detector":
        print(defend(env_id, algo, "value_func", False, True).describe())
    elif task == "defend_enchanting_without_detector":
        print(defend(env_id, algo, "enchanting", False, True).describe())
    elif task == "defend_optimal_without_detector":
        print(defend(env_id, algo, "optimal", False, True).describe())
    elif task == "defend_mad_attack_repairer":
        print(defend(env_id, algo, "mad", False, True, True).describe())
    elif task == "defend_value_func_attack_repairer":
        print(defend(env_id, algo, "value_func", False, False, True).describe())
    elif task == "defend_enchanting_attack_repairer":
        print(defend(env_id, algo, "enchanting", False, False, True).describe())
    elif task == "defend_optimal_attack_repairer":
        print(defend(env_id, algo, "optimal", False, False, True).describe())
    elif task == "defend_enchanting":
        print(defend(env_id, algo, "enchanting", False, False).describe())
    elif task == "defend_mad":
        print(defend(env_id, algo, "mad", False, False).describe())
    elif task == "defend_value_func":
        print(defend(env_id, algo, "value_func", False, False).describe())
    elif task == "defend_optimal":
        print(defend(env_id, algo, "optimal", False, False).describe())


if __name__ == "__main__":
    env_ids = [
        "Pendulum-v0",
        "Hopper-v3",
        "HalfCheetah-v3",
        "Walker2d-v3",
        "Ant-v3",
        "Humanoid-v3"
    ]
    algos = [
        "td3",
        "ppo"
    ]

    for env_id in env_ids:
        for algo in algos:
            prepare("inspect_pretrain", env_id, algo)
            prepare("train_cem_planner", env_id, algo)
            prepare("inspect_cem_planner", env_id, algo)
            prepare("train_optimal_attack", env_id, algo)

            online_attack("mad", env_id, algo, "obs")
            online_attack("value_func", env_id, algo, "obs")
            online_attack("enchanting", env_id, algo, "obs")
            online_attack("optimal", env_id, algo, "obs")

            online_attack("exact_enchanting", env_id, algo, "action")
            online_attack("enchanting", env_id, algo, "action")

            defense("collect_normal_transitions", env_id, algo)
            defense("train_detector", env_id, algo)
            defense("test_detector", env_id, algo)

            offline_attack("mad", env_id, algo)
            offline_attack("value_func", env_id, algo)
            offline_attack("enchanting", env_id, algo)
            offline_attack("optimal", env_id, algo)

            defense("train_repairer", env_id, algo)

            defense("defend_mad", env_id, algo)
            defense("defend_value_func", env_id, algo)
            defense("defend_enchanting", env_id, algo)
            defense("defend_optimal", env_id, algo)
