n_cpu = 30
default_device = "cuda"

# control the max batch size load to GPU
max_batch_size = 10240

data_path = "/data/neural_shield"
pretrained_path = f"{data_path}/pretrained"
traj_history_path = f"{data_path}/traj_history"
defense_model_path = f"{data_path}/defense_models"


# All the file IO pathes are summarized here.
def get_path(task, env_id="", algo=""):
    if task == "pretrained":
        return {
            "model": f"{pretrained_path}/{algo}/{env_id}_1/{env_id}.zip"
        }
    elif task == "cem_planner":
        return {
            "model": f"{data_path}/cem_planner/{env_id}.pkl"
        }
    elif task == "optimal_attack_policy":
        return {
            "model": f"{data_path}/optimal_attack/{algo}/{env_id}.pkl"
        }
    elif task == "data_collection":
        base = f"{traj_history_path}/{env_id}/{algo}"
        return {
            "trajs": f"{base}/trajs.npy",
            "actions": f"{base}/actions.npy"
        }
    elif task == "q_func":
        base = f"{data_path}/q_func/{env_id}/{algo}"
        return {
            "model": f"{base}/q_func.pth",
        }
    elif task == "detector":
        base = f"{defense_model_path}/{env_id}/{algo}"
        return {
            "model": f"{base}/detector.pth",
            "loss_plot": f"{base}/detector_loss.pdf",
        }
    elif task == "offline_attack":
        base = f"{traj_history_path}/{env_id}/{algo}"
        return {
            "mad_trajs": f"{base}/mad_trajs.npy",
            "value_func_trajs": f"{base}/value_func_trajs.npy",
            "enchanting_trajs": f"{base}/enchanting_trajs.npy",
            "optimal_trajs": f"{base}/optimal_trajs.npy",
        }
    elif task == "repairer":
        base = f"{defense_model_path}/{env_id}/{algo}"
        return {
            "model": f"{base}/repairer.pth",
            "loss_plot": f"{base}/repairer_loss.pdf"
        }
    elif task == "online_adv_train":
        base = f"{defense_model_path}/{env_id}/{algo}"
        return {
            "model": f"{base}/atla.pth",
        }
    elif task == "summary":
        return {
            "pretrain": f"{data_path}/stats/pretrained.csv",
            "attack": f"{data_path}/stats/attack.csv",
        }


# simulation configuration
simulation_config = {
    "Pendulum-v0": {"n_step": 200, "n_rollout": 5, "render": False},
    "Hopper-v3": {"n_step": 1000, "n_rollout": 5, "render": False},
    "HalfCheetah-v3": {"n_step": 1000, "n_rollout": 5, "render": False},
    "Walker2d-v3": {"n_step": 1000, "n_rollout": 5, "render": False},
    "Ant-v3": {"n_step": 1000, "n_rollout": 5, "render": False},
    "Humanoid-v3": {"n_step": 1000, "n_rollout": 5, "render": False},
}

# attack configuration
attack_config = {
    "cem_planner": {
        "Pendulum-v0": {
            "max_horizon": 100,
            "elite_frac": 0.1,
            "population_num": 30,
            "exploration_noise": 0.2,
            "verbose": 0,
            "n_gen": 500,
            "mode": "min",
        },
        "Hopper-v3": {
            "max_horizon": 100,
            "elite_frac": 0.1,
            "population_num": 60,
            "exploration_noise": 0.2,
            "verbose": 0,
            "n_gen": 100,
            "mode": "min",
        },
        "HalfCheetah-v3": {
            "max_horizon": 100,
            "elite_frac": 0.1,
            "population_num": 60,
            "exploration_noise": 0.2,
            "verbose": 0,
            "n_gen": 3000,
            "mode": "min",
        },
        "Walker2d-v3": {
            "max_horizon": 100,
            "elite_frac": 0.1,
            "population_num": 60,
            "exploration_noise": 0.2,
            "verbose": 0,
            "n_gen": 3000,
            "mode": "min",
        },
        "Ant-v3": {
            "max_horizon": 100,
            "elite_frac": 0.1,
            "population_num": 60,
            "exploration_noise": 0.2,
            "verbose": 0,
            "n_gen": 3000,
            "mode": "min",
        },
        "Humanoid-v3": {
            "max_horizon": 100,
            "elite_frac": 0.1,
            "population_num": 60,
            "exploration_noise": 0.2,
            "verbose": 0,
            "n_gen": 3000,
            "mode": "min",
        }
    },
    "optimal_attack_policy": {
        "Pendulum-v0": {
            "max_horizon": 200,
            "elite_frac": 0.1,
            "population_num": 30,
            "exploration_noise": 0.2,
            "verbose": 0,
            "n_gen": 100,
            "initial_state": None,
            "mode": "min",
        },
        "Hopper-v3": {
            "max_horizon": 1000,
            "elite_frac": 0.1,
            "population_num": 60,
            "exploration_noise": 0.4,
            "verbose": 2,
            "n_gen": 500,
            "initial_state": None,
            "mode": "min",
        },
        "HalfCheetah-v3": {
            "max_horizon": 1000,
            "elite_frac": 0.1,
            "population_num": 60,
            "exploration_noise": 0.2,
            "verbose": 0,
            "n_gen": 100,
            "initial_state": None,
            "mode": "min",
        },
        "Walker2d-v3": {
            "max_horizon": 1000,
            "elite_frac": 0.1,
            "population_num": 60,
            "exploration_noise": 0.2,
            "verbose": 0,
            "n_gen": 100,
            "initial_state": None,
            "mode": "min",
        },
        "Ant-v3": {
            "max_horizon": 1000,
            "elite_frac": 0.1,
            "population_num": 60,
            "exploration_noise": 0.2,
            "verbose": 0,
            "n_gen": 100,
            "initial_state": None,
            "mode": "min",
        },
        "Humanoid-v3": {
            "max_horizon": 1000,
            "elite_frac": 0.1,
            "population_num": 60,
            "exploration_noise": 0.2,
            "verbose": 0,
            "n_gen": 100,
            "initial_state": None,
            "mode": "min",
        }
    },
    "vulnerable_threshold": {
        "Pendulum-v0": {
            "td3": -20,
            "ppo": -20,
        },
        "Hopper-v3": {
            "td3": 500,
            "ppo": 350,
        },
        "HalfCheetah-v3": {
            "td3": 580,
            "ppo": 700,
        },
        "Walker2d-v3": {
            "td3": 425,
            "ppo": 310,
        },
        "Ant-v3": {
            "td3": 550,
            "ppo": 580,
        },
        "Humanoid-v3": {
            "td3": 550,
            "ppo": 570,
        },
    },
    "q_function": {
        "Pendulum-v0": {
            "rollout_num": 1000,
            "lr": 1e-3,
            "batch_size": 64
        },
        "Hopper-v3": {
            "rollout_num": 1000,
            "lr": 1e-3,
            "batch_size": 64
        },
        "HalfCheetah-v3": {
            "rollout_num": 1000,
            "lr": 1e-3,
            "batch_size": 64
        },
        "Walker2d-v3": {
            "rollout_num": 1000,
            "lr": 1e-3,
            "batch_size": 64
        },
        "Ant-v3": {
            "rollout_num": 1000,
            "lr": 1e-3,
            "batch_size": 64
        },
        "Humanoid-v3": {
            "rollout_num": 1000,
            "lr": 1e-3,
            "batch_size": 64
        },
    },
    "attack_constraint": {
        "Pendulum-v0": {
            "obs": {"epsilon": 0.3,
                    "attack_norm": 1,
                    "attack_lr": 1,
                    "pgd_max_iter": 20,
                    "attack_time_limit": 200},
            "action": {"epsilon": 0.2,
                       "attack_time_limit": 200},
        },
        "Hopper-v3": {
            "obs": {"epsilon": 0.075,
                    "attack_norm": 1,
                    "attack_lr": 1,
                    "pgd_max_iter": 20,
                    "attack_time_limit": 200,
                    "initial_safe_step": 50},
            "action": {"epsilon": 0.2,
                       "attack_time_limit": 200},
        },
        "HalfCheetah-v3": {
            "obs": {"epsilon": 0.15,
                    "attack_norm": 1,
                    "attack_lr": 1,
                    "pgd_max_iter": -1,
                    "attack_time_limit": 1000},
            "action": {"epsilon": 0.15,
                       "attack_time_limit": 1000},
        },
        "Walker2d-v3": {
            "obs": {"epsilon": 0.075,
                    "attack_norm": 1,
                    "attack_lr": 1,
                    "pgd_max_iter": -1,
                    "attack_time_limit": 1000},
            "action": {"epsilon": 0.05,
                       "attack_time_limit": 1000},
        },
        "Ant-v3": {
            "obs": {"epsilon": 0.15,
                    "attack_norm": 1,
                    "attack_lr": 1,
                    "pgd_max_iter": -1,
                    "attack_time_limit": 500},
            "action": {"epsilon": 0.2,
                       "attack_time_limit": 500},
        },
        "Humanoid-v3": {
            "obs": {"epsilon": 0.15,
                    "attack_norm": 1,
                    "attack_lr": 1,
                    "pgd_max_iter": -1,
                    "attack_time_limit": 500},
            "action": {"epsilon": 0.2,
                       "attack_time_limit": 500},
        }
    }
}

# defense configuration
defense_config = {
    "data_collection": {
        "Pendulum-v0": {
            "n_epoch": 6000,
            "n_step": 200,
        },
        "Hopper-v3": {
            "n_epoch": 6000,
            "n_step": 1000,
        },
        "HalfCheetah-v3": {
            "n_epoch": 6000,
            "n_step": 1000,
        },
        "Walker2d-v3": {
            "n_epoch": 6000,
            "n_step": 1000,
        },
        "Ant-v3": {
            "n_epoch": 6000,
            "n_step": 1000,
        },
        "Humanoid-v3": {
            "n_epoch": 6000,
            "n_step": 1000,
        },
    },
    "model": {
        "Pendulum-v0": {
            "detector": {
                "batch_size": 128,
                "lr": 1e-3,
                "n_ep": 30,
                "obs_shape": 3,
                "test_data_size": 100,
                "beta": 1.0,
                "detector_thr": {"td3": 0.3,
                                 "ppo": 0.15}
            },
            "repairer": {
                "batch_size": 128,
                "lr": 1e-3,
                "n_ep": 200,
                "obs_shape": 3,
                "test_data_size": 100,
                "beta": 1.0,
            },
            "gru_vae_hyper_param": {
                "encoder": {
                    "hidden_size": 64,
                    "num_layers": 1,
                    "batch_first": True
                },
                "decoder": {
                    "hidden_size": 64,
                    "num_layers": 1,
                    "batch_first": True
                },
                "latent_shape": 64,
            }
        },
        "Hopper-v3": {
            "detector": {
                "batch_size": 128,
                "lr": 1e-3,
                "n_ep": 50,
                "obs_shape": 11,
                "test_data_size": 100,
                "beta": 1.0,
                "detector_thr": {"td3": 0.2,
                                 "ppo": 0.15}
            },
            "repairer": {
                "batch_size": 128,
                "lr": 1e-3,
                "n_ep": 50,
                "test_data_size": 100,
                "beta": 1.0,
            },
            "gru_vae_hyper_param": {
                "encoder": {
                    "hidden_size": 64,
                    "num_layers": 1,
                    "batch_first": True
                },
                "decoder": {
                    "hidden_size": 64,
                    "num_layers": 1,
                    "batch_first": True
                },
                "latent_shape": 64,
            }
        },
        "HalfCheetah-v3": {
            "detector": {
                "batch_size": 128,
                "lr": 1e-3,
                "n_ep": 50,
                "obs_shape": 17,
                "test_data_size": 100,
                "beta": 1.0,
                "detector_thr": {"td3": 0.3,
                                 "ppo": 0.15}
            },
            "repairer": {
                "batch_size": 128,
                "lr": 1e-3,
                "n_ep": 50,
                "test_data_size": 100,
                "beta": 1.0,
            },
            "gru_vae_hyper_param": {
                "encoder": {
                    "hidden_size": 64,
                    "num_layers": 1,
                    "batch_first": True
                },
                "decoder": {
                    "hidden_size": 64,
                    "num_layers": 1,
                    "batch_first": True
                },
                "latent_shape": 64,
            }
        },
        "Walker2d-v3": {
            "detector": {
                "batch_size": 128,
                "lr": 1e-3,
                "n_ep": 50,
                "obs_shape": 17,
                "test_data_size": 100,
                "beta": 1.0,
                "detector_thr": {"td3": 0.3,
                                 "ppo": 0.75}
            },
            "repairer": {
                "batch_size": 128,
                "lr": 1e-3,
                "n_ep": 50,
                "test_data_size": 100,
                "beta": 1.0,
            },
            "gru_vae_hyper_param": {
                "encoder": {
                    "hidden_size": 64,
                    "num_layers": 1,
                    "batch_first": True
                },
                "decoder": {
                    "hidden_size": 64,
                    "num_layers": 1,
                    "batch_first": True
                },
                "latent_shape": 64,
            }
        },
        "Ant-v3": {
            "detector": {
                "batch_size": 128,
                "lr": 1e-3,
                "n_ep": 50,
                "obs_shape": 111,
                "test_data_size": 100,
                "beta": 1.0,
                "detector_thr": {"td3": 2.5,
                                 "ppo": 2.5}
            },
            "repairer": {
                "batch_size": 32,
                "lr": 1e-3,
                "n_ep": 50,
                "test_data_size": 100,
                "beta": 1.0,
            },
            "gru_vae_hyper_param": {
                "encoder": {
                    "hidden_size": 256,
                    "num_layers": 1,
                    "batch_first": True
                },
                "decoder": {
                    "hidden_size": 256,
                    "num_layers": 1,
                    "batch_first": True
                },
                "latent_shape": 64,
            }
        },
        "Humanoid-v3": {
            "detector": {
                "batch_size": 128,
                "lr": 1e-3,
                "n_ep": 50,
                "obs_shape": 376,
                "test_data_size": 100,
                "beta": 1.0,
                "detector_thr": {"td3": 0.3,
                                 "ppo": 1.5}
            },
            "repairer": {
                "batch_size": 32,
                "lr": 1e-3,
                "n_ep": 50,
                "test_data_size": 100,
                "beta": 1.0,
            },
            "gru_vae_hyper_param": {
                "encoder": {
                    "hidden_size": 256,
                    "num_layers": 1,
                    "batch_first": True
                },
                "decoder": {
                    "hidden_size": 256,
                    "num_layers": 1,
                    "batch_first": True
                },
                "latent_shape": 128,
            }
        }
    },
    "atla": {
        "Pendulum-v0": {
            "obs_dim": 3,
            "action_dim": 1,
            "hidden_size": 16,
            "ppo_var": 0.05
        },
        "Hopper-v3": {
            "obs_dim": 11,
            "action_dim": 3,
            "hidden_size": 32,
            "ppo_var": 0.01
        },
        "HalfCheetah-v3": {
            "obs_dim": 17,
            "action_dim": 6,
            "hidden_size": 32,
            "ppo_var": 0.05
        },
        "Walker2d-v3": {
            "obs_dim": 17,
            "action_dim": 6,
            "hidden_size": 32,
            "ppo_var": 0.01
        },
        "Ant-v3": {
            "obs_dim": 111,
            "action_dim": 8,
            "hidden_size": 128,
            "ppo_var": 0.05
        },
        "Humanoid-v3": {
            "obs_dim": 376,
            "action_dim": 17,
            "hidden_size": 512,
            "ppo_var": 0.05
        },
    }
}
