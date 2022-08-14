import sys

from neural_shield.config import get_path
from neural_shield.controller.zoo_utils import ALGOS
from neural_shield.defense.model import AtlaPPO, AtlaTD3


def load_model(env_id: str, algo: str, device: str = "cpu"):
    """Stable baselines3 zoo and atla model loading interface."

    Args:
        env_id (str): environment id, e.g., Pendulum-v0
        algo (str): support ddpg, sac, ppo, td3, etc,
            see https://github.com/DLR-RM/rl-baselines3-zoo#current-collection-100-trained-agents.
            if ends with -atla load atla model
        device (str): device to store and compute model

    Returns:
        [stabel baseline3 model]: pretrained model
    """

    if algo.endswith("-atlt"):
        _algo = algo[:-5]
        if _algo == "ppo":
            model = AtlaPPO(env_id, _algo)
        elif _algo == "td3":
            model = AtlaTD3(env_id, _algo)

        model_path = get_path("online_adv_train", env_id, _algo)["model"]
        model.load_state_dict(model_path)

        return model

    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    model_path = get_path("pretrained", env_id, algo)["model"]
    model = ALGOS[algo].load(path=model_path,
                             custom_objects=custom_objects, device=device)
    return model
