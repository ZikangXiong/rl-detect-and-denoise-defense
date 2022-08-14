from .ant import Ant
from .base import BenchBase
from .halfcheetah import HalfCheetah
from .hopper import Hopper
from .humanoid import Humanoid
from .pendulum import Pendulum
from .walker2d import Walker2d


def get_env(env_id) -> BenchBase:
    if env_id == "Pendulum-v0":
        env = Pendulum()
    elif env_id == "Ant-v3":
        env = Ant()
    elif env_id == "HalfCheetah-v3":
        env = HalfCheetah()
    elif env_id == "Hopper-v3":
        env = Hopper()
    elif env_id == "Walker2d-v3":
        env = Walker2d()
    elif env_id == "Humanoid-v3":
        env = Humanoid()

    return env
