import numpy as np
import torch as th


def normalize_action(env, action: np.ndarray, clip_first=False) -> np.ndarray:
    if clip_first:
        action = action.clip(env.action_space.low, env.action_space.high)
    action_mean = (env.action_space.low + env.action_space.high) / 2
    action_radius = (env.action_space.high - env.action_space.low) / 2

    return (action - action_mean) / action_radius


def scale_action(env, action: np.ndarray) -> np.ndarray:
    action_mean = (env.action_space.low + env.action_space.high) / 2
    action_radius = (env.action_space.high - env.action_space.low) / 2

    return action * action_radius + action_mean


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}i{suffix}"


def to_batches(inpt, batch_size):
    total_size = len(inpt)
    batch_num = total_size // batch_size

    batches = [inpt[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]

    if batch_num != total_size / batch_size:
        batches.append(inpt[batch_num*batch_size:])

    return batches


def batch_forward(callable, batch_args, batch_size, callable_kwargs={}):
    assert isinstance(batch_args, tuple)
    args_batches = []
    for args in batch_args:
        batches = to_batches(args, batch_size)
        args_batches.append(batches)

    new_x = []
    for i in range(len(args_batches[0])):
        callable_args = []
        for args_batch in args_batches:
            callable_args.append(args_batch[i])
        new_x.append(callable(*callable_args, **callable_kwargs))

    return new_x
