"""DOMINO eval plugin for SmolVLA (isolated-environment / double-env mode).

Experiment 1: SmolVLA fine-tuned on DOMINO with no reconstruction loss.

`script/eval_policy_client.py` (DOMINO env) imports this module and calls `eval()`,
where `model` is a `ModelClient` proxy; `script/policy_model_server.py` (SmolVLA env)
imports the same module and calls `get_model()` to build the real policy. The heavy
imports below therefore have to stay optional — they only resolve server side.

The policy implementation lives in `policy/DynamicVLA`, which is the shared LeRobot
training/inference tree for this repo (SmolVLA, SmolVLA+world-model and DynamicVLA
are all trained by `policy/DynamicVLA/run.py`). Only the eval entry point is split
out per policy, because `policy_name` is what partitions `eval_result/`.
"""

import os
import sys

TRAINER_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "DynamicVLA"
)
if TRAINER_DIR not in sys.path:
    sys.path.append(TRAINER_DIR)

try:
    from smolvla_model import SmolVLA
except ImportError:
    # Client side (DOMINO env): torch / lerobot are absent and not needed, since
    # every model interaction below goes over the socket.
    pass


def encode_obs(observation):
    obs = {
        camera: observation["observation"][camera]["rgb"]
        for camera in ("head_camera", "left_camera", "right_camera")
        if camera in observation["observation"]
    }
    obs["state"] = observation["joint_action"]["vector"]
    return obs


def get_model(usr_args):
    return SmolVLA(
        ckpt_dir=usr_args["ckpt_dir"],
        n_action_steps=usr_args.get("n_action_steps"),
    )


def eval(TASK_ENV, model, observation):
    # Seed the observation at the start of an episode; afterwards every observation
    # is pushed exactly once, in the action loop below.
    if model.call(func_name="n_obs") == 0:
        model.call(func_name="set_instruction", obs=str(TASK_ENV.get_instruction()))
        model.call(func_name="update_obs", obs=encode_obs(observation))

    actions = model.call(func_name="get_action")

    for action in actions:
        TASK_ENV.take_action(action, action_type="qpos")
        model.call(func_name="update_obs", obs=encode_obs(TASK_ENV.get_obs()))


def reset_model(model):
    model.call(func_name="reset_model")
