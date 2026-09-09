"""Convert DOMINO / RoboTwin episodes to a LeRobot v2.1 dataset.

Single hop, unlike PUMA's RoboTwin -> ALOHA -> LeRobot pipeline: DynamicVLA does not
need ALOHA's motor renaming, and going straight from the raw HDF5 keeps the bimanual
14-D joint layout intact.

State / action layout (aloha-agilex, from envs/_base_task.py:get_obs):
    [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
Following the RoboTwin convention, action[t] is the *next* observed joint state, i.e.
an absolute qpos target compatible with TASK_ENV.take_action(action, "qpos").

Usage:
    python convert_robotwin_to_lerobot.py \
        --raw-dir ../../../../data/adjust_bottle/demo_clean \
        --repo-id domino/adjust_bottle
"""

import argparse
import json
import pathlib
import shutil

import cv2
import h5py
import numpy as np
import tqdm
from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset

DEFAULT_CAMERAS = ("head_camera", "left_camera", "right_camera")


def discover_cameras(ep_path: pathlib.Path) -> list[str]:
    with h5py.File(ep_path, "r") as ep:
        return [c for c in ep["/observation"] if "rgb" in ep[f"/observation/{c}"]]


def state_dim(ep_path: pathlib.Path) -> int:
    with h5py.File(ep_path, "r") as ep:
        return ep["/joint_action/left_arm"].shape[1] * 2 + 2  # + one gripper per arm


def load_episode(ep_path: pathlib.Path, cameras: list[str], image_size: tuple[int, int]):
    """Returns (images per camera, state, action)."""
    with h5py.File(ep_path, "r") as ep:
        left_arm = ep["/joint_action/left_arm"][()]
        left_gripper = ep["/joint_action/left_gripper"][()]
        right_arm = ep["/joint_action/right_arm"][()]
        right_gripper = ep["/joint_action/right_gripper"][()]

        qpos = np.concatenate(
            [
                left_arm,
                left_gripper.reshape(-1, 1),
                right_arm,
                right_gripper.reshape(-1, 1),
            ],
            axis=1,
        ).astype(np.float32)

        images = {}
        for camera in cameras:
            frames = []
            for encoded in ep[f"/observation/{camera}/rgb"][()]:
                bgr = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
                bgr = cv2.resize(bgr, image_size)
                # The simulator hands the policy RGB at eval time, so the training
                # frames must be RGB too — cv2.imdecode returns BGR.
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            images[camera] = np.stack(frames)

    # action[t] == state[t + 1]
    return {k: v[:-1] for k, v in images.items()}, qpos[:-1], qpos[1:]


def load_instruction(ep_path: pathlib.Path, ep_idx: int, kind: str) -> str:
    path = ep_path.parent.parent / "instructions" / f"episode{ep_idx}.json"
    if not path.exists():
        return "complete the task"

    with open(path) as f:
        instructions = json.load(f)[kind]
    return str(np.random.choice(instructions))


def build_features(cameras: list[str], state_dim: int, image_size: tuple[int, int]):
    height, width = image_size[1], image_size[0]
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": ["state"],
        },
        "action": {"dtype": "float32", "shape": (state_dim,), "names": ["action"]},
    }
    for camera in cameras:
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def convert(
    raw_dir: pathlib.Path,
    repo_id: str,
    root: pathlib.Path | None,
    fps: int,
    image_size: tuple[int, int],
    instruction_kind: str,
    cameras: list[str] | None,
    overwrite: bool,
) -> None:
    episode_paths = sorted(
        (raw_dir / "data").glob("episode*.hdf5"),
        key=lambda p: int("".join(filter(str.isdigit, p.stem))),
    )
    if not episode_paths:
        raise FileNotFoundError(f"No episode*.hdf5 found under {raw_dir / 'data'}")

    cameras = cameras or [
        c for c in DEFAULT_CAMERAS if c in discover_cameras(episode_paths[0])
    ]
    print(f"Converting {len(episode_paths)} episodes, cameras: {cameras}")

    target = root if root is not None else HF_LEROBOT_HOME / repo_id
    if target.exists():
        if not overwrite:
            raise FileExistsError(
                f"{target} already exists; pass --overwrite to replace it"
            )
        shutil.rmtree(target)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=root,
        robot_type="aloha-agilex",
        features=build_features(cameras, state_dim(episode_paths[0]), image_size),
        use_videos=True,
    )

    for ep_idx, ep_path in enumerate(tqdm.tqdm(episode_paths)):
        images, state, action = load_episode(ep_path, cameras, image_size)
        instruction = load_instruction(ep_path, ep_idx, instruction_kind)

        for i in range(state.shape[0]):
            frame = {"observation.state": state[i], "action": action[i]}
            for camera, frames in images.items():
                frame[f"observation.images.{camera}"] = frames[i]
            dataset.add_frame(frame, task=instruction)

        dataset.save_episode()

    print(f"Done. Dataset at {dataset.root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=pathlib.Path,
        required=True,
        help="DOMINO data dir, e.g. data/adjust_bottle/demo_clean",
    )
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--root", type=pathlib.Path, default=None)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--instruction-kind", type=str, default="seen", choices=["seen", "unseen"]
    )
    parser.add_argument("--cameras", type=str, nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    convert(
        raw_dir=args.raw_dir,
        repo_id=args.repo_id,
        root=args.root,
        fps=args.fps,
        image_size=(args.width, args.height),
        instruction_kind=args.instruction_kind,
        cameras=args.cameras,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
