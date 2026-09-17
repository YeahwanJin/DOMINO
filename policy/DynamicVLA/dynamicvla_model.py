"""Server-side DynamicVLA wrapper for DOMINO evaluation.

Runs in the isolated `dynamicvla` environment, driven over TCP by
`script/policy_model_server.py`; every public method here is reachable as a
`ModelClient.call(func_name=...)` from the simulation process.
"""

import json
import logging
import pathlib
from collections import deque

import numpy as np
import torch

import utils.helpers

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


class DynamicVLA:
    def __init__(self, ckpt_dir: str, n_action_steps: int | None = None):
        ckpt_dir = pathlib.Path(ckpt_dir).expanduser().resolve()
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint dir does not exist: {ckpt_dir}")

        cfg_path = ckpt_dir / "config.json"
        with open(cfg_path) as f:
            model_cfg = json.load(f)

        cfg = utils.helpers.get_policy_cfg(cfg_file=cfg_path)
        # The checkpoint was trained with the world-model branch on; it is a
        # training-only loss, so keep it out of the inference graph entirely.
        # Without this the branch (and its CUDA rasterizer) would be constructed.
        cfg.use_world_model_aux = False
        cfg.enable_streaming = False
        if n_action_steps is not None:
            cfg.n_action_steps = min(n_action_steps, cfg.chunk_size)

        dataset_stats = self._load_dataset_stats(ckpt_dir)
        policy_class = utils.helpers.get_policy_class(model_cfg["type"])
        self.policy = policy_class.from_pretrained(ckpt_dir, config=cfg)
        if dataset_stats is not None:
            from lerobot.policies.normalize import Normalize, Unnormalize
            self.policy.normalize_inputs = Normalize(
                cfg.input_features, cfg.normalization_mapping, dataset_stats
            )
            self.policy.normalize_targets = Normalize(
                cfg.output_features, cfg.normalization_mapping, dataset_stats
            )
            self.policy.unnormalize_outputs = Unnormalize(
                cfg.output_features, cfg.normalization_mapping, dataset_stats
            )
            print("✅ Successfully injected dataset_stats into normalizers!")
        self.policy.eval()
        if torch.cuda.is_available():
            self.policy = self.policy.cuda()

        self.cfg = cfg
        # Observation offsets used during training, e.g. [-2, 0].
        self.obs_deltas = (model_cfg.get("delta_timestamps") or {}).get(
            "observation", [0]
        )[-cfg.n_obs_steps :]
        self.cameras = [
            k.rsplit(".", 1)[-1] for k in cfg.image_features if k in cfg.input_features
        ] or list(CAMERAS)
        self.obs_window = deque(maxlen=max(1, -min(self.obs_deltas) + 1))
        logging.info(
            "DynamicVLA ready: cameras=%s obs_deltas=%s n_action_steps=%d",
            self.cameras,
            self.obs_deltas,
            cfg.n_action_steps,
        )

        self.instruction = ""

    # --- remote API -------------------------------------------------------
    def reset_model(self):
        self.obs_window.clear()
        self.policy.reset()

    def set_instruction(self, instruction):
        self.instruction = str(instruction)

    def n_obs(self):
        return len(self.obs_window)

    def update_obs(self, obs):
        self.obs_window.append(obs)

    def get_action(self, obs=None):
        if obs is not None:
            self.update_obs(obs)
        if not self.obs_window:
            raise RuntimeError("get_action called before any observation was pushed.")

        batch = self._build_batch()
        with torch.no_grad():
            actions = self.policy.predict_action_chunk(batch)

        if self.cfg.use_delta_action:
            # Mirrors DynamicVLAPolicy._get_non_streaming_action.
            action_dim = actions.shape[-1] - 1
            latest_state = batch["observation.state"][:, -1:, :]
            actions[..., :action_dim] += latest_state[..., :action_dim]

        actions = actions[0, : self.cfg.n_action_steps]
        return actions.float().cpu().numpy()

    # --- internals --------------------------------------------------------
    def _frame_at(self, delta: int):
        """Observation `delta` steps back, clamped to the oldest one available."""
        index = max(0, len(self.obs_window) - 1 + delta)
        return self.obs_window[index]

    def _build_batch(self) -> dict:
        device = next(self.policy.parameters()).device
        frames = [self._frame_at(d) for d in self.obs_deltas]

        batch = {}
        _cam_aliases = {
            "cam_high": "head_camera",
            "cam_left_wrist": "left_camera",
            "cam_right_wrist": "right_camera",
            "head_camera": "cam_high",
            "left_camera": "cam_left_wrist",
            "right_camera": "cam_right_wrist",
        }
        for camera in self.cameras:
            images = np.stack([
                f.get(camera) if camera in f else f.get(_cam_aliases.get(camera, camera))
                for f in frames
            ])  # (T, H, W, 3) uint8
            images = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous().float() / 255.0
            batch[f"observation.images.{camera}"] = images.unsqueeze(0).contiguous().to(device)

        state = np.stack([np.asarray(f["state"], dtype=np.float32) for f in frames])
        batch["observation.state"] = (
            torch.from_numpy(state).unsqueeze(0).contiguous().to(device)
        )
        batch["task"] = [self.instruction]
        return batch

    @staticmethod
    def _load_dataset_stats(ckpt_dir: pathlib.Path):
        # 1. Check if stats.json or dataset_stats.json exists in checkpoint dir
        for fname in ("stats.json", "dataset_stats.json"):
            p = ckpt_dir / fname
            if p.exists():
                with open(p) as f:
                    raw = json.load(f)
                return {
                    k: {sk: torch.tensor(sv, dtype=torch.float32) for sk, sv in v.items()}
                    for k, v in raw.items()
                }

        # 2. Fallback: aggregate from data/lerobot_data/*/meta/stats_gr00t.json
        data_dir = pathlib.Path("data/lerobot_data")
        if not data_dir.exists():
            data_dir = pathlib.Path("/workspace/DOMINO/data/lerobot_data")
        if data_dir.exists():
            all_stats = []
            for td in sorted(data_dir.iterdir()):
                if td.is_dir() and td.name != "local":
                    sf = td / "meta" / "stats_gr00t.json"
                    if sf.exists():
                        with open(sf) as f:
                            all_stats.append(json.load(f))
            if all_stats:
                keys = list(all_stats[0].keys())
                merged = {}
                for k in keys:
                    mean_val = np.mean([s[k]["mean"] for s in all_stats], axis=0)
                    std_val = np.mean([s[k]["std"] for s in all_stats], axis=0)
                    std_val = np.where(std_val < 1e-6, 1.0, std_val)
                    min_val = np.min([s[k]["min"] for s in all_stats], axis=0)
                    max_val = np.max([s[k]["max"] for s in all_stats], axis=0)
                    merged[k] = {
                        "mean": torch.tensor(mean_val, dtype=torch.float32),
                        "std": torch.tensor(std_val, dtype=torch.float32),
                        "min": torch.tensor(min_val, dtype=torch.float32),
                        "max": torch.tensor(max_val, dtype=torch.float32),
                    }
                print(f"✅ DynamicVLA: Successfully aggregated dataset_stats from {len(all_stats)} tasks in {data_dir}!")
                return merged
            else:
                print(f"⚠️ DynamicVLA: No stats_gr00t.json found in {data_dir}")
        else:
            print(f"⚠️ DynamicVLA: data_dir {data_dir} does not exist!")
        return None
