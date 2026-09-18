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

import utils.dataset_stats
import utils.domino_obs
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

        policy_class = utils.helpers.get_policy_class(model_cfg["type"])
        self.policy = policy_class.from_pretrained(ckpt_dir, config=cfg)
        utils.dataset_stats.install_normalizers(
            self.policy,
            cfg,
            utils.dataset_stats.load_dataset_stats(ckpt_dir, label="DynamicVLA"),
        )
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
        for camera in self.cameras:
            images = np.stack(
                [utils.domino_obs.frame_camera(f, camera) for f in frames]
            )  # (T, H, W, 3) uint8
            images = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous().float() / 255.0
            batch[f"observation.images.{camera}"] = images.unsqueeze(0).contiguous().to(device)

        state = np.stack([np.asarray(f["state"], dtype=np.float32) for f in frames])
        batch["observation.state"] = (
            torch.from_numpy(state).unsqueeze(0).contiguous().to(device)
        )
        batch["task"] = [self.instruction]
        return batch
