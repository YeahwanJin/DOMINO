"""Server-side SmolVLA wrapper for DOMINO evaluation.

Serves both SmolVLA experiments — the plain fine-tune (`type: smolvla`) and the
world-model variant (`type: smolvla_wm`) — because the world-model branch is
training-only: it is switched off when the config is rebuilt here, so the two
checkpoints run through an identical inference graph.

Runs in the isolated `smolvla` environment, driven over TCP by
`script/policy_model_server.py`; every public method here is reachable as a
`ModelClient.call(func_name=...)` from the simulation process.
"""

import json
import logging
import pathlib

import numpy as np
import torch

import utils.dataset_stats
import utils.domino_obs
import utils.helpers

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


class SmolVLA:
    def __init__(self, ckpt_dir: str, n_action_steps: int | None = None):
        ckpt_dir = pathlib.Path(ckpt_dir).expanduser().resolve()
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint dir does not exist: {ckpt_dir}")

        cfg_path = ckpt_dir / "config.json"
        with open(cfg_path) as f:
            model_cfg = json.load(f)

        cfg = utils.helpers.get_policy_cfg(cfg_file=cfg_path)
        # A `smolvla_wm` checkpoint carries the reconstruction branch's weights, but
        # the branch is a training-only loss. Leaving it off here keeps it (and its
        # optional CUDA rasterizer) out of the inference graph; the loader drops the
        # corresponding `world_model.*` tensors.
        if hasattr(cfg, "use_world_model_aux"):
            cfg.use_world_model_aux = False
        # The base model's weights come from the checkpoint, not from the Hub.
        if hasattr(cfg, "load_vlm_weights"):
            cfg.load_vlm_weights = False
        if n_action_steps is not None:
            cfg.n_action_steps = min(n_action_steps, cfg.chunk_size)

        policy_class = utils.helpers.get_policy_class(model_cfg["type"])
        self.policy = policy_class.from_pretrained(ckpt_dir, config=cfg)
        utils.dataset_stats.install_normalizers(
            self.policy,
            cfg,
            utils.dataset_stats.load_dataset_stats(ckpt_dir, label="SmolVLA"),
        )
        self.policy.eval()
        if torch.cuda.is_available():
            self.policy = self.policy.cuda()

        self.cfg = cfg
        # SmolVLA conditions on a single frame; `save_checkpoint` records whether the
        # dataset delta'd its actions so absolute qpos targets can be restored.
        self.use_delta_action = bool(model_cfg.get("dataset_use_delta_action", False))
        self.cameras = [
            k.rsplit(".", 1)[-1] for k in cfg.image_features if k in cfg.input_features
        ] or list(CAMERAS)
        self.latest_obs = None
        self.instruction = ""
        logging.info(
            "SmolVLA ready: type=%s cameras=%s n_action_steps=%d delta_action=%s",
            model_cfg["type"],
            self.cameras,
            cfg.n_action_steps,
            self.use_delta_action,
        )

    # --- remote API -------------------------------------------------------
    def reset_model(self):
        self.latest_obs = None
        self.policy.reset()

    def set_instruction(self, instruction):
        self.instruction = str(instruction)

    def n_obs(self):
        # Single-frame policy: 0 before the first observation, 1 afterwards. Keeps
        # the same episode-priming contract the DOMINO eval plugins share.
        return 0 if self.latest_obs is None else 1

    def update_obs(self, obs):
        self.latest_obs = obs

    def get_action(self, obs=None):
        if obs is not None:
            self.update_obs(obs)
        if self.latest_obs is None:
            raise RuntimeError("get_action called before any observation was pushed.")

        batch = self._build_batch()
        with torch.no_grad():
            actions = self.policy.predict_action_chunk(batch)

        if self.use_delta_action:
            # Mirrors utils.datasets.LeRobotDataset: every dimension but the trailing
            # gripper was stored relative to the current state.
            action_dim = actions.shape[-1] - 1
            actions[..., :action_dim] += batch["observation.state"][
                :, None, :action_dim
            ]

        actions = actions[0, : self.cfg.n_action_steps]
        return actions.float().cpu().numpy()

    # --- internals --------------------------------------------------------
    def _build_batch(self) -> dict:
        device = next(self.policy.parameters()).device
        frame = self.latest_obs

        batch = {}
        for camera in self.cameras:
            image = np.asarray(utils.domino_obs.frame_camera(frame, camera))
            image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float() / 255.0
            batch[f"observation.images.{camera}"] = image.unsqueeze(0).to(device)

        state = np.asarray(frame["state"], dtype=np.float32)
        batch["observation.state"] = torch.from_numpy(state).unsqueeze(0).to(device)
        batch["task"] = [self.instruction]
        return batch
