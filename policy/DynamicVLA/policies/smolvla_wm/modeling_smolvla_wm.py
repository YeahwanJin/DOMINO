# -*- coding: utf-8 -*-
"""SmolVLA with the GaussianDream-style world-model auxiliary loss.

Only two things are added on top of LeRobot's `SmolVLAPolicy`:

1. the visual tokens produced by `SmolVLMWithExpertModel.embed_image` are captured
   during a training forward pass (the method is wrapped once, per instance), and
2. `forward` adds `world_model_aux_weight * WorldModelAuxLoss(...)` to the action
   flow-matching loss.

The branch supervises a depth map decoded from those very tokens against Depth
Anything V2 pseudo-depth, so the geometry gradient lands on the representation the
policy actually conditions on. Nothing here runs at inference: `use_world_model_aux`
is forced to False when the eval wrapper rebuilds the config, and the branch is not
even constructed in that case.
"""

import logging
import os

import safetensors.torch
import torch
from lerobot.constants import ACTION, OBS_STATE
from lerobot.policies.smolvla.modeling_smolvla import (
    SmolVLAPolicy,
    rename_checkpoint_keys,
    standardise_state_dict,
)

from policies.smolvla_wm.configuration_smolvla_wm import SmolVLAWMConfig


def load_smolvla_wm(
    model: torch.nn.Module,
    filename: str | os.PathLike,
    *,
    device: str = "cpu",
    checkpoint_keys_mapping: str = "",
) -> torch.nn.Module:
    """Tolerant variant of `lerobot...modeling_smolvla.load_smolvla`.

    Upstream raises on any unexpected or non-normalisation missing key, which is too
    strict for both directions this policy needs:

    * initialising from `lerobot/smolvla_base`, where the checkpoint has no
      `world_model.*` weights but the model does, and
    * evaluating a world-model checkpoint, where the model is built without the
      branch but the checkpoint still carries its weights.
    """
    state_dict = safetensors.torch.load_file(filename, device=device)

    if checkpoint_keys_mapping and "//" in checkpoint_keys_mapping:
        state_dict = rename_checkpoint_keys(state_dict, checkpoint_keys_mapping)

    state_dict, _ = standardise_state_dict(state_dict, set(model.state_dict().keys()))

    # Normalization buffers must come from the dataset, never from the checkpoint.
    norm_keys = ("normalize_inputs", "normalize_targets", "unnormalize_outputs")
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith(norm_keys)}

    # Training-only branch: drop its weights when the target model has none.
    if not any(k.startswith("world_model.") for k in model.state_dict()):
        state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith("world_model.")
        }

    # Deliberately dropped tensors (e.g. an action head resized for a different
    # embodiment) keep their freshly initialised values instead of failing the load.
    model_state = model.state_dict()
    filtered, skipped = {}, set()
    for k, v in state_dict.items():
        if k in model_state and v.shape != model_state[k].shape:
            logging.warning(
                "Skipping %s due to size mismatch: checkpoint %s vs model %s",
                k,
                tuple(v.shape),
                tuple(model_state[k].shape),
            )
            skipped.add(k)
        else:
            filtered[k] = v

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    missing = [
        k
        for k in missing
        if k not in skipped
        and not k.startswith(norm_keys)
        and not k.startswith("world_model.")
    ]
    if missing or unexpected:
        raise RuntimeError(
            f"SmolVLA-WM checkpoint does not match the model: {len(missing)} missing / "
            f"{len(unexpected)} unexpected keys. "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    return model


class SmolVLAWMPolicy(SmolVLAPolicy):
    config_class = SmolVLAWMConfig
    name = "smolvla_wm"

    def __init__(self, config: SmolVLAWMConfig, dataset_stats=None):
        super().__init__(config, dataset_stats)

        # Filled with one [B, N, D] tensor per camera while a world-model forward is
        # in flight; None everywhere else so the wrapper below costs nothing.
        self._collected_img_embs = None
        self._wrap_embed_image()

        # Training-only world-model branch. The import is deliberately lazy: on the
        # eval path `use_world_model_aux` is False, so the world_model package (and
        # its CUDA rasterizer) is never imported.
        self.world_model = None
        if config.use_world_model_aux:
            from policies.dynamicvla.world_model import WorldModelAuxLoss

            self.world_model = WorldModelAuxLoss(
                config, token_dim=self._visual_token_dim()
            )

    # --- checkpoint IO -----------------------------------------------------
    @classmethod
    def _load_as_safetensor(cls, model, model_file, map_location, strict):
        return load_smolvla_wm(
            model,
            model_file,
            device=map_location,
            checkpoint_keys_mapping="model._orig_mod.//model.",
        )

    # --- world-model plumbing ---------------------------------------------
    def _visual_token_dim(self) -> int:
        """Width of the tokens `embed_image` returns (post modality connector)."""
        return self.model.vlm_with_expert.config.text_config.hidden_size

    def _wrap_embed_image(self) -> None:
        """Tee `embed_image` so a training forward can see its output.

        A module forward hook would be the tidier tool, but `embed_image` is a plain
        method on `SmolVLMWithExpertModel` rather than a submodule call, so the
        method itself is wrapped. Only the instance is touched, never the class.
        """
        vlm_with_expert = self.model.vlm_with_expert
        inner = vlm_with_expert.embed_image

        def embed_image(image):
            img_emb = inner(image)
            if self._collected_img_embs is not None:
                self._collected_img_embs.append(img_emb)
            return img_emb

        vlm_with_expert.embed_image = embed_image

    def _present_image_keys(self, batch) -> list[str]:
        return [key for key in self.config.image_features if key in batch]

    def _primary_image_key(self, batch) -> str:
        """The camera the depth labels belong to, e.g. observation.images.cam_high."""
        camera = self.config.world_model_depth_key.rsplit(".", 1)[-1]
        key = f"observation.images.{camera}"
        present = self._present_image_keys(batch)
        if key not in present:
            raise ValueError(
                f"{key} is required by the world-model loss (derived from "
                f"world_model_depth_key) but the batch only has {present}."
            )
        return key

    def _current_visual_tokens(self, batch) -> torch.Tensor:
        """Visual tokens of the depth-labelled camera, [B, N, D]."""
        collected = self._collected_img_embs
        if not collected:
            raise RuntimeError(
                "No visual tokens were captured. This should not happen when the "
                "world-model branch is active."
            )
        # embed_prefix consumes images in the order prepare_images emitted them.
        index = self._present_image_keys(batch).index(self._primary_image_key(batch))
        return collected[index]

    def _current_frame(self, batch) -> torch.Tensor:
        """Primary camera's current RGB frame in [0, 1], [B, 3, H, W]."""
        frames = batch[self._primary_image_key(batch)]
        return frames if frames.ndim == 4 else frames[:, -1]

    def _current_depth(self, batch) -> torch.Tensor:
        """Pseudo-depth label for the current frame, [B, 1, H, W]."""
        depth = batch[self.config.world_model_depth_key]
        if depth.ndim == 5:  # (B, T, 1, H, W)
            depth = depth[:, -1]
        elif depth.ndim == 3:  # (B, H, W)
            depth = depth.unsqueeze(1)
        return depth

    # --- training ----------------------------------------------------------
    def forward(self, batch: dict[str, torch.Tensor], noise=None, time=None):
        """Full training forward pass; mirrors SmolVLAPolicy.forward plus the aux loss."""
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("actions_id_pad")
        loss_dict = {}

        run_world_model = (
            self.training
            and self.world_model is not None
            and self.config.world_model_depth_key in batch
        )
        if run_world_model:
            self._collected_img_embs = []

        try:
            losses = self.model.forward(
                images, img_masks, lang_tokens, lang_masks, state, actions, noise, time
            )
            loss_dict["losses_after_forward"] = losses.clone()

            if actions_is_pad is not None:
                in_episode_bound = ~actions_is_pad
                losses = losses * in_episode_bound.unsqueeze(-1)
                loss_dict["losses_after_in_ep_bound"] = losses.clone()

            # Remove padding
            losses = losses[:, :, : self.config.max_action_dim]
            loss_dict["losses_after_rm_padding"] = losses.clone()

            loss = losses.mean()
            loss_dict["action_loss"] = loss.item()

            if run_world_model:
                aux_loss, components = self.world_model(
                    tokens=self._current_visual_tokens(batch),
                    images=self._current_frame(batch),
                    depth_gt=self._current_depth(batch),
                )
                loss = loss + self.config.world_model_aux_weight * aux_loss
                loss_dict.update({k: v.item() for k, v in components.items()})
        finally:
            self._collected_img_embs = None

        loss_dict["loss"] = loss.item()
        return loss, loss_dict
