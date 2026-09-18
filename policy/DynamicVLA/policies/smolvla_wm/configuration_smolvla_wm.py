# -*- coding: utf-8 -*-
"""Configuration for SmolVLA + world-model auxiliary loss.

Everything SmolVLA needs is inherited from `SmolVLAConfig`; the fields added here
mirror `DynamicVLAConfig`'s `world_model_*` block one-for-one so the two experiments
can be compared under identical reconstruction settings.
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("smolvla_wm")
@dataclass
class SmolVLAWMConfig(SmolVLAConfig):
    # World-model auxiliary loss (training only, see policies/dynamicvla/world_model).
    # Must stay False at inference: it is what gates constructing the branch at all.
    use_world_model_aux: bool = False
    world_model_aux_weight: float = 1.0
    world_model_depth_weight: float = 1.0
    world_model_render_weight: float = 0.0
    world_model_smooth_weight: float = 0.0
    world_model_grid_size: int = 16
    world_model_use_image_fusion: bool = True
    world_model_sh_degree: int = 1
    # Key of the depth label in the batch, produced by scripts/generate_depth.py.
    world_model_depth_key: str = "observation.depth.cam_high"
    # Camera the depth labels come from, used to derive rendering intrinsics.
    # Defaults match DOMINO's D435 head camera (task_config/_camera_config.yml).
    world_model_camera_fovy: float = 37.0
    world_model_camera_width: int = 320
    world_model_camera_height: int = 240
