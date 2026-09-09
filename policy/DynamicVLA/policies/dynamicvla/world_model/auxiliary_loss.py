"""Training-only world-model auxiliary loss.

The decoder reads the policy's own visual tokens and predicts a depth map plus a set
of 3D Gaussians for the current frame. Supervising that prediction against
Depth Anything V2 pseudo-depth (and, optionally, against the frame itself through a
differentiable rasterizer) pushes geometry into the representation the policy uses.

The branch runs only inside `DynamicVLAPolicy.forward`, never on the inference path,
and is not constructed at all unless `config.use_world_model_aux` is set.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from policies.dynamicvla.world_model.camera import build_camera_params
from policies.dynamicvla.world_model.decoder import StaticGaussianDecoder


class WorldModelAuxLoss(nn.Module):
    def __init__(self, config, token_dim: int):
        super().__init__()
        self.config = config
        self.grid_size = config.world_model_grid_size
        # SharedGaussianBackbone upsamples the token grid by 8x.
        self.render_size = self.grid_size * 8

        self.renderer = None
        if config.world_model_render_weight > 0:
            from policies.dynamicvla.world_model.renderer import (  # noqa: PLC0415
                GaussianRenderer,
                is_rasterizer_available,
            )

            if is_rasterizer_available():
                self.renderer = GaussianRenderer(
                    image_size=self.render_size,
                    sh_degree=config.world_model_sh_degree,
                )
            else:
                # Depth supervision alone is still useful, so degrade instead of dying.
                logging.warning(
                    "diff-gaussian-rasterization is not available; the world-model "
                    "render loss is disabled and only the depth loss will be used."
                )

        # The splat heads only receive gradient through the render loss.
        self.decoder = StaticGaussianDecoder(
            token_dim=token_dim,
            grid_size=self.grid_size,
            predict_gaussians=self.renderer is not None,
            use_image_fusion=config.world_model_use_image_fusion,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        images: torch.Tensor,
        depth_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            tokens: [B, N, token_dim] visual tokens for the current frame.
            images: [B, 3, H, W] RGB frame in [0, 1].
            depth_gt: [B, 1, H', W'] pseudo-depth in [0, 1] (0 = near).

        Returns:
            (scalar loss, dict of detached components for logging)
        """
        size = (self.render_size, self.render_size)
        images = F.interpolate(
            images.float(), size=size, mode="bilinear", align_corners=False
        )
        depth_gt = F.interpolate(
            depth_gt.float(), size=size, mode="bilinear", align_corners=False
        )

        camera_params = build_camera_params(
            render_size=self.render_size,
            fovy_deg=self.config.world_model_camera_fovy,
            src_width=self.config.world_model_camera_width,
            src_height=self.config.world_model_camera_height,
            batch_size=images.shape[0],
            device=images.device,
            dtype=images.dtype,
        )

        prediction = self.decoder(
            tokens.to(images.dtype),
            images=images,
            intrinsics=camera_params["intrinsics"],
        )

        loss = images.new_zeros(())
        components: dict[str, float] = {}

        depth_loss = F.l1_loss(prediction["depth"], depth_gt)
        loss = loss + self.config.world_model_depth_weight * depth_loss
        components["world_model/depth_l1"] = depth_loss.detach()

        if self.config.world_model_smooth_weight > 0:
            from policies.dynamicvla.world_model.renderer import (  # noqa: PLC0415
                compute_edge_smooth_loss,
            )

            smooth_loss = compute_edge_smooth_loss(images, prediction["depth"])
            loss = loss + self.config.world_model_smooth_weight * smooth_loss
            components["world_model/smooth"] = smooth_loss.detach()

        if self.renderer is not None:
            from policies.dynamicvla.world_model.renderer import (  # noqa: PLC0415
                compute_photometric_loss,
            )

            rendered = self.renderer(prediction, camera_params)
            render_loss = compute_photometric_loss(rendered, images)
            loss = loss + self.config.world_model_render_weight * render_loss
            components["world_model/render"] = render_loss.detach()

        return loss, components
