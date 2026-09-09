"""Minimal static Gaussian decoder.

Adapted from GaussianDream (`src/gaussiandream/models_pytorch/decoder.py`), keeping
only the current-frame ("static") decode path. The future/velocity rollout heads,
the multi-horizon curriculum and the auxiliary future-depth head are intentionally
not ported.

Unlike GaussianDream, the tokens fed in here come from the policy's *own* vision
tower rather than a separate VGGT backbone, so the depth/render gradients reach the
representation the policy actually uses.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _UpsampleBlock(nn.Module):
    """ConvTranspose upsample block with GroupNorm + GELU + residual."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = F.interpolate(
            self.skip(x), scale_factor=2, mode="bilinear", align_corners=False
        )
        h = F.gelu(self.norm1(self.up(x)))
        h = F.gelu(self.norm2(self.conv(h)))
        return h + skip


class _FeatureFusionBlock(nn.Module):
    """DPT-style feature fusion block for multi-scale feature integration."""

    def __init__(self, features: int, has_residual: bool = True):
        super().__init__()
        self.has_residual = has_residual
        if has_residual:
            self.residual_conv = nn.Sequential(
                nn.Conv2d(features, features, 3, padding=1, bias=True),
                nn.GroupNorm(min(32, features), features),
                nn.GELU(),
                nn.Conv2d(features, features, 3, padding=1, bias=True),
            )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1, bias=True),
            nn.GroupNorm(min(32, features), features),
            nn.GELU(),
            nn.Conv2d(features, features, 1, bias=True),
        )

    def forward(self, x, residual=None, size=None):
        if self.has_residual and residual is not None:
            if residual.shape[2:] != x.shape[2:]:
                residual = F.interpolate(
                    residual, size=x.shape[2:], mode="bilinear", align_corners=True
                )
            x = x + self.residual_conv(residual)

        x = self.fusion_conv(x)
        if size is not None and x.shape[2:] != size:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)

        return x


class SharedGaussianBackbone(nn.Module):
    """Token grid -> dense feature map (8x spatial upsampling)."""

    def __init__(self, token_dim: int, feature_dim: int = 128):
        super().__init__()
        self.layer1 = _UpsampleBlock(token_dim, 512)
        self.layer2 = _UpsampleBlock(512, 256)
        self.layer3 = _UpsampleBlock(256, feature_dim)

        self.fusion1 = _FeatureFusionBlock(feature_dim, has_residual=False)
        self.fusion2 = _FeatureFusionBlock(feature_dim, has_residual=True)
        self.fusion3 = _FeatureFusionBlock(feature_dim, has_residual=True)

        self.proj_feat2 = nn.Conv2d(256, feature_dim, 1)
        self.proj_feat1 = nn.Conv2d(512, feature_dim, 1)

    def forward(self, token_grid: torch.Tensor) -> torch.Tensor:
        feat1 = self.layer1(token_grid)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)

        fused = self.fusion1(feat3)
        fused = self.fusion2(fused, residual=self.proj_feat2(feat2))
        fused = self.fusion3(fused, residual=self.proj_feat1(feat1))
        return fused


class GeometryHead(nn.Module):
    """Raw depth map, plus per-pixel rotation(4) + scale(3) + opacity(1).

    `predict_params` is off when only the depth loss is active, so the unused
    Gaussian parameters are neither allocated nor computed.
    """

    def __init__(self, feature_dim: int = 128, predict_params: bool = True):
        super().__init__()
        self.param_head = (
            nn.Conv2d(feature_dim, 8, 3, padding=1) if predict_params else None
        )
        self.depth_refine = nn.Sequential(
            nn.Conv2d(feature_dim, 64, 3, padding=1),
            nn.GroupNorm(min(32, 64), 64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(min(32, 64), 64),
            nn.GELU(),
            nn.Conv2d(64, 1, 3, padding=1),
        )
        for module in self.depth_refine.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        if self.param_head is not None:
            nn.init.xavier_uniform_(self.param_head.weight, gain=0.01)
            nn.init.zeros_(self.param_head.bias)

    def forward(self, shared_features: torch.Tensor) -> dict[str, torch.Tensor]:
        result = {"depth": self.depth_refine(shared_features)}
        if self.param_head is not None:
            result["geom_params"] = self.param_head(shared_features)
        return result


class AppearanceHead(nn.Module):
    """Per-pixel SH coefficients, optionally fused with the RGB frame."""

    def __init__(
        self,
        use_image_fusion: bool = True,
        img_dim: int = 3,
        feature_dim: int = 128,
        image_fusion_alpha: float = 1.0,
        stop_grad_geometry: bool = True,
    ):
        super().__init__()
        self.use_image_fusion = bool(use_image_fusion)
        self.image_fusion_alpha = float(image_fusion_alpha)
        self.stop_grad_geometry = bool(stop_grad_geometry)

        if use_image_fusion:
            self.img_merger = nn.Sequential(
                nn.Conv2d(img_dim, feature_dim, 7, padding=3),
                nn.GELU(),
            )

        self.sh_head = nn.Conv2d(feature_dim, 9, 3, padding=1)
        nn.init.xavier_uniform_(self.sh_head.weight, gain=0.01)
        nn.init.zeros_(self.sh_head.bias)

    def forward(
        self, shared_features: torch.Tensor, images: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Appearance must not be able to explain away geometry errors through the
        # RGB shortcut, hence the detach.
        geom_context = (
            shared_features.detach() if self.stop_grad_geometry else shared_features
        )
        appearance_features = geom_context
        if self.use_image_fusion and images is not None:
            if images.shape[2:] != geom_context.shape[2:]:
                images = F.interpolate(
                    images,
                    size=geom_context.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                )
            appearance_features = (
                appearance_features + self.image_fusion_alpha * self.img_merger(images)
            )
        return self.sh_head(appearance_features)


class StaticGaussianDecoder(nn.Module):
    """Visual tokens -> per-pixel depth map + 3D Gaussian parameters.

    Args:
        token_dim: width of the incoming visual tokens.
        grid_size: tokens are resampled onto a ``grid_size x grid_size`` grid; the
            decoded maps are ``8 * grid_size`` on a side.
        predict_gaussians: also emit splat parameters. Only needed for the render
            loss — with depth supervision alone those heads would get no gradient.
        use_image_fusion: let the appearance head see the RGB frame.
    """

    # Damps the higher-order SH bands, which are poorly constrained by a single view.
    SH_MASK = (1.0, 1.0, 1.0, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025)

    def __init__(
        self,
        token_dim: int,
        grid_size: int = 16,
        feature_dim: int = 128,
        predict_gaussians: bool = True,
        use_image_fusion: bool = True,
        depth_scale: float = 1.0,
        depth_offset: float = 0.1,
        max_gaussian_scale: float = 0.01,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.grid_size = int(grid_size)
        self.predict_gaussians = bool(predict_gaussians)
        self.depth_scale = float(depth_scale)
        self.depth_offset = float(depth_offset)
        self.max_gaussian_scale = float(max_gaussian_scale)

        self.shared_backbone = SharedGaussianBackbone(
            token_dim=token_dim, feature_dim=feature_dim
        )
        self.geometry_head = GeometryHead(
            feature_dim=feature_dim, predict_params=self.predict_gaussians
        )
        self.appearance_head = (
            AppearanceHead(use_image_fusion=use_image_fusion, feature_dim=feature_dim)
            if self.predict_gaussians
            else None
        )
        self.register_buffer(
            "sh_mask", torch.tensor(self.SH_MASK, dtype=torch.float32), persistent=False
        )

    def _tokens_to_grid(self, tokens: torch.Tensor) -> torch.Tensor:
        """[B, N, D] -> [B, D, grid_size, grid_size].

        The policy's vision tower emits an arbitrary token count, which is not
        necessarily a perfect square, so the sequence is resampled along the token
        axis before being folded into a square grid.
        """
        if tokens.ndim != 3:
            raise ValueError(f"Expected token tensor [B, N, D], got {tuple(tokens.shape)}")

        b, n, d = tokens.shape
        target = self.grid_size * self.grid_size
        src_grid = math.isqrt(n)
        if src_grid * src_grid == n:
            grid = tokens.transpose(1, 2).reshape(b, d, src_grid, src_grid)
            if src_grid != self.grid_size:
                grid = F.interpolate(
                    grid,
                    size=(self.grid_size, self.grid_size),
                    mode="bilinear",
                    align_corners=False,
                )
            return grid

        seq = tokens.transpose(1, 2)  # [B, D, N]
        if n != target:
            seq = F.interpolate(seq, size=target, mode="linear", align_corners=False)
        return seq.reshape(b, d, self.grid_size, self.grid_size)

    def forward(
        self,
        tokens: torch.Tensor,
        images: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            tokens: [B, N, token_dim] visual tokens from the policy.
            images: [B, 3, H, W] RGB frame in [0, 1] for appearance fusion.
            intrinsics: [B, 4] (fx, fy, cx, cy) already rescaled to the decoded
                map resolution. Falls back to a centred pinhole guess if None.

        Returns:
            dict with `depth` [B, 1, h, w] in [0, 1] and flattened Gaussian params.
        """
        shared_features = self.shared_backbone(self._tokens_to_grid(tokens))
        geometry = self.geometry_head(shared_features)

        # Relative depth in [0, 1] (0 = near). Depth Anything V2 labels are
        # normalised the same way, see scripts/generate_depth.py.
        depth = torch.sigmoid(geometry["depth"].squeeze(1))
        if not self.predict_gaussians:
            return {"depth": depth.unsqueeze(1)}

        sh_logits = self.appearance_head(shared_features, images=images)
        rot_raw, scale_raw, opa_raw = geometry["geom_params"].split([4, 3, 1], dim=1)

        b, h, w = depth.shape
        n = h * w

        rotations = rot_raw.permute(0, 2, 3, 1)
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + 1e-8)
        scales = (
            F.softplus(scale_raw.permute(0, 2, 3, 1), beta=1) * self.max_gaussian_scale
        )
        opacity = torch.sigmoid(opa_raw.permute(0, 2, 3, 1))
        sh = sh_logits.permute(0, 2, 3, 1) * self.sh_mask.view(1, 1, 1, 9)

        # The rasterizer needs strictly positive, well-scaled z. A global scale on
        # depth does not change the 2D projection (u = fx*x/z + cx is scale
        # invariant), so relative depth is fine here.
        metric_depth = depth * self.depth_scale + self.depth_offset
        xyz = self.depth_to_points(metric_depth, intrinsics)
        xyz = torch.clamp(xyz, min=-100.0, max=100.0)
        xyz = torch.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "xyz": xyz,
            "scales": torch.clamp(scales.reshape(b, n, 3), min=1e-7, max=10.0),
            "opacity": opacity.reshape(b, n, 1),
            "sh": sh.reshape(b, n, 9),
            "rotations": rotations.reshape(b, n, 4),
            "depth": depth.unsqueeze(1),
        }

    @staticmethod
    def depth_to_points(
        depth: torch.Tensor, intrinsics: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Unproject [B, H, W] depth to [B, H*W, 3] camera-space points."""
        b, h, w = depth.shape
        device, dtype = depth.device, depth.dtype

        if intrinsics is None:
            fx = fy = torch.full((b,), float(max(h, w)), device=device, dtype=dtype)
            cx = torch.full((b,), w / 2.0, device=device, dtype=dtype)
            cy = torch.full((b,), h / 2.0, device=device, dtype=dtype)
        else:
            intrinsics = intrinsics.to(device=device, dtype=dtype)
            fx, fy, cx, cy = intrinsics.unbind(dim=-1)

        u = torch.arange(0.5, w + 0.5, device=device, dtype=dtype)
        v = torch.arange(0.5, h + 0.5, device=device, dtype=dtype)
        v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")

        fx = fx.view(b, 1, 1)
        fy = fy.view(b, 1, 1)
        cx = cx.view(b, 1, 1)
        cy = cy.view(b, 1, 1)

        x = (u_grid[None] - cx) * depth / fx
        y = (v_grid[None] - cy) * depth / fy
        return torch.stack([x, y, depth], dim=-1).reshape(b, h * w, 3)
