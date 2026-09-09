"""Differentiable Gaussian rasterization + photometric losses.

Trimmed from GaussianDream's `gaussian_renderer.py`. The CUDA extension is imported
lazily so that the depth-only auxiliary loss stays usable on machines where
`diff-gaussian-rasterization` has not been compiled.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_RASTERIZER = None


def _load_rasterizer():
    """Import the CUDA rasterizer on first use."""
    global _RASTERIZER
    if _RASTERIZER is None:
        from diff_gaussian_rasterization import (  # noqa: PLC0415
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        _RASTERIZER = (GaussianRasterizationSettings, GaussianRasterizer)
    return _RASTERIZER


def is_rasterizer_available() -> bool:
    try:
        _load_rasterizer()
    except ImportError:
        return False
    return True


class GaussianRenderer(nn.Module):
    """Rasterize 3D Gaussians into an image so a photometric loss can be taken."""

    def __init__(self, image_size: int = 224, sh_degree: int = 1):
        super().__init__()
        self.image_size = image_size
        self.sh_degree = sh_degree
        # Fail at construction rather than mid-training.
        _load_rasterizer()

    def forward(
        self,
        gaussian_params: dict[str, torch.Tensor],
        camera_params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Returns [B, 3, image_size, image_size] renders."""
        settings_cls, rasterizer_cls = _load_rasterizer()

        xyz = torch.nan_to_num(gaussian_params["xyz"], nan=0.0, posinf=0.0, neginf=0.0)
        opacity = torch.clamp(gaussian_params["opacity"], min=0.0, max=1.0)
        scales = torch.clamp(gaussian_params["scales"], min=1e-6, max=10.0)
        rotations = gaussian_params["rotations"]
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + 1e-8)
        shs = self._reshape_sh(gaussian_params["sh"])

        device = xyz.device
        screenspace_points = torch.zeros_like(
            xyz, dtype=xyz.dtype, device=device, requires_grad=xyz.requires_grad
        )

        tanfovx = _as_float(camera_params["tanfovx"])
        tanfovy = _as_float(camera_params["tanfovy"])

        rendered = []
        for b in range(xyz.shape[0]):
            raster_settings = settings_cls(
                image_height=self.image_size,
                image_width=self.image_size,
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=torch.zeros(3, device=device),
                scale_modifier=1.0,
                # diff-gaussian-rasterization expects column-major matrices.
                viewmatrix=camera_params["viewmatrix"][b].transpose(0, 1),
                projmatrix=camera_params["projmatrix"][b].transpose(0, 1),
                sh_degree=self.sh_degree,
                campos=camera_params["campos"][b],
                prefiltered=False,
                debug=False,
            )
            rasterizer = rasterizer_cls(raster_settings)

            valid = (
                torch.isfinite(xyz[b]).all(dim=-1)
                & torch.isfinite(scales[b]).all(dim=-1)
                & torch.isfinite(rotations[b]).all(dim=-1)
                & (opacity[b].squeeze(-1) > 0.0)
                & (scales[b].min(dim=-1)[0] > 0.0)
            )
            if not valid.any():
                rendered.append(
                    torch.zeros(3, self.image_size, self.image_size, device=device)
                )
                continue

            color, _ = rasterizer(
                means3D=xyz[b][valid],
                means2D=screenspace_points[b][valid],
                opacities=opacity[b][valid],
                shs=shs[b][valid],
                scales=scales[b][valid],
                rotations=rotations[b][valid],
            )
            rendered.append(color)

        return torch.stack(rendered, dim=0)

    def _reshape_sh(self, sh: torch.Tensor) -> torch.Tensor:
        """[B, N, 9] -> [B, N, (sh_degree+1)^2, 3], zero-padding missing bands."""
        num_coeffs = (self.sh_degree + 1) ** 2
        expected = num_coeffs * 3
        b, n, dim = sh.shape
        if dim > expected:
            sh = sh[..., :expected]
        elif dim < expected:
            sh = F.pad(sh, (0, expected - dim))
        return sh.reshape(b, n, num_coeffs, 3)


def _as_float(value) -> float:
    return value.item() if isinstance(value, torch.Tensor) else float(value)


def compute_ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """SSIM loss with a 3x3 kernel. Returns a per-pixel loss map."""
    pad = nn.ReflectionPad2d(1)
    pred, target = pad(pred), pad(target)

    mu_pred = F.avg_pool2d(pred, kernel_size=3, stride=1)
    mu_target = F.avg_pool2d(target, kernel_size=3, stride=1)

    musq_pred, musq_target = mu_pred.pow(2), mu_target.pow(2)
    mu_cross = mu_pred * mu_target

    sigma_pred = F.avg_pool2d(pred.pow(2), kernel_size=3, stride=1) - musq_pred
    sigma_target = F.avg_pool2d(target.pow(2), kernel_size=3, stride=1) - musq_target
    sigma_cross = F.avg_pool2d(pred * target, kernel_size=3, stride=1) - mu_cross

    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mu_cross + c1) * (2 * sigma_cross + c2)) / (
        (musq_pred + musq_target + c1) * (sigma_pred + sigma_target + c2) + 1e-8
    )
    return torch.clamp((1 - ssim) / 2, 0, 1)


def compute_photometric_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """0.85 * SSIM + 0.15 * L1, both [B, 3, H, W]."""
    l1 = (target - pred).abs().mean(1, True)
    ssim = compute_ssim_loss(pred, target).mean(1, True)
    return (0.85 * ssim + 0.15 * l1).mean()


def compute_edge_smooth_loss(rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    """Edge-aware depth smoothness. rgb: [B,3,H,W], depth: [B,1,H,W]."""
    grad_rgb_x = (rgb[:, :, :, :-1] - rgb[:, :, :, 1:]).abs().mean(1, True)
    grad_rgb_y = (rgb[:, :, :-1, :] - rgb[:, :, 1:, :]).abs().mean(1, True)

    grad_depth_x = (depth[:, :, :, :-1] - depth[:, :, :, 1:]).abs()
    grad_depth_y = (depth[:, :, :-1, :] - depth[:, :, 1:, :]).abs()

    grad_depth_x = grad_depth_x * (-grad_rgb_x).exp()
    grad_depth_y = grad_depth_y * (-grad_rgb_y).exp()
    return grad_depth_x.mean() + grad_depth_y.mean()
