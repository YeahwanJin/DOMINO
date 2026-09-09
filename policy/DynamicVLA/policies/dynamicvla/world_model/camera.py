"""Camera parameters for the world-model reconstruction loss.

The auxiliary loss reconstructs the *same* view it decoded from, so camera space is
taken to be world space (identity view matrix) — the unprojected points already live
in camera space. Only the intrinsics of the source camera matter, and DOMINO defines
those in `task_config/_camera_config.yml` as a vertical FOV plus a resolution
(see `envs/camera/camera.py`), so they are derived rather than stored per frame.
"""

import math

import torch


def intrinsics_for_square_render(
    render_size: int, fovy_deg: float, src_width: int, src_height: int
) -> tuple[float, float, float, float]:
    """Intrinsics (fx, fy, cx, cy) for a square `render_size` render.

    Assumes the source frame is resized (without padding) to `render_size` on both
    axes, which keeps the principal point centred — the rasterizer assumes a centred
    principal point.
    """
    tan_half_fovy = math.tan(math.radians(fovy_deg) / 2.0)
    tan_half_fovx = tan_half_fovy * (src_width / src_height)

    half = render_size / 2.0
    return half / tan_half_fovx, half / tan_half_fovy, half, half


def build_camera_params(
    render_size: int,
    fovy_deg: float,
    src_width: int,
    src_height: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    znear: float = 0.01,
    zfar: float = 100.0,
) -> dict[str, torch.Tensor]:
    """Renderer inputs for reconstructing the source view."""
    fx, fy, cx, cy = intrinsics_for_square_render(
        render_size, fovy_deg, src_width, src_height
    )
    half = render_size / 2.0
    tanfovx, tanfovy = half / fx, half / fy

    view = torch.eye(4, device=device, dtype=dtype)

    # OpenGL-style symmetric perspective projection, column-vector convention.
    proj = torch.zeros(4, 4, device=device, dtype=dtype)
    proj[0, 0] = 1.0 / tanfovx
    proj[1, 1] = 1.0 / tanfovy
    proj[2, 2] = zfar / (zfar - znear)
    proj[2, 3] = -(zfar * znear) / (zfar - znear)
    proj[3, 2] = 1.0

    return {
        "viewmatrix": view.expand(batch_size, 4, 4).contiguous(),
        # full projection; the view matrix is identity so this is just `proj`.
        "projmatrix": proj.expand(batch_size, 4, 4).contiguous(),
        "campos": torch.zeros(batch_size, 3, device=device, dtype=dtype),
        "tanfovx": tanfovx,
        "tanfovy": tanfovy,
        "intrinsics": torch.tensor(
            [fx, fy, cx, cy], device=device, dtype=dtype
        ).expand(batch_size, 4),
    }
