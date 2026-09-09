"""Generate Depth Anything V2 pseudo-depth labels for a LeRobot dataset.

The labels are written as per-episode sidecars rather than new parquet columns, so
they can be regenerated without rewriting the dataset:

    <dataset_root>/depth/<camera>/episode_XXXXXX.npy   float16, (T, H, W)

`utils.datasets.LeRobotDataset` picks them up through `aux_features`, and they are
consumed only by the training-time world-model loss — never as a model input.

Values are depth-ordered and normalised to [0, 1] (0 = near, 1 = far), matching the
`sigmoid` output of the world-model depth head. The relative Depth Anything V2
checkpoints predict *inverse* depth, so their output is inverted here; pass
`--metric` when using a Metric-Depth checkpoint.

Usage:
    python scripts/generate_depth.py --repo-id domino/adjust_bottle
"""

import argparse
import pathlib

import numpy as np
import torch
import torchcodec.decoders
import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def decode_video(path: pathlib.Path) -> torch.Tensor:
    """Decode a whole episode video to uint8 [T, 3, H, W]."""
    decoder = torchcodec.decoders.VideoDecoder(str(path), seek_mode="approximate")
    return decoder[:]


@torch.no_grad()
def predict_depth(
    frames: torch.Tensor,
    model,
    processor,
    device: torch.device,
    batch_size: int,
    out_size: int,
) -> torch.Tensor:
    """uint8 [T, 3, H, W] -> float32 [T, out_size, out_size] raw model output."""
    outputs = []
    for start in range(0, frames.shape[0], batch_size):
        chunk = frames[start : start + batch_size]
        inputs = processor(images=list(chunk), return_tensors="pt").to(device)
        predicted = model(**inputs).predicted_depth  # [B, h, w]
        predicted = torch.nn.functional.interpolate(
            predicted.unsqueeze(1).float(),
            size=(out_size, out_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        outputs.append(predicted.cpu())

    return torch.cat(outputs, dim=0)


def to_normalized_depth(
    raw: torch.Tensor, metric: bool, scope: str
) -> torch.Tensor:
    """Map raw predictions to depth-ordered values in [0, 1] (0 = near)."""
    if scope == "episode":
        lo, hi = raw.min(), raw.max()
    elif scope == "frame":
        lo = raw.amin(dim=(-2, -1), keepdim=True)
        hi = raw.amax(dim=(-2, -1), keepdim=True)
    else:
        raise ValueError(f"Unknown normalization scope: {scope}")

    normalized = (raw - lo) / (hi - lo + 1e-8)
    # Relative checkpoints output inverse depth: large == near.
    return normalized if metric else 1.0 - normalized


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--root", type=pathlib.Path, default=None)
    parser.add_argument("--camera", type=str, default="head_camera")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--out-size",
        type=int,
        default=128,
        help="Stored resolution; keep equal to WORLD_MODEL_GRID_SIZE * 8.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--metric",
        action="store_true",
        help="The checkpoint predicts metric depth rather than inverse depth.",
    )
    parser.add_argument(
        "--normalize", type=str, default="episode", choices=["episode", "frame"]
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    meta = LeRobotDatasetMetadata(args.repo_id, args.root)
    video_key = f"observation.images.{args.camera}"
    if video_key not in meta.video_keys:
        raise ValueError(f"{video_key} not in dataset video keys: {meta.video_keys}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForDepthEstimation.from_pretrained(args.model).to(device).eval()

    out_dir = meta.root / "depth" / args.camera
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep_idx in tqdm.tqdm(range(meta.total_episodes)):
        out_path = out_dir / ("episode_%06d.npy" % ep_idx)
        if out_path.exists() and not args.overwrite:
            continue

        frames = decode_video(meta.root / meta.get_video_file_path(ep_idx, video_key))
        raw = predict_depth(
            frames, model, processor, device, args.batch_size, args.out_size
        )
        depth = to_normalized_depth(raw, args.metric, args.normalize)
        np.save(out_path, depth.numpy().astype(np.float16))

    print(f"Wrote depth labels to {out_dir}")


if __name__ == "__main__":
    main()
