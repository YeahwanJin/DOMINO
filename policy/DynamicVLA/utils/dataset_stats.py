# -*- coding: utf-8 -*-
"""Dataset statistics for the eval-time normalizers.

`utils.helpers.save_checkpoint` writes the policy config but not the dataset stats,
and `PreTrainedPolicy._load_as_safetensor` deliberately drops the normalisation
buffers from the checkpoint. The eval-side wrappers therefore have to rebuild the
Normalize / Unnormalize modules from the stats of the dataset the policy was trained
on, which is what this module resolves.

Shared by `dynamicvla_model.py` and `smolvla_model.py` so both policies normalise
observations and actions identically.
"""

import json
import pathlib

import numpy as np
import torch

DATA_DIRS = (
    pathlib.Path("data/lerobot_data"),
    pathlib.Path("/workspace/DOMINO/data/lerobot_data"),
)
STATS_FILENAMES = ("stats.json", "dataset_stats.json")
PER_TASK_STATS_FILENAME = "stats_gr00t.json"


def load_dataset_stats(ckpt_dir: pathlib.Path, label: str = "policy"):
    """Stats as `{feature: {mean|std|min|max: tensor}}`, or None if unresolvable.

    Looks for a stats file written next to the checkpoint first; falls back to
    aggregating `data/lerobot_data/*/meta/stats_gr00t.json` the same way
    `utils.datasets.get_dataset("all")` merges its per-task stats at training time.
    """
    ckpt_dir = pathlib.Path(ckpt_dir)
    for fname in STATS_FILENAMES:
        path = ckpt_dir / fname
        if path.exists():
            with open(path) as f:
                raw = json.load(f)
            print(f"✅ {label}: loaded dataset_stats from {path}")
            return {
                k: {sk: torch.tensor(sv, dtype=torch.float32) for sk, sv in v.items()}
                for k, v in raw.items()
            }

    data_dir = next((d for d in DATA_DIRS if d.exists()), None)
    if data_dir is None:
        print(f"⚠️ {label}: none of {[str(d) for d in DATA_DIRS]} exist")
        return None

    all_stats = []
    for task_dir in sorted(data_dir.iterdir()):
        if not task_dir.is_dir() or task_dir.name == "local":
            continue
        stats_file = task_dir / "meta" / PER_TASK_STATS_FILENAME
        if stats_file.exists():
            with open(stats_file) as f:
                all_stats.append(json.load(f))

    if not all_stats:
        print(f"⚠️ {label}: no {PER_TASK_STATS_FILENAME} found in {data_dir}")
        return None

    merged = {}
    for key in all_stats[0]:
        std = np.mean([s[key]["std"] for s in all_stats], axis=0)
        merged[key] = {
            "mean": torch.tensor(
                np.mean([s[key]["mean"] for s in all_stats], axis=0), dtype=torch.float32
            ),
            # A zero std would blow up the normalizer; the feature is constant anyway.
            "std": torch.tensor(np.where(std < 1e-6, 1.0, std), dtype=torch.float32),
            "min": torch.tensor(
                np.min([s[key]["min"] for s in all_stats], axis=0), dtype=torch.float32
            ),
            "max": torch.tensor(
                np.max([s[key]["max"] for s in all_stats], axis=0), dtype=torch.float32
            ),
        }

    print(f"✅ {label}: aggregated dataset_stats from {len(all_stats)} tasks in {data_dir}")
    return merged


def install_normalizers(policy, cfg, dataset_stats) -> None:
    """Replace the policy's normalizers with ones built from `dataset_stats`."""
    if dataset_stats is None:
        return

    from lerobot.policies.normalize import Normalize, Unnormalize

    policy.normalize_inputs = Normalize(
        cfg.input_features, cfg.normalization_mapping, dataset_stats
    )
    policy.normalize_targets = Normalize(
        cfg.output_features, cfg.normalization_mapping, dataset_stats
    )
    policy.unnormalize_outputs = Unnormalize(
        cfg.output_features, cfg.normalization_mapping, dataset_stats
    )
    print("✅ Successfully injected dataset_stats into normalizers!")
