import json
import pathlib
import numpy as np

def main():
    base_dir = pathlib.Path("data/lerobot_data")
    if not base_dir.exists():
        base_dir = pathlib.Path("/workspace/DOMINO/data/lerobot_data")

    task_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name != "local"])
    print(f"Found {len(task_dirs)} tasks.")

    all_stats = []
    for td in task_dirs:
        stats_file = td / "meta" / "stats_gr00t.json"
        if stats_file.exists():
            with open(stats_file) as f:
                all_stats.append(json.load(f))
        else:
            print(f"Warning: {stats_file} does not exist.")

    print(f"Loaded stats for {len(all_stats)} tasks.")
    keys = list(all_stats[0].keys())

    merged_stats = {}
    for key in keys:
        merged_stats[key] = {
            "mean": np.mean([s[key]["mean"] for s in all_stats], axis=0).tolist(),
            "std": np.mean([s[key]["std"] for s in all_stats], axis=0).tolist(),
            "min": np.min([s[key]["min"] for s in all_stats], axis=0).tolist(),
            "max": np.max([s[key]["max"] for s in all_stats], axis=0).tolist(),
        }

    # Target output directories. Anything listed here gets a stats.json next to its
    # checkpoint, which utils.dataset_stats.load_dataset_stats prefers over
    # re-aggregating at eval time. Keep in sync with the `ckpt_dir` defaults in
    # policy/*/deploy_policy.yml.
    target_dirs = [
        pathlib.Path("policy/DynamicVLA/runs/checkpoints/domino_all_tasks_with_wm"),
        pathlib.Path("policy/DynamicVLA/runs/checkpoints/domino_adjust_bottle"),
        pathlib.Path("policy/DynamicVLA/runs/checkpoints/domino_smolvla"),
        pathlib.Path("policy/DynamicVLA/runs/checkpoints/domino_smolvla_wm"),
    ]

    for target_dir in target_dirs:
        target_dir.mkdir(parents=True, exist_ok=True)
        for fname in ("stats.json", "dataset_stats.json"):
            out_path = target_dir / fname
            with open(out_path, "w") as f:
                json.dump(merged_stats, f, indent=4)
            print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
