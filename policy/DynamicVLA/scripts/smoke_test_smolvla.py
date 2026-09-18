"""Checks for the two SmolVLA experiments. Run from policy/DynamicVLA.

    python scripts/smoke_test_smolvla.py            # wiring only, no downloads
    python scripts/smoke_test_smolvla.py --policy   # also builds the full policy

Experiment 1 is stock LeRobot `SmolVLAPolicy`, so what is worth checking is the
wiring around it: that both policy types resolve, that the configs parse into the
right dataclasses, and that a single-frame delta_timestamps falls out.

Experiment 2 additionally has to prove that the auxiliary loss reaches SmolVLA's
vision tower and that it stays out of the inference graph entirely.
"""

import argparse
import pathlib
import sys

import easydict
import torch
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import utils.helpers  # noqa: E402

CONFIG_DIR = pathlib.Path(__file__).resolve().parent.parent / "configs"


def _load_yaml(name: str) -> easydict.EasyDict:
    with open(CONFIG_DIR / name) as f:
        return easydict.EasyDict(yaml.load(f, Loader=yaml.FullLoader))


def test_registry() -> None:
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    from policies.smolvla_wm import SmolVLAWMConfig, SmolVLAWMPolicy

    assert utils.helpers.get_policy_class("smolvla") is SmolVLAPolicy
    assert utils.helpers.get_policy_class("smolvla_wm") is SmolVLAWMPolicy
    # The world-model config must stay a drop-in SmolVLA config.
    assert issubclass(SmolVLAWMConfig, SmolVLAConfig)
    assert issubclass(SmolVLAWMPolicy, SmolVLAPolicy)
    print("  both policy types resolve; smolvla_wm subclasses smolvla")


def test_configs() -> None:
    from policies.smolvla_wm import SmolVLAWMConfig

    baseline = _load_yaml("domino_smolvla.yaml")
    world_model = _load_yaml("domino_smolvla_wm.yaml")

    # The experiments must differ only in the policy type and the world-model block.
    shared = {
        k: v
        for k, v in baseline.POLICY.items()
        if not k.startswith("WORLD_MODEL") and k != "TYPE"
    }
    for key, value in shared.items():
        assert world_model.POLICY.get(key) == value, (
            f"POLICY.{key} differs between the experiments: "
            f"{value} vs {world_model.POLICY.get(key)} — the reconstruction loss "
            "would no longer be the only variable"
        )
    for section in ("DATASET", "TRAIN"):
        assert baseline[section] == world_model[section], f"{section} differs"

    cfg = utils.helpers.get_policy_cfg(world_model.POLICY)
    assert isinstance(cfg, SmolVLAWMConfig), type(cfg)
    assert cfg.use_world_model_aux is True
    assert cfg.world_model_depth_key == "observation.depth.cam_high"
    assert cfg.chunk_size == 20 and cfg.n_action_steps == 10
    assert cfg.freeze_vision_encoder is False and cfg.train_expert_only is False, (
        "the world-model loss only shapes the representation if the vision tower "
        "and the VLM are trainable"
    )

    # SmolVLA is single-frame: DELTA_TIMESTAMPS is null, so the policy's own
    # observation_delta_indices ([0]) has to be what the dataset gets.
    deltas = utils.helpers.get_delta_timestamps(
        world_model.POLICY, world_model.DATASET.DELTA_TIMESTAMPS
    )
    assert deltas["observation"] == [0], deltas
    assert len(deltas["action"]) == cfg.chunk_size, deltas
    print(f"  configs agree; observation deltas={deltas['observation']} chunk={cfg.chunk_size}")


def test_disabled_does_not_import() -> None:
    for name in [m for m in sys.modules if "world_model" in m]:
        del sys.modules[name]

    from policies.smolvla_wm.modeling_smolvla_wm import SmolVLAWMPolicy  # noqa: F401

    leaked = [m for m in sys.modules if "world_model" in m]
    assert not leaked, f"world_model imported on the inference path: {leaked}"
    print("  world_model is not imported when the branch is disabled")


def test_policy() -> None:
    from lerobot.configs.types import FeatureType, PolicyFeature

    from policies.smolvla_wm import SmolVLAWMConfig, SmolVLAWMPolicy

    config = SmolVLAWMConfig()
    config.use_world_model_aux = True
    config.world_model_grid_size = 8  # 64x64 maps, keeps the test fast
    config.world_model_smooth_weight = 0.1
    config.world_model_depth_key = "observation.depth.cam_high"
    config.n_obs_steps = 1
    config.chunk_size = 4
    config.n_action_steps = 4
    config.num_vlm_layers = 2
    config.load_vlm_weights = False
    config.freeze_vision_encoder = False
    config.train_expert_only = False
    # cam_high deliberately second, so the token/camera indexing is exercised.
    config.input_features = {
        "observation.images.cam_left_wrist": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 240, 320)
        ),
        "observation.images.cam_high": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 240, 320)
        ),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(14,))
    }

    policy = SmolVLAWMPolicy(config)
    policy.train()
    assert policy.world_model is not None, "world model was not constructed"

    batch = {
        "observation.images.cam_left_wrist": torch.rand(2, 3, 240, 320),
        "observation.images.cam_high": torch.rand(2, 3, 240, 320),
        "observation.state": torch.rand(2, 14),
        "action": torch.rand(2, config.chunk_size, 14),
        "observation.depth.cam_high": torch.rand(2, 1, 64, 64),
        "task": ["pick up the bottle", "pick up the bottle"],
    }
    loss, loss_dict = policy.forward(batch)
    assert torch.isfinite(loss), f"loss is not finite: {loss}"
    assert "world_model/depth_l1" in loss_dict, loss_dict
    assert loss_dict["loss"] != loss_dict["action_loss"], (
        "the auxiliary loss did not change the total loss"
    )
    loss.backward()

    vision_grads = [
        n
        for n, p in policy.model.vlm_with_expert.get_vlm_model()
        .vision_model.named_parameters()
        if p.grad is not None and p.grad.any()
    ]
    assert vision_grads, (
        "no gradient reached SmolVLA's vision tower — the reconstruction loss "
        "would not shape the representation the policy conditions on"
    )
    decoder_grads = [
        n
        for n, p in policy.world_model.named_parameters()
        if p.grad is not None and p.grad.any()
    ]
    assert decoder_grads, "no decoder parameter received a gradient"

    print(f"  loss={loss.item():.4f} keys={sorted(loss_dict)}")
    print(f"  {len(vision_grads)} vision-tower tensors received gradients")
    print(f"  {len(decoder_grads)} decoder tensors received gradients")

    # The depth key is absent at inference, so the branch must not fire.
    policy.eval()
    del batch["observation.depth.cam_high"]
    with torch.no_grad():
        chunk = policy.predict_action_chunk(batch)
    assert chunk.shape == (2, config.chunk_size, 14), chunk.shape
    print(f"  inference chunk shape={tuple(chunk.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        action="store_true",
        help="Also build the full policy (downloads a VLM config).",
    )
    args = parser.parse_args()

    print("[1] policy registry")
    test_registry()
    print("[2] experiment configs")
    test_configs()
    print("[3] disabled branch is not imported")
    test_disabled_does_not_import()
    if args.policy:
        print("[4] full policy forward")
        test_policy()
    print("OK")


if __name__ == "__main__":
    main()
