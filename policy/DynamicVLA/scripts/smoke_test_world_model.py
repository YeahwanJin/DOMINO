"""Checks for the world-model auxiliary branch. Run from policy/DynamicVLA.

    python scripts/smoke_test_world_model.py            # branch only, no downloads
    python scripts/smoke_test_world_model.py --policy   # also builds the full policy

Verifies that the loss is finite and differentiable, that gradients reach the tokens
(i.e. the policy's vision tower would be trained by it), and that nothing from
`world_model` is imported when the branch is disabled.
"""

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from policies.dynamicvla.configuration_dynamicvla import DynamicVLAConfig  # noqa: E402


def make_config(**overrides) -> DynamicVLAConfig:
    config = DynamicVLAConfig()
    config.use_world_model_aux = True
    config.world_model_grid_size = 8  # 64x64 maps, keeps the test fast
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_branch() -> None:
    from policies.dynamicvla.world_model import WorldModelAuxLoss

    config = make_config()
    token_dim, batch, num_tokens = 64, 2, 197  # 197 = non-square on purpose
    branch = WorldModelAuxLoss(config, token_dim=token_dim)

    tokens = torch.randn(batch, num_tokens, token_dim, requires_grad=True)
    images = torch.rand(batch, 3, 240, 320)
    depth = torch.rand(batch, 1, 128, 128)

    loss, components = branch(tokens=tokens, images=images, depth_gt=depth)
    assert torch.isfinite(loss), f"loss is not finite: {loss}"
    loss.backward()

    assert tokens.grad is not None and tokens.grad.abs().sum() > 0, (
        "no gradient reached the visual tokens — the auxiliary loss would not "
        "shape the policy's representation"
    )
    trained = [
        n for n, p in branch.named_parameters() if p.grad is not None and p.grad.any()
    ]
    assert trained, "no decoder parameter received a gradient"

    print(f"  loss={loss.item():.4f} components={ {k: round(v.item(), 4) for k, v in components.items()} }")
    print(f"  {len(trained)} decoder tensors received gradients")
    print(f"  token grad norm={tokens.grad.norm().item():.4f}")


def test_disabled_does_not_import() -> None:
    for name in [m for m in sys.modules if "world_model" in m]:
        del sys.modules[name]

    from policies.dynamicvla.modeling_dynamicvla import DynamicVLAPolicy  # noqa: F401

    leaked = [m for m in sys.modules if "world_model" in m]
    assert not leaked, f"world_model imported on the inference path: {leaked}"
    print("  world_model is not imported when the branch is disabled")


def test_policy() -> None:
    from lerobot.configs.types import FeatureType, PolicyFeature

    from policies.dynamicvla.modeling_dynamicvla import DynamicVLAPolicy

    config = make_config(
        n_obs_steps=1,
        chunk_size=4,
        n_action_steps=4,
        temporal_fusion="attn",
        resize_imgs_with_padding=(384, 384),
        vlm_model_name="HuggingFaceTB/SmolLM2-360M",
        num_vlm_layers=2,
    )
    config.input_features = {
        "observation.images.head_camera": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 240, 320)
        ),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(14,))
    }

    policy = DynamicVLAPolicy(config)
    policy.train()
    assert policy.world_model is not None, "world model was not constructed"

    batch = {
        "observation.images.head_camera": torch.rand(2, 1, 3, 240, 320),
        "observation.state": torch.rand(2, 1, 14),
        "action": torch.rand(2, config.chunk_size, 14),
        "observation.depth.head_camera": torch.rand(2, 1, 128, 128),
        "task": ["pick up the bottle", "pick up the bottle"],
    }
    loss, loss_dict = policy.forward(batch)
    assert torch.isfinite(loss), f"loss is not finite: {loss}"
    assert "world_model/depth_l1" in loss_dict, loss_dict
    loss.backward()

    vision_grads = [
        n
        for n, p in policy.model.vlm_with_expert.named_parameters()
        if p.grad is not None and p.grad.any()
    ]
    print(f"  loss={loss.item():.4f} keys={sorted(loss_dict)}")
    print(f"  {len(vision_grads)} VLM tensors received gradients")

    # The depth key is absent at inference, so the branch must not fire.
    policy.eval()
    del batch["observation.depth.head_camera"]
    with torch.no_grad():
        chunk = policy.predict_action_chunk(batch)
    print(f"  inference chunk shape={tuple(chunk.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        action="store_true",
        help="Also build the full policy (downloads a VLM backbone).",
    )
    args = parser.parse_args()

    print("[1] world-model branch")
    test_branch()
    print("[2] disabled branch is not imported")
    test_disabled_does_not_import()
    if args.policy:
        print("[3] full policy forward")
        test_policy()
    print("OK")


if __name__ == "__main__":
    main()
