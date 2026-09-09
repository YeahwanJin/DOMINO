"""Training-only world-model auxiliary branch (adapted from GaussianDream).

Nothing in this package is imported unless `DynamicVLAConfig.use_world_model_aux`
is True, which is never the case on the inference/eval path.
"""

from policies.dynamicvla.world_model.auxiliary_loss import WorldModelAuxLoss

__all__ = ["WorldModelAuxLoss"]
