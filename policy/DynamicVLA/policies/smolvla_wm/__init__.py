"""SmolVLA with the training-only world-model auxiliary loss.

`SmolVLAWMPolicy` is a thin subclass of LeRobot's stock `SmolVLAPolicy`: the action
flow-matching path is inherited verbatim, and the only addition is the GaussianDream-
style reconstruction branch already used by DynamicVLA
(`policies/dynamicvla/world_model`), attached to the visual tokens SmolVLA's own
SigLIP tower + connector produce.

The branch is training-only. `use_world_model_aux` must be False at inference, which
is what keeps the branch (and its optional CUDA rasterizer) out of the graph.
"""

from policies.smolvla_wm.configuration_smolvla_wm import SmolVLAWMConfig
from policies.smolvla_wm.modeling_smolvla_wm import SmolVLAWMPolicy

__all__ = ["SmolVLAWMConfig", "SmolVLAWMPolicy"]
