"""DOMINO eval plugin for SmolVLA + world-model loss (double-env mode).

Experiment 2: SmolVLA fine-tuned on DOMINO with the GaussianDream-style depth
reconstruction loss (`policies/smolvla_wm`).

The reconstruction branch is training-only — `smolvla_model.SmolVLA` switches it off
and drops its weights when rebuilding the config — so at inference this policy is
byte-for-byte the same graph as experiment 1. The only reason this is a separate
module is `policy_name`: it is what partitions `eval_result/`, keeping the two
experiments' metrics side by side rather than interleaved under one directory.
"""

from SmolVLA.deploy_policy import encode_obs, eval, get_model, reset_model

__all__ = ["encode_obs", "eval", "get_model", "reset_model"]
