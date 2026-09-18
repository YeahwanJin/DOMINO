# -*- coding: utf-8 -*-
"""Bridging DOMINO's observation dict to LeRobot feature names.

The simulator hands out `head_camera` / `left_camera` / `right_camera`
(`envs/_base_task.py:get_obs`), while LeRobot datasets built from DOMINO episodes
may name the same three streams `cam_high` / `cam_left_wrist` / `cam_right_wrist`
depending on how `scripts/robotwin2lerobot` was invoked. Eval has to work either
way, so lookups go through the alias table below rather than assuming one spelling.
"""

CAMERA_ALIASES = {
    "cam_high": "head_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
    "head_camera": "cam_high",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}


def frame_camera(frame: dict, camera: str):
    """RGB array for `camera` in an encoded observation, trying its alias too."""
    if camera in frame:
        return frame[camera]

    alias = CAMERA_ALIASES.get(camera)
    if alias is not None and alias in frame:
        return frame[alias]

    raise KeyError(
        f"Camera '{camera}' (alias '{alias}') is missing from the observation; "
        f"got {sorted(frame)}."
    )
