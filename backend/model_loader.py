"""
Loads the trained policy.

It picks the model by matching the observation space, not by the highest step
number in the filename. Several training runs write to checkpoints/ and reuse
the same names, so the biggest number can belong to an old run with a network
of a different shape.

model/packing_policy.zip is the model that ships with the repo, so a fresh
clone can run the app without training anything.
"""

from __future__ import annotations

import glob
import os
import re

from sb3_contrib import MaskablePPO

from paths import CHECKPOINTS, MODEL_DIR

SHIPPED = str(MODEL_DIR / "packing_policy.zip")


def load_policy(reference_env, verbose: bool = True):
    """Return a MaskablePPO whose observation space matches reference_env."""
    if os.path.exists(SHIPPED):
        model = MaskablePPO.load(SHIPPED, device="cpu")
        if model.observation_space == reference_env.observation_space:
            if verbose:
                print(f"model: {SHIPPED} ({model.num_timesteps:,} steps)")
            return model
        if verbose:
            print(f"{SHIPPED} does not match this environment; scanning checkpoints/")

    files = sorted(
        glob.glob(str(CHECKPOINTS / "*.zip")),
        key=lambda f: int(re.search(r"(\d+)_steps", f).group(1)),
        reverse=True,
    )
    for f in files:
        try:
            model = MaskablePPO.load(f, device="cpu")
        except Exception:
            continue
        if model.observation_space == reference_env.observation_space:
            if verbose:
                print(f"model: {f} ({model.num_timesteps:,} steps)")
            return model

    raise RuntimeError(
        "No policy matches this environment. Train one with "
        "`python train.py --randomize --box-source perfect --grid-min 8 --grid-size 12`, "
        "then copy the best checkpoint to " + SHIPPED
    )
