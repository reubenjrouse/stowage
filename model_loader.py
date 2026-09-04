"""Find the trained policy.

Everything that runs the agent -- the app, the solver demo, the exporters --
needs the same model, and picking it by "highest step number in checkpoints/"
was a trap: several training runs share that directory and their filenames
collide by step count, so a bigger number can belong to an older run with a
different network shape. Matching the observation space is what makes the
choice correct.

model/packing_policy.zip is the shipped model, so a fresh clone works with no
training and no 889MB of checkpoints. If it is missing (say you have just
trained something new) this falls back to scanning checkpoints/, newest first.
"""

from __future__ import annotations

import glob
import os
import re

from sb3_contrib import MaskablePPO

SHIPPED = os.path.join("model", "packing_policy.zip")


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
        glob.glob(os.path.join("checkpoints", "*.zip")),
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
