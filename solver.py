"""
Inference-time solving: what the app actually calls.

A single greedy pass of the policy scores ~85% on 8-12 puzzles. Sampling
several packings and keeping the best scores ~90% at 16 samples, for about
1.5 seconds of compute -- a bigger gain than any amount of extra training
bought, because the remaining loss is a GLOBAL ARRANGEMENT problem (which
box where, in what order) rather than a local placement one. The reference
papers do the same thing: sample multiple solutions, keep the best.

    greedy heuristic      83.6%
    policy, single pass   84.9%
    policy, best of 4     88.6%   0.39s
    policy, best of 16    90.3%   1.48s
    policy, best of 64    91.8%   6.00s
"""

from __future__ import annotations

import numpy as np


def rollout(model, env, seed: int, deterministic: bool):
    """One packing attempt. Returns (fill, placements).

    placements are (box, rot, x, y, z, l, w, h) in the order they were
    placed -- everything the app needs to animate the bot packing.
    """
    obs, _ = env.reset(seed=seed)
    placements, done = [], False
    while not done:
        action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=deterministic)
        box, rot, x, y = env.decode_action(int(action))
        l, w, h = env.oriented_dims(box, rot)
        z = env._landing_height(x, y, l, w)  # record where it lands, before it does
        placements.append((box, rot, x, y, int(z), l, w, h))
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return info["fill_fraction"], placements


def solve(model, env, seed: int, n_samples: int = 16):
    """Best of n_samples stochastic packings, plus one greedy pass.

    The greedy pass is included because it is occasionally better than every
    sample, and it costs one extra rollout.
    """
    best_fill, best_placements = rollout(model, env, seed, deterministic=True)
    for _ in range(n_samples):
        fill, placements = rollout(model, env, seed, deterministic=False)
        if fill > best_fill:
            best_fill, best_placements = fill, placements
    return best_fill, best_placements


if __name__ == "__main__":
    import glob
    import re

    from sb3_contrib import MaskablePPO

    from environment import BinPackingEnv

    # pick the newest checkpoint matching the current env, not the highest
    # step number -- different runs share the checkpoints/ directory and
    # their filenames collide by step count.
    env = BinPackingEnv(grid_size=12, max_height=12, max_boxes=30, n_rotations=6, randomize=True,
                        grid_range=(8, 12), height_range=(8, 12), boxes_range=(8, 30), box_source="perfect")
    best = None
    for f in glob.glob("checkpoints/*.zip"):
        try:
            m = MaskablePPO.load(f, device="cpu")
            if m.observation_space == env.observation_space:
                step = int(re.search(r"(\d+)_steps", f).group(1))
                if best is None or step > best[0]:
                    best = (step, f, m)
        except Exception:
            pass
    step, path, model = best
    print(f"model: {path} ({step:,} steps)\n")

    for seed in range(3):
        fill, placements = solve(model, env, seed, n_samples=16)
        env.reset(seed=seed)
        print(f"puzzle {seed}: container {env.cur_gx}x{env.cur_gy}x{env.cur_h}, "
              f"{env.n_active} pieces -> filled {fill:.1%} with {len(placements)} placed")
        for p in placements[:3]:
            print(f"    box {p[0]:2d} rot {p[1]} at (x={p[2]}, y={p[3]}, z={p[4]}) size {p[5]}x{p[6]}x{p[7]}")
        print("    ...")
