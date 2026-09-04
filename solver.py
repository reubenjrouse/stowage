"""
Playing a puzzle with a trained policy.

Three ways to do it:
  rollout      one pass, taking the best-looking move each time
  best_of_n    play the whole puzzle several times over, keep the best result
  beam_search  keep several part-finished packings going, drop the poor ones

Which is better depends on the size of the puzzle:
  5-8 pieces     beam solves 60% of them, best-of-32 only 32%
  17-30 pieces   best-of-16 fills 91.3%, beam only 87.7%

Beam search wins when it can tell a good half-finished packing from a bad one.
On big puzzles it can't, so plain sampling does better.
"""

from __future__ import annotations

import numpy as np


def rollout(model, env, seed: int, deterministic: bool):
    """One attempt at a puzzle. Returns how full it got, and the moves.

    Each move is (box, rotation, x, y, z, length, width, height) in the order
    it was played, which is what the app needs to animate the bot.
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
    """Play the puzzle n_samples times and keep the best result.

    The straight greedy pass is thrown in too, since it sometimes beats all
    the random ones and only costs one extra attempt.
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
    from model_loader import load_policy

    # pick the newest checkpoint matching the current env, not the highest
    # step number -- different runs share the checkpoints/ directory and
    # their filenames collide by step count.
    env = BinPackingEnv(grid_size=12, max_height=12, max_boxes=30, n_rotations=6, randomize=True,
                        grid_range=(8, 12), height_range=(8, 12), boxes_range=(8, 30), box_source="perfect")
    model = load_policy(env)

    for seed in range(3):
        fill, placements = solve(model, env, seed, n_samples=16)
        env.reset(seed=seed)
        print(f"puzzle {seed}: container {env.cur_gx}x{env.cur_gy}x{env.cur_h}, "
              f"{env.n_active} pieces -> filled {fill:.1%} with {len(placements)} placed")
        for p in placements[:3]:
            print(f"    box {p[0]:2d} rot {p[1]} at (x={p[2]}, y={p[3]}, z={p[4]}) size {p[5]}x{p[6]}x{p[7]}")
        print("    ...")


def beam_search(model, env, seed: int, beam_width: int = 16, top_k: int = 6):
    """Beam search: the bot, but able to change its mind.

    best_of_n starts from scratch every time, so all its attempts end up
    being variations on the same idea -- 200 restarts only moved the solve
    rate from 28% to 35%. A beam keeps several different part-finished
    packings going at once and extends the promising ones, so a bad early
    move gets dropped instead of ruining every later attempt.

    Packings are ranked by space gained so far, minus dead air created.
    """
    import copy

    import torch

    def policy_top_k(e, mask):
        obs_t, _ = model.policy.obs_to_tensor(e._get_obs())
        dist = model.policy.get_distribution(obs_t, action_masks=torch.as_tensor(mask).unsqueeze(0))
        probs = dist.distribution.probs[0].detach().cpu().numpy()
        return np.argsort(-probs)[:top_k]

    env.reset(seed=seed)
    beams = [(0.0, copy.deepcopy(env), [])]
    finished = []

    while beams:
        candidates = []
        for score, e, placed in beams:
            mask = e.action_masks()
            if not mask.any():
                finished.append((score, e, placed))
                continue
            for action in policy_top_k(e, mask):
                if not mask[action]:
                    continue
                e2 = copy.deepcopy(e)
                box, rot, x, y = e2.decode_action(int(action))
                l, w, h = e2.oriented_dims(box, rot)
                z = e2._landing_height(x, y, l, w)
                _, reward, terminated, truncated, info = e2.step(int(action))
                entry = (score + reward, e2, placed + [(box, rot, x, y, int(z), l, w, h)])
                (finished if (terminated or truncated) else candidates).append(entry)

        # keep the most promising partial packings, and stop early on a
        # perfect fill since nothing can beat it
        candidates.sort(key=lambda c: -c[0])
        beams = candidates[:beam_width]
        for sc, e, pl in finished:
            if e.filled_volume / e.container_volume > 0.9999:
                return 1.0, pl

    best = max(finished, key=lambda c: c[1].filled_volume / c[1].container_volume)
    return best[1].filled_volume / best[1].container_volume, best[2]
