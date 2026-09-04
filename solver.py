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

WHICH SEARCH TO USE DEPENDS ON PUZZLE SIZE -- measured, and it flips:

  EASY (3-8 pieces): beam_search wins decisively. It can back out of a bad
  early placement, which best_of_n cannot -- restarts just repeat the same
  mistake, which is why 200 restarts only moved solves 28% -> 35%.
    3-4 pieces  beam(w=48)   solved 97%   fill 99.7%   0.24s
    5-8 pieces  beam(w=96)   solved 67%   fill 94.7%   2.46s
    5-8 pieces  best-of-32   solved 32%   fill 88.7%   0.72s

  HARD (17-30 pieces): best_of_n wins; the beam is WORSE and slower.
    best-of-16   fill 91.3%   1.53s
    beam(w=32)   fill 87.7%   5.85s
  The beam scores PARTIAL packings by volume-minus-dead-air so far, which is
  myopic: with 25 pieces it commits to arrangements that look tidy early but
  box in later pieces, and a width-32 beam covers almost nothing of that
  space. best_of_n only ever judges FINISHED packings, so it avoids the trap.

So: beam_search for small puzzles, best_of_n for large ones.
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


def beam_search(model, env, seed: int, beam_width: int = 16, top_k: int = 6):
    """Policy-guided beam search -- the bot WITH backtracking.

    best_of_n replays the whole puzzle from scratch each time, so its
    attempts are all variations on the same idea (200 restarts moved the
    solve rate on small puzzles only 28% -> 35%). A beam keeps several
    DIFFERENT partial packings alive at once and extends the most promising
    ones, so a bad early placement gets abandoned instead of poisoning every
    later attempt.

    Scores partial packings by cumulative reward, which is exactly
    "volume placed minus dead air created" -- the thing we want to maximise.
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
