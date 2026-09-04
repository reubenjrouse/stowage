"""
Runs the bot on a few puzzles and writes the results to packing_runs.json,
which viewer.html replays as a standalone page with no server needed.
"""

from __future__ import annotations

import json
import re

import numpy as np
from sb3_contrib import MaskablePPO

from environment import BinPackingEnv
from model_loader import load_policy
from solver import beam_search, rollout


def best_of_n(model, env, seed, n):
    fill, placements = rollout(model, env, seed, deterministic=True)
    for _ in range(n):
        f, p = rollout(model, env, seed, deterministic=False)
        if f > fill:
            fill, placements = f, p
    return fill, placements


def main():
    base = dict(grid_size=12, max_height=12, max_boxes=30, n_rotations=6, randomize=True,
                grid_range=(8, 12), height_range=(8, 12), box_source="perfect",
                split_variety=0.7)   # a real mix of big and small pieces
    probe = BinPackingEnv(**base, boxes_range=(5, 8))
    model = load_policy(probe)
    step = model.num_timesteps

    # a spread of difficulties, each solved with whichever search suits it
    jobs = [
        ("easy",   (4, 5),   "beam",   dict(beam_width=48, top_k=8)),
        ("medium", (8, 10),  "beam",   dict(beam_width=96, top_k=8)),
        ("hard",   (20, 26), "sample", dict(n=16)),
    ]

    out = []
    for label, (lo, hi), method, kw in jobs:
        env = BinPackingEnv(**base, boxes_range=(lo, hi))
        # scan seeds for a representative puzzle rather than cherry-picking a fluke
        for seed in range(40):
            env.reset(seed=seed)
            if method == "beam":
                fill, placements = beam_search(model, env, seed, **kw)
            else:
                fill, placements = best_of_n(model, env, seed, kw["n"])
            if len(placements) >= lo - 1:
                break

        env.reset(seed=seed)  # replay to recover this puzzle's metadata
        solution = [dict(box=int(b), x=int(x), y=int(y), z=int(z), l=int(l), w=int(w), h=int(h))
                    for (b, x, y, z, l, w, h) in env.solution]
        out.append(dict(
            label=label, seed=seed, method=method,
            container=dict(x=env.cur_gx, y=env.cur_gy, z=env.cur_h),
            n_pieces=int(env.n_active),
            fill=round(float(fill), 4),
            solved=bool(fill > 0.9999),
            placements=[dict(box=int(b), rot=int(r), x=int(x), y=int(y), z=int(z),
                             l=int(l), w=int(w), h=int(h))
                        for (b, r, x, y, z, l, w, h) in placements],
            solution=solution,
        ))
        print(f"  {label:6s} seed {seed:2d}  {env.cur_gx}x{env.cur_gy}x{env.cur_h}  "
              f"{env.n_active} pieces -> {fill:.1%} ({len(placements)} placed)")

    with open("packing_runs.json", "w") as fh:
        json.dump(dict(model_steps=step, runs=out), fh)
    print(f"\nwrote packing_runs.json ({sum(len(r['placements']) for r in out)} placements)")


if __name__ == "__main__":
    main()
