"""
The "obvious simple strategy" baseline the trained AI needs to beat.

Greedy first-fit: go through the boxes (biggest volume first), and for each
one, try every orientation at every legal position and take whichever keeps
the stack lowest (a common, sensible, but not learned, packing rule). No AI
involved -- this is the bar the agent has to clear.

The baseline gets the same rotations the agent does (it reads
env.n_rotations), so turning rotation off for an ablation keeps the
comparison fair automatically.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from environment import BinPackingEnv


def _best_placement(env: BinPackingEnv, l: int, w: int, h: int, cache: dict):
    """Lowest legal resting height for footprint (l, w, h), or None.

    Same sliding-window-max trick the env uses for its mask: one pass gives
    the landing height at every (x, y) at once, instead of re-scanning the
    footprint position by position.
    """
    if l > env.grid_size or w > env.grid_size:
        return None

    key = (l, w)
    if key not in cache:
        cache[key] = sliding_window_view(env.heightmap, (l, w)).max(axis=(-2, -1))
    window_max = cache[key]

    feasible = window_max + h <= env.cur_h   # current container, not the cap
    if not feasible.any():
        return None

    # argmin over the feasible positions, scanning row-major -- so ties break
    # toward the lowest x, then the lowest y.
    z = np.where(feasible, window_max, np.inf)
    x, y = np.unravel_index(int(np.argmin(z)), z.shape)
    return float(z[x, y]), int(x), int(y)


def greedy_first_fit_episode(env: BinPackingEnv, seed: int) -> float:
    env.reset(seed=seed)

    # Try bigger boxes first -- a standard, simple packing heuristic.
    order = np.argsort(-env.boxes.prod(axis=1))

    for box_idx in order:
        box_idx = int(box_idx)
        cache: dict = {}  # heightmap changes after every placement

        best = None  # (z, rot, x, y) -- prefer the lowest resulting stack
        for rot in range(env.n_rotations):
            l, w, h = env.oriented_dims(box_idx, rot)
            found = _best_placement(env, l, w, h, cache)
            if found is None:
                continue
            z, x, y = found
            candidate = (z, rot, x, y)
            if best is None or candidate < best:
                best = candidate

        if best is None:
            continue  # this box doesn't fit anywhere -- same rule the AI lives with

        _, rot, x, y = best
        _, _, terminated, truncated, _ = env.step(env.encode_action(box_idx, rot, x, y))
        if terminated or truncated:
            break  # nothing legal left for any remaining box

    return env.filled_volume / env.container_volume


def evaluate_baseline(num_episodes: int = 200, **env_kwargs) -> tuple[float, float]:
    env = BinPackingEnv(**env_kwargs)
    fills = [greedy_first_fit_episode(env, seed=i) for i in range(num_episodes)]
    return float(np.mean(fills)), float(np.std(fills))


if __name__ == "__main__":
    mean_fill, std_fill = evaluate_baseline(num_episodes=200)
    print(f"Greedy first-fit baseline over 200 episodes: {mean_fill:.1%} avg fill (+/- {std_fill:.1%})")
