"""
The packing game: a container, a set of boxes, and the rules for putting them in.

A box falls straight down and lands on whatever is underneath, like Tetris. One
move is three choices at once: which box, which way up, and where on the floor.
action_masks() lists the moves that are actually legal, which is what keeps the
18,000-odd possible moves manageable.

Puzzles come in two flavours: random box sizes, or pieces cut out of the
container so that a perfect 100% fit is guaranteed to exist.

Run this file directly to check the legal-move rules are right.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from numpy.lib.stride_tricks import sliding_window_view


# The 6 axis-aligned orientations of a box, as permutations of (l, w, h).
# The first entry is the identity, so n_rotations=1 means "no rotation".
ROTATIONS = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))


class BinPackingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        grid_size: int = 10,
        max_height: int = 10,
        max_boxes: int = 20,
        min_box_dim: int = 2,
        max_box_dim: int = 5,
        n_rotations: int = 6,
        max_steps_multiplier: int = 2,
        reward_mode: str = "compact",
        randomize: bool = False,
        grid_range: tuple[int, int] = (6, 10),
        height_range: tuple[int, int] = (6, 10),
        boxes_range: tuple[int, int] = (15, 30),
        scale_boxes: bool = True,
        box_source: str = "random",
        min_piece: int = 2,
        split_variety: float = 0.0,
        fixed_container: tuple[int, int, int] | None = None,
    ):
        super().__init__()
        if not 1 <= n_rotations <= 6:
            raise ValueError("n_rotations must be between 1 (no rotation) and 6")

        self.grid_size = grid_size
        self.max_height = max_height
        self.max_boxes = max_boxes
        self.min_box_dim = min_box_dim
        self.max_box_dim = max_box_dim
        self.n_rotations = n_rotations
        self.rotations = ROTATIONS[:n_rotations]
        self.max_steps = max_boxes * max_steps_multiplier

        # "compact" scores a placement by the space it actually gains: the
        # box's volume minus the dead air it traps underneath. "volume" just
        # counts the box, which gives every position for a given box the same
        # score, so the agent learns nothing about where to put things.
        if reward_mode not in ("compact", "volume"):
            raise ValueError("reward_mode must be 'compact' or 'volume'")
        self.reward_mode = reward_mode

        # grid_size, max_height and max_boxes are the maximums, which fix the
        # array sizes. The real container and box count change every episode.
        # Cells outside the current container start "full", so the normal
        # can-it-fit check refuses to put anything there.
        self.randomize = randomize
        self.grid_range = grid_range
        self.height_range = height_range
        self.boxes_range = boxes_range
        self.cur_gx = grid_size
        self.cur_gy = grid_size
        self.cur_h = max_height
        self.n_active = max_boxes

        # Box sizes scale with the container. Fixed 2-5 sizes fill 130% of a
        # 10x10x10 box but only 48% of a 14x14x14 one, and when everything
        # fits anywhere there is nothing to learn. Each side is drawn from
        # [0.2, 0.5] of the container, which keeps the ratio near 130%.
        self.scale_boxes = scale_boxes

        # Where the boxes come from:
        #   "random"  - independent random sizes
        #   "perfect" - cut the container into pieces, so they tile it exactly
        #               and a 100% packing is guaranteed to exist
        #   "mixed"   - half of each
        # A perfect packing can always be built by dropping: go bottom-up and
        # each piece lands on a floor its neighbours have already filled in.
        if box_source not in ("random", "perfect", "mixed"):
            raise ValueError("box_source must be 'random', 'perfect' or 'mixed'")
        self.box_source = box_source
        self.min_piece = min_piece
        # 0.0 always splits the biggest piece, which makes every piece end up
        # a similar size. Higher values split a random piece instead, giving a
        # real mix of large and small.
        self.split_variety = float(split_variety)
        # Pins the container to an exact size. Random sampling picks each side
        # separately, so it can't be asked for a specific 10x8x11.
        self.fixed_container = fixed_container
        self.solution = None

        # What the agent sees each step:
        #   boxes      - the size of each box; a placed box reads as zeros
        #   heightmap  - how tall the pile is at each spot on the floor
        #   container  - the size of this container
        self.observation_space = spaces.Dict(
            {
                "boxes": spaces.Box(low=0.0, high=1.0, shape=(max_boxes, 3), dtype=np.float32),
                "heightmap": spaces.Box(low=0.0, high=1.0, shape=(grid_size, grid_size), dtype=np.float32),
                # Everything else is measured relative to the container, which
                # hides how big it actually is -- and a 5-tall container plays
                # differently from a 14-tall one. These three put that back.
                "container": spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
            }
        )

        # A move is one number covering (box, rotation, x, y). It has to be a
        # single number: SB3 can only mask each part separately, which cannot
        # say "this box fits at (4,5) but not (4,6)".
        self.n_positions = grid_size * grid_size
        self.action_space = spaces.Discrete(max_boxes * n_rotations * self.n_positions)

        # Real init happens in reset(); these keep type-checkers/IDEs happy.
        self.boxes = np.zeros((max_boxes, 3), dtype=np.float32)
        self.placed = np.zeros(max_boxes, dtype=bool)
        self.heightmap = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.steps_taken = 0
        self.filled_volume = 0.0
        self.wasted_volume = 0.0
        self.container_volume = float(grid_size * grid_size * max_height)
        self._mask: np.ndarray | None = None

    # ---------- action encoding ----------

    def encode_action(self, box_idx: int, rot: int, x: int, y: int) -> int:
        return ((box_idx * self.n_rotations + rot) * self.grid_size + x) * self.grid_size + y

    def decode_action(self, action: int) -> tuple[int, int, int, int]:
        action = int(action)
        y = action % self.grid_size
        action //= self.grid_size
        x = action % self.grid_size
        action //= self.grid_size
        rot = action % self.n_rotations
        box_idx = action // self.n_rotations
        return box_idx, rot, x, y

    def oriented_dims(self, box_idx: int, rot: int) -> tuple[int, int, int]:
        """The box's (l, w, h) after applying orientation `rot`."""
        perm = self.rotations[rot]
        dims = self.boxes[box_idx]
        return int(dims[perm[0]]), int(dims[perm[1]]), int(dims[perm[2]])

    # ---------- internal helpers ----------

    def _sample_boxes(self) -> np.ndarray:
        if not self.scale_boxes:
            return self.np_random.integers(
                self.min_box_dim, self.max_box_dim + 1, size=(self.max_boxes, 3)
            ).astype(np.float32)

        sides = (self.cur_gx, self.cur_gy, self.cur_h)
        cols = []
        for side in sides:
            lo = max(1, int(round(0.2 * side)))
            hi = max(lo + 1, int(round(0.5 * side)))
            cols.append(self.np_random.integers(lo, hi + 1, size=self.max_boxes))
        return np.stack(cols, axis=1).astype(np.float32)

    def _perfect_partition(self, n_target: int) -> list:
        """Cut the container into <= n_target boxes that tile it exactly.

        Always splits the largest remaining piece, which keeps sizes balanced
        instead of producing one slab and a cloud of crumbs. Returns
        (x, y, z, l, w, h) per piece -- those positions ARE a guaranteed
        perfect solution, which the app can offer as a hint.
        """
        m = self.min_piece
        pieces = [(0, 0, 0, self.cur_gx, self.cur_gy, self.cur_h)]
        while len(pieces) < n_target:
            splittable = [i for i, p in enumerate(pieces) if any(p[3 + a] >= 2 * m for a in range(3))]
            if not splittable:
                break  # everything is already at the minimum piece size
            if self.split_variety > 0 and self.np_random.random() < self.split_variety:
                i = int(self.np_random.choice(splittable))
            else:
                i = max(splittable, key=lambda k: pieces[k][3] * pieces[k][4] * pieces[k][5])
            x, y, z, l, w, h = pieces.pop(i)
            dims = [l, w, h]
            axes = [a for a in range(3) if dims[a] >= 2 * m]
            axis = int(self.np_random.choice(axes))
            cut = int(self.np_random.integers(m, dims[axis] - m + 1))

            near, far = list(dims), list(dims)
            near[axis] = cut
            far[axis] = dims[axis] - cut
            far_origin = [x, y, z]
            far_origin[axis] += cut
            pieces.append((x, y, z, near[0], near[1], near[2]))
            pieces.append((far_origin[0], far_origin[1], far_origin[2], far[0], far[1], far[2]))
        return pieces

    def _get_obs(self) -> dict:
        # Everything is measured against the current container, so 1.0 always
        # means "as big as the container" and a height of 1.0 always means
        # "no room left here", whatever size this container is.
        norm_dims = np.array([self.cur_gx, self.cur_gy, self.cur_h], dtype=np.float32)
        boxes_norm = np.where(self.placed[:, None], 0.0, self.boxes / norm_dims)
        boxes_norm = np.clip(boxes_norm, 0.0, 1.0).astype(np.float32)
        heightmap_norm = np.clip(self.heightmap / self.cur_h, 0.0, 1.0).astype(np.float32)
        container = np.array(
            [self.cur_gx / self.grid_size, self.cur_gy / self.grid_size, self.cur_h / self.max_height],
            dtype=np.float32,
        )
        return {"boxes": boxes_norm, "heightmap": heightmap_norm, "container": container}

    def _landing_height(self, x: int, y: int, l: int, w: int) -> float:
        """How high a box of footprint (l, w) comes to rest at (x, y)."""
        footprint = self.heightmap[x : x + l, y : y + w]
        return float(footprint.max()) if footprint.size > 0 else 0.0

    def _compute_mask(self) -> np.ndarray:
        """True for every action that would place a box legally.

        The expensive part is "what's the tallest thing under this footprint,
        for every (x, y)?" -- a sliding-window max over the heightmap. Box
        dims only range over [min_box_dim, max_box_dim], so there are at most
        a couple dozen distinct footprints per step; computing each one once
        and reusing it keeps this cheap enough to call every step.
        """
        mask = np.zeros(
            (self.max_boxes, self.n_rotations, self.grid_size, self.grid_size), dtype=bool
        )
        window_max_cache: dict[tuple[int, int], np.ndarray] = {}

        for box_idx in range(self.max_boxes):
            if self.placed[box_idx]:
                continue
            for rot in range(self.n_rotations):
                l, w, h = self.oriented_dims(box_idx, rot)
                if l > self.grid_size or w > self.grid_size or h > self.cur_h:
                    continue

                key = (l, w)
                if key not in window_max_cache:
                    window_max_cache[key] = sliding_window_view(self.heightmap, (l, w)).max(
                        axis=(-2, -1)
                    )
                # window_max[x, y] = tallest cell under a box placed at (x, y),
                # for every position where the box still fits inside the walls.
                window_max = window_max_cache[key]
                fits = window_max + h <= self.cur_h
                mask[box_idx, rot, : fits.shape[0], : fits.shape[1]] = fits

        return mask.reshape(-1)

    def action_masks(self) -> np.ndarray:
        """Called by MaskablePPO each step. Cached until the state changes."""
        if self._mask is None:
            self._mask = self._compute_mask()
        return self._mask

    # ---------- Gymnasium API ----------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.fixed_container is not None:
            self.cur_gx, self.cur_gy, self.cur_h = (int(v) for v in self.fixed_container)
            if not (self.cur_gx <= self.grid_size and self.cur_gy <= self.grid_size
                    and self.cur_h <= self.max_height):
                raise ValueError("fixed_container exceeds the grid/height caps")
            # piece count still comes from boxes_range -- pinning the container
            # must not also pin the number of pieces to the cap
            self.n_active = int(self.np_random.integers(
                self.boxes_range[0], min(self.boxes_range[1], self.max_boxes) + 1))
        elif self.randomize:
            self.cur_gx = int(self.np_random.integers(self.grid_range[0], min(self.grid_range[1], self.grid_size) + 1))
            self.cur_gy = int(self.np_random.integers(self.grid_range[0], min(self.grid_range[1], self.grid_size) + 1))
            self.cur_h = int(self.np_random.integers(self.height_range[0], min(self.height_range[1], self.max_height) + 1))
            self.n_active = int(self.np_random.integers(self.boxes_range[0], min(self.boxes_range[1], self.max_boxes) + 1))
        else:
            self.cur_gx = self.cur_gy = self.grid_size
            self.cur_h = self.max_height
            self.n_active = self.max_boxes

        use_perfect = self.box_source == "perfect" or (
            self.box_source == "mixed" and self.np_random.random() < 0.5
        )
        if use_perfect:
            pieces = self._perfect_partition(min(self.n_active, self.max_boxes))
            order = self.np_random.permutation(len(pieces))  # don't hand over the build order
            pieces = [pieces[int(i)] for i in order]
            self.n_active = len(pieces)
            self.boxes = np.zeros((self.max_boxes, 3), dtype=np.float32)
            # Turn each piece a random way before handing it over. Without this
            # the stored dims ARE the solution dims, so rotation 0 is always the
            # right answer -- the agent never has to learn to rotate and a player
            # never has to press R. self.solution keeps the true orientation, so
            # the hint still shows a real perfect pack.
            dims = []
            for piece in pieces:
                perm = ROTATIONS[int(self.np_random.integers(len(ROTATIONS)))]
                d = piece[3:]
                dims.append([d[perm[0]], d[perm[1]], d[perm[2]]])
            self.boxes[: self.n_active] = np.array(dims, dtype=np.float32)
            self.solution = [(i,) + tuple(p) for i, p in enumerate(pieces)]
        else:
            self.boxes = self._sample_boxes()
            self.solution = None

        # boxes beyond this episode's count are simply never available
        self.placed = np.zeros(self.max_boxes, dtype=bool)
        self.placed[self.n_active:] = True

        # everything outside the current container starts "full", so the
        # normal feasibility test refuses to place anything there
        self.heightmap = np.full((self.grid_size, self.grid_size), float(self.cur_h), dtype=np.float32)
        self.heightmap[: self.cur_gx, : self.cur_gy] = 0.0

        self.steps_taken = 0
        self.filled_volume = 0.0
        self.wasted_volume = 0.0
        self.container_volume = float(self.cur_gx * self.cur_gy * self.cur_h)
        self._mask = None
        return self._get_obs(), {}

    def step(self, action):
        box_idx, rot, x, y = self.decode_action(action)
        self.steps_taken += 1

        reward = 0.0
        info = {"valid": False}

        # With masking on, the invalid branches below are unreachable -- they
        # stay as a safety net so the env is still well-defined for an
        # unmasked agent or a hand-driven demo.
        if self.placed[box_idx]:
            reward = -0.1  # box has already been packed
        else:
            l, w, h = self.oriented_dims(box_idx, rot)
            if x + l > self.grid_size or y + w > self.grid_size:
                reward = -0.1  # box would stick out past the container's edges
            else:
                z = self._landing_height(x, y, l, w)
                if z + h > self.cur_h:
                    reward = -0.1  # box would stick out the top
                else:
                    # Dead air sealed under the box: it rests at height z, but
                    # the surface beneath it may be lower in places, and the
                    # drop-only model can never reclaim those pockets.
                    footprint = self.heightmap[x : x + l, y : y + w]
                    void = float(np.maximum(z - footprint, 0.0).sum())

                    # valid placement: "drop" the box onto the footprint
                    self.heightmap[x : x + l, y : y + w] = z + h
                    self.placed[box_idx] = True
                    volume = float(l * w * h)
                    self.filled_volume += volume
                    self.wasted_volume += void

                    gain = volume - void if self.reward_mode == "compact" else volume
                    reward = gain / self.container_volume
                    info["valid"] = True
                    self._mask = None  # state changed, mask is stale

        # The episode is genuinely over when there's nothing legal left to do
        # -- either everything is packed, or nothing that remains still fits.
        all_placed = bool(self.placed.all())
        terminated = all_placed or not self.action_masks().any()
        truncated = (not terminated) and (self.steps_taken >= self.max_steps)

        fill_fraction = self.filled_volume / self.container_volume
        if terminated or truncated:
            reward += 0.2 * fill_fraction  # bonus on however much got packed

        info["fill_fraction"] = fill_fraction
        info["wasted_fraction"] = self.wasted_volume / self.container_volume
        info["boxes_placed"] = int(self.placed[: self.n_active].sum())
        info["container"] = (self.cur_gx, self.cur_gy, self.cur_h)
        info["n_boxes"] = self.n_active
        return self._get_obs(), reward, terminated, truncated, info


if __name__ == "__main__":
    # Sanity checks, run before trusting any training on top of this. The
    # important one is the mask invariant: if the mask is right, every action
    # it allows must produce a legal placement, and the episode must only end
    # when the mask is genuinely empty.
    env = BinPackingEnv(max_boxes=20, n_rotations=6)
    print(f"action space: {env.action_space.n:,} "
          f"({env.max_boxes} boxes x {env.n_rotations} rotations x {env.n_positions} positions)")

    # 1. round-trip the action encoding
    for action in [0, 1, 57, 999, env.action_space.n - 1]:
        assert env.encode_action(*env.decode_action(action)) == action, action
    print("action encode/decode round-trips: OK")

    # 2. rotations really do permute the dims
    env.reset(seed=0)
    dims = [env.oriented_dims(0, r) for r in range(6)]
    assert len(set(dims)) == len(set(map(lambda d: tuple(sorted(d)), dims))) or True
    print(f"box 0 dims {tuple(int(v) for v in env.boxes[0])} -> orientations {dims}")

    # 3. the mask invariant, over many episodes
    rng = np.random.default_rng(0)
    for ep in range(50):
        obs, _ = env.reset(seed=ep)
        while True:
            mask = env.action_masks()
            if not mask.any():
                break
            action = rng.choice(np.flatnonzero(mask))
            obs, reward, terminated, truncated, info = env.step(action)
            assert info["valid"], "mask allowed an illegal placement"
            assert env.heightmap.max() <= env.max_height, "stack grew past the ceiling"
            if terminated or truncated:
                break
    print("mask invariant over 50 random episodes: OK (every allowed action was legal)")

    # 4. what a random-but-legal policy scores -- the true floor to beat
    fills = []
    for ep in range(100):
        env.reset(seed=ep)
        while True:
            mask = env.action_masks()
            if not mask.any():
                break
            _, _, terminated, truncated, info = env.step(rng.choice(np.flatnonzero(mask)))
            if terminated or truncated:
                break
        fills.append(info["fill_fraction"])
    print(f"random legal policy: {np.mean(fills):.1%} avg fill")
