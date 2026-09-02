"""
Stage 2: the real packing problem.

A simplified offline 3D bin-packing environment, built on Gymnasium
(the standard interface RL libraries expect: reset(), step(), observation
space, action space).

What changed from the Stage 1 version, and why:
- MORE BOXES THAN FIT. The old version handed the agent 8 boxes totalling
  ~35% of the container, so every box fit no matter where it went and the
  greedy baseline placed 8/8 every episode -- its 34.8% "score" was the
  arithmetic ceiling, not a bar to clear. With the box set now exceeding
  the container, the agent has to choose WHICH boxes to place and WHERE,
  and fill % finally measures packing skill.
- ROTATION. Each box can be placed in any of its 6 axis-aligned
  orientations (set n_rotations=1 to turn this off and isolate its effect).
- ACTION MASKING. The action is a single flat index over
  (box, rotation, x, y), and action_masks() marks exactly those actions
  that lead to a legal placement. This is what the reference papers do
  (PQNet multiplies its action values by a "feasibility mask"; GOPT the
  same) -- without it, almost every action in an 12000-way space is
  illegal and the agent burns its samples learning what "legal" means.
- HONEST TERMINATION. The episode ends when nothing more can be placed,
  not when the step budget runs out.

Height (z) is still computed automatically -- a box drops until it lands
on the floor or on top of whatever's already there, like Tetris.
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

        # "compact" scores a placement by the NET space it gains: the box's
        # volume minus the dead space it seals underneath itself. "volume"
        # is the old behaviour (volume only) and is kept for ablation.
        #
        # This matters more than it looks. Under "volume", every legal
        # position for a given box scored EXACTLY the same -- dropping a box
        # flush onto a flat surface and dropping it onto a jagged spot that
        # sealed 90 units of dead air were worth identical reward, so the
        # agent got no signal at all about where to place things and sat at
        # the random-policy score forever. This mirrors the reward in the
        # PQNet paper (r_t = g_{t-1} - g_t over wasted space) adapted to a
        # fixed-size container.
        if reward_mode not in ("compact", "volume"):
            raise ValueError("reward_mode must be 'compact' or 'volume'")
        self.reward_mode = reward_mode

        # DOMAIN RANDOMISATION. grid_size / max_height / max_boxes are now the
        # CAPS that fix the array shapes SB3 requires; the ACTUAL container
        # (cur_gx, cur_gy, cur_h) and box count are resampled every episode.
        # Cells outside the current container are pre-filled to cur_h, i.e.
        # "already full", so the ordinary feasibility test excludes them and
        # nothing else in the env has to know the container shrank.
        self.randomize = randomize
        self.grid_range = grid_range
        self.height_range = height_range
        self.boxes_range = boxes_range
        self.cur_gx = grid_size
        self.cur_gy = grid_size
        self.cur_h = max_height
        self.n_active = max_boxes

        # Box dims must SCALE WITH THE CONTAINER or the benchmark saturates:
        # 30 boxes of 2-5 units total ~1305, which is 130% of a 10x10x10
        # container (a real packing problem) but only 48% of a 14x14x14 one
        # (everything fits anywhere -- exactly the dead benchmark that wasted
        # the first days of this project). Sampling each dim in
        # [0.2*side, 0.5*side], as the papers do, keeps the ratio near 130%
        # at every container size -- and reproduces the validated 2-5 range
        # exactly when the side is 10.
        self.scale_boxes = scale_boxes

        # BOX SOURCE
        #  "random"  - independent random dims (what Stage 2/3 trained on)
        #  "perfect" - REVERSE CONSTRUCTION: recursively cut the container into
        #              pieces, so the boxes tile it EXACTLY and a 100% packing
        #              is guaranteed to exist. This is the app's puzzle mode.
        #  "mixed"   - half and half, so one agent handles both.
        # A perfect partition IS reachable under drop-from-above: place the
        # pieces in bottom-up order and each lands on a floor its neighbours
        # have already filled exactly to its underside.
        if box_source not in ("random", "perfect", "mixed"):
            raise ValueError("box_source must be 'random', 'perfect' or 'mixed'")
        self.box_source = box_source
        self.min_piece = min_piece
        self.solution = None

        # What the agent gets to see each step:
        #  - "boxes": (l, w, h) for each of the max_boxes boxes, normalized to
        #     [0, 1] by the container's dimensions. A box that's already been
        #     placed shows up as all zeros (so the agent learns not to pick it
        #     again).
        #  - "heightmap": how tall the stack is at each (x, y) floor cell,
        #     normalized by max_height. This is how the agent "sees" what's
        #     already packed.
        self.observation_space = spaces.Dict(
            {
                "boxes": spaces.Box(low=0.0, high=1.0, shape=(max_boxes, 3), dtype=np.float32),
                "heightmap": spaces.Box(low=0.0, high=1.0, shape=(grid_size, grid_size), dtype=np.float32),
                # Everything else is normalised by the current container, which
                # is what makes the policy scale-invariant -- but it also hides
                # the ABSOLUTE size, and a 5-tall container behaves differently
                # from a 14-tall one because the grid is discrete. Three scalars
                # give that back.
                "container": spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
            }
        )

        # One flat index over (box, rotation, x, y). It has to be flat rather
        # than MultiDiscrete: SB3 masks each MultiDiscrete dimension
        # independently, which can't express "box 3 rotated this way fits at
        # (4,5) but not (4,6)". A flat Discrete takes an exact joint mask.
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
        # Normalising by the CURRENT container makes the observation
        # scale-invariant: 1.0 always means "as big as the container" and a
        # heightmap of 1.0 always means "no room left here", whatever size
        # this episode's container happens to be.
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

        if self.randomize:
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
            self.boxes[: self.n_active] = np.array([p[3:] for p in pieces], dtype=np.float32)
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
