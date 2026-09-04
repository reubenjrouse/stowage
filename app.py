"""Backend for the packing app.

Two endpoints. /api/puzzle builds a puzzle to the player's chosen container
and piece count -- instant, no model involved -- so Play mode starts without
waiting. /api/solve runs the trained policy on that same puzzle, which is
the slow part, so Watch and Compete fetch it separately and can show the
puzzle while the bot thinks.

Search method follows what we measured: beam search for small puzzles (it
can abandon a bad early placement), sampling for large ones (scoring
half-finished packings is myopic and the beam loses to plain sampling).

    python app.py      then open http://127.0.0.1:8000
"""

from __future__ import annotations

import random
import re
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sb3_contrib import MaskablePPO

from environment import BinPackingEnv
from model_loader import load_policy
from solver import beam_search, rollout

GRID_CAP, BOX_CAP = 12, 30
MIN_SIDE = 6          # below the trained 8-12 the bot is weak; 6 is the floor we allow
TRAINED_MIN = 8

app = FastAPI()
MODEL = None


def make_env(gx: int, gy: int, gz: int, n: int) -> BinPackingEnv:
    return BinPackingEnv(
        grid_size=GRID_CAP, max_height=GRID_CAP, max_boxes=BOX_CAP, n_rotations=6,
        randomize=True, box_source="perfect", split_variety=0.7,
        boxes_range=(n, n), fixed_container=(gx, gy, gz),
    )


class Spec(BaseModel):
    gx: int = 10
    gy: int = 10
    gz: int = 10
    n: int = 8
    seed: int | None = None


def validate(s: Spec) -> int:
    for v, name in ((s.gx, "width"), (s.gy, "depth"), (s.gz, "height")):
        if not MIN_SIDE <= v <= GRID_CAP:
            raise HTTPException(400, f"{name} must be between {MIN_SIDE} and {GRID_CAP}")
    if not 3 <= s.n <= BOX_CAP:
        raise HTTPException(400, f"pieces must be between 3 and {BOX_CAP}")
    return s.seed if s.seed is not None else random.randrange(1 << 30)


@app.get("/api/info")
def info():
    return dict(model_steps=app.state.model_steps, grid_cap=GRID_CAP,
                box_cap=BOX_CAP, min_side=MIN_SIDE, trained_min=TRAINED_MIN)


@app.get("/")
def index():
    return FileResponse("app.html")


@app.post("/api/puzzle")
def puzzle(spec: Spec):
    """Generate the puzzle only. No model, so this returns immediately."""
    seed = validate(spec)
    env = make_env(spec.gx, spec.gy, spec.gz, spec.n)
    env.reset(seed=seed)
    return dict(
        seed=seed,
        container=[env.cur_gx, env.cur_gy, env.cur_h],
        # dims only, in the shuffled order the player sees them
        pieces=[[int(l), int(w), int(h)] for l, w, h in env.boxes[: env.n_active]],
        # the arrangement the puzzle was cut from -- a guaranteed 100% answer.
        # leading index says which piece, so the UI can retire it from the tray
        solution=[[int(b), int(x), int(y), int(z), int(l), int(w), int(h)]
                  for (b, x, y, z, l, w, h) in env.solution],
        below_trained_range=min(spec.gx, spec.gy, spec.gz) < TRAINED_MIN,
    )


@app.post("/api/solve")
def solve(spec: Spec):
    """Run the policy on the puzzle with this seed."""
    if spec.seed is None:
        raise HTTPException(400, "solve needs the seed returned by /api/puzzle")
    validate(spec)
    env = make_env(spec.gx, spec.gy, spec.gz, spec.n)
    t0 = time.time()

    if spec.n <= 12:
        method = "beam"
        fill, placements = beam_search(MODEL, env, spec.seed, beam_width=48, top_k=7)
    else:
        method = "best-of-16"
        fill, placements = rollout(MODEL, env, spec.seed, deterministic=True)
        for _ in range(16):
            f, p = rollout(MODEL, env, spec.seed, deterministic=False)
            if f > fill:
                fill, placements = f, p

    return dict(
        seed=spec.seed, method=method,
        fill=round(float(fill), 4), solved=bool(fill > 0.9999),
        seconds=round(time.time() - t0, 2),
        placements=[[int(b), int(x), int(y), int(z), int(l), int(w), int(h)]
                    for (b, _r, x, y, z, l, w, h) in placements],
    )


if __name__ == "__main__":
    import uvicorn
    MODEL = load_policy(make_env(10, 10, 10, 8))
    app.state.model_steps = MODEL.num_timesteps
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
