"""
Draws the ablation figures from the tensorboard logs.

An ablation run is the same training with exactly ONE thing changed, so the
two curves on the same axes are the evidence for what that thing was doing.

    python plot_ablations.py --runs "full=tb_logs/reference_1" \
                                    "flat head=tb_logs/abl_flat_1" \
                                    "volume reward=tb_logs/abl_volume_1"

Curves are plotted against TIMESTEPS, not wall-clock, because the arms run at
different speeds and equal compute is not the comparison being made.
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator as EA

from paths import ROOT

# tag -> (panel title, y label, whether to negate)
# SB3 logs entropy_loss = -entropy, so flip it back to read as entropy.
PANELS = [
    ("rollout/avg_fill_fraction", "How full the container ends up", "fill fraction", False),
    ("train/entropy_loss", "Policy entropy (has it formed an opinion?)", "entropy (nats)", True),
    ("rollout/solve_rate", "Exact solves (the puzzle has a known 100%)", "fraction solved", False),
]


def load(run_dir: str) -> dict:
    """Every scalar in a run, as {tag: (steps, values)}."""
    files = sorted(glob.glob(os.path.join(run_dir, "events*")))
    if not files:
        raise SystemExit(f"no tensorboard event file in {run_dir}")
    acc = EA.EventAccumulator(files[-1], size_guidance={"scalars": 0})
    acc.Reload()
    out = {}
    for tag in acc.Tags()["scalars"]:
        pts = acc.Scalars(tag)
        out[tag] = (np.array([p.step for p in pts]), np.array([p.value for p in pts]))
    return out


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    """Running mean, with the leading edge averaged over what exists so far."""
    if window <= 1 or len(y) < 2:
        return y
    csum = np.cumsum(np.insert(y.astype(float), 0, 0.0))
    n = np.minimum(np.arange(1, len(y) + 1), window)
    lo = np.maximum(np.arange(1, len(y) + 1) - window, 0)
    return (csum[1:] - csum[lo]) / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help='one or more "label=path/to/tb_run_dir"; the first is the reference')
    ap.add_argument("--out", default=str(ROOT / "assets" / "ablations.png"))
    ap.add_argument("--smooth", type=int, default=5, help="running-mean window, in logged points")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="truncate every curve here, so arms of different lengths compare fairly")
    args = ap.parse_args()

    runs = []
    for spec in args.runs:
        if "=" not in spec:
            raise SystemExit(f'expected "label=path", got {spec!r}')
        label, path = spec.split("=", 1)
        runs.append((label, load(path)))

    fig, axes = plt.subplots(1, len(PANELS), figsize=(5.2 * len(PANELS), 4.2))
    axes = np.atleast_1d(axes)

    for ax, (tag, title, ylabel, negate) in zip(axes, PANELS):
        for i, (label, scalars) in enumerate(runs):
            if tag not in scalars:
                continue
            steps, vals = scalars[tag]
            if negate:
                vals = -vals
            if args.max_steps:
                keep = steps <= args.max_steps
                steps, vals = steps[keep], vals[keep]
            # the reference arm is drawn solid and heavier; ablations dashed
            ax.plot(steps, smooth(vals, args.smooth), label=label,
                    linewidth=2.2 if i == 0 else 1.6,
                    linestyle="-" if i == 0 else "--")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("environment steps")
        # raw step counts overlap badly on a shared axis; 200k reads better
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _p: f"{v/1000:.0f}k" if v else "0"))
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6))
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")

    # the same comparison as a number, for the blog caption
    print("\nfinal values (mean of the last 5 logged points):")
    for tag, title, _ylabel, negate in PANELS:
        print(f"  {title}")
        for label, scalars in runs:
            if tag not in scalars:
                print(f"    {label:<22} --")
                continue
            steps, vals = scalars[tag]
            if negate:
                vals = -vals
            if args.max_steps:
                keep = steps <= args.max_steps
                steps, vals = steps[keep], vals[keep]
            print(f"    {label:<22} {vals[-5:].mean():.4f}   (at {steps[-1]:,} steps)")


if __name__ == "__main__":
    main()
