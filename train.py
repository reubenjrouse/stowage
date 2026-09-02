"""
Stage 2: hook up the learner.

Trains a maskable PPO agent (via sb3-contrib -- a proven, off-the-shelf
algorithm, not hand-rolled) on the BinPackingEnv, then compares it against
the greedy baseline at the end.

Why MaskablePPO rather than plain PPO: the action space is
(box x rotation x position), 18,000 entries at the default settings, and at
any given step the overwhelming majority are illegal. The env hands the
policy an exact feasibility mask so it only ever samples legal placements
and spends its samples on packing well instead of on rediscovering the
rules. This is what the reference papers do (PQNet multiplies its action
values by a feasibility mask; GOPT the same).

Usage:
    python train.py                        # default: 1M timesteps, 30 boxes, rotation on
    python train.py --timesteps 2000000    # train longer
    python train.py --n-rotations 1        # ablation: rotation off, to isolate its effect
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from baseline import evaluate_baseline
from environment import BinPackingEnv
from policy import PackingPolicy


class FillFractionCallback(BaseCallback):
    """Logs the agent's average fill % (over the last 50 finished episodes)
    to TensorBoard, so you can watch it trend up during training."""

    def __init__(self, log_every: int = 500, verbose: int = 0):
        super().__init__(verbose)
        self.log_every = log_every
        self.episode_fills: list[float] = []
        self.episode_placed: list[int] = []

    def _on_step(self) -> bool:
        for done, info in zip(self.locals.get("dones", []), self.locals.get("infos", [])):
            if done and "fill_fraction" in info:
                self.episode_fills.append(info["fill_fraction"])
                self.episode_placed.append(info.get("boxes_placed", 0))

        if self.episode_fills and self.n_calls % self.log_every == 0:
            self.logger.record("rollout/avg_fill_fraction", float(np.mean(self.episode_fills[-50:])))
            self.logger.record("rollout/avg_boxes_placed", float(np.mean(self.episode_placed[-50:])))

        return True


def check_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(
            "  WARNING: CUDA not detected -- training on CPU. If you expected "
            "your GPU to be used, check that torch was installed with CUDA support."
        )
    return device


def paired_evaluate(model, num_episodes: int, **env_kwargs):
    """Score agent and greedy on the SAME seeds, i.e. the SAME box sets.

    The earlier version evaluated them on different seed ranges, which is
    valid but noisy and invites the worry that one range was easier. Paired
    differencing cancels the episode-to-episode variation entirely.
    """
    from baseline import greedy_first_fit_episode

    genv = BinPackingEnv(**env_kwargs)
    aenv = BinPackingEnv(**env_kwargs)
    g, a, wins = [], [], 0
    for ep in range(num_episodes):
        g.append(greedy_first_fit_episode(genv, seed=ep))

        obs, _ = aenv.reset(seed=ep)
        done, info = False, {}
        while not done:
            action, _ = model.predict(obs, action_masks=aenv.action_masks(), deterministic=True)
            obs, _, terminated, truncated, info = aenv.step(action)
            done = terminated or truncated
        a.append(info["fill_fraction"])
        if a[-1] > g[-1]:
            wins += 1

    g, a = np.array(g), np.array(a)
    diff = a - g
    stderr = diff.std(ddof=1) / np.sqrt(num_episodes)
    return dict(greedy=g.mean(), agent=a.mean(), diff=diff.mean(),
                stderr=stderr, t=diff.mean() / stderr if stderr else 0.0,
                win_rate=wins / num_episodes)



def main(
    total_timesteps: int,
    grid_size: int,
    max_boxes: int,
    log_dir: str,
    n_envs: int,
    ent_coef: float,
    n_epochs: int,
    embed_dim: int,
    n_rotations: int,
    target_kl: float,
    checkpoint_freq: int,
    resume: str,
    randomize: bool,
    box_source: str,
    grid_range: tuple,
    boxes_range: tuple,
    lr: float,
):
    device = check_device()
    env_kwargs = dict(grid_size=grid_size, max_height=grid_size, max_boxes=max_boxes,
                      n_rotations=n_rotations, randomize=randomize, box_source=box_source,
                      grid_range=grid_range, height_range=grid_range, boxes_range=boxes_range)

    # Profiling says ~95% of the per-step cost is env work (stepping plus
    # building the feasibility mask), which is pure Python/numpy and can't be
    # moved to the GPU. DummyVecEnv would run all n_envs of it sequentially in
    # this process, so spread it across cores instead -- this is the single
    # biggest wall-clock win available, bigger than the GPU.
    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    vec_env = make_vec_env(lambda: BinPackingEnv(**env_kwargs), n_envs=n_envs, vec_env_cls=vec_cls)

    if resume:
        print(f"Resuming from {resume}")
        model = MaskablePPO.load(resume, env=vec_env, device=device, tensorboard_log=log_dir)
    else:
        model = MaskablePPO(
            PackingPolicy,  # dot-product policy: box embeddings x position embeddings
            vec_env,
            verbose=1,
            device=device,
            tensorboard_log=log_dir,
                learning_rate=lambda progress: lr * progress,  # linear decay to 0
        ent_coef=ent_coef,
            n_epochs=n_epochs,
            target_kl=target_kl,  # early-stop an update once it leaves the trust region
            policy_kwargs=dict(n_rotations=n_rotations, embed_dim=embed_dim),
        )

    n_actions = max_boxes * n_rotations * grid_size * grid_size
    print(f"\n{max_boxes} boxes, {n_rotations} orientations, {grid_size}x{grid_size} grid "
          f"-> {n_actions:,} actions (masked to the legal ones each step)")
    print(f"Training for {total_timesteps:,} timesteps...")
    print(f"Watch progress live with: tensorboard --logdir {log_dir}\n")

    # Checkpoint often. A 540k-step run that had already passed the greedy
    # baseline was lost outright when the laptop powered off mid-run, because
    # the only save happened after learn() returned. save_freq counts steps
    # PER ENV, so divide by n_envs to get an interval in total timesteps.
    callback = CallbackList([
        FillFractionCallback(),
        CheckpointCallback(
            save_freq=max(1, checkpoint_freq // n_envs),
            save_path="checkpoints",
            name_prefix="ppo_bin_packing",
        ),
    ])

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True,
            reset_num_timesteps=not resume,
        )
    finally:
        # runs on Ctrl+C and on crashes too, not only clean completion
        model.save("ppo_bin_packing_stage2")
        print("saved -> ppo_bin_packing_stage2.zip")

    # Always score on the FIXED canonical setting so the headline number stays
    # comparable with every earlier run, and additionally on randomised
    # containers when we trained that way, to show it generalises.
    settings = [("puzzle: perfect-fit pieces", dict(env_kwargs, box_source="perfect")),
                ("random boxes", dict(env_kwargs, box_source="random"))]

    for label, kw in settings:
        print("")
        print(f"[{label}] paired over 200 episodes (agent and greedy on identical box sets)")
        r = paired_evaluate(model, 200, **kw)
        print(f"  Greedy  : {r['greedy']:.1%}")
        print(f"  Agent   : {r['agent']:.1%}")
        print(f"  Diff    : {r['diff']:+.2%}  (se {r['stderr']:.2%}, t={r['t']:.1f})")
        print(f"  Agent wins {r['win_rate']:.0%} of episodes")
        if r['t'] > 2:
            print("  PASS: the agent beats greedy.")
        elif r['t'] > -2:
            print("  TIE.")
        else:
            print("  BEHIND.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--grid-size", type=int, default=14, help="container cap; also the height cap")
    parser.add_argument("--max-boxes", type=int, default=30)
    parser.add_argument("--log-dir", type=str, default="./tb_logs")
    parser.add_argument("--n-envs", type=int, default=8, help="parallel environments (subprocesses) for faster data collection")
    # 0.01 is the usual default, but it assumes a handful of actions. With
    # ~9,800 legal moves per step the bonus is 0.01*ln(9800) ~= 0.09 against
    # episode returns of ~0.6 -- i.e. ~15% of the agent's income was paid for
    # staying undecided, which is part of why entropy never fell.
    parser.add_argument("--ent-coef", type=float, default=0.001, help="entropy bonus; scale it down as the action space grows")
    parser.add_argument("--n-epochs", type=int, default=5, help="PPO epochs per rollout; lower reduces over-clipping")
    parser.add_argument("--embed-dim", type=int, default=128, help="embedding width for the box and position encoders")
    parser.add_argument("--box-source", type=str, default="mixed", choices=["random", "perfect", "mixed"],
                        help="perfect = reverse-construction puzzles (the app mode); mixed handles both")
    parser.add_argument("--grid-min", type=int, default=5, help="smallest container side to train on")
    parser.add_argument("--boxes-min", type=int, default=8, help="fewest boxes per puzzle")
    parser.add_argument("--lr", type=float, default=3e-4, help="initial learning rate (decays linearly to 0)")
    parser.add_argument("--randomize", action="store_true", help="randomise container size and box count each episode (Stage 3)")
    parser.add_argument("--checkpoint-freq", type=int, default=25_000, help="save a checkpoint every N timesteps")
    parser.add_argument("--resume", type=str, default=None, help="path to a checkpoint .zip to continue from")
    parser.add_argument("--n-rotations", type=int, default=6, help="6 = full rotation, 1 = rotation off (ablation)")
    # Off by default. At 0.03 this fired on every single rollout ("Early
    # stopping at step 1"), cutting ~1280 gradient steps down to 2 and pinning
    # the policy at its random init -- 230k steps of training that scored
    # exactly the 52% random-legal floor. clip_range already bounds the update;
    # this second brake is what strangled it.
    parser.add_argument("--target-kl", type=float, default=None, help="optional extra brake on update size; off by default")
    args = parser.parse_args()
    main(
        args.timesteps,
        args.grid_size,
        args.max_boxes,
        args.log_dir,
        args.n_envs,
        args.ent_coef,
        args.n_epochs,
        args.embed_dim,
        args.n_rotations,
        args.target_kl,
        args.checkpoint_freq,
        args.resume,
        args.randomize,
        args.box_source,
        (args.grid_min, args.grid_size),
        (args.boxes_min, args.max_boxes),
        args.lr,
    )
