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


def evaluate_agent(model: MaskablePPO, num_episodes: int, **env_kwargs) -> tuple[float, float]:
    env = BinPackingEnv(**env_kwargs)
    fills = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=10_000 + ep)
        done = False
        info = {}
        while not done:
            action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        fills.append(info["fill_fraction"])
    return float(np.mean(fills)), float(np.std(fills))


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
):
    device = check_device()
    env_kwargs = dict(grid_size=grid_size, max_boxes=max_boxes, n_rotations=n_rotations, randomize=randomize)

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
    settings = [("fixed: 30 boxes, 10x10x10", dict(env_kwargs, randomize=False))]
    if randomize:
        settings.append(("randomised: varied container + box count", dict(env_kwargs, randomize=True)))

    for label, kw in settings:
        print("")
        print(f"Evaluating [{label}] over 200 episodes each...")
        baseline_mean, baseline_std = evaluate_baseline(num_episodes=200, **kw)
        agent_mean, agent_std = evaluate_agent(model, num_episodes=200, **kw)

        gap = agent_mean - baseline_mean
        # 200 episodes of a ~6% spread puts the standard error near 0.4pp, so
        # anything under about 1pp of separation is a tie, not a win.
        stderr = (agent_std**2 / 200 + baseline_std**2 / 200) ** 0.5
        print(f"  Greedy baseline : {baseline_mean:.1%} avg fill (+/- {baseline_std:.1%})")
        print(f"  PPO agent       : {agent_mean:.1%} avg fill (+/- {agent_std:.1%})")
        print(f"  Difference      : {gap:+.1%} (standard error {stderr:.1%})")
        if gap > 2 * stderr:
            print("  PASS: the agent beats the greedy baseline.")
        elif gap > -2 * stderr:
            print("  TIE: matches the baseline but doesn't clearly beat it.")
        else:
            print("  BEHIND: the baseline still wins here.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--max-boxes", type=int, default=30)
    parser.add_argument("--log-dir", type=str, default="./tb_logs")
    parser.add_argument("--n-envs", type=int, default=8, help="parallel environments (subprocesses) for faster data collection")
    # 0.01 is the usual default, but it assumes a handful of actions. With
    # ~9,800 legal moves per step the bonus is 0.01*ln(9800) ~= 0.09 against
    # episode returns of ~0.6 -- i.e. ~15% of the agent's income was paid for
    # staying undecided, which is part of why entropy never fell.
    parser.add_argument("--ent-coef", type=float, default=0.001, help="entropy bonus; scale it down as the action space grows")
    parser.add_argument("--n-epochs", type=int, default=5, help="PPO epochs per rollout; lower reduces over-clipping")
    parser.add_argument("--embed-dim", type=int, default=64, help="embedding width for the box and position encoders")
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
    )
