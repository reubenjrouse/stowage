"""
Trains the packing policy with MaskablePPO.

    python train.py --randomize --box-source perfect --grid-min 8 --grid-size 12

Checkpoints are written to checkpoints/ every 25k steps, and the model is saved
even if you stop it with Ctrl+C. At the end it plays the trained agent and the
greedy baseline on the same puzzles and reports the difference.
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
from paths import CHECKPOINTS, TB_LOGS
from policy import PackingPolicy


class FillFractionCallback(BaseCallback):
    """Logs fill %, BUCKETED BY CONTAINER SIZE.

    A single average is close to useless once containers are randomised:
    the agent scores ~100% on a 5x5x5 and ~84% on a 14x14x14, so a running
    mean over 50 episodes mostly measures which containers happened to be
    drawn, not whether the policy improved. Three buckets are each
    comparable with themselves over time, so progress is actually visible.
    Also logs the exact-solve rate, which is the number that matters for a
    puzzle with a known 100% optimum.
    """

    def __init__(self, grid_cap: int = 14, log_every: int = 500, window: int = 200, verbose: int = 0):
        super().__init__(verbose)
        self.log_every = log_every
        self.window = window
        # thresholds relative to the biggest possible container, so the three
        # buckets stay populated whatever range we train on
        self.max_vol = grid_cap ** 3
        self.buckets: dict = {"small": [], "mid": [], "large": []}
        self.solved: list = []
        self.placed: list = []

    def _on_step(self) -> bool:
        for done, info in zip(self.locals.get("dones", []), self.locals.get("infos", [])):
            if not (done and "fill_fraction" in info):
                continue
            fill = info["fill_fraction"]
            gx, gy, h = info.get("container", (10, 10, 10))
            frac = (gx * gy * h) / self.max_vol
            key = "small" if frac < 0.4 else ("mid" if frac < 0.7 else "large")
            self.buckets[key].append(fill)
            self.solved.append(1.0 if fill > 0.9999 else 0.0)
            self.placed.append(info.get("boxes_placed", 0))

        if self.n_calls % self.log_every == 0:
            for key, vals in self.buckets.items():
                if vals:
                    self.logger.record(f"rollout/fill_{key}", float(np.mean(vals[-self.window:])))
            if self.solved:
                self.logger.record("rollout/solve_rate", float(np.mean(self.solved[-self.window:])))
                self.logger.record("rollout/avg_boxes_placed", float(np.mean(self.placed[-self.window:])))
                allf = [v for vals in self.buckets.values() for v in vals[-self.window:]]
                self.logger.record("rollout/avg_fill_fraction", float(np.mean(allf)))
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
    reward_mode: str,
    policy: str,
    split_variety: float,
    run_name: str,
):
    device = check_device()
    env_kwargs = dict(grid_size=grid_size, max_height=grid_size, max_boxes=max_boxes,
                      n_rotations=n_rotations, randomize=randomize, box_source=box_source,
                      grid_range=grid_range, height_range=grid_range, boxes_range=boxes_range,
                      reward_mode=reward_mode, split_variety=split_variety)

    # Roughly 95% of the time per step goes on the environment, not the
    # network, and that part cannot use the GPU. Running the environments in
    # separate processes is a bigger speed-up than the GPU is.
    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    vec_env = make_vec_env(lambda: BinPackingEnv(**env_kwargs), n_envs=n_envs, vec_env_cls=vec_cls)

    if resume:
        print(f"Resuming from {resume}")
        model = MaskablePPO.load(resume, env=vec_env, device=device, tensorboard_log=log_dir)
    else:
        # "factored" is policy.py: box embeddings x position embeddings, scored
        # by dot product. "flat" is the ablation arm -- SB3's stock MLP ending
        # in one Linear(64 -> n_actions) layer, which has to learn every action
        # score as an independent number with nothing shared between them.
        if policy == "flat":
            policy_cls, policy_kwargs = "MultiInputPolicy", {}
        else:
            policy_cls = PackingPolicy
            policy_kwargs = dict(n_rotations=n_rotations, embed_dim=embed_dim)

        model = MaskablePPO(
            policy_cls,
            vec_env,
            verbose=1,
            device=device,
            tensorboard_log=log_dir,
                learning_rate=lambda progress: lr * progress,  # linear decay to 0
        ent_coef=ent_coef,
            n_epochs=n_epochs,
            target_kl=target_kl,  # early-stop an update once it leaves the trust region
            policy_kwargs=policy_kwargs,
        )

    n_actions = max_boxes * n_rotations * grid_size * grid_size
    print(f"\n{max_boxes} boxes, {n_rotations} orientations, {grid_size}x{grid_size} grid "
          f"-> {n_actions:,} actions (masked to the legal ones each step)")
    print(f"run '{run_name}': policy={policy}, reward={reward_mode}, "
          f"rotations={n_rotations}, split_variety={split_variety}")
    print(f"Training for {total_timesteps:,} timesteps...")
    print(f"Watch progress live with: tensorboard --logdir {log_dir}\n")

    # Save often, so losing power costs 25k steps rather than the whole run.
    # save_freq counts steps per environment, hence the division.
    callback = CallbackList([
        FillFractionCallback(grid_cap=grid_size),
        CheckpointCallback(
            save_freq=max(1, checkpoint_freq // n_envs),
            save_path=str(CHECKPOINTS),
            name_prefix=run_name,
        ),
    ])

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True,
            reset_num_timesteps=not resume,
            tb_log_name=run_name,
        )
    finally:
        # runs on Ctrl+C and on crashes too, not only clean completion
        model.save(run_name)
        print(f"saved -> {run_name}.zip")

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
    parser.add_argument("--log-dir", type=str, default=str(TB_LOGS))
    parser.add_argument("--n-envs", type=int, default=8, help="parallel environments (subprocesses) for faster data collection")
    # The usual 0.01 assumes a handful of actions. With ~9,800 legal moves the
    # bonus works out at about 15% of what an episode is worth, which pays the
    # agent to stay undecided. Scale it down as the action space grows.
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
    # Off by default. At 0.03 it cut every update short and the policy barely
    # moved. clip_range already limits how far an update can go.
    parser.add_argument("--target-kl", type=float, default=None, help="optional extra brake on update size; off by default")
    # --- ablation knobs: change ONE of these against a reference run ---
    parser.add_argument("--reward-mode", type=str, default="compact", choices=["compact", "volume"],
                        help="compact = box volume minus the void it seals underneath; "
                             "volume = position-blind, every spot for a given box scores the same (ablation)")
    parser.add_argument("--policy", type=str, default="factored", choices=["factored", "flat"],
                        help="factored = policy.py's box x position dot-product head; "
                             "flat = stock MLP into one Linear(64 -> n_actions) layer (ablation)")
    parser.add_argument("--split-variety", type=float, default=0.0,
                        help="0 = always split the largest piece; >0 = chance of splitting a random one, "
                             "which is what the app serves")
    parser.add_argument("--run-name", type=str, default="ppo_bin_packing",
                        help="names the tensorboard subfolder, the checkpoint prefix and the final .zip, "
                             "so concurrent runs cannot overwrite each other")
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
        args.reward_mode,
        args.policy,
        args.split_variety,
        args.run_name,
    )
