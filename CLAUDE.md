PROJECT BRIEF — RL Bin Packing "Puzzle" App

GOAL
Build an RL agent that solves 3D bin packing (given a set of boxes + a
container, decide what order to place them in AND where/how to place
each one, to maximize space utilization). Wrap it in a demo app framed
as a puzzle/game: watch the bot pack, or compete against it manually.
Longer-term stretch: let users measure real boxes/trunks via phone
camera (v2, not now), and have the bot learn from human demonstrations
that beat it (v2, via offline imitation learning / fine-tuning, not
live/real-time learning).

WHO'S BUILDING THIS
Me: ~2 years production ML experience (fine-tuning/LoRA, RAG,
distributed training, agentic pipelines) but NEW to reinforcement
learning specifically. Learning RL as I build this. Building this as
a portfolio/SOP project for grad school applications (Waterloo MDSAI,
UofT MScAC, TU Delft, UvA — Fall 2027 intake) — want a genuinely
differentiated project, not a generic classification/regression demo.

=====================================================================
CURRENT STATE (Stage 2 VALIDATED -- agent beats the baseline)
=====================================================================

WHAT'S BUILT AND VERIFIED
- Gymnasium env with rotation, exact action masking, honest termination.
- policy.py: dot-product policy (box embeddings x position embeddings).
- MaskablePPO training with checkpointing + --resume.
- Greedy first-fit baseline that also uses rotation (fair comparison).
- Self-tests (python environment.py): mask invariant over 50 episodes.

NUMBERS (30 boxes, 10x10x10 container, 6 orientations)
- Random-but-legal:   52.0% fill   (the floor)
- Greedy baseline:    85.2% fill   (the bar)
- Trained agent:      87.5% fill   at 540k steps -- BEATS THE BASELINE
- Curve: 0.510 -> 0.875, crossed greedy at ~475k steps, still climbing.
- boxes_placed stayed flat at ~17 the whole time: all the gain is
  placement QUALITY, not grabbing more boxes.
NOTE: that 87.5% is the training running-average. A completed run with a
deterministic eval on held-out seeds is still needed for a citable number
(the 540k run died to a power cut before saving -- hence checkpointing).

THE THREE BUGS THAT COST THIS PROJECT DAYS -- DO NOT REPEAT
1. SATURATED BENCHMARK. Original env: 8 boxes = 35% of container volume,
   so everything fit anywhere and greedy placed 8/8 every episode. Its
   "34.8%" was the ARITHMETIC CEILING, not a bar. Hours of tuning chased
   an impossible target.
   LESSON: measure random / baseline / theoretical-max before tuning.
2. POSITION-BLIND REWARD. Reward was volume/container_volume, which is
   IDENTICAL for every position of a given box. Dropping a box flush on a
   flat surface and dropping it on a jagged spot sealing 90 units of dead
   air paid exactly the same. The agent had no placement signal at all and
   sat at the random score. Fixed by reward_mode="compact": volume minus
   the void sealed underneath (the PQNet r_t = g_{t-1} - g_t idea).
   LESSON: check that the reward actually varies across the decisions you
   want the agent to learn. Measure the spread.
3. FLAT-LOGIT POLICY. An MLP ending in Linear(128 -> 18000) has to learn
   18,000 unrelated numbers; "box 3 at (4,5)" and "box 3 at (4,6)" share
   nothing. Policy entropy sat at ln(n_legal) for 250k+ steps -- it never
   formed an opinion. Fixed by policy.py: embed boxes (attention) and
   positions (CNN over the heightmap), score each pair by DOT PRODUCT,
   exactly as PQNet builds its matrix M. 71k params instead of 2.4M, and
   it learns. THIS WAS THE ACTUAL BLOCKER.

SMALLER LESSONS
- ent_coef must scale DOWN as the action space grows. At 0.01 with ~9,800
  legal actions the entropy bonus was ~15% of episode return -- the agent
  was paid to stay random. Now 0.001.
- target_kl=0.03 fired every rollout ("Early stopping at step 1") and cut
  epochs 5 -> 2. clip_range already bounds updates; leave target_kl off.
- Do NOT wrap training in `timeout`; do NOT pipe it through `grep|tail`
  (buffers, so you can't watch it). Read tb_logs event files instead.
- Laptop sleeps on battery after 15 min and killed two runs. Plug in and
  `powercfg /change standby-timeout-ac 0` before a long run.

=====================================================================
PROBLEM FORMULATION (decided and implemented — don't relitigate)
=====================================================================
- OFFLINE bin packing, not online: the full set of boxes is known
  upfront. Confirmed the right and EASIER choice vs. online/buffer
  approaches — no partial observability, cleaner reward signal.
- SINGLE neural network, not hierarchical/multi-agent. Explicitly
  rejected: separate manager/worker networks (hierarchical RL with
  Dueling DQN) — too hard to debug for a first RL project.
- ONE policy network decides, per step: (a) which remaining box, (b)
  what orientation, (c) where to place it. Implemented as a single
  FLAT action index over (box x rotation x x-position x y-position),
  18,000 entries at current settings.
- ACTION MASKING is what makes that tractable: the env exposes
  action_masks() marking exactly the legal placements, so the agent
  only ever samples legal moves. This is what the reference papers do
  (PQNet multiplies action values by a feasibility mask; GOPT same).
  NOTE: the action space must be flat Discrete, NOT MultiDiscrete —
  SB3 masks each MultiDiscrete dimension independently, which can't
  express "box 3 rotated this way fits here but not there".
- ROTATION: implemented, 6 axis-aligned orientations.
  Use --n-rotations 1 to ablate it.
- PLACEMENT MODEL: heightmap / "drop from above", like Tetris. A box
  falls until it rests on the floor or whatever's below. Known
  limitation: cannot slide a box into a cavity under an overhang.
  This is the standard simplification in this literature (Wang & Dong).

=====================================================================
WHAT'S IN THE REPO
=====================================================================
environment.py  The game: container, boxes, rules, rewards. Gymnasium
                API (reset/step). Also action_masks() (legal moves) and
                self-tests under `if __name__ == "__main__"`.
baseline.py     The non-AI opponent: greedy first-fit, biggest box
                first, placed wherever it rests lowest. The bar to beat.
policy.py       The policy network: box embeddings (attention) x position
                embeddings (CNN over the heightmap), scored by dot product.
                This is what made the agent able to learn at all.
train.py        The trainer: MaskablePPO, checkpointing (--resume), and
                agent-vs-baseline evaluation with a significance check.
                --randomize turns on Stage 3 domain randomisation.
checkpoints/    Auto-saved every 25k steps (a power cut used to cost a
                whole run; now it costs 25k steps).
requirements.txt Dependencies.
reference_code_papers.txt  Source papers (see below).
tb_logs/        TensorBoard output.   ppo_bin_packing_stage2.zip  Saved model.

=====================================================================
KEY REFERENCE PAPERS / SOURCES (in priority order)
=====================================================================
1. "Deep Reinforcement Learning for Scalable Offline Three-Dimensional
   Packing" (2026) — single network simultaneously selects object +
   placement, attention encoders, closest match to what we want.
2. Attend2Pack (arXiv 2107.04333) — attention-based, decomposes action
   space but single shared reward.
3. GOPT (github.com/Xiong5Heng/GOPT, IEEE RA-L 2024) — single
   actor-critic (shared "Packing Transformer" backbone), real public
   CODE, real robot validation. Their EMS (Empty Maximal Space)
   approach is the next lever if the action space needs shrinking.
4. Jiang et al. 2021 (AAMAS) — single encoder-decoder agent outputs
   sequence, orientation, position from ONE network. NOT hierarchical.
5. Wang & Dong (arXiv 2403.12420) — clean height-map placement math
   (Section II-C); this is the placement model currently implemented.

=====================================================================
REJECTED APPROACHES (don't suggest these again)
=====================================================================
- Hierarchical RL / manager-worker split / Dueling DQN.
- Buffer-based limited lookahead (online-style framing).
- Fine-tuning a pretrained model, benchmark tables, plain
  classification/regression — wanted "built a system" substance.
- MultiDiscrete action space (masking is per-dimension, too weak).

=====================================================================
NEXT STEPS, IN ORDER
=====================================================================
NOW: the Stage 2 + Stage 3 run, in one go:
     powercfg /change standby-timeout-ac 0      (plug in first!)
     python train.py --timesteps 1000000 --randomize

     It evaluates twice at the end: on the FIXED canonical setting
     (comparable with every earlier run, greedy = 85.2%) and on
     randomised containers (greedy = 84.7%). Checkpoints every 25k in
     checkpoints/; continue a killed run with --resume <path>.

EVAL FLOORS (deterministic argmax, 200 episodes, fixed setting)
- Untrained network:  48.2%   <- the real floor for the eval protocol
- Uniform random:     52.1%
- Greedy baseline:    85.2%   <- the bar
- 8k steps of the dot-product policy already gives 79.0%, i.e. it is
  roughly 10x more sample-efficient than the old MLP (which needed
  ~400k steps to reach the same place).

STAGE 3 IS IMPLEMENTED (not just planned)
- Domain randomisation: container (6-10)^2 x (6-10) and 15-30 boxes,
  resampled every episode. Observation shapes stay FIXED (SB3 requires
  it); cells outside the current container are pre-filled to "full" so
  the ordinary feasibility mask excludes them, and surplus boxes are
  marked already-placed. Verified: 1176 random steps, 0 illegal, never
  placed outside the container, obs always inside the declared space.
- Attention encoder: DONE, shipped as policy.py.
- EMS: still only if fill plateaus. Measured upside ~5.7pp.

LATER (app layer, after core RL is validated):
- Gradio + HF Space, Plotly/Three.js 3D visualization.
- Watch mode / Compete mode / Puzzle mode (reverse construction,
  guarantees a perfect fit exists) / Free-play mode (custom dims,
  doubles as a real "does this fit in my trunk" tool).
- v2 stretch: imitation learning on winning human sequences (batch,
  not live). v2 stretch: AR measurement (ARKit/ARCore) — deferred,
  needs native mobile, separate stack.

=====================================================================
TECH STACK
=====================================================================
Python, Gymnasium (custom env), Stable-Baselines3 + sb3-contrib
(MaskablePPO — do NOT hand-roll the algorithm), TensorBoard,
eventually Gradio + HF Space and Plotly/Three.js.
Local GPU: RTX 4060 Laptop (8GB). NOTE: ~95% of per-step cost is env
work (mask building) which the GPU can't touch — SubprocVecEnv across
cores matters more than the GPU until the policy gets heavier (i.e.
until the attention encoder lands).

=====================================================================
HOW TO WORK WITH ME
=====================================================================
Be honest about what's realistic vs. scope creep. I have a tendency to
jump to the most sophisticated version (hierarchical RL, full EMS,
rotation, game UI all at once); keep me anchored to incremental,
working-first scope. Verify claims with measurements rather than
asserting them — the saturated-benchmark bug above was found by
measuring the ceiling, not by reasoning about the training curves.
