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

FINAL VALIDATED RESULTS (1M steps, randomised training, Sep 2026)
Paired evaluation -- agent and greedy on IDENTICAL seeds/box sets, 200 eps
(the built-in eval in train.py compares on DIFFERENT seed ranges, which is
valid but noisier; the paired numbers below are the ones to quote):

  setting                greedy   agent   paired diff    t     agent wins
  fixed 30 / 10x10x10    85.2%    88.2%   +2.95pp       6.9    66% of eps
  randomised             84.7%    89.4%   +4.70pp       7.5    67% of eps

MECHANISM (why it wins -- not a fluke):
  trapped void   greedy 5.7%  -> agent 2.6%   (seals half the dead air)
  boxes placed   greedy 14.7  -> agent 16.2   (so more boxes fit)
  Exactly the causal chain the compaction reward was built to create.

TRAINING HEALTH (all normal): explained_variance 0.09 -> 0.972, approx_kl
steady 0.013-0.029, clip_fraction 0.19-0.27, value_loss falling monotonically,
entropy -6.03 -> -1.35. Fill plateaued at 0.89-0.90 from ~670k steps, so
1M was enough and more training buys little.

KNOWN LIMITATION -- GENERALISATION IS RANGE-BOUND
Trained on containers 6-10 per side and 15-30 boxes. Outside that:
  tiny 5x5x5 containers (below range)  greedy 83.4%  agent 80.1%  -3.3pp  LOSES
  few boxes, 5-12 (below range)        greedy 62.8%  agent 64.1%  +1.3pp  thin
  fixed 10x10x10, 30 boxes (in range)  greedy 85.2%  agent 87.9%  +2.8pp  fine
The reference paper reports the same shape of failure (its Fig. 6: utilisation
drops once item count moves far from training). IF free-play mode must accept
small containers, widen grid_range/height_range and retrain (~1 hour);
otherwise constrain the app UI to the trained range.
NOT TESTED: the rotation ablation needs its OWN trained model, because
n_rotations=1 changes the action space (3,000 vs 18,000).

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

STAGE 3 DONE. NOW BUILDING FOR THE APP (puzzle mode)
=====================================================================
TRAINING TARGET (decided Sep 2, after the 2.9M mixed run)
box_source="perfect" ONLY, containers 8-12 per side, grid cap 12
(25,920 actions). In puzzle mode "pack efficiently" and "solve the puzzle"
are the SAME objective, because the pieces tile the container exactly:
fill 100% <=> every box placed <=> solved. Verified: the perfect solution
earns 1.200, the maximum possible reward (greedy 0.904, random 0.359).
The reward is NOT the problem -- don't redesign it.

WHY NARROWER: breadth was costing more than it bought.
  6-10 cubic-ish, one box source, 18k actions -> beat greedy +2.95pp
  5-14 any aspect, mixed sources, 35k actions -> PARITY (+0.55pp, t=0.8)
~1000 container shapes at 2.9M steps is only ~180 episodes per shape. The
2.9M mixed model beat greedy on exactly-cubic containers (+2.7 to +5.0pp)
but fell to greedy level as soon as dimensions varied, and LOST on small
containers (5-7: -5.1pp, both near-cubic and elongated). 8-12 is 125
shapes, ~8x more practice each, and perfect-only doubles the samples that
count. Headroom at 8-12: greedy 83.6% vs a true 100% ceiling.

METRICS WERE MISLEADING -- FIXED. A single avg_fill_fraction is nearly
useless under randomisation: the agent scores ~100% on 5x5x5 and ~84% on
14x14x14, so a 50-episode running mean mostly tracks WHICH CONTAINERS WERE
DRAWN, not policy quality. That is why the curve looked noisy and flat.
Now logged: fill_small / fill_mid / fill_large (thresholds relative to the
grid cap, so each is comparable with itself) plus solve_rate, the fraction
of puzzles solved to exactly 100% -- the number that actually matters when
the optimum is known. Solve rate on the 2.9M model: 13% small, 2% mid, 0%
large.

RESULTS AT 2.6M STEPS (8-12, perfect puzzles, paired on identical seeds)
  greedy 83.4%   agent 85.5%   +2.06pp (t=3.2)   agent wins 62%
Loss decomposition -- this is the important part:
  agent : filled 85.5%  trapped void 2.6%  unplaced boxes 14.5%
  greedy: filled 83.4%  trapped void 6.7%  unplaced boxes 16.6%
The agent has ESSENTIALLY SOLVED tight packing (less than half greedy's
dead air). All remaining loss is boxes it never fits, i.e. a GLOBAL
ARRANGEMENT problem (which box, in what order), not placement quality.
Training plateaued at ~1.6M: entropy flat at -1.23, fill flat, expl_var
0.96. More steps will not fix an arrangement problem.

BEST-OF-N SAMPLING IS THE BIG WIN (solver.py) -- no retraining needed
  greedy heuristic      83.6%
  policy, single pass   84.9%
  policy, best of 4     88.6%   0.39s per puzzle
  policy, best of 16    90.3%   1.48s per puzzle   <- use this in the app
  policy, best of 64    91.8%   6.00s per puzzle
Sampling several packings and keeping the best beats greedy by 6.7pp and
lands in the range the reference papers report (89-93%). The app has no
real-time constraint, so the bot can afford to think for a second. This is
what the papers do too ("sample multiple solutions and select the best").
solver.py returns the full placement sequence (box, rot, x, y, z, dims) in
placement order, which is exactly what the app needs to animate the bot.

DOES THE BOT FIND THE PERFECT SOLUTION? ALMOST NEVER -- and that is fine.
Best-of-16 over 60 puzzles: average fill 90.4%, exact solves 0/60. Fill
distribution: 80-90% x21, 90-95% x29, 95-100% x8, 100% x0. Exact 3D packing
is NP-hard (~20 pieces is already 20! ~ 2.4e18 orderings before rotations
and positions), so sampling 16 attempts explores almost nothing. Best-of-N
raises the AVERAGE (84.9 -> 90.3) but barely moves exact solves (1 -> 2%).

SOLVE RATE DEPENDS ON PIECE COUNT -- USE THIS FOR APP DIFFICULTY TIERS
  pieces   avg fill   solved exactly (best-of-16)
   6-10     90.2%      28%
  11-16     89.8%       8%
  17-24     91.7%       0%
  25-30     92.1%       0%
More pieces = higher average fill but essentially no perfect solves (one
misplacement among 30 ruins it); fewer, larger pieces = lower average fill
but genuinely solvable. So:
  - FEW pieces (6-12)  -> "can you solve it perfectly?" The bot sometimes
    does, and a human realistically can too.
  - MANY pieces (20-30) -> "can you beat the bot's fill %?" Nobody solves
    these, so score-based competition is the honest framing.
Either way the app can always reveal env.solution, the guaranteed-perfect
arrangement, after the attempt.

EMS IS PROBABLY NOT NEEDED ANY MORE. Its trigger ("only if fill plateaus")
did fire, but best-of-N already reached paper-level numbers for a fraction
of the effort. Revisit only if single-pass quality becomes important.

CHECKPOINT COLLISION -- WATCH OUT. Every run writes to checkpoints/ with
filenames keyed by step number, so a new run silently overwrites an older
one's files and any surviving higher-numbered files belong to the OLD run
and the OLD architecture. Select checkpoints by matching observation_space
(solver.py does this), not by highest step number.

THE APP FLOW (decided): user picks a container size within limits ->
boxes are generated by REVERSE CONSTRUCTION so they tile it EXACTLY ->
user watches the bot, competes against it, or solves it alone.

REVERSE CONSTRUCTION (environment.py, box_source="perfect")
Recursively cuts the container into pieces, always splitting the largest,
so the pieces tile it exactly and a 100% packing is guaranteed. reset()
also stores env.solution = [(box, x, y, z, l, w, h)] -- the app can use it
as a hint or a "show me" button. VERIFIED over 200 episodes:
  - pieces tile the container exactly: 200/200
  - 100% IS reachable by drop-from-above: 200/200, worst fill = 1.0000
    (replay the solution bottom-up; each piece lands on a floor its
    neighbours have already filled exactly to its underside)
  - greedy on these puzzles: 6^3 91.0%, 10^3 83.5%, 14^3 80.3% vs a real
    100% ceiling, so there is 9-20pp of genuine headroom.

CRITICAL APP RULE: the agent can only DROP boxes from above -- it cannot
slide one sideways into a pocket under an overhang. The human player must
be held to the SAME rule (pick x, y and an orientation; gravity does the
rest) or compete mode is unfair in the human's favour.

BOX DIMS NOW SCALE WITH THE CONTAINER. 30 boxes of 2-5 units are 130% of a
10^3 container but only 48% of a 14^3 one -- which would have silently
recreated bug 1. Dims are sampled in [0.2*side, 0.5*side] (the papers use
[L/10, L/2]), which holds the ratio near 130% at every size and reproduces
the validated 2-5 range exactly at side 10.

ARCHIVED MODEL: stage3_validated_1M_grid10.zip is the model behind the
+2.95pp / +4.70pp results above. It pairs with commit f793093 and will NOT
load against the current env (grid 14, and the observation gained a
"container" key). Keep it as the artifact for those numbers.

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
