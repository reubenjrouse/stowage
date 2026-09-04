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

THE RL CORE
environment.py   The game. Container, pieces, rules, reward, Gymnasium API
                 (reset/step). Also action_masks() (which moves are legal),
                 the reverse-construction puzzle generator, and self-tests
                 under `python environment.py`.
policy.py        The policy network. Box embeddings (attention) x position
                 embeddings (CNN over the heightmap), scored by dot product.
                 This is what made the agent able to learn at all.
train.py         Training: MaskablePPO, checkpointing, --resume, and a paired
                 agent-vs-greedy evaluation at the end.
baseline.py      The greedy first-fit heuristic the agent has to beat.
solver.py        Inference. rollout(), best-of-N, and beam_search(); run it
                 directly for a quick demo. Beam for small puzzles, sampling
                 for large -- see the numbers in its docstring.
model_loader.py  Finds the trained policy. ALWAYS matches on observation
                 space, never on "highest step number": training runs share
                 checkpoints/ and their filenames collide by step count, so a
                 bigger number can be an older run with a different network.

THE APP
app.py           FastAPI backend. /api/puzzle builds a puzzle to the player's
                 chosen size (instant, no model); /api/solve runs the policy.
                 `python app.py`, then open http://127.0.0.1:8000
app.html         The whole front end: three.js scene, Watch / Play / Compete,
                 3D staging area, split-screen compete, settings, results.
check_js.py      Rough syntax check for app.html's inline JS (balanced
                 delimiters, functions defined, element ids exist). It cannot
                 catch runtime errors -- only a browser can.

STANDALONE DEMO (no server needed)
export_run.py    Runs the bot on a few puzzles and writes packing_runs.json.
viewer.html      Replays those runs as a self-contained page; this is what
                 gets published as a shareable artifact.
packing_runs.json  The exported runs viewer.html embeds.

DATA AND WEIGHTS
model/packing_policy.zip  The shipped trained policy (3.4M steps). Committed,
                 so a fresh clone runs the app with no training.
checkpoints/     Training checkpoint stream, ~900MB. GITIGNORED.
tb_logs/         TensorBoard output. GITIGNORED.
reference_code_papers.txt  Source papers.

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
THE APP IS BUILT AND RUNNING (python app.py -> http://127.0.0.1:8000)
Three modes, all real-time against the live model:
  Watch   the bot packs puzzle after puzzle, endlessly
  Play    pick a box off the ground, R turns it, click to drop, undo freely
  Compete split screen, same pieces; FIRST TO 100% WINS. Because the pieces
          are cut from the container, placing them all IS a perfect pack, so
          finishing and winning are the same event. Short of that the tighter
          pack wins, exact ties go on time. The bot is deliberately slowed to
          one box every 2.8s so a human can keep up, and says so on screen.

OPEN QUESTION -- COMPETE FAIRNESS (raised, not yet done)
The bot plays its BEST OF 16 attempts (or a beam search) worked out before
the race starts, while the player gets one attempt with undo. Slowing the
bot fixes the pace but not that asymmetry. The fix is to give compete a
single deterministic pass: 84.9% instead of 90.3%, one honest attempt like
the player's, with undo as the player's compensating edge. Decide before
calling the app finished.

THEN, IF THE MODEL MATTERS MORE THAN THE APP
1. Train on split_variety>0 puzzles. The shipped model trained on
   equalised piece sizes (always split the largest), but the app now serves
   varied ones. It handles them fine -- better, in fact -- but matching the
   distributions would be cleaner.
2. Visualise the beam search: show candidate packings developing side by
   side and dimming as they are abandoned. This is the most interesting
   thing in the system and nothing on screen currently shows it.
3. EMS -- still only if fill plateaus. Measured upside ~5.7pp.

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
