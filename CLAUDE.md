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

ABLATIONS SETTLE WHICH BUG ACTUALLY MATTERED (ablations.png, 400k each)
Four arms, one change each, plotted from tb_logs by plot_ablations.py:
  full model      fill 0.74, entropy 6.4 -> 4.2
  flat head       fill 0.49 FLAT, entropy barely moves (6.4 -> 6.2)
  volume reward   fill 0.77, entropy tracks the full model
  no rotation     fill 0.735, but the best exact-solve rate (4.3% vs 0.2%)

READ THIS AGAINST BUG 2 AND BUG 3 BELOW. The flat head is confirmed as the
real blocker -- it never forms an opinion at all. But the position-blind
"volume" reward trains FINE here, which contradicts the bug-2 story. The two
faults were present at the same time during debugging and were confounded:
with a flat head nothing learns whatever the reward is, and once the
dot-product head is in, the delayed signal (a bad placement means fewer boxes
fit later) is enough on its own at this budget. The compaction reward is
still the better-shaped one and is what the shipped model trained on, but it
is not what unblocked the project. Do not repeat the original claim.

Rotation is a genuine trade, not a free win: dropping it shrinks the action
space so small puzzles get solved outright far more often, at a small cost in
average fill.

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

LAYOUT
backend/   all the Python. Flat imports work because running any of these
           puts backend/ on sys.path; paths.py anchors everything else to the
           repo root via __file__, so the cwd does not matter.
frontend/  app.html and viewer.html
model/     packing_policy.zip, the shipped policy

backend/environment.py   The game. Container, pieces, rules, reward, Gymnasium
                 API (reset/step), action_masks() (which moves are legal), the
                 reverse-construction puzzle generator, and self-tests under
                 `python backend/environment.py`.
backend/policy.py        The policy network. Box embeddings (attention) x
                 position embeddings (CNN over the heightmap), scored by dot
                 product. This is what made the agent able to learn at all.
backend/train.py         MaskablePPO, checkpointing, --resume, and a paired
                 agent-vs-greedy evaluation at the end.
backend/baseline.py      The greedy first-fit heuristic the agent has to beat.
backend/solver.py        Inference: rollout(), best-of-N, beam_search().
backend/model_loader.py  Finds the trained policy. ALWAYS matches on
                 observation space, never on "highest step number": runs share
                 checkpoints/ and their filenames collide by step count.
backend/paths.py         ROOT/FRONTEND/MODEL_DIR/CHECKPOINTS/TB_LOGS.
backend/app.py           FastAPI. /api/puzzle builds a puzzle (instant, no
                 model); /api/solve runs the policy.
                 `python backend/app.py` -> http://127.0.0.1:8000
backend/check_js.py      Rough syntax check for frontend/app.html's inline JS.
                 Cannot catch runtime errors -- only a browser can.
backend/export_run.py    Writes frontend/packing_runs.json for viewer.html.
backend/plot_ablations.py  Draws ablations.png from tb_logs.

frontend/app.html        The whole game: three.js scene, Watch / Play /
                 Compete, 3D staging area, split screen, settings, results.
frontend/viewer.html     Standalone replay page, no server needed.

DEPLOYMENT (live on Google Cloud Run)
Dockerfile       Installs the CPU torch wheel (500MB, not the 2.5GB CUDA one)
                 and pins the thread pools to 1 -- torch otherwise sizes them
                 to the HOST core count and the arenas alone blew past the
                 1GiB container limit. CMD is `python backend/app.py`.
.dockerignore    Keeps venv/ and checkpoints/ out of the image; model/ is NOT
                 excluded.
requirements.txt        runtime only (what the image installs)
requirements-train.txt  the above plus tensorboard/tqdm/rich
run_graph.ps1    Runs the ablation arms. Equal budgets so they share an LR
                 schedule.
assets/          ablations.png (the figure the README embeds) and demo.mp4
                 (7.6MB clip). Committed but .dockerignored -- no business in
                 a runtime image. NOTE: GitHub will not play a relative-path
                 .mp4 inline; drag the file into the web editor to get a
                 user-attachments URL that does.

checkpoints/     Training checkpoint stream, ~1.8GB. GITIGNORED.
tb_logs/         TensorBoard output. GITIGNORED.
(the papers themselves are cited in README.md, not vendored)

=====================================================================
KEY REFERENCE PAPERS / SOURCES (in priority order)
=====================================================================
1. Yin, He, Chen. "Deep Reinforcement Learning for Scalable Offline
   Three-Dimensional Packing." AAAI 2026. github.com/Ashenone511/BBMP-DCS
   Single network selects object + placement; the dot-product scoring and
   the r_t = g_{t-1} - g_t reward both come from here. Closest match.
2. Attend2Pack (arXiv 2107.04333) — attention-based, decomposes action
   space but single shared reward.
3. Xiong et al. "GOPT: Generalizable Online 3D Bin Packing via
   Transformer-Based Deep RL." IEEE RA-L 9(11), 2024.
   github.com/Xiong5Heng/GOPT -- single masked actor-critic on PPO, real
   public code, real robot validation. Their EMS (Empty Maximal Space)
   approach is the next lever if the action space needs shrinking.
4. Jiang et al. 2021 (AAMAS) — single encoder-decoder agent outputs
   sequence, orientation, position from ONE network. NOT hierarchical.
5. Wang, Lin, Kong, Dong. "Bin Packing Optimization via Deep
   Reinforcement Learning." IEEE RA-L 10(3), 2025 (arXiv 2403.12420).
   Clean height-map placement math; this is the model implemented here.

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
