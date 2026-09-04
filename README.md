# Packing Puzzle

A reinforcement learning agent that packs 3D boxes into a container, and a browser game where you can watch it, play the same puzzle yourself, or race it.

Every puzzle is made by cutting a container into pieces, so a **perfect 100% packing always exists**. That makes it a fair challenge and gives a real yardstick: not "did the agent beat a heuristic", but "how close did it get to a known perfect answer".

## Results

Measured over 200 puzzles, with the agent and the baseline given identical box sets.

| | fill | notes |
|---|---|---|
| Random legal moves | 52.0% | the floor |
| Greedy heuristic | 83.6% | biggest box first, lowest spot |
| **Agent, single pass** | **84.9%** | one move at a time, no second guesses |
| **Agent, best of 16** | **90.3%** | plays it 16 times, keeps the best |
| Perfect packing | 100% | always exists, by construction |

The agent beats the greedy heuristic by **+2.06 points** on a single pass (t = 6.9, winning 66% of individual puzzles), and by **+6.7 points** when allowed to sample.

Where the space goes is more revealing than the totals:

| | filled | dead air trapped | boxes left over |
|---|---|---|---|
| Greedy | 83.4% | 6.7% | 16.6% |
| Agent | 85.5% | **2.6%** | 14.5% |

The agent traps less than half the dead air the heuristic does. That is the specific skill it was trained for, and it is why it fits more boxes.

## How it works

**The problem.** A move is three choices at once: which box, which way up, and where to drop it. At the default size that is 25,920 possible moves per turn, nearly all illegal. The environment hands the network a mask of the legal ones, so it only ever chooses among moves that work.

**The network.** Scoring 25,920 unrelated moves is hopeless to learn. Instead the network learns to *describe a box* and to *describe a spot*, and scores a pair by how well the two descriptions match. What it learns about one spot then carries to similar ones. Boxes are read by an attention layer, the surface of the pile by a small CNN. 315k parameters.

**The reward.** A placement is worth the space it gains: the box's volume, minus the dead air it seals underneath. Rewarding volume alone gives every position for a given box the same score, so the agent never learns *where* to put things.

**Playing.** At play time the policy can be used three ways: one straight pass, sampling several attempts and keeping the best, or beam search — carrying several part-finished packings and dropping the poor ones. Beam search wins on small puzzles (60% solved vs 32% for sampling) and loses on large ones (87.7% vs 91.3%), because judging a half-finished packing is only reliable when the puzzle is small.

## Running it

```bash
pip install -r requirements.txt
python app.py          # then open http://127.0.0.1:8000
```

A trained model ships with the repo, so this works without training anything.

**Watch** the bot pack puzzle after puzzle. **Play** one yourself: click a box off the ground, `R` turns it, click the container to drop it, `U` undoes. **Compete** splits the screen and gives you both the same pieces — first to a perfect pack wins, and the bot is deliberately slowed so you can keep up.

To train from scratch:

```bash
python train.py --randomize --box-source perfect --grid-min 8 --grid-size 12 --timesteps 4000000
```

About four hours on a CPU. Checkpoints are saved every 25k steps and the model survives a Ctrl+C.

## Files

| | |
|---|---|
| `environment.py` | the game: container, boxes, rules, reward, legal moves |
| `policy.py` | the network |
| `train.py` | training |
| `baseline.py` | the greedy heuristic to beat |
| `solver.py` | playing a puzzle: single pass, sampling, beam search |
| `app.py` / `app.html` | the game server and front end |
| `viewer.html` | standalone replay page, no server needed |

## Limitations

**Boxes only drop straight down.** They cannot slide sideways into a gap under an overhang. This is the standard simplification in the literature, and it costs about 5.7 points of fill — though the agent recovers most of that by packing flatter in the first place.

**Sizes are capped.** Containers 6-12 per side, up to 30 boxes, because the network's input is a fixed size. It was trained on 8-12 and gets noticeably worse below that.

**It rarely finds the perfect answer.** On 3-4 piece puzzles it solves 97%; on 5-8, about 60%; on 17 or more, essentially never. Exact 3D packing is NP-hard — 20 pieces is already more orderings than could be checked in a lifetime — so "very good, quickly" is the goal, not optimality.

## Reference

Built following the approach in *Deep Reinforcement Learning for Scalable Offline Three-Dimensional Packing* (PQNet) and GOPT, in particular scoring moves as a dot product between item and space descriptions, and masking infeasible placements. Sources are listed in `reference_code_papers.txt`.
