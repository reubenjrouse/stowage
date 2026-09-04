# Packing Puzzle

A reinforcement learning agent that packs 3D boxes into a container, and a browser game where you can watch it, play the same puzzle yourself, or race it.

Every puzzle is made by cutting a container into pieces, so a **perfect 100% packing always exists**. That gives a real yardstick — not "did it beat a heuristic", but "how close did it get to a known perfect answer".

## Results

200 puzzles, agent and baseline given identical box sets.

| | fill |
|---|---|
| Random legal moves | 52.0% |
| Greedy heuristic | 83.6% |
| **Agent, single pass** | **84.9%** |
| **Agent, best of 16** | **90.3%** |
| Perfect packing | 100% |

On a single pass it beats the greedy heuristic by **+2.06 points** (t = 6.9, winning 66% of individual puzzles). The reason shows up in the waste: it traps **2.6%** of the container as dead air, against greedy's 6.7%.

## Run it

```bash
pip install -r requirements.txt
python app.py          # then open http://127.0.0.1:8000
```

A trained model ships with the repo, so this works without training anything.

**Watch** the bot pack puzzle after puzzle · **Play** one yourself (click a box, `R` turns it, click to drop, `U` undoes) · **Compete** for the same pieces on a split screen.

To train from scratch — about four hours on a CPU:

```bash
python train.py --randomize --box-source perfect --grid-min 8 --grid-size 12 --timesteps 4000000
```

## How it works

A move is three choices at once: which box, which way up, and where to drop it — 25,920 options per turn, nearly all illegal. The environment masks it down to the legal ones.

Rather than learn 25,920 unrelated scores, the network learns to *describe a box* and to *describe a spot*, then scores a pair by how well the two match. What it learns about one spot carries to similar ones, which is what makes it trainable. A placement is rewarded with the space it gains: the box's volume minus the dead air it seals underneath.

At play time it can take one pass, sample several attempts and keep the best, or run a beam search that carries several part-finished packings and drops the poor ones.

## Files

| | |
|---|---|
| `environment.py` | the game: container, boxes, rules, reward, legal moves |
| `policy.py` | the network |
| `train.py` | training |
| `baseline.py` | the greedy heuristic to beat |
| `solver.py` | playing a puzzle: single pass, sampling, beam search |
| `app.py` / `app.html` | game server and front end |

## Limitations

- **Boxes only drop straight down** — they can't slide into a gap under an overhang. Costs about 5.7 points of fill.
- **Sizes are capped** at 6-12 per side and 30 boxes; the network's input is a fixed size.
- **It rarely finds the perfect answer** — 97% of 3-4 piece puzzles, ~60% at 5-8, essentially never above 17. Exact 3D packing is NP-hard.

Built following the approach in PQNet and GOPT; sources in `reference_code_papers.txt`. A longer writeup — the dead ends, and what actually fixed them — is separate.
