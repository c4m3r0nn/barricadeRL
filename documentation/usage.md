# Barricade RL Usage

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

Install the RL extras later when training work starts:

```bash
.venv/bin/python -m pip install -e '.[dev,rl]'
```

## Playable UI

```bash
.venv/bin/barricade-play
```

Controls:

- Arrow keys or WASD move the current pawn.
- Drag a horizontal or vertical wall from the right panel onto a border line between squares.
- Walls are continuous two-edge pieces; separate end-to-end walls leave a small gap for a perpendicular wall.
- Click a reachable board dot to move there.
- `R` resets the game.

Open a training replay:

```bash
.venv/bin/barricade-play --replay runs/maskable_ppo_barricade/replays/replay_1000.json
```

## Training Environment

```python
from barricade_rl import BarricadeSingleAgentEnv

env = BarricadeSingleAgentEnv()
obs, info = env.reset()
mask = env.action_masks()
obs, reward, terminated, truncated, info = env.step(mask.nonzero()[0][0])
```

The environment keeps rendering and UI out of the training loop. The core state is compact: pawn coordinates, two 8x8 wall arrays, wall counts, current player, and winner.

## Benchmark And Evaluation

```bash
.venv/bin/barricade-benchmark --episodes 100 --opponent random
.venv/bin/barricade-evaluate --episodes 100 --policy greedy --opponent random
```

## MaskablePPO Smoke Training

First install RL dependencies:

```bash
.venv/bin/python -m pip install -e '.[dev,rl]'
```

Then run:

```bash
.venv/bin/barricade-train-maskable-ppo --timesteps 10000 --opponent random
```

Watch metrics during or after training:

```bash
.venv/bin/tensorboard --logdir runs
```

On Windows, replace `.venv/bin/` with `.\.venv\Scripts\` and use `.exe` console scripts, for example:

```powershell
.\.venv\Scripts\barricade-play.exe
.\.venv\Scripts\tensorboard.exe --logdir runs
```
