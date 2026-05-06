# Barricade RL

Barricade RL is a Python project for building and testing a reinforcement-learning environment for the board game Barricade. The goal is to keep the training environment small and fast while also providing a human-playable UI so the rules can be checked before spending compute on training.

The project currently includes:

- A plain Python and NumPy game engine for Barricade rules.
- A Gymnasium environment with a compact `Discrete(132)` action space.
- Legal action masks for pawn moves and wall placements.
- BFS path-preservation checks so walls cannot fully block either player.
- A Tkinter UI for human testing with drag-and-drop walls.
- Random and greedy scripted opponents.
- A single-agent training wrapper where the learner plays player 0.
- Benchmark, evaluation, and MaskablePPO smoke-training commands.
- Pytest coverage for important movement, wall, jump, opponent, and win rules.

## Project Layout

```text
barricade_rl/
  core.py      # Minimal rules engine used by both training and UI
  env.py       # Gymnasium environment wrapper
  single_agent.py # Learner-vs-scripted-opponent Gymnasium wrapper
  opponents.py # Random and greedy scripted opponents
  benchmark.py # Environment speed benchmark command
  evaluate.py  # Scripted policy evaluation command
  train_maskable_ppo.py # MaskablePPO smoke-training command
  ui_tk.py     # Human-playable Tkinter UI
documentation/
  rules.md     # Rule notes used to implement the environment
  usage.md     # Short usage reference
tests/
  test_core.py # Core correctness tests
  test_single_agent.py # Opponent and single-agent tests
  test_evaluate.py # Evaluation helper tests
```

## Windows Support

This project should work on Windows. The core code is pure Python plus NumPy/Gymnasium, and the UI uses Tkinter, which is included with the standard Python installer from python.org.

Use PowerShell commands on Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m pytest -q
```

Run the UI on Windows:

```powershell
.\.venv\Scripts\barricade-play.exe
```

Install RL extras on Windows:

```powershell
.\.venv\Scripts\python -m pip install -e ".[dev,rl]"
```

If Tkinter is missing, reinstall Python from python.org and make sure the Tcl/Tk option is enabled. If PyTorch install fails, install PyTorch using the command from the official PyTorch selector for your Windows machine, then rerun the project install.

## Requirements

You need Python 3.10 or newer. Python 3.11 is a conservative choice across macOS and Windows.

The basic environment only needs:

- `gymnasium`
- `numpy`
- `pytest` for tests

Training dependencies such as PyTorch, Stable-Baselines3, and `sb3-contrib` are optional for now.

## Setup

From the project root:

```bash
python3 -m venv .venv
```

Install the project and test dependencies:

```bash
.venv/bin/python -m pip install -e '.[dev]'
```

Run the tests:

```bash
.venv/bin/python -m pytest -q
```

You should see all tests pass.

## Play The Game

Start the UI:

```bash
.venv/bin/barricade-play
```

Controls:

- Arrow keys or WASD move the current pawn.
- Click a reachable square to move there.
- Drag a horizontal or vertical wall from the right panel onto a border line between squares.
- `R` resets the game.

The right panel shows whose turn it is, how many walls the current player has left, and the draggable wall pieces. The active pawn also has a colored ring.

Open a saved training replay in the same UI:

```bash
.venv/bin/barricade-play --replay runs/maskable_ppo_barricade/replays/replay_1000.json
```

Replay controls:

- Right arrow or `N` steps forward.
- Left arrow or `P` steps backward.
- Space plays or pauses.
- `R` restarts the replay.

## Wall Behavior

Walls are continuous two-edge pieces placed on border lines between squares.

Important placement rules:

- A wall cannot overlap another wall.
- A wall cannot cross another wall at the same anchor.
- Same-direction adjacent anchors are illegal because they would overlap one of the two occupied edges.
- Same-direction walls can be placed end-to-end when they do not overlap.
- End-to-end wall pieces leave an endpoint gap where a perpendicular wall can be placed.
- Every wall placement must leave both players with at least one path to their goal row.

## Training Environment

The Gymnasium environment is separate from the UI, so training does not pay for rendering.

Basic example:

```python
from barricade_rl import BarricadeEnv

env = BarricadeEnv()
obs, info = env.reset()

mask = env.action_masks()
legal_action = mask.nonzero()[0][0]

obs, reward, terminated, truncated, info = env.step(legal_action)
```

Observation shape:

```text
(6, 9, 9)
```

Observation planes:

1. Current player pawn.
2. Opponent pawn.
3. Horizontal wall anchors.
4. Vertical wall anchors.
5. Current player walls remaining as a constant plane.
6. Opponent walls remaining as a constant plane.

Action space:

```text
Discrete(132)
```

Actions:

- `0` move up.
- `1` move down.
- `2` move left.
- `3` move right.
- `4` through `67` place horizontal wall anchors.
- `68` through `131` place vertical wall anchors.

The environment exposes both `action_mask()` and `action_masks()` for compatibility with masking libraries such as `sb3-contrib` MaskablePPO.

For initial training, use the single-agent wrapper:

```python
from barricade_rl import BarricadeSingleAgentEnv
from barricade_rl.opponents import GreedyOpponent

env = BarricadeSingleAgentEnv(opponent=GreedyOpponent())
obs, info = env.reset()
```

In this wrapper, the learner is always player 0. After the learner acts, the scripted opponent automatically takes player 1's turn. Rewards are from the learner's perspective:

- learner win: `+1`
- opponent win: `-1`
- non-terminal step: `0`

## Benchmarking

Run a quick environment-speed benchmark:

```bash
.venv/bin/barricade-benchmark --episodes 100 --opponent random
```

This reports learner steps per second. Early training is likely to be CPU-bound because wall legality uses BFS path checks.

## Evaluation

Evaluate a simple scripted learner against a scripted opponent:

```bash
.venv/bin/barricade-evaluate --episodes 100 --policy greedy --opponent random
```

Available policies and opponents:

- `random`
- `greedy`

## Optional RL Dependencies

Install these when you are ready to start training:

```bash
.venv/bin/python -m pip install -e '.[dev,rl]'
```

The intended first training stack is:

- PyTorch
- Stable-Baselines3
- `sb3-contrib`
- MaskablePPO

After installing RL dependencies, run a short smoke-training job:

```bash
.venv/bin/barricade-train-maskable-ppo --timesteps 10000 --opponent random
```

Outputs are written under `runs/maskable_ppo_barricade/` by default.

The smoke trainer currently uses `MlpPolicy`, which flattens the small `(6, 9, 9)` observation. A custom small CNN can be added later once the training loop and evaluation workflow are stable.

## Visualize Training Progress

Training writes TensorBoard logs under the run directory:

```bash
.venv/bin/tensorboard --logdir runs
```

Then open the URL TensorBoard prints, usually:

```text
http://localhost:6006
```

Useful charts:

- `rollout/ep_rew_mean`: average training episode reward.
- `rollout/ep_len_mean`: average episode length.
- `eval/mean_reward`: evaluation reward at checkpoints.
- `train/entropy_loss`: policy randomness.
- `train/value_loss`: value function fit.

The trainer also saves replay files at milestones:

```text
runs/maskable_ppo_barricade/replays/replay_1000.json
runs/maskable_ppo_barricade/replays/replay_2000.json
...
```

View one in the UI:

```bash
.venv/bin/barricade-play --replay runs/maskable_ppo_barricade/replays/replay_1000.json
```

Control replay frequency with:

```bash
.venv/bin/barricade-train-maskable-ppo --timesteps 10000 --opponent random --replay-freq 1000
```

Use `--replay-freq 0` to disable replay saving.

## Current Status

Implemented:

- Core two-player turn logic.
- Pawn movement.
- Straight jumps and side jumps.
- Wall counts.
- Wall overlap and crossing checks.
- End-to-end wall placement behavior.
- BFS path preservation.
- Sparse terminal rewards.
- Action masks.
- Human playable UI.
- Scripted opponents.
- Training script.
- Evaluation script.

Not implemented yet:

- PettingZoo self-play wrapper.
- Checkpointing and opponent pools.

## Suggested Next Steps

1. Install RL dependencies and run a short MaskablePPO smoke train.
2. Benchmark CPU speed before optimizing rule checks.
3. Add a saved-model evaluation command for trained MaskablePPO checkpoints.
4. Add more rule tests from played UI edge cases.
5. Add a PettingZoo AEC wrapper once the single-agent training loop is stable.
6. Add checkpoint self-play and opponent pools.
