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
- Replay summary tooling, checkpoint opponent pools, optional shaped rewards, and an AEC-style PettingZoo wrapper.
- Experiment runner and Tkinter experiment dashboard for launching runs and viewing live metrics.
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
  replay.py    # Replay recording and summary command
  experiments.py # Experiment specs, runner, JSONL metrics helpers
  experiment_ui.py # Local Tkinter training dashboard
  pettingzoo_env.py # AEC-style two-agent wrapper
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

## Windows Tutorial

This project should work on Windows. The core code is pure Python plus NumPy/Gymnasium, and the UI uses Tkinter, which is included with the normal Python installer from python.org.

These instructions assume PowerShell and Python 3.11.

### 1. Install Python

1. Download Python 3.11 from <https://www.python.org/downloads/windows/>.
2. During install, check `Add python.exe to PATH`.
3. Keep `tcl/tk and IDLE` enabled. This is needed for the playable UI.
4. Open PowerShell and check Python:

```powershell
py -3.11 --version
```

### 2. Open The Project

In PowerShell, move into the project folder. Example:

```powershell
cd C:\Users\YourName\Projects\barricadeRL
```

### 3. Create A Virtual Environment

```powershell
py -3.11 -m venv .venv
```

You can either use the full `.venv\Scripts\python` path in commands, or activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation scripts, run this once for the current terminal session:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

### 4. Install The Basic Project

```powershell
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

### 5. Run Tests

```powershell
.\.venv\Scripts\python -m pytest -q
```

You should see all tests pass.

### 6. Play The Game UI

```powershell
.\.venv\Scripts\barricade-play.exe
```

If this fails with a Tkinter error, reinstall Python from python.org and make sure `tcl/tk and IDLE` is selected.

### 7. Install RL Dependencies

```powershell
.\.venv\Scripts\python -m pip install -e ".[dev,rl]"
```

This installs PyTorch, Stable-Baselines3, `sb3-contrib`, and TensorBoard.

If PyTorch install fails, use the official command from <https://pytorch.org/get-started/locally/> for your Windows machine, then rerun:

```powershell
.\.venv\Scripts\python -m pip install -e ".[dev,rl]"
```

### 8. Run A Short Training Job

```powershell
.\.venv\Scripts\barricade-train-maskable-ppo.exe --timesteps 10000 --opponent random --replay-freq 1000
```

Outputs are written under:

```text
runs\maskable_ppo_barricade\
```

### 9. Watch Training In TensorBoard

In a second PowerShell window from the project folder:

```powershell
.\.venv\Scripts\tensorboard.exe --logdir runs
```

Open the URL TensorBoard prints, usually:

```text
http://localhost:6006
```

### 10. Watch A Saved Training Replay

Training saves replay JSON files at milestones:

```text
runs\maskable_ppo_barricade\replays\replay_1000.json
runs\maskable_ppo_barricade\replays\replay_2000.json
```

Open one in the UI:

```powershell
.\.venv\Scripts\barricade-play.exe --replay runs\maskable_ppo_barricade\replays\replay_1000.json
```

Replay controls:

- Right arrow or `N`: next frame.
- Left arrow or `P`: previous frame.
- Space: play or pause.
- `R`: restart replay.

### 11. Useful Windows Commands

Benchmark environment speed:

```powershell
.\.venv\Scripts\barricade-benchmark.exe --episodes 100 --opponent random
```

Evaluate a scripted policy:

```powershell
.\.venv\Scripts\barricade-evaluate.exe --episodes 100 --policy greedy --opponent random
```

Evaluate a trained model:

```powershell
.\.venv\Scripts\barricade-evaluate-model.exe --model runs\maskable_ppo_barricade\best\best_model.zip --episodes 100 --opponent random
```

Run all tests again:

```powershell
.\.venv\Scripts\python -m pytest -q
```

## Requirements

You need Python 3.10 or newer. Python 3.11 is a conservative choice across macOS and Windows.

The basic environment only needs:

- `gymnasium`
- `numpy`
- `pytest` for tests

Training dependencies such as PyTorch, Stable-Baselines3, and `sb3-contrib` are optional for now. PettingZoo is optional unless you are using the AEC self-play wrapper.

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
- `mixed` for opponents, which samples between random and greedy play

Evaluate a trained MaskablePPO checkpoint:

```bash
.venv/bin/barricade-evaluate-model --model runs/maskable_ppo_barricade/best/best_model.zip --episodes 100 --opponent random
```

This reports wins, losses, truncations, win rate, and average learner steps.
It also reports loss rate, truncation rate, min/max learner steps, and average walls placed by each side. Watch wall usage closely: if the learner always burns all walls, reward shaping or opponent curriculum may need adjustment.

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

Train against a stronger mixed opponent:

```bash
.venv/bin/barricade-train-maskable-ppo --timesteps 10000 --opponent mixed --out runs/maskable_ppo_mixed
```

`mixed` samples between random and greedy opponent moves. It is slower than `random`, but it is a better first curriculum step because the learner sees less brittle play.

Train against a checkpoint opponent pool:

```bash
.venv/bin/barricade-train-maskable-ppo \
  --timesteps 10000 \
  --checkpoint-opponents "runs/maskable_ppo_barricade/best/*.zip" \
  --out runs/maskable_ppo_pool
```

Add optional shortest-path shaped rewards:

```bash
.venv/bin/barricade-train-maskable-ppo --timesteps 10000 --opponent mixed --shaped-reward
```

Sparse win/loss reward remains the default. Use shaped rewards as an experiment, not as the baseline.

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

Summarize saved replays side by side:

```bash
.venv/bin/barricade-replay-summary runs/maskable_ppo_barricade/replays/replay_*.json
```

## Experiment Runner And Dashboard

Run the default experiment set from the command line:

```bash
.venv/bin/barricade-experiments --root runs/experiments --timesteps 10000 --seed 0
```

Open the local experiment dashboard:

```bash
.venv/bin/barricade-experiment-ui
```

The dashboard can:

- launch one experiment at a time
- stop the active process
- choose opponent, timesteps, seed, replay frequency, shaped reward, and checkpoint glob
- graph live `metrics.jsonl` values such as reward, episode length, FPS, and train losses when available
- open the latest replay for a selected run
- record a replay from a completed `final_model.zip`

Each UI run is written under:

```text
runs/ui_experiments/<experiment-name>/
```

Important files:

```text
metrics.jsonl
experiment.json
final_model.zip
best/best_model.zip
replays/*.json
```

Record a replay from any checkpoint manually:

```bash
.venv/bin/barricade-record-model-game \
  --model runs/maskable_ppo_barricade/best/best_model.zip \
  --out runs/manual_replay.json \
  --opponent random
```

## PettingZoo Wrapper

Install the optional multi-agent dependency:

```bash
.venv/bin/python -m pip install -e '.[multiagent]'
```

Then use the AEC-style wrapper:

```python
from barricade_rl import BarricadeAECEnv

env = BarricadeAECEnv()
env.reset(seed=0)
agent = env.agent_selection
obs = env.observe(agent)
action = obs["action_mask"].nonzero()[0][0]
env.step(int(action))
```

This wrapper is for the next self-play phase. The single-agent SB3 path remains the fastest route for laptop experiments.

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
- PettingZoo self-play wrapper.
- Checkpointing and opponent pools.
- Optional shaped rewards.

## Suggested Next Steps

1. Run longer baseline experiments: random, mixed, shaped mixed, checkpoint pool.
2. Compare checkpoints with model evaluation plus replay summaries.
3. Add a richer opponent-pool scheduler instead of uniform random checkpoint sampling.
4. Move from the AEC wrapper scaffold to full PettingZoo self-play training.
5. Optimize BFS/path checks further if long runs are still CPU-bound.
