# Barricade RL

Barricade RL is a Python project for building and testing a reinforcement-learning environment for the board game Barricade. The goal is to keep the training environment small and fast while also providing a human-playable UI so the rules can be checked before spending compute on training.

For a detailed learning-oriented explanation of the full workflow and design rationale, read [architecture.md](architecture.md).

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
  test_experiments.py # Experiment runner tests
  test_pettingzoo_env.py # AEC wrapper tests
  test_replay.py # Replay tests
  test_training.py # Training smoke tests
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

Open the experiment dashboard:

```powershell
.\.venv\Scripts\barricade-experiment-ui.exe
```

Run the default experiment set:

```powershell
.\.venv\Scripts\barricade-experiments.exe --root runs\experiments --timesteps 10000 --seed 0
```

Record one model replay:

```powershell
.\.venv\Scripts\barricade-record-model-game.exe --model runs\maskable_ppo_barricade\best\best_model.zip --out runs\manual_replay.json
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
- `anti_rush_lite` for a softer walling opponent
- `anti_rush` for a stronger walling opponent

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

By default, the learner is player `0` in the single-agent training path. Use `--randomize-learner-side` to train from both sides. Rewards are always learner-relative, so a learner win is `+1` whether the model is playing as player `0` or player `1`.

The default trainer uses `MlpPolicy`, which flattens the small `(6, 9, 9)` observation. You can now switch to a compact board CNN:

```bash
.venv/bin/barricade-train-maskable-ppo --timesteps 10000 --opponent mixed --policy cnn
```

The CNN uses small 3x3 convolutions over the six board planes, then feeds the result into MaskablePPO's policy and value heads. This gives the model a better chance to learn local board patterns such as blocked lanes and wall shapes.

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

Run foundation CNN training before self-play:

```bash
.venv/bin/barricade-train-maskable-ppo \
  --timesteps 50000 \
  --opponent curriculum \
  --policy cnn \
  --randomize-learner-side \
  --wall-penalty 0.05 \
  --reverse-move-penalty 0.02 \
  --progress-reward-scale 0.03 \
  --out runs/cnn_foundation
```

The foundation stage is deliberately not self-play. Its job is to teach basic racing behavior: move toward the goal, avoid wasting walls, and beat random/greedy opponents from both sides. Move to self-play only after `eval_greedy_p0_win_rate` and `eval_greedy_p1_win_rate` are consistently useful.

Run the light anti-rush stage after the foundation stage is working:

```bash
.venv/bin/barricade-train-maskable-ppo \
  --timesteps 50000 \
  --opponent curriculum_stage2 \
  --policy cnn \
  --initial-model runs/ui_experiments/cnn_foundation/final_model.zip \
  --randomize-learner-side \
  --wall-penalty 0.03 \
  --reverse-move-penalty 0.02 \
  --progress-reward-scale 0.02 \
  --out runs/cnn_anti_rush_lite
```

This stage continues from the foundation model and introduces `anti_rush_lite`, a softer walling opponent. Its job is to teach basic defensive-wall timing without immediately jumping to the full anti-rush opponent. Watch `eval_anti_rush_lite_p0_win_rate` and `eval_anti_rush_lite_p1_win_rate`, but also make sure `eval_greedy_*` does not collapse.

Run the bridge anti-rush stage after the lite stage is still strong against greedy:

```bash
.venv/bin/barricade-train-maskable-ppo \
  --timesteps 50000 \
  --opponent curriculum_stage3_bridge \
  --policy cnn \
  --initial-model runs/ui_experiments/cnn_anti_rush_lite/best/best_model.zip \
  --randomize-learner-side \
  --wall-penalty 0.03 \
  --reverse-move-penalty 0.02 \
  --progress-reward-scale 0.02 \
  --out runs/cnn_anti_rush_bridge
```

This stage uses `anti_rush_medium`: the same trigger distance and self-cost allowance as full anti-rush, but with a lower wall frequency. It is a targeted bridge between `anti_rush_lite` and full `anti_rush`. It uses the best lite checkpoint because the final lite checkpoint overtrained against the lite curriculum.

Run the shaped anti-rush bridge when sparse win/loss reward is not moving full anti-rush above zero:

```bash
.venv/bin/barricade-train-maskable-ppo \
  --timesteps 50000 \
  --opponent curriculum_stage3_bridge \
  --policy cnn \
  --initial-model runs/ui_experiments/cnn_anti_rush_lite/best/best_model.zip \
  --randomize-learner-side \
  --wall-penalty 0.03 \
  --reverse-move-penalty 0.02 \
  --progress-reward-scale 0.02 \
  --survival-reward 0.004 \
  --opponent-wall-value-penalty-scale 0.04 \
  --out runs/cnn_anti_rush_shaped
```

This adds two anti-rush-specific hints. `--survival-reward` gives a tiny reward for surviving the opponent turn. `--opponent-wall-value-penalty-scale` subtracts reward when an opponent wall increases the learner's shortest path. In plain terms, it rewards lasting longer while teaching the model that a good response makes the opponent's wall less valuable.

Run the endgame anti-rush curriculum when the model still cannot learn from normal opening positions:

```bash
.venv/bin/barricade-train-maskable-ppo \
  --timesteps 50000 \
  --opponent curriculum_stage3_bridge \
  --policy cnn \
  --initial-model runs/ui_experiments/cnn_anti_rush_shaped/best/best_model.zip \
  --randomize-learner-side \
  --wall-penalty 0.03 \
  --reverse-move-penalty 0.02 \
  --progress-reward-scale 0.02 \
  --survival-reward 0.004 \
  --opponent-wall-value-penalty-scale 0.04 \
  --endgame-start-probability 0.30 \
  --out runs/cnn_anti_rush_endgame
```

`--endgame-start-probability 0.30` means about 30% of training episodes start with the learner already 2-4 moves from goal. This is backward curriculum: instead of asking the model to discover rare anti-rush situations from the opening, it practices the failing phase directly.

The older `cnn anti-rush gentle` preset is still available for comparison, but the bridge preset is the recommended next run because the gentle stage did not move full anti-rush above 0%.

Run the harder full anti-rush stage only after the bridge stage improves full anti-rush results:

```bash
.venv/bin/barricade-train-maskable-ppo \
  --timesteps 50000 \
  --opponent curriculum_stage3 \
  --policy cnn \
  --initial-model runs/ui_experiments/cnn_anti_rush_bridge/best/best_model.zip \
  --randomize-learner-side \
  --wall-penalty 0.03 \
  --reverse-move-penalty 0.02 \
  --progress-reward-scale 0.02 \
  --out runs/cnn_anti_rush
```

Run refreshing self-play after the stronger anti-rush stage is working:

```bash
.venv/bin/barricade-train-maskable-ppo \
  --timesteps 250000 \
  --opponent curriculum_stage3 \
  --policy cnn \
  --initial-model runs/ui_experiments/cnn_anti_rush/final_model.zip \
  --self-play \
  --self-play-save-freq 5000 \
  --randomize-learner-side \
  --checkpoint-probability 0.35 \
  --wall-penalty 0.03 \
  --reverse-move-penalty 0.02 \
  --progress-reward-scale 0.03 \
  --checkpoint-opponents "runs/ui_experiments/*/best/*.zip" \
  --out runs/cnn_self_play
```

Self-play saves snapshots under:

```text
runs/cnn_self_play/self_play_pool/checkpoint_<timesteps>.zip
```

During self-play, the opponent samples from the previous snapshots plus any checkpoint glob you supplied. `--checkpoint-probability 0.35` means about 35% of opponent turns come from checkpoints and about 65% still come from the scripted curriculum. This is intentionally gentle early on: the model still sees older versions of itself, but it is not crushed by checkpoint opponents before it has learned basic winning play.

The `curriculum` opponent samples several scripted styles:

```text
random: occasional noisy play
greedy: direct shortest-path racing
anti_rush: wall placements that try to lengthen the learner's path
anti_rush_lite: a softer anti-rush version used as the bridge into full anti-rush
anti_rush_medium: a harder bridge that uses full anti-rush timing but walls less often
mixed: the earlier random/greedy blend
```

The default curriculum is now a foundation curriculum: mostly greedy play, some random play, a small amount of mixed play, and no anti-rush pressure by default. Anti-rush still exists for evaluation and later experiments, but the base curriculum focuses on learning the race before learning heavy wall defense.

`curriculum_stage2` is the light anti-rush curriculum. It keeps greedy play as the majority opponent but adds `anti_rush_lite` exposure:

```text
random: 15%
greedy: 60%
anti_rush_lite: 15%
mixed: 10%
```

`curriculum_stage3_bridge` is the targeted bridge from lite anti-rush to full anti-rush:

```text
random: 12%
greedy: 58%
anti_rush_medium: 20%
mixed: 10%
```

`curriculum_stage3_gentle` is the safer bridge from lite anti-rush to full anti-rush:

```text
random: 15%
greedy: 67%
anti_rush: 8%
mixed: 10%
```

`curriculum_stage3` increases the full anti-rush share:

```text
random: 10%
greedy: 55%
anti_rush: 20%
mixed: 15%
```

Use `--randomize-learner-side` for symmetric training. At the start of each episode the learner is randomly assigned to player `0` or player `1`. If the learner is player `1`, the scripted/checkpoint opponent makes the opening player `0` move first, then the model receives a canonical observation from player `1`'s perspective. Rewards are always from the learner's side.

Add optional shortest-path shaped rewards:

```bash
.venv/bin/barricade-train-maskable-ppo --timesteps 10000 --opponent mixed --shaped-reward
```

Sparse win/loss reward remains the default. Use shaped rewards as an experiment, not as the baseline.

Three extra reward controls are useful for CNN curriculum training:

```bash
--wall-penalty 0.05
--reverse-move-penalty 0.02
--progress-reward-scale 0.03
```

`--wall-penalty` subtracts a tiny reward when the learner places a wall. It does not make walls illegal; it just tells the model that a wall should have a purpose. `--reverse-move-penalty` subtracts a tiny reward when the learner immediately undoes its previous pawn move, such as `up` then `down`. `--progress-reward-scale` gives a tiny bonus when a pawn move shortens the learner's path to goal. These are training hints, not rule changes.

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

- `rollout/ep_rew_mean`: average reward in the current training environment. This answers "is the learner getting reward against the opponent distribution it is training on?"
- `rollout/ep_len_mean`: average episode length.
- `eval/mean_reward`: SB3's evaluation reward against the same evaluation environment used by the trainer.
- `train/entropy_loss`: policy randomness.
- `train/value_loss`: value function fit.

The project also writes explicit scripted evaluation win rates into `metrics.jsonl`:

```text
eval_random_p0_win_rate
eval_random_p1_win_rate
eval_greedy_p0_win_rate
eval_greedy_p1_win_rate
eval_mixed_p0_win_rate
eval_mixed_p1_win_rate
eval_anti_rush_lite_p0_win_rate
eval_anti_rush_lite_p1_win_rate
eval_anti_rush_p0_win_rate
eval_anti_rush_p1_win_rate
eval_balanced_win_rate
```

The `p0` and `p1` metrics are the clearest "is the learner actually winning from both sides?" metrics. `eval_balanced_win_rate` averages the side-specific win rates across the scripted evaluation opponents. It is a better headline metric than rollout reward when comparing checkpoints.

The trainer also saves replay files at milestones:

```text
runs/maskable_ppo_barricade/replays/replay_1000.json
runs/maskable_ppo_barricade/replays/replay_2000.json
...
```

When `--randomize-learner-side` is enabled, milestone replays are saved for both sides:

```text
runs/cnn_self_play/replays/replay_5000_p0.json
runs/cnn_self_play/replays/replay_5000_p1.json
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
- choose from the main experiment presets: `random`, `mixed`, `mixed + shaped reward`, `checkpoint pool`, `cnn foundation`, `cnn anti-rush lite`, `cnn anti-rush bridge`, `cnn anti-rush shaped`, `cnn anti-rush endgame`, `cnn anti-rush gentle`, `cnn anti-rush`, `cnn anti-rush+`, and `cnn self-play`
- still edit opponent, policy, timesteps, seed, replay frequency, shaped reward, self-play settings, learner-side randomization, wall/reverse-move/progress rewards, scripted evaluation settings, checkpoint glob, and initial model directly
- graph multiple live `metrics.jsonl` values at once, including training reward, scripted win rates, episode length, FPS, episodes, self-play pool size, and train losses when available
- change the selected graph metrics while training is running
- save the current graph as `graphs/metrics.svg` inside the selected run directory
- open the selected run directory in Finder, Explorer, or the Linux file manager
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
self_play_pool/checkpoint_*.zip
replays/*.json
graphs/metrics.svg
```

`metrics.jsonl` is the dashboard's live data source. It is written after rollouts and may include:

```text
timesteps
fps
episodes
ep_rew_mean
train_env_ep_rew_mean
ep_len_mean
eval_random_p0_win_rate
eval_random_p1_win_rate
eval_greedy_p0_win_rate
eval_greedy_p1_win_rate
eval_mixed_p0_win_rate
eval_mixed_p1_win_rate
eval_anti_rush_lite_p0_win_rate
eval_anti_rush_lite_p1_win_rate
eval_anti_rush_p0_win_rate
eval_anti_rush_p1_win_rate
eval_balanced_win_rate
train_loss
train_value_loss
train_entropy_loss
train_policy_gradient_loss
train_survival_reward_mean
train_opponent_wall_value_delta_mean
train_opponent_wall_value_reward_mean
train_endgame_start_mean
self_play_pool_size
```

`train_env_ep_rew_mean` is an alias for the rollout reward so the dashboard label is explicit. Use `eval_balanced_win_rate` plus the `eval_*_p0_win_rate` and `eval_*_p1_win_rate` metrics when comparing what the model does from each side against fixed opponents. The dashboard writes `graphs/metrics.svg` automatically whenever it has enough data to draw the selected metrics. When several metrics are selected, each line is scaled independently. That means reward, FPS, win rate, and loss can be viewed together even though they use very different numeric ranges.

For initial testing, use short runs that prove the pipeline works before spending time on longer comparisons:

```text
random: 25,000 timesteps
mixed: 25,000 timesteps
mixed + shaped reward: 25,000 timesteps
checkpoint pool: 25,000 timesteps after at least one earlier run has produced a checkpoint
cnn foundation: 50,000 timesteps for a first smoke test, then 100,000+ if greedy win rates are improving
cnn anti-rush lite: 50,000 timesteps after a good foundation checkpoint exists
cnn anti-rush bridge: 50,000 timesteps from the lite best checkpoint
cnn anti-rush shaped: 50,000 timesteps from the lite best checkpoint when bridge stays at 0% full anti-rush
cnn anti-rush endgame: 50,000 timesteps from the shaped best checkpoint
cnn anti-rush gentle: optional comparison from the lite best checkpoint
cnn anti-rush: 50,000 timesteps after the bridge stage improves full anti-rush
cnn anti-rush+: optional extra hardening after full anti-rush starts improving
cnn self-play: only after the full anti-rush model keeps greedy wins and improves anti-rush wins
```

These are smoke-test lengths. They are enough to check that metrics, replays, checkpoints, and the UI are behaving, but not enough to judge the best training strategy. For the first meaningful comparison, rerun the same four presets at around `100,000` timesteps each. Use the checkpoint pool after you have a usable checkpoint glob, for example:

```text
runs/ui_experiments/*/best/*.zip
```

Record a replay from any checkpoint manually:

```bash
.venv/bin/barricade-record-model-game \
  --model runs/maskable_ppo_barricade/best/best_model.zip \
  --out runs/manual_replay.json \
  --opponent random \
  --learner-side 1
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
- PettingZoo AEC-style wrapper.
- Checkpointing and opponent pools.
- Compact CNN policy option.
- Refreshing self-play checkpoint pool.
- Foundation curriculum opponent focused on random/greedy racing.
- Light anti-rush curriculum and initial-model continuation.
- Full anti-rush and stronger anti-rush continuation stages.
- Optional shaped rewards.
- Optional wall-use, reverse-move, and forward-progress training hints.
- Experiment runner and local dashboard.

## Suggested Next Steps

1. Run the new `cnn anti-rush endgame` preset for `50,000` timesteps using `runs/ui_experiments/cnn_anti_rush_shaped/best/best_model.zip` as the initial model.
2. Check that `eval_greedy_p0_win_rate` and `eval_greedy_p1_win_rate` stay above roughly 80% while `eval_anti_rush_p0_win_rate` and `eval_anti_rush_p1_win_rate` start moving above zero.
3. Inspect p0/p1 replays for purposeful wall responses rather than renewed wall spam.
4. Use the best anti-rush checkpoint as the initial model for `cnn self-play`.
5. Move from the AEC wrapper scaffold to full PettingZoo self-play training once the single-agent staged curriculum is stable.
