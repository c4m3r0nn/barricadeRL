# Barricade RL Architecture

This document explains how the Barricade RL project is structured, why it is structured this way, and how the pieces fit together. It is written as a learning example: the goal is not only to describe the code, but also to explain the reasoning behind each design decision.

## Project Goal

The project has two jobs:

1. Let a human play Barricade in a UI so the game rules can be checked.
2. Provide a fast reinforcement-learning environment so agents can learn the game.

Those two jobs need different things. A human UI needs buttons, drag-and-drop walls, colors, and replay controls. A training environment needs to step thousands or millions of times without wasting time drawing anything. The architecture keeps these concerns separate.

A useful real-life analogy is a car factory. The factory floor has machines optimized for production speed. The showroom has lighting, signs, and polished displays. They both deal with the same car, but they should not be the same room. In this project, `core.py` is the car, `env.py` and `single_agent.py` are the factory floor, and `ui_tk.py` is the showroom.

## High-Level Workflow

The normal project workflow is:

```text
1. Implement rules in the core game engine.
2. Verify rules with tests and the playable UI.
3. Wrap the core engine as a Gymnasium environment.
4. Train with MaskablePPO and action masks.
5. Save metrics, checkpoints, and replay files.
6. Evaluate trained checkpoints.
7. Compare experiments in the dashboard.
8. Move to stronger opponents, checkpoint pools, and PettingZoo self-play.
```

Each step builds on the previous one. The project intentionally avoids jumping straight to complex self-play. In board-game RL, bad rules or bad evaluation can waste far more time than a small neural network.

## Core Rule Engine

File:

```text
barricade_rl/core.py
```

The core engine owns the real game rules:

- pawn positions
- wall positions
- wall counts
- current player
- legal movement
- jumping
- wall placement
- path preservation
- win detection

The main class is:

```python
BarricadeGame
```

It contains a `BarricadeState`, which stores the board:

```text
pawns: two (row, col) coordinates
h_walls: 8x8 boolean array
v_walls: 8x8 boolean array
walls_remaining: [player_0_walls, player_1_walls]
current_player: 0 or 1
winner: None, 0, or 1
```

This representation is deliberately small. It is like writing a shopping list instead of drawing a picture of the kitchen. The training loop needs the facts, not the decoration.

## Board Representation

The pawn board is a 9x9 grid of squares.

Walls use two 8x8 grids:

```text
h_walls[row, col]
v_walls[row, col]
```

Each wall anchor represents a continuous two-edge wall. A horizontal wall blocks movement across a row boundary for two adjacent columns. A vertical wall blocks movement across a column boundary for two adjacent rows.

Why not store every blocked edge directly?

That would also work, but the 8x8 wall-anchor representation matches the action space cleanly:

```text
64 horizontal wall actions
64 vertical wall actions
```

The code can still ask, "Can this pawn cross from square A to square B?" through `can_cross()`.

## Action Space

The training action space is:

```text
Discrete(132)
```

That means the model chooses one integer from 0 to 131.

The mapping is:

```text
0   move up
1   move down
2   move left
3   move right
4-67    horizontal wall placements
68-131  vertical wall placements
```

This compact action space is easier for the model than a large menu of every possible jump and diagonal move. Jumping is handled dynamically. For example, if "move up" means "jump over the opponent" in the current position, the core engine maps action `0` to that jump.

A real-life example is a keyboard shortcut. Pressing the right arrow in a text editor might move one character, jump over a selected block, or do nothing at the end of the line. The key is the same, but the context decides the result.

## Action Masks

An action mask is a list of true/false values saying which actions are legal.

Example:

```text
[true, false, true, true, true, ...]
```

If action 1 is false, the model should not choose it.

This matters because most wall placements are illegal in many board states. Teaching the model through punishment would waste training time. It would be like teaching someone chess by letting them try illegal moves for weeks. The better method is to show only legal options.

The core method is:

```python
BarricadeGame.legal_actions_mask()
```

Gymnasium and SB3 access this through:

```python
env.action_masks()
```

## Canonical Perspective

The observation is usually encoded from the current player's perspective.

Player 0 starts at the bottom and moves upward. Player 1 starts at the top and moves downward. Without canonical encoding, the model has to learn the same idea twice:

```text
player 0 wants to go up
player 1 wants to go down
```

Canonical encoding rotates player 1's view so the active player always appears to move "up".

This is like rotating a map so "the way forward" is always at the top. Drivers do this with GPS maps all the time.

Important detail: actions must be rotated too. If player 1's canonical action says "move up", the real board action is "move down". The function:

```python
canonical_action_to_absolute()
```

performs that conversion.

## Observations

The model sees a tensor shaped:

```text
(6, 9, 9)
```

Think of this as six transparent sheets stacked on top of the board:

```text
Plane 1: current player pawn
Plane 2: opponent pawn
Plane 3: horizontal walls
Plane 4: vertical walls
Plane 5: current player walls remaining
Plane 6: opponent walls remaining
```

This is similar to how weather maps can layer temperature, wind, rain, and pressure over the same geography.

## Speed Optimizations

The most expensive rule is wall legality. A wall is legal only if both players still have a path to their goal. Checking that requires graph search.

The project uses BFS for pathfinding.

BFS means breadth-first search. It explores nearby squares first, then farther squares. If you imagine searching a building for an exit, BFS means checking every room one step away, then every room two steps away, and so on.

Current speed improvements:

1. Legal action masks are cached per exact state.
2. Shortest-path results are cached per exact state and player.
3. `apply_action()` can reuse cached wall legality after a mask has already been computed.

This avoids asking the same expensive question repeatedly.

Real-life example: if you already checked that the front door is locked, you do not need to check it again one second later unless someone changed the lock or opened the door.

The cache key includes:

```text
pawn positions
wall grids
wall counts
current player
winner
move count
```

If the state changes, the cache is no longer used.

## Gymnasium Environments

Gymnasium is a standard Python API for reinforcement-learning environments.

An environment provides:

```python
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

Key terms:

- `obs`: what the agent sees
- `reward`: the score signal
- `terminated`: the game ended naturally, such as a win
- `truncated`: the game stopped because of a limit, such as max moves
- `info`: extra debugging data

The project has two Gymnasium-style wrappers.

## Two-Player Turn Environment

File:

```text
barricade_rl/env.py
```

Class:

```python
BarricadeEnv
```

This is a direct wrapper around the two-player game. The active player changes every step.

It is useful for validating the core environment, but it is not the first training path because Stable-Baselines3 is mainly built around single-agent training.

## Single-Agent Training Environment

File:

```text
barricade_rl/single_agent.py
```

Class:

```python
BarricadeSingleAgentEnv
```

This is the main first training environment.

The learner is always player 0. After the learner acts, a scripted or checkpoint opponent automatically acts as player 1. Then the observation returns to the learner.

This gives Stable-Baselines3 a normal single-agent problem:

```text
learner acts
environment handles opponent
learner receives reward
learner acts again
```

This is like practicing tennis against a ball machine. You still learn the game, but the machine handles the other side.

## Rewards

Default rewards are sparse:

```text
win: +1
loss: -1
non-terminal step: 0
```

Sparse means the model gets feedback only at important moments. This is clean but can be slow.

Optional shaped rewards can be enabled:

```bash
barricade-train-maskable-ppo --shaped-reward
```

Shaped reward means adding small hints. In this project, the hints are based on shortest path changes:

```text
small bonus if learner path gets shorter
small penalty if opponent path gets shorter
```

This is like a coach saying, "Good, you moved closer to the goal," even before the game is won.

Shaping is optional because it can distort learning. If the hint is too strong, the model may optimize "short path" instead of "win the game."

The trainer also has small optional action-level training hints:

```bash
barricade-train-maskable-ppo --wall-penalty 0.05 --reverse-move-penalty 0.02 --progress-reward-scale 0.03
```

The wall penalty is like charging a small price for each wall. A useful wall is still worth playing, but the model is discouraged from spending all walls just because wall actions look important. The reverse-move penalty is aimed at pawn oscillation. If the model moves `up` and immediately moves `down`, it pays a small cost because it has gone nowhere.

The progress reward is a small bonus when a pawn move shortens the learner's shortest path to goal. In real-life terms, this is like rewarding a beginner for walking toward the finish line before asking them to master advanced blocking tactics.

These penalties are not game rules. They are training hints. The playable UI and the core rules still allow legal wall placements and legal backtracking.

Anti-rush training also has two targeted hints:

```bash
barricade-train-maskable-ppo --survival-reward 0.004 --opponent-wall-value-penalty-scale 0.04
```

The survival reward is a tiny bonus for getting through the opponent turn without losing. The opponent wall value penalty measures how much an opponent wall increased the learner's shortest path and subtracts a small amount. In real-life terms, if someone blocks your route, the training signal asks: "Did that block actually hurt you?" If it did, the model pays a small cost.

These hints are meant for the anti-rush cliff only. They should stay small because the real goal is still winning the game, not merely surviving forever.

The trainer can also use endgame starts:

```bash
barricade-train-maskable-ppo --endgame-start-probability 0.30
```

This means some episodes do not begin from the normal opening. Instead, the learner starts 2-4 moves from goal with both players still having legal paths. This is called backward curriculum because training begins closer to the hard ending and later connects that skill back to full games.

In real-life terms, it is like practicing penalty kicks directly instead of only playing full matches and hoping enough penalty situations appear.

## Opponents

File:

```text
barricade_rl/opponents.py
```

Implemented opponents:

```text
RandomOpponent
GreedyOpponent
MixedOpponent
AntiRushLiteOpponent
AntiRushMediumOpponent
AntiRushOpponent
CurriculumOpponent
StageTwoCurriculumOpponent
StageThreeBridgeCurriculumOpponent
StageThreeGentleCurriculumOpponent
StageThreeCurriculumOpponent
CheckpointPoolOpponent
RefreshingCheckpointPoolOpponent
```

`RandomOpponent` chooses a random legal move.

`GreedyOpponent` tries to reduce its shortest path to goal.

`MixedOpponent` randomly alternates between random and greedy behavior.

`AntiRushLiteOpponent` is a softer walling opponent. It only considers walls when the runner is closer to goal, it refuses walls that slow itself down, and it only chooses a wall some of the time. In real-life terms, this is like practicing against a cautious defender before facing someone who blocks aggressively every time.

`AntiRushMediumOpponent` uses the full anti-rush timing and self-cost rules but does not wall every time. In real-life terms, it is a sparring partner who knows the real defensive pattern but does not apply maximum pressure on every turn.

`AntiRushOpponent` uses walls only when the learner is close enough to goal and the wall meaningfully increases the learner's path without hurting the opponent too much. This keeps it from teaching wall-heavy play at the start of every game.

`CurriculumOpponent` samples from random, greedy, anti-rush, and mixed opponents. The current default is a foundation curriculum: mostly greedy, some random, a little mixed, and no anti-rush pressure. This is like learning to race cleanly before adding a defender.

`StageTwoCurriculumOpponent` is the next lesson after foundation training. It keeps greedy play as the main opponent but introduces `AntiRushLiteOpponent`. This is like adding a moderate defender after the player can already run the basic route.

`StageThreeBridgeCurriculumOpponent` uses `AntiRushMediumOpponent` as targeted drills. This is the current bridge from lite anti-rush to full anti-rush, because a small amount of full anti-rush was still too hard to learn from directly.

`StageThreeGentleCurriculumOpponent` is the safer bridge from lite anti-rush to full anti-rush. It keeps greedy play dominant and adds only a small amount of full anti-rush. This is like letting a learner face a serious defender for a few drills per session instead of making every drill high pressure.

`StageThreeCurriculumOpponent` raises full anti-rush pressure further while keeping greedy play as the majority opponent. It is meant for continuing from a successful gentle-stage checkpoint, not for training from scratch.

`CheckpointPoolOpponent` loads trained model checkpoints and samples one as the opponent.

`RefreshingCheckpointPoolOpponent` watches checkpoint glob patterns during training, so new self-play snapshots can become opponents without restarting the run.

The checkpoint pool is the first step toward self-play. Self-play means a model improves by playing against current or older versions of itself.

A real-life example is a chess player reviewing and playing against their own earlier games. The older versions are not perfect, but they provide increasingly relevant practice.

## Training

File:

```text
barricade_rl/train_maskable_ppo.py
```

The first training algorithm is:

```text
MaskablePPO
```

PPO means Proximal Policy Optimization. It is a reinforcement-learning method that updates the policy gradually instead of making huge unstable changes.

MaskablePPO is PPO with invalid-action masking. That makes it a good fit for Barricade because illegal actions are common.

The current model policy is:

```text
MlpPolicy
```

MLP means multilayer perceptron, a standard feed-forward neural network. It flattens the board tensor into numbers. This is simple and good for smoke training.

The project intentionally delays custom CNN architecture work. CNN means convolutional neural network, which is often good for spatial board data. That may help later, but the current priority is a reliable experiment loop.

## Training Outputs

A training run writes files under a run directory:

```text
final_model.zip
best/best_model.zip
metrics.jsonl
eval/evaluations.npz
tb/
replays/
```

`final_model.zip` is the model at the end of training.

`best/best_model.zip` is the best evaluation checkpoint.

`metrics.jsonl` is a line-by-line metrics file for the dashboard.

JSONL means JSON Lines. Each line is one JSON object. This makes it easy to append data while training is still running.

Example:

```json
{"timesteps": 512, "fps": 250.0, "ep_rew_mean": 0.2}
{"timesteps": 1024, "fps": 245.0, "ep_rew_mean": 0.4}
```

This is like a lab notebook where each row records one measurement.

## Experiment Runner

File:

```text
barricade_rl/experiments.py
```

The experiment runner defines:

```python
ExperimentSpec
```

An experiment spec says:

```text
name
timesteps
opponent
seed
whether shaped reward is enabled
checkpoint opponent patterns
initial model to continue from
replay frequency
```

This allows repeatable experiments. Instead of remembering a long command from yesterday, the experiment config records the important details.

Run default experiments:

```bash
barricade-experiments --root runs/experiments --timesteps 10000 --seed 0
```

## Experiment Dashboard

File:

```text
barricade_rl/experiment_ui.py
```

Start it with:

```bash
barricade-experiment-ui
```

The dashboard lets you:

- launch an experiment
- stop the active process
- choose opponent and training settings
- view live graphs from `metrics.jsonl`
- open the latest replay
- record a replay from a completed model

This is intentionally local and simple. It is not meant to replace TensorBoard or a full experiment platform. It is a control panel for laptop development.

Real-life example: TensorBoard is like a detailed medical chart. The dashboard is like the monitor beside the bed showing the main vital signs.

## Replay System

File:

```text
barricade_rl/replay.py
```

Replays are saved as JSON files. A replay contains:

```text
metadata
frames
```

Each frame stores a board state:

```text
pawns
walls
wall counts
current player
winner
move count
actions
reward
```

The same Tkinter UI can open a replay:

```bash
barricade-play --replay runs/.../replay_1000.json
```

This matters because metrics can say a model is improving, but a replay shows how it is improving. A model might win by clever pathing, or it might win because the opponent made silly mistakes. Replays reveal that.

## Evaluation

File:

```text
barricade_rl/evaluate.py
```

Evaluation answers the question:

```text
How good is this model against a fixed opponent?
```

Example:

```bash
barricade-evaluate-model --model runs/.../best_model.zip --episodes 100 --opponent random
```

Evaluation reports:

```text
wins
losses
truncations
win rate
loss rate
truncation rate
average game length
wall usage
```

This is separate from training because training reward is noisy. Evaluation is the scoreboard.

## PettingZoo Wrapper

File:

```text
barricade_rl/pettingzoo_env.py
```

PettingZoo is a standard API for multi-agent reinforcement learning.

Multi-agent means more than one learning agent acts in the environment. In Barricade, player 0 and player 1 can both be agents.

AEC means Agent Environment Cycle. It is PettingZoo's turn-based API:

```text
agent 0 observes
agent 0 acts
agent 1 observes
agent 1 acts
repeat
```

This matches Barricade naturally because players alternate turns.

The project provides:

```python
BarricadeAECEnv
raw_env()
env()
```

The wrapper passes PettingZoo's API test. It returns dictionary observations:

```python
{
    "observation": board_tensor,
    "action_mask": legal_actions
}
```

PettingZoo warns that observations are dictionaries, but this is expected for action-masked environments.

This wrapper is ready for future self-play training, but the current main training path is still single-agent SB3 because it is simpler and easier to measure.

## Testing Strategy

Tests live in:

```text
tests/
```

The project uses red-green TDD:

1. Write a failing test.
2. Implement the smallest fix.
3. Run tests until green.

Red-green TDD is like writing a checklist before fixing a machine. First you prove the machine is broken in a specific way. Then you fix it. Then you prove that exact problem is gone.

The tests cover:

- movement
- jumps
- wall legality
- path preservation
- rewards
- action masks
- replay serialization
- model evaluation
- training smoke runs
- experiment helpers
- PettingZoo API behavior

## Current Recommended Research Workflow

Use this sequence:

```text
1. Open the experiment dashboard.
2. Run the CNN foundation preset.
3. Run the CNN anti-rush lite preset from the foundation checkpoint.
4. Run the CNN anti-rush shaped preset from the lite best checkpoint.
5. Run the CNN anti-rush endgame preset from the shaped best checkpoint.
6. Run the CNN anti-rush preset from the endgame best checkpoint only if the endgame stage improves full anti-rush.
7. Evaluate all final and best models with the same episode count and seed.
8. Compare replay summaries.
9. Watch selected games in the replay UI.
10. Start self-play only after the scripted curriculum is stable.
11. Only then change model architecture.
```

This keeps the project honest. If architecture changes happen before the experiment loop is reliable, it becomes hard to know whether improvement came from the architecture or from noise.

## Why This Architecture

The most important design decision is separation of concerns.

```text
core.py: rules
env.py: Gymnasium interface
single_agent.py: SB3 training interface
opponents.py: opponent behavior
train_maskable_ppo.py: training
evaluate.py: measurement
replay.py: visual evidence
experiment_ui.py: local control panel
pettingzoo_env.py: future multi-agent training
```

Each part has one main responsibility.

Real-life example: a restaurant separates kitchen, menu, waiter, cashier, and accounting. They all support dinner, but mixing them into one job would be chaotic. This project does the same with game rules, training, evaluation, and UI.

## What Comes Next

The next major work should be:

1. Run controlled experiments with the dashboard.
2. Compare results with model evaluation and replay summaries.
3. Add a better checkpoint-pool scheduler.
4. Add architecture experiments, such as a small CNN.
5. Move from the PettingZoo wrapper to full PettingZoo self-play training.

The current architecture is ready for those steps because it now has:

- correct rules
- playable UI
- training loop
- evaluation loop
- replay inspection
- experiment dashboard
- PettingZoo-compatible wrapper
