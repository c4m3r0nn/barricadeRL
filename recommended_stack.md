Recommended stack
Language: Python
Best ecosystem for RL, Gym-style environments, PyTorch, logging, and experimentation.
Environment API: Gymnasium
Build the game as a custom gymnasium.Env.
Gymnasium has official guidance for custom environments and action masking, which is important here because many wall placements and pawn moves will be illegal in a given state.
Multi-agent wrapper: PettingZoo
Use this once you want proper self-play.
Barricade is a turn-based two-player game, and PettingZoo’s AEC API is designed for sequential multi-agent environments. Its docs also explicitly support action masks.
ML framework: PyTorch
Use PyTorch as the model/training backend.
On Apple Silicon, PyTorch can use Apple’s MPS backend for GPU acceleration through Metal.
First RL algorithm library: Stable-Baselines3 + sb3-contrib
Start with MaskablePPO from sb3-contrib.
This is a strong fit because your action space needs legality masking: pawn moves, jumps, wall placements, and path-preservation constraints. sb3-contrib documents MaskablePPO specifically for invalid action masking.
Research / custom training later: CleanRL
Once the environment is correct, move to a custom PPO based on CleanRL if you want better control over self-play, opponent pools, curriculum learning, or custom rewards.
CleanRL is useful because it provides compact, single-file RL implementations that are easier to modify than large frameworks.
Scaling later: Ray RLlib
Only use RLlib if you later want distributed training, many parallel environments, or experiment orchestration.
RLlib is powerful but heavier than you need for a first laptop-based version. It uses Gymnasium as its main single-agent environment interface.
Best practical setup for your MacBook
Core stack
python
gymnasium
pettingzoo
torch
stable-baselines3
sb3-contrib
numpy
pytest
tensorboard or wandb
Environment implementation
Write the game rules in plain Python + NumPy.
Use a compact board representation:
Pawn positions as integer coordinates.
Horizontal walls as an 8×8 boolean array.
Vertical walls as an 8×8 boolean array.
Wall counts as small integers.
Implement BFS manually for “both players must still have a path.”
Avoid rendering during training.
Training implementation
Start with single-agent framing:
Current learner plays against a random, greedy, or scripted opponent.
Then move to self-play:
Learner vs previous checkpoint.
Learner vs pool of older checkpoints.
Learner vs itself.
M4 Pro-specific advice
Use PyTorch with mps available, but benchmark both CPU and MPS.
For small RL networks, the environment stepping can dominate runtime, and CPU training may sometimes be as fast or faster.
MPS becomes more useful once the network or batch size is large enough.
Use many vectorized CPU environments.
Your M4 Pro has strong CPU performance, and this kind of board-game environment is usually CPU-bound.
Parallel rollout collection will probably matter more than raw GPU speed early on.
Keep the neural net modest.
A small CNN or MLP is enough initially.
Example:
Input: board planes.
Trunk: 2–4 small convolution layers.
Policy head: 132 logits.
Value head: scalar value.
Suggested architecture
Observation
Tensor planes, for example:
Current player pawn.
Opponent pawn.
Horizontal walls.
Vertical walls.
Current player walls remaining.
Opponent walls remaining.
Optional shortest-path maps.
Action space
Discrete(132)
4 pawn movement actions.
128 wall placement actions.
Provide an action mask every step.
Algorithm
Start with:
MaskablePPO
action masking
sparse reward: +1, -1, 0
Then try:
shaped reward using shortest path difference.
self-play checkpoints.
curriculum from smaller boards, such as 5×5, then 7×7, then 9×9.
Recommended path
Phase 1: Implement BarricadeEnv(gymnasium.Env) with full legality tests.
Phase 2: Train MaskablePPO against random/scripted opponents.
Phase 3: Wrap with PettingZoo for cleaner self-play.
Phase 4: Move to a custom CleanRL-style PPO loop when you need opponent pools and more control.
Phase 5: Consider RLlib only if you outgrow the MacBook workflow.
I would avoid initially
MLX for this project, despite being excellent on Apple Silicon. MLX is Apple’s NumPy-like ML framework optimized for Apple Silicon, but the RL ecosystem around Gymnasium, PettingZoo, and SB3 is much stronger in PyTorch today.
Complex distributed tooling too early.
Get correctness and self-play working first.
Board-game RL usually fails first from environment bugs, reward design, or weak self-play setup, not from lack of compute.