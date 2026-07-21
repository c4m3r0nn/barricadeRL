# barricadeRL

This repository is being rebuilt as a pure-self-play AlphaZero system for standard two-player 9x9 Quoridor. The authoritative engineering specification is [quoridor_alphazero_handover_v2.md](quoridor_alphazero_handover_v2.md).

The previous MaskablePPO, dense-reward, tactical imitation, GUI, and PettingZoo-first implementation has been removed. It used a 132-action encoding, context-dependent jump actions, a mutable NumPy state, 180-degree canonicalization, and a 500-ply cap. Those choices conflict with the handover and would invalidate AlphaZero replay data.

## Current status

Milestones M0 and M1 are complete. M2 is in progress. Implemented foundation:

- Dependency-free Rust rules engine with a thin Python binding and correctness-first Python oracle.
- Immutable 20-byte state/transposition key.
- Fixed 140-action canonical action space with distinct moves, straight jumps, diagonal jumps, and wall placements.
- Exact jump, wall overlap/crossing, reachability, and instant-win rules.
- Stateless `Game` interface: initial state, legal actions, successor, terminal status/value, canonical observation, mirroring, and state keys.
- Six-plane mover-relative observation with vertical-only canonicalization.
- Left-right state and policy symmetry.
- 200-ply cap with zero value for capped games.
- Handwritten jump and wall-legality batteries, random-game invariants, deterministic replay checks, and native/Python-oracle differential verification.
- Independent full-game differential testing against [`pyquoridor` 0.0.5](https://github.com/playquoridor/python-quoridor).
- A committed perft corpus at depths 1 through 4 for the opening and five legally reconstructed midgames.

The remaining M2 milestone work is:

- Run proper self-play training, checkpoint gating, and the M2 acceptance evaluations against the exact validation corpus.

Latest local test verification (2026-07-14):

- 184 Python tests and 2 Rust tests passed.

Latest heavy rules verification (2026-07-07):

- 10,000 randomized invariant games passed across 1,756,757 plies.
- 10,000 native/Python-oracle states matched.
- 10,000 independent-oracle games matched across 267,303 plies and 277,303 states.
- Native random-playout throughput was 42,903 plies/second against a 20,000 target.

Generated data from the removed 132-action implementation is incompatible with this code. The current milestone is M2: the 5x5 solver, AlphaZero network, MCTS, replay buffer, and learner.

## M1 progress

Implemented:

- Registered Gymnasium environment `BarricadeRL/Quoridor-v0`.
- Automatic opponent reply inside each `step`.
- Legal masks in every `reset` and `step` info dictionary.
- Strict terminal rewards (`+1`, `-1`, or `0` at the ply cap), with no shaping.
- Frozen ladder version 1: random, greedy racer, heuristic-1, and alpha-beta depths 3 and 5.
- Deterministic evaluation harness with balanced colours, per-game records, cap-as-draw match scoring, and Elo estimates anchored at `random = 0`.
- Dashboard JSONL event schema plus a minimal HTML renderer exposing ladder Elo, average game length, cap fraction, and wall usage, with placeholders for the later learner metrics required by the handover.
- Dependency-free masked DQN smoke baseline trained through the Gym wrapper against random. It uses replay, a target network, legal-action masking before every argmax, terminal rewards from the wrapper, and a short greedy-racer behaviour warm start to make the sparse-reward M1 sanity check cheap.

M1 acceptance run on 2026-07-07:

- Command: `barricade-train-dqn --episodes 200 --expert-episodes 120 --evaluation-games-per-color 25 --seed 0`.
- Random-opponent evaluation: 46 wins, 0 losses, 4 draws over 50 games.
- Score rate: 0.96 against the 0.80 gate.
- Artifact path: `artifacts/m1/masked_dqn_smoke.npz` locally; artifacts are ignored by git.

## M2 progress

Implemented:

- Generic Python small-board rules surface for the solved-board curriculum, defaulting to 5x5 with 3 walls per player.
- 5x5 immutable `SmallState` with 12-byte keys.
- 5x5 action encoding: the same 12 pawn actions, then `(n-1)^2` horizontal walls and `(n-1)^2` vertical walls; for 5x5 this is 44 actions.
- Mover-canonical vertical flips, left-right mirror symmetry, six-plane observations, exact jumps, wall conflicts, path preservation, terminal/cap status, shortest-path distances, and perft.
- Bounded exact solver API returning `WIN`, `LOSS`, `DRAW`, or `UNKNOWN` for depth-limited proof searches.
- Opening 5x5 perft pinned at depths 1 through 3: `35`, `1109`, `31540`.
- Versioned M2 config at `configs/m2_5x5.json`, including the explicit decision to use board-size-dependent flat policy heads for M2/M3 transfer because the optional size-agnostic policy head was not implemented before M2.
- Budgeted 5x5 oracle corpus generator that writes JSONL labels with state keys, proof config, config hash, best action, distances, exact/unknown status, and node-budget exhaustion flags.
- Proof-number search oracle method with explicit proof/disproof numbers and budget exhaustion reporting.
- Reusable proof-number solved-state cache for exact win/loss labels, shared across corpus/refinement batches and persisted via JSONL without caching draws or budget-exhausted unknowns.
- Deterministic oracle sharding via `--shard-index` / `--shard-count`, preserving unsharded sample order through `record_index`.
- Deterministic shard merge mode via `--merge-from`, sorting by `record_index` and rejecting duplicate shard records before the validation corpus is consumed.
- Exact-corpus compaction via `--compact-exact-from`, combining independent phase runs, filtering inexact and terminal states, deduplicating state keys, reindexing records, and selecting balanced opening/midgame/endgame quotas.
- Validation-corpus audit mode via `--audit-corpus`, checking record count, config hash, duplicate state keys, terminal positions, exact-label coverage, phase distribution, label methods, and outcomes.
- Exact no-wall endgame solver for positions where both wall counters are zero, including the reduced no-wall initial race.
- Exact low-wall endgame solver for positions with a configured small number of unplaced walls remaining, plus `sampling=low-wall` corpus generation and hybrid-oracle dispatch before proof-number fallback.
- Tested PUCT MCTS with evaluator protocol, legal-logit masking, FPU reduction, root Dirichlet noise, forced root playouts, pruned visit-count policy targets, value sign flips, and immediate-win handling.
- AlphaZero network inference/checkpoint contract with batch-normalized global-pooling residual blocks, flat board-size-dependent policy head, opponent-policy head, tanh value head, auxiliary distance head, MCTS evaluator compatibility, and EMA checkpoint state.
- Versioned AlphaZero replay sample/buffer contract with policy-mask validation, value bounds, shortest-path and following-ply opponent-policy targets, run/config/git provenance, game and scoring metadata, true gradient-samples-per-generated-position accounting, batch sampling, and NPZ persistence.
- Self-play actor contract with mandatory 25/75 full/fast search randomization, full-search-only replay recording, the 16-ply temperature schedule, weak raw-policy opening diversification, pre-injection target invalidation, final mover-perspective values, and replay-buffer ingestion.
- PyTorch gradient learner covering the full trunk and all heads, combined policy/value/distance/opponent-policy/L2 loss, momentum SGD, fixed learning-rate drops, on-the-fly mirror augmentation, EMA updates, stale-replay overconsumption protection, resumable momentum checkpoints, and the `barricade-train-az` command.
- Deterministic checkpoint gate using 100 unique legal prefixes paired with colours swapped (200 noise-free 800-simulation games), promotion at a 0.55 score rate, full start-state audit metadata, permanent gated-checkpoint manifests, and a generic evaluation harness that supports the 5x5 action/wall contract.
- Correctness-first `barricade-run-az-cycle` coordinator: persisted cycle indices, independent deterministic seed streams, globally unique game IDs and artifact paths, self-play from the gated incumbent EMA network, replay persistence, bounded continuous learning, candidate checkpointing, gating, and promotion archival. Rejection leaves the self-play incumbent unchanged without discarding learner weights, momentum, EMA, step, or RNG state. Cycle records include the final-step policy, value, auxiliary, opponent-policy and L2 losses, root-policy entropy, learning rate, and replay-consumption ratio.
- Supervisor-compliant 200-ply cap with per-cycle cap telemetry and a run-latched automatic shortest-path adjudication switch on the cycle after three consecutive self-play cap fractions exceed 5%; every replay sample records the scoring scheme used.
- Training-readiness preflight via `barricade-training-readiness`, reporting oracle, replay, MCTS, network, self-play, and learner blockers before any proper training run.
- Handover compliance tests for M2 config constants, Gymnasium usage, masked softmax call sites, terminal reward/gamma choices, dashboard metrics, and the flat policy-head decision.
- Local held-out oracle artifact `artifacts/m2/oracle_5x5_exact_5000_cap200.jsonl`: 5,000 exact, non-terminal, unique positions under the 200-ply rules, balanced 1,667/1,667/1,666 across opening/midgame/endgame, with a passing strict audit and training-readiness preflight.

Not yet complete:

- Full 5x5 solver/proof-number or retrograde oracle for the entire solved state space.
- MCTS tree reuse, batched inference, transposition sharing, Gumbel mode, multi-process inference-service scaling, and M2 training acceptance.

The compacted 5x5 validation corpus now passes `--audit-corpus` and the full training-readiness preflight, so proper M2 training may begin. The synchronous cycle is suitable for correctness-first M2 runs; multi-process batched inference remains the scaling step for high-throughput training.

## Setup

Rust and Python 3.10+ are required.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,training]'
cargo build --release
```

## Tests and blocking verification

Fast development suite:

```bash
.venv/bin/python -m pytest -q
cargo test
```

Full M0 randomized and differential run:

```bash
.venv/bin/barricade-verify \
  --games 10000 \
  --differential-states 10000 \
  --external-games 10000 \
  --seed 0
```

Throughput gate:

```bash
.venv/bin/barricade-benchmark --plies 100000
```

The benchmark exits non-zero below 20,000 plies/second.

M1 ladder evaluation and dashboard skeleton:

```bash
.venv/bin/barricade-evaluate \
  --candidate greedy-racer \
  --games-per-color 2 \
  --seed 0 \
  --compact \
  --output artifacts/evaluations/greedy.json \
  --dashboard-events artifacts/dashboard/events.jsonl

.venv/bin/barricade-dashboard artifacts/dashboard/events.jsonl \
  --output artifacts/dashboard/index.html
```

M1 masked DQN smoke baseline:

```bash
.venv/bin/barricade-train-dqn \
  --episodes 200 \
  --expert-episodes 120 \
  --evaluation-games-per-color 25 \
  --seed 0 \
  --output artifacts/m1/masked_dqn_smoke.npz \
  --result-json artifacts/m1/masked_dqn_smoke.json \
  --dashboard-events artifacts/dashboard/events.jsonl
```

M2 budgeted 5x5 oracle corpus:

```bash
.venv/bin/barricade-generate-5x5-oracle \
  --config configs/m2_5x5.json \
  --output artifacts/m2/oracle_5x5_smoke.jsonl \
  --positions 128 \
  --random-plies 8 \
  --method proof-number \
  --sampling random \
  --max-depth 8 \
  --max-nodes 50000 \
  --seed 0
```

Generate independent exact phase pools with enough headroom for terminal-state filtering and deduplication:

```bash
.venv/bin/barricade-generate-5x5-oracle \
  --config configs/m2_5x5.json \
  --output artifacts/m2/oracle_200_scale/opening.jsonl \
  --positions 4200 \
  --random-plies 64 \
  --method hybrid \
  --sampling random \
  --max-depth 12 \
  --max-nodes 100000 \
  --low-wall-max-remaining 1 \
  --low-wall-max-nodes 500000 \
  --seed 801
```

Repeat with `--positions 6500 --random-plies 72 --seed 802` for `midgame.jsonl`, with `--positions 10000 --random-plies 136 --seed 803` for `endgame.jsonl`, and with `--positions 3000 --random-plies 136 --seed 804` for `endgame_extra.jsonl`. These near-boundary depths provide opening/midgame/endgame coverage under the 200-ply phase definition, with measured headroom for terminal filtering. If a long run is sharded, use distinct seeds or the existing `--shard-index` / `--shard-count` protocol; the compactor safely handles colliding source `record_index` values from independent runs.

Compact the independent pools into the held-out validation corpus:

```bash
.venv/bin/barricade-generate-5x5-oracle \
  --config configs/m2_5x5.json \
  --output artifacts/m2/oracle_5x5_exact_5000_cap200.jsonl \
  --compact-exact-from \
    artifacts/m2/oracle_200_scale/opening.jsonl \
    artifacts/m2/oracle_200_scale/midgame.jsonl \
    artifacts/m2/oracle_200_scale/endgame.jsonl \
    artifacts/m2/oracle_200_scale/endgame_extra.jsonl \
  --compact-records 5000
```

Audit it before any proper training run. The 1,600-per-phase floor leaves room for the 5,000-record balanced split of 1,667/1,667/1,666:

```bash
.venv/bin/barricade-generate-5x5-oracle \
  --config configs/m2_5x5.json \
  --audit-corpus artifacts/m2/oracle_5x5_exact_5000_cap200.jsonl \
  --audit-min-records 5000 \
  --audit-min-exact-fraction 1.0 \
  --audit-min-phase-records 1600
```

Then run the full training-readiness preflight:

```bash
.venv/bin/barricade-training-readiness \
  --config configs/m2_5x5.json \
  --oracle-corpus artifacts/m2/oracle_5x5_exact_5000_cap200.jsonl \
  --min-records 5000 \
  --min-exact-fraction 1.0 \
  --min-phase-records 1600
```

Initialize the reproducible step-zero incumbent for a fresh run:

```bash
.venv/bin/barricade-train-az \
  --config configs/m2_5x5.json \
  --steps 0 \
  --run-id m2-run-001 \
  --git-commit YOUR_GIT_COMMIT \
  --output artifacts/m2/m2-run-001/incumbent-step-000000000.npz
```

Once a self-play replay file exists, run a bounded learner phase with full provenance:

```bash
.venv/bin/barricade-train-az \
  --config configs/m2_5x5.json \
  --replay artifacts/m2/self_play_replay.npz \
  --steps 100 \
  --run-id m2-smoke-001 \
  --git-commit YOUR_GIT_COMMIT \
  --device cpu \
  --output artifacts/m2/checkpoints/m2-smoke-001.npz
```

Use `--resume` with a learner checkpoint to restore raw and EMA weights, momentum, step count, and RNG state. The command refuses to consume replay beyond the configured four gradient samples per generated position.

Run one complete self-play, learner, and gating cycle only after the oracle audit passes:

```bash
.venv/bin/barricade-run-az-cycle \
  --config configs/m2_5x5.json \
  --oracle-corpus artifacts/m2/oracle_5x5_exact_5000_cap200.jsonl \
  --incumbent artifacts/m2/m2-run-001/incumbent-step-000000000.npz \
  --output-directory artifacts/m2/m2-run-001 \
  --self-play-games 128 \
  --learner-steps 1 \
  --run-id m2-run-001 \
  --git-commit YOUR_GIT_COMMIT \
  --seed 0 \
  --device cpu
```

On the first cycle, omitting `--learner-checkpoint` initializes the learner from
`--incumbent`. On every later cycle, pass the preceding cycle's
`candidate_checkpoint` as `--learner-checkpoint`, even when that candidate was
not promoted. The learner is continuous; gating controls only which network
generates self-play and serves as the evaluation incumbent. Every cycle records
the learner input checkpoint separately from the incumbent checkpoint.

For exact late-game labels covered by the no-wall tablebase, use `--method hybrid --sampling no-wall`.
For low-wall endgames, enable the conservative exact low-wall solver explicitly:

```bash
.venv/bin/barricade-generate-5x5-oracle \
  --config configs/m2_5x5.json \
  --output artifacts/m2/oracle_5x5_low_wall.jsonl \
  --positions 128 \
  --random-plies 4 \
  --method hybrid \
  --sampling low-wall \
  --low-wall-max-remaining 1 \
  --low-wall-max-nodes 500000 \
  --max-nodes 50000 \
  --seed 0
```

Refine an existing corpus in place of resampling:

```bash
.venv/bin/barricade-generate-5x5-oracle \
  --config configs/m2_5x5.json \
  --refine-from artifacts/m2/oracle_5x5_smoke.jsonl \
  --output artifacts/m2/oracle_5x5_refined.jsonl \
  --method hybrid \
  --max-nodes 50000
```

## Functional API

```python
import numpy as np
from barricade_rl import Game, TerminalStatus

game = Game()
state = game.initial_state()

while game.is_terminal(state) is TerminalStatus.NOT_TERMINAL:
    mask = game.legal_actions(state)
    action = int(np.flatnonzero(mask)[0])
    state = game.next_state(state, action)

value_for_mover = game.terminal_value(state)
```

`State` is immutable. `next_state` never mutates its input. Actions accepted and returned by the functional interface are always canonical for the mover.

## Gymnasium API

```python
import gymnasium as gym
import barricade_rl

env = gym.make("BarricadeRL/Quoridor-v0")
observation, info = env.reset(seed=0)
observation, reward, terminated, truncated, info = env.step(int(info["action_mask"].nonzero()[0][0]))
```

## Action encoding

- `0..11`: `N, S, E, W, NN, SS, EE, WW, NE, NW, SE, SW`.
- `12..75`: horizontal wall at anchor `(r, c)`, index `12 + r*8 + c`.
- `76..139`: vertical wall at anchor `(r, c)`, index `76 + r*8 + c`.

`N` is always forward toward the mover's goal in the canonical frame. Player B conversion flips rows only; columns are not reversed.
