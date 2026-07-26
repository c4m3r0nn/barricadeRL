# Architecture

The handover specification is authoritative. This document records the code that currently exists, not the eventual full system.

## M0 rules boundary

`barricade_rl.game.Game` is the only public game interface. It is stateless: every operation takes an immutable `State` and returns data or a new state. Search code must not own a mutable environment.

`State.data` is the exact 20-byte key:

| Bytes | Value |
| --- | --- |
| 0..7 | horizontal-wall `u64` bitboard |
| 8..15 | vertical-wall `u64` bitboard |
| 16..17 | two 7-bit pawn cells plus side-to-move bit |
| 18 | two 4-bit walls-remaining counters |
| 19 | ply counter |

The native implementation is in `src/lib.rs` and uses a small C ABI consumed by `barricade_rl/_native.py`; it has no third-party Rust dependency. `barricade_rl/_engine.py` is a deliberately independent, slower test oracle with the same low-level API. Python orchestration and tensor encoding are in `barricade_rl/game.py`.

## Coordinates and perspective

Rows and columns are zero-indexed. Player A starts at `(0, 4)` and aims for row 8. Player B starts at `(8, 4)` and aims for row 0.

All public actions are mover-canonical. For Player B, canonical conversion flips the row axis and keeps columns unchanged. The observation always places the mover in plane 0 and its opponent in plane 1. Wall anchors use the corresponding `r -> 7-r` transform.

The only augmentation symmetry is left-right reflection. State transposition keys use the lexicographically smaller of the original and mirrored packed keys.

## Correctness strategy

The test order follows red-green TDD:

1. Contract tests were added against the absent functional API and observed failing.
2. The Python oracle made the rules tests green.
3. The native engine implements the same contract.
4. Differential verification compares every mask, distance, and selected successor between native and Python-oracle implementations.
5. An adapter compares complete games against the separately maintained `pyquoridor` 0.0.5 package.
6. Random games assert path preservation, wall conservation, non-overlap, terminal correctness, and deterministic replay.
7. Native perft pins depths 1 through 4 for the opening and five midgame positions reconstructed from legal actions.

The full verification covers 10,000 invariant games, 10,000 native/Python-oracle states, and 10,000 complete external-oracle games. It is intentionally heavier than the normal test suite and is also wired into CI.

## Milestone boundary

M0 passed its correctness, independent differential, perft, and throughput gates on 2026-07-05, and M1 passed on 2026-07-07. M2 is active: the 5,000-position exact validation corpus and proper-training preflight pass; training, acceptance evaluation, and multi-process inference scaling remain.

## M1 environment and ladder

`QuoridorEnv` owns only the current immutable state. A learner action is applied through `Game`, then the configured opponent policy supplies the reply before control returns to the learner. Rewards are terminal-only and legal masks are returned on every transition.

The reference ladder is versioned in `barricade_rl.opponents` and ordered as random, greedy racer, heuristic-1, alpha-beta depth 3, and alpha-beta depth 5. The scripted policies consume the same canonical action masks and successor interface as future MCTS code.

`barricade_rl.evaluate` runs deterministic, colour-balanced matches between any policy using the ladder policy protocol. Game records preserve seeds, actions, final state keys, termination mode, and wall usage. Match records score caps as draws for Elo purposes and `estimate_elos` anchors ratings at `random = 0` with a pseudo-draw prior so smoke evaluations produce finite ratings.

`barricade_rl.dashboard` is the M1 dashboard skeleton. It appends JSONL events and renders a minimal HTML table for ladder Elo, average game length, cap fraction, and wall usage. The schema already reserves the learner metrics called out in the handover: policy/value/auxiliary losses, root entropy, value calibration, samples-per-position ratio, games per hour, and GPU utilization.

`barricade_rl.baseline_dqn` is the M1 masked DQN smoke baseline. It is deliberately dependency-free and not reused by the AlphaZero stack. The learner interacts only through `QuoridorEnv`, consumes `info["action_mask"]`, masks illegal Q-values before every exploratory or greedy action, stores replay, uses a target network for TD targets, and mixes in Monte Carlo returns so terminal-only rewards propagate quickly in this smoke setting. A short greedy-racer behaviour warm start is used only to avoid spending M1 compute on sparse-reward exploration.

M1 is complete as of 2026-07-07. The recorded smoke acceptance run trained for 200 episodes against random and evaluated over 50 games, scoring 46 wins, 0 losses, and 4 draws for a 0.96 score rate against the 0.80 gate. The smoke DQN is not a production training component.

## M2 5x5 oracle foundation

`barricade_rl.small_board` is the M2 rules surface for the solved-board curriculum. It is a separate Python implementation rather than a modification of the 9x9 Rust ABI, so the verified M0/M1 engine remains stable while M2 solver work progresses.

The default spec is 5x5 with 3 walls per player and the supervisor-mandated 200-ply cap. The action encoding preserves the production semantics: 12 canonical pawn actions followed by horizontal and vertical wall anchors. On 5x5 that gives 44 actions: `12 + 2 * 4 * 4`.

`SmallState` is immutable and serializes to a compact 12-byte key: 32-bit horizontal walls, 32-bit vertical walls, two pawn-cell bytes, packed wall counters/current-player byte, and ply byte. `SmallGame` exposes the same search-oriented operations as `Game`: initial state, legal mask, successor, terminal status/value, canonical observation, mirror, state key, transposition key, shortest-path distance, render, and perft.

The bounded solver hook, `solve_state`, is not the final M2 oracle. It performs exact depth-limited minimax over the small-board rules and returns `WIN`, `LOSS`, `DRAW`, or `UNKNOWN`. This gives immediate proof labels for terminal, forced-win, and short-race positions while leaving the full-state retrograde/proof-number solver as the next M2 task.

`barricade_rl.oracle5x5` adds the first oracle-data layer over that hook. `prove_state` runs budgeted exact search and records whether the result is exact, unknown because of depth, or unknown because the node budget was exhausted. `generate_oracle_corpus` writes JSONL labels containing the state key, config hash, proof parameters, best action when known, shortest-path distances, legal action count, and exactness flags. This is intentionally not claimed as the final full-board oracle; it is the reproducible data path that the proof-number or retrograde solver will fill with exact labels.

The same module now includes `proof_number_search`, a best-first proof/disproof search for the root mover's win objective. It reports proof and disproof numbers directly and only returns exact loss labels when the disproof tree consists of opponent-win terminals rather than cap draws. It also includes `solve_no_wall_endgame`, an exact finite negamax tablebase for positions where both wall counters are zero; ply remains part of the state key, so cap draws are represented correctly.

Corpus generation supports three labelling methods: `minimax`, `proof-number`, and `hybrid`. Hybrid uses the exact no-wall tablebase first, an optional exact low-wall endgame solver for configured small remaining-wall counts, and proof-number search otherwise. It supports `sampling=no-wall`, which forces legal random wall placement until both players have exhausted their walls before sampling additional random moves, and `sampling=low-wall`, which stops once the configured unplaced-wall threshold is reached. Proof-number-backed batch runs share a solved-state cache that stores only exact win/loss labels, not draws or budget-exhausted unknowns, and that cache can be loaded/saved as JSONL for reuse across runs. Deterministic `--shard-index` / `--shard-count` splitting emits only the assigned `record_index` values while still advancing the same sampling stream as an unsharded run. `--merge-from` rebuilds true shards in `record_index` order; `--compact-exact-from` combines independent runs, removes inexact, terminal, and duplicate states, reindexes them, and selects balanced phase quotas. `--audit-corpus` checks record count, config hash, duplicate keys and indices, terminal positions, exact-label coverage, opening/midgame/endgame coverage, methods, and outcomes. The 200-ply local corpus `artifacts/m2/oracle_5x5_exact_5000_cap200.jsonl` passes this gate with 1,667/1,667/1,666 exact non-terminal positions across the three phases and no unknown or exhausted labels.

`refine_oracle_corpus` is the resumability path for scale runs. It reads an existing JSONL corpus, preserves exact records by default, and relabels only unknown/exhausted records with a stronger method or larger node budget. The CLI exposes this through `--refine-from`, so broad corpus generation and expensive proof attempts can be scheduled separately.

`barricade_rl.mcts` is the tested PUCT search core. It is evaluator-agnostic, masks illegal network logits before normalization, applies the configured first-play-urgency reduction, adds legal-only Dirichlet noise at self-play roots, forces exploratory root visits, removes those forced visits from the policy target, backs values up with sign flips, and detects immediate wins before network evaluation. Fast self-play searches disable noise and forced playouts. Production scaling still needs tree reuse, batched inference, transposition sharing, Gumbel mode, and the 800-simulation oracle-accuracy path.

`barricade_rl.az_network` defines the AlphaZero inference/checkpoint contract. It implements batch normalization in the stem, both convolutions of every residual block, and the policy/value heads; configurable global-pooling blocks; the committed flat main and opponent-policy heads; tanh mover value; and normalized shortest-path outputs. NumPy inference uses checkpointed running statistics and EMA parameters, while the learner maintains the matching differentiable PyTorch graph.

`barricade_rl.az_replay` defines the AlphaZero replay contract that self-play and the learner must use: versioned canonical observation, board size, legal-action mask, visit-count policy target, mover-perspective value target, state key, ply, current player, scoring scheme, game/run identity, git commit, config hash, target origin, samples-per-position reporting, and NPZ persistence. It rejects policy mass on illegal actions and values outside `[-1, 1]`, which pins the replay data invariants before any learner is introduced.

`barricade_rl.az_self_play` owns the single-game actor contract. At each ply it chooses a full 200-simulation search with probability 0.25 or an unrecorded 50-simulation fast search otherwise. Only full-search positions enter replay. For the first eight plies it can make one raw-policy move with probability 0.04 per ply; if that happens, earlier pending targets are discarded because their outcome was contaminated by the intervention. Finished-game values are assigned in each stored state's mover frame. Capped games initially remain zero-valued; after the monitored trigger, shortest-path adjudication assigns the closer player the win, with equal distance broken by the mover's half-tempo advantage.

`barricade_rl.az_learner` trains every network parameter with momentum SGD and the configured sum of main-policy cross-entropy, mover-value mean square, 0.1-weighted distance loss, 0.15-weighted following-ply opponent-policy loss, and L2 decay. It applies independent 0.5 mirror augmentation per row, maintains EMA inference weights and batch-normalization statistics, follows fixed learning-rate steps, refuses updates above four consumed gradient samples per generated position, and checkpoints raw/EMA weights, momentum, RNG state, run identity, git commit, and config hash. `barricade-train-az` exposes this path for persisted replay.

`barricade_rl.az_gating` wraps EMA networks in deterministic evaluation MCTS and compares each candidate with the incumbent over exactly 200 colour-balanced games at 800 simulations with no root noise or move sampling. To avoid reducing a deterministic second-player-win game to 100 copies of each colour assignment, the gate samples 100 unique reproducible legal prefixes from plies 1 through the configured temperature horizon and plays each position twice with network colours swapped. Gate JSON records the sampling protocol, seed, ply range, and every start-state key. A candidate is promoted at a score rate of at least 0.55. Promoted checkpoints are copied into an append-only gated archive with a JSONL manifest; rejected candidates remain auditable but cannot replace the self-play network.

`barricade_rl.az_pipeline` is the local correctness-first coordinator. It persists monotonically increasing cycle indices; derives separate deterministic self-play, gate-start, and gate-match seeds per cycle; gives every self-play game and rejected candidate a collision-free identity; and records self-play cap fraction, scoring scheme, learner device, requested/completed learner steps, final-step loss components, root-policy entropy, learning rate, and replay-consumption ratio in `cycles.jsonl`. If the cap fraction exceeds 5% for three consecutive cycles, the following cycle switches to recorded shortest-path adjudication for the remainder of that run, preventing scoring-scheme oscillation after cap rates recover. Each cycle generates replay using only the gated incumbent, computes the legal learner-step headroom from the resulting replay size, clamps oversized requests to the four-samples-per-position limit, advances the continuous learner from its preceding candidate checkpoint, persists the next candidate, runs the gate, and archives promotions. A rejection leaves the self-play incumbent unchanged but does not discard learner weights, momentum, EMA, step, or RNG state; cycle provenance records the learner input and incumbent separately. The CLI requires a passing oracle audit before starting.

`barricade_rl.apple_benchmark` is a read-only performance probe for choosing the Apple Silicon scaling path without changing training semantics. It compares EMA inference across the production NumPy implementation and equivalent PyTorch CPU/MPS graphs, reports output parity and both compute-only and device-round-trip throughput, separates learner cold-step latency from steady-state throughput, and measures deterministic independent-MCTS scaling with spawned worker processes. It SHA-256 hashes the source checkpoint and replay before and after every command. The benchmark deliberately does not claim that batched Metal inference is integrated into MCTS: current self-play and gating still request one NumPy evaluation at a time, while `--device mps` affects only the learner.

`barricade_rl.training_readiness` is the proper-training preflight. It runs the oracle corpus audit and imports and validates replay, MCTS, network, self-play, learner, gating, and pipeline contracts. The local preflight passes; high-throughput production orchestration remains engineering work before an unattended run.

The 5x5 opening perft is pinned at depths 1 through 3 as `35`, `1109`, and `31540`.

`configs/m2_5x5.json` is the committed source-of-truth config for this milestone. It fixes the 5x5 board spec, MCTS budgets, replay window, gating rule, learner constants, and acceptance thresholds. It also records the head-layout decision: because the optional size-agnostic policy head was not implemented and tested before M2, the AlphaZero network will use board-size-dependent flat policy heads and transfer only the convolutional trunk across 5x5, 7x7, and 9x9.

`tests/test_handover_compliance.py` pins the main implementation decisions against the supervisor guidance: Gymnasium rather than deprecated Gym, masked softmax call sites, M2 config constants, terminal reward/gamma choices, gating settings, dashboard metric reservations, and the flat-head decision.
