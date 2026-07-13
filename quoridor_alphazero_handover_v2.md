# Quoridor (barricade.gg) AlphaZero Training System
## Senior Engineering Handover Specification

**Status:** Final handover document, revision 2. Revision 2 incorporates the post-AlphaZero state of the art following a literature audit: the KataGo training-efficiency suite (Wu, "Accelerating Self-Play Learning in Go", arXiv:1902.10565), Gumbel AlphaZero (Danihelka et al., "Policy Improvement by Planning with Gumbel", ICLR 2022), reanalyze-style buffer refresh (MuZero Unplugged lineage; ReZero, arXiv:2404.16364), and self-play start-state diversification (Go-Exploit, AAMAS 2023). Every adopted technique is integrated inline at the point where it applies and catalogued with its evidence base and adoption tier in Appendix A. The author will not be available after this document is delivered. Read it end to end before writing a line of code, then read Section 12 (Contingency Playbook) again after your first failed training run, because you will have one.

---

## 1. North Star

The north star is a single sentence: **produce an agent that plays 9x9 Quoridor at a superhuman level, trained purely from self-play with no human game data, whose strength is demonstrated by a monotonically rising Elo curve against a fixed ladder of reference opponents.**

Everything in this document serves that sentence. When you face a design decision not covered here, resolve it by asking three questions in order: does this preserve the exact game-theoretic structure of Quoridor (the game we are solving must be the game people play), does this keep the training signal unbiased (the value function must estimate the true probability of winning, not a proxy we invented), and does this keep the system measurable (if we cannot see strength on the Elo ladder, the change does not exist). If a proposed change fails any of these, do not make it.

A secondary north star, and your single most important de-risking tool: **the 5x5 board is solved.** Perfect play on 5x5 is known (it is a second-player win, established by retrograde analysis and independently by proof-number search). This gives us something Go and chess teams never had: ground truth. Our pipeline must first reproduce perfect play on 5x5, verified move-by-move against a solver, before we spend a single GPU-hour on 9x9. This is the acceptance gate for the entire training pipeline. If the architecture, MCTS, and training loop cannot find the known-perfect policy on a solved board, they will not find a strong policy on the unsolved one.

---

## 2. Scope and Terminology

We are building an AlphaZero-style system, not literally AlphaGo. AlphaGo used human game data and separate policy/value networks; AlphaZero uses pure self-play, a single two-headed network, and no rollouts in MCTS. AlphaZero is the correct choice here because we have no large corpus of expert Quoridor games and the game is perfect-information, deterministic, two-player, and zero-sum, which is exactly AlphaZero's home turf.

"Gym environment" in this document means a Gymnasium-compatible environment (the maintained successor to OpenAI Gym; do not build against the deprecated `gym` package). One warning up front: the Gym API is single-agent and step-centric, which is a poor fit for the training loop itself. AlphaZero's MCTS does not "step an environment", it asks arbitrary states for their legal moves and successors. So we build two interfaces over one rules engine, as specified in Section 4.

Throughout this document, "ply" means one move by one player, and "the mover" means the player whose turn it is.

---

## 3. The Rules Engine (the foundation everything sits on)

### 3.1 Exact rules to implement

The game is standard two-player Quoridor. Implement these rules exactly, with no house variants:

1. Board of 9x9 cells. Player A's pawn starts at the centre cell of row 0 (cell e1 in algebraic terms, coordinate (0,4)), Player B's at the centre of row 8 (coordinate (8,4)). Player A wins by reaching any cell in row 8, Player B any cell in row 0.
2. Each player begins with 10 walls. A wall is two cells long and sits in the grooves between cells.
3. On a turn, the mover either moves their pawn or places one wall. There is no pass. (If a player has no legal move the position is unreachable under correct rules; assert on it.)
4. Pawn movement: one cell orthogonally, not through walls, not off the board, not onto the opponent's cell.
5. Jumps: if the opponent's pawn is orthogonally adjacent and no wall separates the two pawns, the mover may jump straight over it to the cell directly beyond, provided that cell is on the board and not blocked by a wall behind the opponent. If the straight-ahead landing cell is blocked by a wall or the board edge, the mover may instead move diagonally to either cell orthogonally adjacent to the opponent's pawn (perpendicular to the jump direction), provided that diagonal destination is not blocked by a wall between it and the opponent's cell. Diagonal jumps are legal only when the straight jump is unavailable. This is the most commonly mis-implemented rule in Quoridor engines (public app reviews of Quoridor implementations complain about exactly this bug), so it gets its own test battery in Section 3.4.
6. Wall placement legality, all four conditions required: the wall lies fully on the board; it does not overlap any segment of an existing wall; it does not cross an existing perpendicular wall at the same central notch; and after placement, both players still have at least one path to their respective goal rows. The last condition requires a reachability search and is the expensive one.
7. Walls, once placed, never move and are never recovered.
8. Termination: a player wins the instant their pawn stands on their goal row. There is no natural draw rule in Quoridor, but infinite games are possible through repetition, so we impose an artificial ply cap (Section 8.4).

### 3.2 Coordinate and wall-anchor conventions (fix these now, never change them)

Cells are addressed (row, col), zero-indexed, with row 0 at Player A's home edge. Wall anchors live on an 8x8 grid of internal notches, also addressed (row, col) with row in 0..7 and col in 0..7.

A **horizontal wall anchored at (r, c)** blocks movement between cells (r, c) and (r+1, c), and between (r, c+1) and (r+1, c+1).

A **vertical wall anchored at (r, c)** blocks movement between cells (r, c) and (r, c+1), and between (r+1, c) and (r+1, c+1).

Overlap rule in this convention: a horizontal wall at (r, c) conflicts with horizontal walls at (r, c-1), (r, c), (r, c+1) and with the vertical wall at (r, c). Symmetrically for vertical walls. Write this as a precomputed conflict mask, not runtime logic.

### 3.3 Board representation and performance requirements

Self-play throughput is the binding constraint on the whole project, so the rules engine must be fast. The representation that works, and which you should implement exactly:

- Horizontal walls: one 64-bit integer, bit index r*8+c for anchor (r, c). Vertical walls: a second 64-bit integer. Pawn positions: two bytes. Walls remaining: two 4-bit counters. Side to move: one bit. The whole state fits in under 20 bytes, which makes the transposition-table key trivial (use the state bytes directly; Zobrist hashing is optional at this size).
- Precompute, at startup, for every anchor index: the conflict bitmask against both wall integers (turns wall-overlap legality into two AND operations), and the pair of blocked cell-adjacencies it creates.
- Maintain adjacency as a per-cell 4-bit "open directions" table updated incrementally when a wall is placed, so pawn move generation is a table lookup plus the jump logic.
- Path-existence checks (wall legality condition 4) dominate runtime if done naively, because every candidate wall placement needs one. Three mitigations, implement all: (a) check reachability with A* toward the goal row using row-distance as the heuristic, not plain BFS, and early-exit on first goal contact; (b) only run the check for candidate walls that touch the current shortest path of either player, since a wall that does not intersect any current shortest path cannot disconnect (this is a sound shortcut only for the "does a path exist" question if you verify it against the player's full reachable set; if in doubt, run the check, correctness beats speed); (c) cache the result per (wall-state, pawn-position) in the search tree, since MCTS revisits states constantly.
- Shortest-path distances for both players (needed by the evaluation baselines, adjudication, and auxiliary training targets) come from a BFS from the goal row backwards, giving distance-to-goal for every cell in one pass per player. Update incrementally on wall placement where convenient, recompute where not; measure before optimizing further.

Performance target: the bare rules engine must sustain at least 20,000 random-playout plies per second per CPU core in the implementation language before you connect any ML. If it cannot, self-play will starve the GPU and the project timeline doubles. Write the engine in a compiled language (Rust or C++ with Python bindings; given the team's Rust experience, Rust with PyO3 is the recommended path) and treat Python as orchestration only.

### 3.4 The correctness test suite (blocking milestone, nothing proceeds without it)

Build a golden test suite before the environment, and wire it into CI so it runs on every commit. Required contents:

1. **Jump geometry battery.** Enumerate by hand at least 25 positions covering: straight jump open; straight jump blocked by wall behind opponent (both diagonals open, one open, neither open); straight jump blocked by board edge; diagonal jump blocked by a wall between the diagonal cell and the opponent; pawns adjacent but separated by a wall (no jump at all); jumps in all four directions. For each, the expected exact legal move set.
2. **Wall legality battery.** Overlap in all orientations, the cross-at-same-notch case, edge-of-board anchors, and at least ten "this wall would seal the last path" positions including sneaky ones where the path exists but only through a jump-dependent square (path existence must be computed on the wall graph ignoring pawns, since pawns move and cannot permanently block; verify your implementation treats the opponent's pawn as non-blocking for reachability purposes).
3. **Property-based tests.** From 10,000 randomly played games: both players always have a path to goal after every legal move; wall counts never go negative; total walls on board plus walls in hand always equals 20; no two walls overlap; every game either terminates with a pawn on its goal row or hits the ply cap; replaying the recorded move list reproduces the identical final state (determinism).
4. **Differential testing.** Play your engine against an independent open-source Quoridor rules implementation on thousands of random games, asserting identical legal-move sets at every ply. Disagreements are bugs in one of the two; investigate every one.
5. **Perft-style move counts.** Record legal-move counts at depths 1..4 from the initial position and from five handcrafted midgame positions, commit these numbers to the test suite, and treat any change as a regression.

The single most expensive class of bug in this project is a rules bug discovered after a training run, because it invalidates every game generated. Budget a full week for this suite and consider it cheap.

---

## 4. Environment Architecture: two interfaces, one engine

### 4.1 The Game interface (what MCTS and the trainer consume)

A stateless functional interface over immutable state objects, with exactly these operations: `initial_state()`; `legal_actions(state)` returning a boolean mask of length 140 (encoding in Section 5); `next_state(state, action)` returning a new state; `is_terminal(state)` returning one of not-terminal, mover-has-lost (opponent just reached goal), or capped; `terminal_value(state)` from the mover's perspective; `canonical_observation(state)` returning the tensor of Section 6; `mirror(state_or_policy)` applying the left-right symmetry (Section 6.3); and `state_key(state)` returning the transposition key. Immutability matters: MCTS holds many states alive simultaneously and in-place mutation is the classic source of silent tree corruption. If profiling later demands a mutable make/unmake pattern, that is an optimization to do behind this same interface with the immutable version kept as the test oracle.

### 4.2 The Gymnasium interface (for baselines, debugging, and external users)

Wrap the same engine in a Gymnasium environment for three purposes: sanity-checking with off-the-shelf algorithms (a DQN or PPO baseline that beats random confirms observation and reward plumbing), human play and rendering, and evaluation harnesses. Because Gymnasium is single-agent, the environment takes an `opponent_policy` at construction (random, scripted heuristic, or a frozen network checkpoint) and internally plays the opponent's reply inside `step`, returning the post-reply observation. Expose the legal-move mask through the `info` dictionary under the key `action_mask` on every step and reset, following the standard action-masking convention. Reward from this wrapper is strictly terminal: +1 win, -1 loss, 0 otherwise, consistent with Section 8. Rendering: an ASCII renderer first (walls drawn in the grooves, pawns as letters), then optionally a minimal web or image renderer; the ASCII one is the debugging workhorse, do not skip it.

Multi-agent framing (PettingZoo AECEnv) is a nice-to-have for external release, not needed for the training loop. Do not spend time on it before Milestone M4.

---

## 5. Action Space (fixed, total 140 actions)

The action space is a fixed discrete space of 140, always interpreted **from the canonical perspective of the mover** (Section 6.2). The encoding, which must be documented in code exactly as here:

- Indices 0 to 11: pawn moves, in the order N, S, E, W, NN, SS, EE, WW, NE, NW, SE, SW, where N is the mover's forward direction (toward their goal row) in the canonical frame. Indices 4 to 7 are the straight jumps, 8 to 11 the diagonal jumps. Encoding jumps as distinct actions (rather than reusing N for a jump) keeps the policy head unambiguous: at most one of "move N" and "jump NN" is ever legal in a position, but they are different motor actions and the network learns them faster as separate logits.
- Indices 12 to 75: horizontal walls, index 12 + r*8 + c for anchor (r, c) in the canonical frame.
- Indices 76 to 139: vertical walls, index 76 + r*8 + c likewise.

Every consumer of actions receives, alongside the policy logits, the legal-action mask from the engine. Illegal logits are set to negative infinity before softmax, always, at every call site, and the training loss only ever sees masked distributions. A single unmasked softmax anywhere in the codebase will silently poison training; grep for `softmax` in code review and verify each occurrence.

---

## 6. Observation Encoding and Canonicalization

### 6.1 The observation tensor

Shape (C, 9, 9), float32, C = 6 planes, always in the canonical frame:

1. Plane 0: one-hot of the mover's pawn.
2. Plane 1: one-hot of the opponent's pawn.
3. Plane 2: horizontal walls; value 1.0 at cell (r, c) if a horizontal wall is anchored at notch (r, c), zeros elsewhere (the 8x8 anchor grid is embedded in the top-left of the 9x9 plane; row 8 and column 8 are always zero).
4. Plane 3: vertical walls, same embedding.
5. Plane 4: constant plane, mover's walls remaining divided by 10.
6. Plane 5: constant plane, opponent's walls remaining divided by 10.

No side-to-move plane is needed because canonicalization removes it. No move-count plane initially; if cap-related pathologies appear (Section 12), add a seventh constant plane of plies-elapsed divided by the cap, retrain, and note the observation version in the config. Version the observation encoding explicitly (an integer in every checkpoint and every replay-buffer record) so a change never silently mixes encodings.

### 6.2 Canonicalization (the mover always plays "up")

The network always sees the position as if the mover is heading toward row 8. When the actual mover is Player B, flip the board vertically before encoding: cell (r, c) maps to (8-r, c); a wall anchored at (r, c) maps to anchor (7-r, c) with unchanged orientation; pawn planes swap so plane 0 is always the mover. Action canonicalization is the same transform on indices: N and S swap in the actual-move decoding, NN/SS swap, NE/SE and NW/SW swap, wall index rows map r to 7-r. Write the canonical-to-actual and actual-to-canonical action maps as two precomputed permutation arrays of length 140 and unit-test that they are inverse permutations, and that canonicalize-then-decanonicalize is the identity on 1,000 random states.

### 6.3 Symmetry augmentation

Quoridor has exactly one nontrivial symmetry in the canonical frame: the left-right mirror (columns c map to 8-c; wall anchors c to 7-c; actions E/W swap, EE/WW swap, NE/NW swap, SE/SW swap). Rotations are not symmetries because the goal direction is fixed. Use the mirror as training-data augmentation: for every stored position, train on both the position and its mirror with the mirrored policy target (either by doubling storage or by flipping on the fly with probability 0.5; do the latter, it is free). Also exploit it in the search transposition table by storing under the lexicographically smaller of key and mirrored key. Expected benefit is a straightforward 2x on effective data.

---

## 7. Network Architecture

A single convolutional ResNet with two heads, standard AlphaZero shape, deliberately small because the board is 9x9:

- **Stem:** 3x3 convolution from C input planes to F filters, batch norm, ReLU. Start with F = 96.
- **Body:** B residual blocks, each two 3x3 convolutions of F filters with batch norm, ReLU, and a skip connection. Start with B = 8. This (8 blocks, 96 filters) is the "small" config; define a "medium" config (12 blocks, 128 filters) in the config file now, and only move to it if the small one plateaus on the Elo ladder while training loss keeps falling (a capacity signal), not before.
- **Global pooling blocks (adopted from KataGo, mandatory):** replace two of the B residual blocks (roughly one-third and two-thirds of the way through the trunk) with global-pooling variants: within the block, split the F channels into a pooled group of F/4 channels and a regular group; compute the channelwise mean and max of the pooled group over the whole board, pass those 2 x F/4 scalars through a small fully connected layer, and add the result as a per-channel bias onto the regular group before the block's second convolution. The rationale is Quoridor-specific and strong: wall counts, overall wall density, and the global race state are whole-board quantities that plain 3x3 convolutions propagate slowly across a 9x9 board, and KataGo's ablations showed global pooling contributes materially to learning speed precisely because it lets local features condition on global context. Without it, the constant walls-remaining planes take many layers to influence spatially distant decisions.
- **Policy head:** 1x1 convolution to 4 filters, batch norm, ReLU, flatten, fully connected to 140 logits. Masking applied outside the network as per Section 5.
- **Value head:** 1x1 convolution to 2 filters, batch norm, ReLU, flatten, fully connected to 96 units, ReLU, fully connected to a scalar, tanh. Output is the expected outcome for the mover in the canonical frame, in [-1, 1].
- **Auxiliary distance head (strongly recommended, see Section 8.6):** 1x1 convolution to 1 filter, flatten, fully connected to 2 linear outputs predicting (mover's shortest-path distance, opponent's shortest-path distance), each normalized by dividing by 20. Trained with mean-squared error at weight 0.1 relative to the value loss. This head is discarded at inference; its only job is to force the trunk to learn pathfinding features early, which it demonstrably will not do quickly from win/loss signal alone.
- **Auxiliary opponent-policy head (adopted from KataGo, mandatory):** a second policy head, identical in shape to the main one, trained to predict the opponent's reply policy (the MCTS visit distribution of the following ply in the stored game) with cross-entropy at weight 0.15 relative to the main policy loss. Also inference-discarded. KataGo's ablations attribute a nontrivial share of its order-of-magnitude efficiency gain to exactly this target: it forces the trunk to model the adversary's intentions, which in Quoridor means learning threatened wall placements one ply before they land.
- **Weight averaging (mandatory, nearly free):** maintain an exponential moving average of the network weights (decay 0.999 per learner step, or equivalently stochastic weight averaging over recent checkpoints) and use the averaged weights for all self-play inference, gating candidates, and evaluation, while gradients continue to update the raw weights. Both KataGo and the strongest community reimplementations ship this; it smooths the strategy-cycling oscillations described in Section 12 at essentially zero cost. The checkpoint format stores both raw and averaged weights.
- **Optional size-agnostic policy head (decide at M1, before any training):** the flat 140-logit head hardwires the 9x9 board into the network and forces the head reinitializations in the curriculum. The alternative, which KataGo's cross-board-size training validates, is a fully convolutional policy output: one 1x1 convolution producing two planes over the (n-1)x(n-1) anchor grid for horizontal and vertical wall logits, plus a global-pooling branch producing the 12 pawn-move logits. Combined with the global-pooling trunk, this makes the entire network board-size-independent, which upgrades the curriculum of Section 10.2 from "transfer trunk, reinitialize heads" to "one network, mixed-size training". Take this option if and only if it is implemented and unit-tested before M2 begins; do not retrofit it mid-curriculum, because it changes the policy-target tensor layout in the replay buffer.

Everything is fully convolutional except the heads, which matters for the curriculum: when moving from 5x5 to 7x7 to 9x9 (Section 10), the trunk weights transfer directly and only the heads are reinitialized. Keep the head parameter shapes board-size-dependent and the trunk board-size-independent, and write the checkpoint loader to do partial restores by parameter name.

Parameter budget sanity check: the small config is roughly 1.5 to 2 million parameters. If your implementation reports 20 million, a head is misconfigured.

---

## 8. Reward Design (read this section twice)

This is the part of the system where a well-meaning "improvement" most reliably destroys everything, so the policy here is conservative and the reasoning is spelled out.

### 8.1 The canonical scheme, which is the default and the final scheme

Terminal reward only: **+1 for a win, -1 for a loss, 0 for a capped game**, no discounting (gamma = 1), no per-step rewards, no shaping. The value target z for every position in a finished game is the final outcome from the perspective of the player to move at that position. That is the entire reward specification.

Why so austere: AlphaZero's correctness rests on the value head converging to the minimax value of the position under the current policy, and MCTS amplifying that into a better policy. The minimax value of Quoridor is defined by win/loss and nothing else. Every reward term you add redefines the game being solved. In a two-player zero-sum self-play loop this is worse than in single-agent RL, because both players adapt to exploit the modified objective against each other, and the failure modes are subtle: reward a player for reducing their path distance and you will train an agent that races beautifully and never learns walls; reward wall efficiency and you get wall spam that harvests the bonus; penalize long games and the agent in a losing position stops resisting because resistance lengthens the game. All three of these have been observed in practice in games-RL projects. Sparse win/loss is not a limitation to engineer around; it is the specification of the game.

### 8.2 The temptation you will feel, and the two sanctioned outlets for it

Around the second week of 9x9 training you will observe that early self-play games are near-random shuffles and learning is slow, and someone will propose dense rewards based on path-length differential. The proposal is understandable and must still be rejected in the form proposed. There are exactly two sanctioned ways to inject domain knowledge about path distance, both of which provably or practically avoid corrupting the objective:

**Outlet one, auxiliary supervised targets (preferred, use this).** As specified in Section 7, add prediction heads for both players' shortest-path distances, trained on values the engine computes for free. This shapes the representation, not the reward: gradients teach the trunk what corridors and detours look like, while the value and policy targets remain pure win/loss and MCTS visit counts. The KataGo line of work established that auxiliary targets of this kind accelerate games-RL by large factors with no objective distortion. Optionally add a second auxiliary target later: predicted final wall counts for both players.

**Outlet two, potential-based shaping, warm-up only, then off.** If and only if 5x5 and 7x7 results suggest the cold start on 9x9 is genuinely wasteful, you may add a shaping term of the strict potential-based form F(s, s') = Phi(s') - Phi(s) with potential Phi(s) = k times (opponent's shortest-path distance minus mover's shortest-path distance) evaluated in a fixed frame, with k no larger than 0.05, applied only to value targets during the first N training iterations, and annealed linearly to exactly zero by a scheduled iteration that is committed in the config before the run starts. Potential-based shaping is the only shaping form with a policy-invariance guarantee, and the guarantee is proven for single-agent MDPs, not adversarial self-play, which is why the anneal-to-zero is mandatory rather than optional. If anyone proposes keeping k above zero permanently because "it seemed to help", the answer is no; the Elo ladder against fixed opponents, not training-loss aesthetics, is the arbiter of whether it helped.

### 8.3 Discounting and the "win fast" question

Gamma is 1.0 and stays 1.0. A common request is to make the agent prefer quick wins and slow losses, usually by discounting or a small per-ply penalty. Both change optimal play (a position where a guaranteed 40-move win exists but a risky 10-move win might be found becomes misvalued) and both interact badly with the ply cap. If quick wins are wanted for product reasons, achieve them in the search, not the reward: at equal root value, break ties toward the child with the shorter proven or estimated distance to termination. Implement this as a final tie-break in move selection only after the system otherwise works.

### 8.4 The ply cap and stalling (the one place the game itself is ambiguous)

Quoridor permits infinite repetition, so a cap is required. Set the cap at 200 plies (well above the observed length of competent games, which cluster far below 100 plies). The default scoring for a capped game is 0 for both value targets.

Anticipated failure mode, and it is a serious one: an agent that is losing learns that shuffling to the cap yields 0 instead of -1, and self-play degenerates into draw-farming; you will see it as a rising fraction of capped games and falling average decisiveness on the monitoring dashboard (Section 11). If the capped-game fraction exceeds 5 percent of self-play games for three consecutive evaluation cycles, switch the capped-game scoring to **adjudication**: the player with the strictly smaller shortest-path distance at the cap (with the tie broken by the mover having effectively half a tempo, and a true tie scored 0) is scored +1 and the opponent -1. Adjudication reintroduces a sliver of heuristic into the objective, which is why it is the fallback rather than the default; it is acceptable because it only ever touches degenerate games that healthy play never reaches. Record which scheme every game in the replay buffer was scored under.

### 8.5 Resignation (a throughput tool that touches reward, handle with care)

To roughly double self-play throughput, allow resignation in self-play: if the root value estimate is below -0.92 for both of a player's last two turns, the game is scored as a loss for that player immediately. Mandatory safeguard, copied from the AlphaGo Zero methodology: 10 percent of self-play games are played with resignation disabled, and the false-positive rate (games that would have been resigned but were ultimately won) is monitored; keep it under 5 percent by tuning the threshold, and disable resignation entirely on 5x5 and 7x7 where games are short and ground truth matters.

### 8.6 Value target composition

Baseline: the value target is the final outcome z, exactly. Sanctioned refinement once the baseline works end to end: mix the target as 0.7z + 0.3q, where q is the MCTS root value of the position when it was played. This reduces target variance and is standard in modern reimplementations. It is a refinement, not a rescue; do not reach for it to fix a broken run.

---

## 9. MCTS Specification

PUCT search, AlphaZero variant, no rollouts; leaf evaluation is the network. Every constant below goes in the config file, and the listed value is the starting point, not scripture.

- **Selection:** at node s choose the action a maximizing Q(s,a) + c_puct * P(s,a) * sqrt(sum of all visit counts at s) / (1 + N(s,a)). c_puct = 1.6 initially. Q of an unvisited child (first-play urgency) is initialized to the parent's Q minus 0.2 rather than zero, which materially matters in a game with 100+ mostly-bad wall moves; make FPU a config value.
- **Expansion and evaluation:** at a leaf, query the network for masked policy priors and value; store priors on the edges; back up the value with sign flips at each ply (the tree alternates movers).
- **Terminal and forced-win handling:** if the mover can reach the goal row this ply, do not consult the network; the node value is an exact win and the search should treat it as proven. Propagate proven values (win/loss) up the tree where all children are proven, chess-solver style. This is cheap and dramatically sharpens endgame play, where Quoridor is effectively a deterministic race.
- **Root exploration noise (self-play only, never in evaluation, PUCT mode only):** mix Dirichlet noise into root priors, P = 0.75 * P + 0.25 * Dir(alpha), with alpha = 0.25 on 9x9. The principled scaling is alpha of roughly 10 divided by the typical legal-move count (Quoridor's is around 80 to 130 early, collapsing near the endgame), and alpha should be set per board size: 0.6 on 5x5, 0.4 on 7x7, 0.25 on 9x9. Sanctioned refinement once baseline works (KataGo's "shaped" noise): allocate a fraction of the Dirichlet concentration proportionally to the network's own prior rather than uniformly, so exploration probability mass is not wasted on the many wall placements the network already knows are absurd; in a 140-action space dominated by junk wall moves this is worth more than it was in Go.
- **Forced playouts and policy target pruning (adopted from KataGo, mandatory in PUCT mode):** during self-play root search, force each root child that has received at least one visit to receive a minimum visit count proportional to the square root of (its prior times total simulations), overriding PUCT selection if necessary; then, when converting root visit counts into the stored policy target, prune away the forced portion of each child's visits (subtract visits down to what PUCT would have given the child, and zero out children whose value estimate is below the best child's) before normalizing. The point is to decouple exploration from the learning target: forcing guarantees noise-seeded moves get enough visits to reveal whether they are genuinely good, and pruning stops the exploration subsidy itself from being imitated by the policy head. Without pruning, Dirichlet noise leaks permanent probability mass onto bad moves; KataGo's ablation shows the pair is a clear win, and Quoridor's junk-heavy action space amplifies the effect.
- **Simulation budget and search modes:** implement both search modes behind one interface, selected in config. Mode A, classic PUCT as described above, budget 600 full simulations per recorded move on 9x9 (200 on 5x5), 800 for ladder evaluations. Mode B, Gumbel root search: sample m = 16 root candidates without replacement via the Gumbel-Top-k trick on the masked policy logits, allocate the simulation budget across them by sequential halving, select the surviving action deterministically, and train the policy toward the "completed Q-values" target (network-estimated Q for unvisited actions, search Q for visited ones) rather than visit counts. Gumbel AlphaZero carries a proven policy-improvement guarantee at arbitrarily small budgets, learning reliably with even a handful of simulations where classic PUCT degrades or fails outright below roughly 100, and it matches PUCT at large budgets. Decision rule, committed here so nobody relitigates it: if sustained self-play throughput at M4 settings supports 600 simulations per move or more, run Mode A with the forced-playout machinery above; if it supports fewer than 300, run Mode B with a budget of 64 to 150 and m = 16; in between, run the M2 pipeline both ways on 5x5 (cheap) and let the ladder decide. Mode B is no longer a degraded fallback; it is the theoretically better-founded algorithm at low budgets and the audit found no evidence against it at moderate ones.
- **Temperature:** during self-play, select the actual move from the visit-count distribution with temperature 1.0 for the first 16 plies, then argmax thereafter. During evaluation, argmax always.
- **Tree reuse:** after a move, retain the chosen child subtree as the new root and discard siblings. Re-apply fresh Dirichlet noise at each new root during self-play.
- **Batched inference:** MCTS workers do not call the GPU one leaf at a time. Use virtual loss (add a temporary loss of 1 along the selected path) to collect batches of 8 to 32 leaves per tree, or run many independent games per worker and batch across games (simpler and equally effective; prefer it). Target GPU inference batch sizes of at least 256 by aggregating across parallel games; the inference server pattern in Section 11.1 is how.
- **Transposition table:** key on the state bytes with the mirror-canonical trick from Section 6.3. Quoridor transposes heavily (move orders of independent walls commute), so this is worth more here than in chess.

---

## 10. Training Loop, Curriculum, and Gating

### 10.1 The loop

Three concurrent components. **Self-play actors** pull the latest (or latest-gated, see 10.3) network, generate games, and write (observation, policy target, z, metadata) records to the replay buffer, applying the two self-play data techniques below.

**Playout cap randomization (adopted from KataGo, mandatory).** The value head and the policy head want opposite things from self-play: value learning wants many games (more win/loss outcomes per compute), policy learning wants deep searches (better visit-count targets per position). Resolve the tension the way KataGo proved out: for each move, with probability p = 0.25 run a full search (the Mode A/B budget from Section 9) and record the position as a training sample; with probability 0.75 run a fast search (about one-quarter of the full budget, and with Dirichlet noise and forced playouts disabled, since these moves are never trained on) and do not record the position for policy training, only letting the game continue toward its outcome. Every position in a game still receives the final z, but only full-search positions enter the buffer. Net effect: roughly 3x more games per GPU-hour, hence 3x more value targets, at a modest cost in policy samples that the improvement in per-sample quality more than repays. KataGo's ablations show this beats every fixed simulation count; it is a headline contributor to its order-of-magnitude efficiency gain over AlphaZero-style baselines, and nothing about it is Go-specific.

**Start-state diversification (adopted from Go-Exploit and KataGo's branching, mandatory in a weaker form, optional in the stronger form).** Quoridor from the fixed initial position risks opening collapse: self-play funnels into one narrow opening tree and the network goes blind elsewhere, which is also how strategy cycling starts. Weak form, mandatory: for the first few plies of each self-play game, with small probability per ply (0.04 works), inject a move sampled directly from the raw network policy at temperature 1 instead of the search result, and mark the game so positions before the injection are excluded from value training (their z is contaminated by the injected move). Strong form, optional after M4 is stable: maintain an archive of positions from recent games and start a fraction of self-play games from states sampled out of it rather than from the initial position, which the Go-Exploit work showed improves sample efficiency by training the value function on a wider, more decision-relevant state distribution.

 **The learner** samples uniformly from the buffer and performs gradient steps on the combined loss: cross-entropy of policy versus visit distribution, plus mean-squared error of value versus target, plus 0.1 times the auxiliary loss, plus L2 weight decay of 1e-4. Optimizer: SGD with momentum 0.9; learning rate 0.02, dropped to 0.002 and then 0.0002 on a schedule fixed in the config (drop when the Elo curve flattens, roughly at one-third and two-thirds of the planned run). Batch size 512 with the mirror augmentation applied on the fly. **The evaluator** periodically plays candidate checkpoints against the reference ladder and (if gating is enabled) against the current best network.

Replay buffer: a sliding window of the most recent games, sized 300,000 positions on 5x5, 1,000,000 on 9x9. The critical ratio to monitor and control is gradient samples consumed per new position generated: keep it between 1 and 4. Below 1 the learner starves; above 4 the network overfits stale self-play and strength oscillates. Make this ratio a first-class dashboard metric with an alert, because on a small cluster it is the number most likely to silently drift.

**Reanalyze (optional, the sanctioned escape hatch when self-play is the bottleneck).** If the samples-per-position ratio is pinned at 4 because game generation cannot keep up, do not overtrain on stale targets; instead refresh them. Reanalyze, from the MuZero Unplugged lineage and made cheap by the ReZero work, re-runs a search on already-buffered positions with the current network and overwrites the stored policy target (value targets z stay as played). Because we have a perfect simulator, this is straightforwardly sound for AlphaZero, and a reanalyze pass costs a fraction of fresh self-play since no games are played to completion. Implement it as a fourth service that consumes buffer positions and rewrites their policy targets in place, activated by a config flag when the dashboard shows learner starvation, and record per-sample whether the target is original or reanalyzed. On the compute profile this team is likely to have, expect to want this.

### 10.2 Curriculum: 5x5, then 7x7, then 9x9

This is the project's spine, because 5x5 gives ground truth.

**Milestone M2, 5x5 to perfection.** Train the pipeline on 5x5 (with walls scaled down to 3 per player, matching the solved configuration you verify against; confirm the exact wall count your solver reference used and match it). Acceptance criteria, all mandatory: (a) the trained agent at evaluation settings, playing as the second player, wins 100 percent of games against any opponent including itself from the initial position; (b) on a held-out set of at least 5,000 solver-labelled positions spanning all game phases, the sign of the value head agrees with the solver's win/loss label at least 99 percent of the time, and MCTS at 800 simulations selects a solver-optimal move at least 99 percent of the time; (c) the Elo curve against the fixed ladder was monotone up to convergence. Building the 5x5 solver oracle is in scope for this milestone: retrograde analysis or proof-number search over the 5x5 state space is a few days of work with the bitboard engine and runs in minutes; it doubles as the ultimate rules-engine test, because a rules bug will show up as a solver result contradicting the published one (second-player win).
**Do not proceed past M2 until all three criteria hold.** Every pipeline bug ever encountered in AlphaZero reimplementations (masking errors, sign errors in value backup, canonicalization mismatches, stale-buffer pathologies) manifests on 5x5 in hours instead of on 9x9 in weeks.

**Milestone M3, 7x7 with transfer.** Restore the trunk from M2, reinitialize heads, train on 7x7 with 5 walls each. No solver here; acceptance is a stable Elo plateau against the ladder and qualitative sanity (the agent uses walls purposefully, defends against traps, wins races it should win). Also use M3 to rehearse the transfer tooling itself.

**Curriculum alternative if the size-agnostic policy head of Section 7 was taken:** instead of sequential transfer, train one network on a mixed stream of board sizes with the proportion of larger boards ramping over time (start 70/30/0 across 5x5, 7x7, 9x9; end 5/15/80). KataGo demonstrated that a single net trained across board sizes generalizes across all of them and that the mixing costs little. The M2 acceptance gate is unchanged and still blocking: the mixed-size network, restricted to 5x5 evaluation, must meet all three perfect-play criteria. The benefit is eliminating two head-reinitialization discontinuities and giving the 9x9 policy a running start; the cost is the more complex head and buffer plumbing, which is why the decision is due at M1 and irrevocable afterwards.

**Milestone M4, 9x9 production run.** Full rules, 10 walls. Transfer the M3 trunk. This is the long run; everything before it exists to make sure this run is not wasted.

### 10.3 Gating versus continuous replacement

AlphaZero proper replaces the network continuously; AlphaGo Zero gated (a candidate had to beat the incumbent at 55 percent to become the self-play network). For a small team without babysitting capacity, **use gating**: evaluate every candidate over 200 games (colours balanced, evaluation settings, no noise) and promote at 55 percent or better. Gating costs some throughput and buys the guarantee that self-play data quality never regresses unnoticed overnight, which is the right trade for you. Revisit only if evaluation compute becomes the bottleneck.

### 10.4 The reference ladder (frozen on day one, never modified)

Fixed opponents, in ascending strength: (1) uniform random legal mover; (2) greedy racer, always steps along its own shortest path, never places a wall; (3) heuristic-1, one-ply search over all legal moves maximizing the classic evaluation of opponent-distance minus own-distance with a 0.7-weighted wall-count difference term; (4) alpha-beta depth 3 with the same evaluation, iterative deepening, move ordering by the heuristic; (5) alpha-beta depth 5, same engine; (6) a frozen network checkpoint from each previous milestone. Report Elo anchored at random = 0. The ladder is versioned, its engines are covered by the rules test suite, and it is never edited after the first 9x9 run begins, because the entire meaning of "progress" depends on it being constant. Expected shape of results, for calibration: the greedy racer beats random almost always; heuristic-1 beats the racer by exploiting walls; a healthy network run passes depth-3 alpha-beta within days on this board size and depth-5 within weeks.

---

## 11. Infrastructure, Monitoring, and Reproducibility

### 11.1 System shape

Four services: an **inference server** owning the GPU(s), exposing a batched evaluate(observations) endpoint over shared memory or a fast local socket, with dynamic batching (flush at 256 requests or 2 milliseconds, whichever first); **self-play workers**, pure CPU, many processes, each running dozens of games concurrently and awaiting inference asynchronously; the **learner**, on its own GPU when available, otherwise time-sliced; and the **evaluator**, scheduled between learner phases. The replay buffer is an append-only on-disk store of fixed-size binary records (observation version, board size, observation tensor, 140-float policy target, value target, scoring-scheme flag, game id, ply index) with an in-memory index for uniform sampling; a memory-mapped flat-file ring buffer is entirely sufficient and preferable to a database.

### 11.2 Configuration and reproducibility discipline

One config file (versioned in git) is the single source of truth for every constant named in this document. Every training run gets a run id; every checkpoint, every replay record, and every evaluation result carries the run id, the git commit hash, and the config hash. All stochastic components take seeds from the config. A run that cannot be attributed to an exact code-plus-config state did not happen, for debugging purposes. Checkpoint every hour and at every gating event; keep all gated checkpoints forever (they are ladder opponents and rollback points) and a rolling window of the rest.

### 11.3 The dashboard (build it in week one, not week six)

Time-series, one page: policy loss, value loss, auxiliary loss; **policy entropy at the root** averaged over self-play moves (the single best early-warning signal: a collapse toward zero entropy in the opening plies means exploration death, usually a noise or temperature bug); value-head calibration (bucket root value predictions, plot actual win rate per bucket; the plot should hug the diagonal); average game length and its distribution; **fraction of games ending at the ply cap** (Section 8.4 trigger); resignation false-positive rate; wall-usage statistics (mean walls placed per game per player; on 9x9, near-zero means the agent is stuck in racing, near-twenty means wall spam); samples-per-position ratio (Section 10.1); games per hour; GPU utilization; and the Elo ladder over time. Alert thresholds on the cap fraction, the samples ratio, and entropy.

---

## 12. Contingency Playbook (read after your first bad run)

Symptoms, causes, and actions, in the order you are likely to meet them.

**Value loss falls, policy loss falls, Elo flat.** The classic. Ninety percent of the time this is a perspective or masking bug: value target sign inconsistent with the canonical frame, or an unmasked softmax, or canonicalization applied to observations but not to actions. Action: run the invariance tests of Section 6.2, then hand-check ten stored replay records end to end (render the position, verify the policy target's top move is plausible and legal, verify z matches the game's actual result from that mover's perspective). Do this by hand, with the ASCII renderer; it takes an hour and finds the bug.

**Self-play games look like both pawns shuffling on their home rows.** Exploration failure or reward plumbing failure. Check Dirichlet noise is actually applied (log the root prior before and after), check temperature schedule, check that z is nonzero for finished games in the buffer.

**Capped-game fraction climbing.** Stalling exploit; execute the adjudication switch per Section 8.4, and consider adding the plies-elapsed observation plane so the network can see the cap coming.

**Elo oscillates: each new network beats the last but loses to the one before.** Strategy cycling, endemic to self-play. Mitigations in order: enable gating if it is off; widen the replay window so old strategies stay represented; add ladder-anchored evaluation to promotion criteria (candidate must not regress by more than 25 Elo against the fixed ladder even if it beats the incumbent).

**Agent never places walls on 9x9 (or places them randomly).** Insufficient simulations to discover wall value, or the auxiliary head is missing so the trunk lacks path features, or FPU is punishing the huge wall-move fan-out too hard. Raise simulations for a diagnostic run, verify the auxiliary head is training (its loss should fall fast), soften FPU.

**GPU idle, games trickling.** Throughput inversion: the rules engine or the batching is too slow. Profile the engine against the Section 3.3 target first; then check inference batch-size telemetry against the 256 target; then raise concurrent games per worker.

**Loss spikes after a gating promotion.** Data distribution shift is normal; a persistent spike with falling Elo means the promoted network was a fluke of a small evaluation sample. Raise gating games from 200 to 400 and require 55 percent with a one-sided binomial test at p < 0.05 rather than a raw fraction.

**Everything works on 5x5, nothing transfers to 7x7.** Almost always the head reinitialization or an observation-shape assumption baked somewhere (a hardcoded 5, a flatten layer sized to the wrong board). The trunk is convolutional precisely so this cannot happen in the trunk; audit the heads and every reshape.

**Compute turns out to be a tenth of what was hoped.** Fall back in this order, sacrificing time-to-strength before correctness: switch to search Mode B (Gumbel, Section 9) at 64 to 128 simulations with m = 16, which is designed for exactly this regime and learns reliably where PUCT fails; enable reanalyze to sweat the buffer harder; smaller network (6 blocks, 64 filters, keep the global-pooling blocks, they are cheap); 9x9 with reduced walls (say 6 each) as an intermediate curriculum stage; and accept a weaker final agent rather than reintroducing reward shaping to "speed things up", which is the one fallback that is never on the table.

---

## 13. Milestones and Definition of Done

**M0, rules engine and test suite.** Engine meets the Section 3.3 throughput target; the entire Section 3.4 suite passes in CI; differential testing shows zero disagreements over 10,000 games. Nothing else starts before this is done.

**M1, environments and ladder.** Both interfaces of Section 4 built; ladder opponents 1 through 5 implemented and covered by tests; a PPO or DQN baseline trained through the Gym wrapper beats the random opponent convincingly (this validates observation, mask, and reward plumbing cheaply); dashboard skeleton live.

**M2, 5x5 solved-play reproduction.** All three acceptance criteria of Section 10.2 met. This is the pipeline's proof of correctness.

**M3, 7x7 transfer.** Trunk transfer tooling exercised; stable Elo plateau; qualitative review of 50 sampled games signed off by the team.

**M4, 9x9 production run.** Gated training to a plateau; agent beats depth-5 alpha-beta ladder opponent at 90 percent or better with both colours; final report includes the full Elo history, calibration plots, and an opening-move visit-distribution analysis (of genuine theoretical interest given the open question of which player wins 9x9 with perfect play).

**M5, hardening and release.** Reproducibility audit (a fresh machine reproduces M2 from config and seed); documentation pass; frozen final checkpoint added to the ladder for whoever comes after you.

---

## 14. Final Word

The order of operations in this document is the risk management strategy: rules correctness before environments, environments before learning, solved-board ground truth before open-board scale. The reward scheme is deliberately boring because boring is what converges; the cleverness budget is spent instead on the auxiliary targets, the search and self-play machinery of Appendix A, the curriculum, and the engineering throughput, which is where cleverness actually pays in this class of system. When in doubt, return to the north star sentence and the three questions under it. Good luck, and trust the Elo ladder over your intuitions about what "looks" stronger; it is the only voice in the room that never lies.

---

## Appendix A: Post-AlphaZero Technique Catalogue, Evidence, and Adoption Tiers

This appendix exists so that when someone questions a design choice in two years, the reasoning and its evidence base are on record, and so that nobody bolts on a technique from a paper without knowing which tier it belongs to.

**Tier 1, mandatory, integrated above.** These have strong published ablations, are domain-independent, and have survived years of replication in open-source systems. From KataGo (Wu, arXiv:1902.10565), which demonstrated roughly a 50x efficiency improvement over AlphaZero-era baselines with each technique contributing measurably in ablation: playout cap randomization (Section 10.1), forced playouts with policy target pruning (Section 9), global pooling blocks (Section 7), and the auxiliary opponent-policy target (Section 7); our shortest-path-distance auxiliary head is the Quoridor analogue of KataGo's game-specific ownership and score targets, which its ablations also validated as a class. Weight averaging for inference (Section 7) is standard across KataGo and the strongest reimplementations.

**Tier 2, conditional, integrated above with explicit decision rules.** Gumbel AlphaZero (Danihelka et al., ICLR 2022) replaces PUCT's heuristics at the root with sampling-without-replacement plus sequential halving and a completed-Q policy target, carries a formal policy-improvement guarantee that visit-count targets lack, matches classic AlphaZero at large budgets, and remains stable down to absurdly small ones (learning reliably with as few as two simulations in published experiments, where standard methods fail below sixteen); independent replication in the MiniZero and LightZero benchmark frameworks confirms the low-budget claim. Adopt per the Section 9 decision rule. Reanalyze (MuZero Unplugged lineage; ReZero, arXiv:2404.16364, shows large wall-clock gains and is benchmarked in LightZero) refreshes stale buffer policy targets with the current network; adopt when the dashboard shows learner starvation, per Section 10.1. Start-state diversification (Go-Exploit, Trudeau and Bowling, AAMAS 2023, which also verified compatibility with the KataGo suite) in the weak injected-move form is mandatory; the archive-sampling strong form is optional after M4 stabilizes.

**Tier 3, monitored, not adopted.** MuZero, EfficientZero, and their descendants learn a dynamics model, solving a problem we do not have: our simulator is perfect and fast, so paying model-learning's complexity tax buys nothing here. Transformer or hybrid trunks show no compelling advantage over CNN-plus-global-pooling at 9x9 scale in the games literature to date. Uncertainty-guided self-play exploration (NeurIPS 2025) is promising but young; revisit if the project outlives this document's assumptions. The rule for anything not listed: it enters through a 5x5 A/B run against the current pipeline with the ladder as judge, or it does not enter at all.
