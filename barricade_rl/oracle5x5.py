from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .config import config_hash, load_config, small_game_from_config
from .game import TerminalStatus
from .small_board import SmallGame, SmallState, SolverOutcome

INFINITY = 10**12


@dataclass(frozen=True, slots=True)
class ProofSearchConfig:
    max_depth: int = 12
    max_nodes: int = 100_000

    def __post_init__(self) -> None:
        if self.max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        if self.max_nodes < 1:
            raise ValueError("max_nodes must be positive")


DEFAULT_PROOF_CONFIG = ProofSearchConfig()


@dataclass(frozen=True, slots=True)
class ProofNumberConfig:
    max_nodes: int = 100_000

    def __post_init__(self) -> None:
        if self.max_nodes < 1:
            raise ValueError("max_nodes must be positive")


DEFAULT_PROOF_NUMBER_CONFIG = ProofNumberConfig()


@dataclass(frozen=True, slots=True)
class LowWallEndgameConfig:
    max_walls_remaining: int = 1
    max_nodes: int = 500_000

    def __post_init__(self) -> None:
        if self.max_walls_remaining < 0:
            raise ValueError("max_walls_remaining must be non-negative")
        if self.max_nodes < 1:
            raise ValueError("max_nodes must be positive")


DEFAULT_LOW_WALL_CONFIG = LowWallEndgameConfig()


@dataclass(frozen=True, slots=True)
class OracleLabel:
    method: str
    board_size: int
    walls_per_player: int
    state_key: str
    ply: int
    current_player: int
    outcome: SolverOutcome
    value: int | None
    exact: bool
    exhausted: bool
    best_action: int | None
    depth: int
    nodes: int
    legal_action_count: int
    distances: tuple[int | None, int | None]
    proof_number: int | None = None
    disproof_number: int | None = None

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "board_size": self.board_size,
            "walls_per_player": self.walls_per_player,
            "state_key": self.state_key,
            "ply": self.ply,
            "current_player": self.current_player,
            "outcome": self.outcome.value,
            "value": self.value,
            "exact": self.exact,
            "exhausted": self.exhausted,
            "best_action": self.best_action,
            "depth": self.depth,
            "nodes": self.nodes,
            "legal_action_count": self.legal_action_count,
            "distances": list(self.distances),
            "proof_number": self.proof_number,
            "disproof_number": self.disproof_number,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "OracleLabel":
        return cls(
            method=str(payload.get("method", "minimax")),
            board_size=int(payload["board_size"]),
            walls_per_player=int(payload["walls_per_player"]),
            state_key=str(payload["state_key"]),
            ply=int(payload["ply"]),
            current_player=int(payload["current_player"]),
            outcome=SolverOutcome(str(payload["outcome"])),
            value=None if payload["value"] is None else int(payload["value"]),
            exact=bool(payload["exact"]),
            exhausted=bool(payload["exhausted"]),
            best_action=None if payload["best_action"] is None else int(payload["best_action"]),
            depth=int(payload["depth"]),
            nodes=int(payload["nodes"]),
            legal_action_count=int(payload["legal_action_count"]),
            distances=tuple(payload["distances"]),  # type: ignore[arg-type]
            proof_number=None if payload.get("proof_number") is None else int(payload["proof_number"]),
            disproof_number=None if payload.get("disproof_number") is None else int(payload["disproof_number"]),
        )


@dataclass(frozen=True, slots=True)
class CorpusSummary:
    path: str
    records: int
    exact_records: int
    unknown_records: int
    exhausted_records: int
    config_path: str
    config_hash: str
    proof: ProofSearchConfig
    seed: int
    random_plies: int
    sampling: str
    method: str
    proof_number: ProofNumberConfig | None = None
    low_wall: LowWallEndgameConfig | None = None
    proof_cache_records: int = 0
    proof_cache_hits: int = 0
    proof_cache_path: str | None = None
    shard_index: int = 0
    shard_count: int = 1
    refined_records: int = 0

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["proof"] = asdict(self.proof)
        if self.proof_number is not None:
            payload["proof_number"] = asdict(self.proof_number)
        if self.low_wall is not None:
            payload["low_wall"] = asdict(self.low_wall)
        return payload


@dataclass(frozen=True, slots=True)
class CorpusMergeSummary:
    path: str
    inputs: tuple[str, ...]
    records: int
    duplicate_records: int
    exact_records: int
    unknown_records: int
    exhausted_records: int
    record_index_min: int | None
    record_index_max: int | None
    config_hashes: tuple[str, ...]
    methods: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CorpusCompactionSummary:
    path: str
    inputs: tuple[str, ...]
    requested_records: int
    input_records: int
    exact_candidates: int
    non_exact_skipped: int
    terminal_records_skipped: int
    duplicate_states_skipped: int
    records: int
    requested_phase_buckets: dict[str, int]
    candidate_phase_buckets: dict[str, int]
    selected_phase_buckets: dict[str, int]
    config_hash: str
    methods: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CorpusAuditSummary:
    path: str
    records: int
    min_records: int
    exact_records: int
    exact_fraction: float
    min_exact_fraction: float
    unknown_records: int
    exhausted_records: int
    terminal_records: int
    duplicate_state_keys: int
    duplicate_record_indices: int
    config_path: str | None
    expected_config_hash: str | None
    config_hash_mismatches: int
    config_hashes: tuple[str, ...]
    methods: dict[str, int]
    outcomes: dict[str, int]
    phase_buckets: dict[str, int]
    exact_phase_buckets: dict[str, int]
    min_phase_records: int
    passed: bool
    failures: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _ProofResult:
    outcome: SolverOutcome
    value: int | None
    best_action: int | None
    nodes: int
    exhausted: bool


@dataclass(frozen=True, slots=True)
class _EndgameResult:
    outcome: SolverOutcome
    value: int | None
    best_action: int | None
    exhausted: bool


@dataclass(frozen=True, slots=True)
class _CachedProofNumberResult:
    value: int
    best_action: int | None


@dataclass(slots=True)
class _PNNode:
    state: SmallState
    proof_number: int = 1
    disproof_number: int = 1
    expanded: bool = False
    terminal: bool = False
    disproof_is_loss: bool = False
    best_action: int | None = None
    children: dict[int, "_PNNode"] | None = None


class ProofNumberSolvedCache:
    """Reusable exact win/loss cache for proof-number searches.

    Only proven binary outcomes are stored. Draws and budget-exhausted unknowns
    are deliberately excluded because the current proof-number search is a
    win/loss prover rather than a full draw classifier.
    """

    def __init__(self, game: SmallGame) -> None:
        self.game = game
        self._values: dict[bytes, _CachedProofNumberResult] = {}
        self.hits = 0

    @property
    def cache_size(self) -> int:
        return len(self._values)

    def lookup(self, state: SmallState) -> _CachedProofNumberResult | None:
        result = self._values.get(self.game.state_key(state))
        if result is not None:
            self.hits += 1
        return result

    def store(self, state: SmallState, *, value: int, best_action: int | None = None) -> None:
        if value not in (-1, 1):
            return
        key = self.game.state_key(state)
        existing = self._values.get(key)
        if existing is None or (existing.best_action is None and best_action is not None):
            self._values[key] = _CachedProofNumberResult(value=value, best_action=best_action)

    def load_jsonl(self, path: str | Path) -> None:
        source = Path(path)
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if int(payload["board_size"]) != self.game.board_size:
                raise ValueError("proof cache board_size does not match the game")
            if int(payload["walls_per_player"]) != self.game.spec.walls_per_player:
                raise ValueError("proof cache walls_per_player does not match the game")
            value = int(payload["value"])
            if value not in (-1, 1):
                raise ValueError("proof cache may only contain exact win/loss values")
            state = SmallState.from_key(self.game.spec, bytes.fromhex(str(payload["state_key"])))
            best_action = None if payload.get("best_action") is None else int(payload["best_action"])
            self.store(state, value=value, best_action=best_action)

    def save_jsonl(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_name(f"{output.name}.tmp")
        with temp.open("w", encoding="utf-8") as handle:
            for key, result in sorted(self._values.items()):
                payload = {
                    "best_action": result.best_action,
                    "board_size": self.game.board_size,
                    "method": "proof-number",
                    "state_key": key.hex(),
                    "value": result.value,
                    "walls_per_player": self.game.spec.walls_per_player,
                }
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        temp.replace(output)


def prove_state(game: SmallGame, state: SmallState, proof: ProofSearchConfig | None = None) -> OracleLabel:
    proof = proof or ProofSearchConfig()
    memo: dict[tuple[bytes, int], _ProofResult] = {}
    nodes = 0

    def search(current: SmallState, depth: int) -> _ProofResult:
        nonlocal nodes
        if nodes >= proof.max_nodes:
            return _ProofResult(SolverOutcome.UNKNOWN, None, None, nodes, True)
        nodes += 1

        status = game.is_terminal(current)
        if status is TerminalStatus.MOVER_LOST:
            return _ProofResult(SolverOutcome.LOSS, -1, None, 1, False)
        if status is TerminalStatus.CAPPED:
            return _ProofResult(SolverOutcome.DRAW, 0, None, 1, False)
        if depth == 0:
            return _ProofResult(SolverOutcome.UNKNOWN, None, None, 1, False)

        key = (game.state_key(current), depth)
        if key in memo:
            return memo[key]

        first_unknown: int | None = None
        first_draw: int | None = None
        any_exhausted = False
        searched_children = 0
        child_node_total = 0
        actions = _ordered_actions(game, current)
        for action in actions:
            child = game.next_state(current, action)
            before = nodes
            child_result = search(child, depth - 1)
            child_node_total += max(0, nodes - before)
            searched_children += 1
            any_exhausted = any_exhausted or child_result.exhausted
            if child_result.outcome is SolverOutcome.LOSS:
                result = _ProofResult(SolverOutcome.WIN, 1, action, 1 + child_node_total, any_exhausted)
                memo[key] = result
                return result
            if child_result.outcome is SolverOutcome.DRAW and first_draw is None:
                first_draw = action
            if child_result.outcome is SolverOutcome.UNKNOWN and first_unknown is None:
                first_unknown = action
            if child_result.exhausted:
                break

        if any_exhausted:
            result = _ProofResult(SolverOutcome.UNKNOWN, None, first_unknown, 1 + child_node_total, True)
        elif first_unknown is not None:
            result = _ProofResult(SolverOutcome.UNKNOWN, None, first_unknown, 1 + child_node_total, False)
        elif first_draw is not None:
            result = _ProofResult(SolverOutcome.DRAW, 0, first_draw, 1 + child_node_total, False)
        elif searched_children == len(actions):
            result = _ProofResult(SolverOutcome.LOSS, -1, actions[0] if actions else None, 1 + child_node_total, False)
        else:
            result = _ProofResult(SolverOutcome.UNKNOWN, None, first_unknown, 1 + child_node_total, True)
        memo[key] = result
        return result

    result = search(state, proof.max_depth)
    legal_count = int(game.legal_actions(state).sum())
    exact = result.outcome is not SolverOutcome.UNKNOWN and not result.exhausted
    return OracleLabel(
        method="minimax",
        board_size=game.board_size,
        walls_per_player=game.spec.walls_per_player,
        state_key=game.state_key(state).hex(),
        ply=state.ply,
        current_player=state.current_player,
        outcome=result.outcome,
        value=result.value,
        exact=exact,
        exhausted=result.exhausted,
        best_action=result.best_action,
        depth=proof.max_depth,
        nodes=min(nodes, proof.max_nodes),
        legal_action_count=legal_count,
        distances=(
            game.shortest_path_distance(state, 0),
            game.shortest_path_distance(state, 1),
        ),
    )


def proof_number_search(
    game: SmallGame,
    state: SmallState,
    config: ProofNumberConfig | None = None,
    solved_cache: ProofNumberSolvedCache | None = None,
) -> OracleLabel:
    config = config or ProofNumberConfig()
    cached_root = solved_cache.lookup(state) if solved_cache is not None else None
    if cached_root is not None:
        return _proof_number_label_from_cached(game, state, cached_root)

    root_player = state.current_player
    root = _PNNode(state)
    nodes_created = 1
    exhausted = False

    def initialize(node: _PNNode) -> None:
        status = game.is_terminal(node.state)
        if status is TerminalStatus.MOVER_LOST:
            node.terminal = True
            node.expanded = True
            winner = 1 - node.state.current_player
            if winner == root_player:
                node.proof_number = 0
                node.disproof_number = INFINITY
                node.disproof_is_loss = False
            else:
                node.proof_number = INFINITY
                node.disproof_number = 0
                node.disproof_is_loss = True
            return
        if status is TerminalStatus.CAPPED:
            node.terminal = True
            node.expanded = True
            node.proof_number = INFINITY
            node.disproof_number = 0
            node.disproof_is_loss = False
            return
        cached = solved_cache.lookup(node.state) if solved_cache is not None else None
        if cached is not None:
            _apply_cached_proof_number_result(node, cached, root_player)
            return
        node.proof_number = 1
        node.disproof_number = 1
        node.disproof_is_loss = False

    def store_if_solved(node: _PNNode) -> None:
        if solved_cache is None:
            return
        value = _node_value_for_mover(node, root_player)
        if value is not None:
            solved_cache.store(node.state, value=value, best_action=node.best_action)

    def update(node: _PNNode) -> None:
        if node.terminal:
            return
        if not node.expanded or not node.children:
            node.proof_number = 1
            node.disproof_number = 1
            node.disproof_is_loss = False
            node.best_action = None
            return

        children = list(node.children.items())
        if node.state.current_player == root_player:
            best_action, best_child = min(children, key=lambda item: (item[1].proof_number, item[0]))
            node.proof_number = best_child.proof_number
            node.disproof_number = _capped_sum(child.disproof_number for _, child in children)
            node.best_action = int(best_action)
            node.disproof_is_loss = (
                node.disproof_number == 0
                and all(child.disproof_number == 0 and child.disproof_is_loss for _, child in children)
            )
        else:
            best_action, best_child = min(children, key=lambda item: (item[1].disproof_number, item[0]))
            node.proof_number = _capped_sum(child.proof_number for _, child in children)
            node.disproof_number = best_child.disproof_number
            node.best_action = int(best_action)
            node.disproof_is_loss = (
                node.disproof_number == 0
                and any(child.disproof_number == 0 and child.disproof_is_loss for _, child in children)
            )

    initialize(root)
    while root.proof_number != 0 and root.disproof_number != 0:
        path = [root]
        node = root
        while node.expanded and node.children and not node.terminal:
            if node.state.current_player == root_player:
                _, node = min(node.children.items(), key=lambda item: (item[1].proof_number, item[0]))
            else:
                _, node = min(node.children.items(), key=lambda item: (item[1].disproof_number, item[0]))
            path.append(node)
            if node.proof_number == 0 or node.disproof_number == 0:
                break

        if node.terminal or node.proof_number == 0 or node.disproof_number == 0:
            break

        actions = _ordered_actions(game, node.state)
        if nodes_created + len(actions) > config.max_nodes:
            exhausted = True
            break
        node.expanded = True
        node.children = {}
        for action in actions:
            child = _PNNode(game.next_state(node.state, action))
            initialize(child)
            store_if_solved(child)
            node.children[int(action)] = child
            nodes_created += 1

        for ancestor in reversed(path):
            update(ancestor)
            store_if_solved(ancestor)

    legal_count = int(game.legal_actions(state).sum())
    outcome = SolverOutcome.UNKNOWN
    value: int | None = None
    exact = False
    if root.proof_number == 0:
        outcome = SolverOutcome.WIN
        value = 1
        exact = not exhausted
    elif root.disproof_number == 0 and root.disproof_is_loss:
        outcome = SolverOutcome.LOSS
        value = -1
        exact = not exhausted

    if exhausted:
        outcome = SolverOutcome.UNKNOWN
        value = None
        exact = False

    label = OracleLabel(
        method="proof-number",
        board_size=game.board_size,
        walls_per_player=game.spec.walls_per_player,
        state_key=game.state_key(state).hex(),
        ply=state.ply,
        current_player=state.current_player,
        outcome=outcome,
        value=value,
        exact=exact,
        exhausted=exhausted,
        best_action=root.best_action,
        depth=0,
        nodes=min(nodes_created, config.max_nodes),
        legal_action_count=legal_count,
        distances=(
            game.shortest_path_distance(state, 0),
            game.shortest_path_distance(state, 1),
        ),
        proof_number=_public_number(root.proof_number),
        disproof_number=_public_number(root.disproof_number),
    )
    if solved_cache is not None and label.exact and label.value in (-1, 1):
        solved_cache.store(state, value=int(label.value), best_action=label.best_action)
    return label


class NoWallTablebase:
    def __init__(self, game: SmallGame) -> None:
        self.game = game
        self._memo: dict[bytes, tuple[int, int | None]] = {}

    @property
    def cache_size(self) -> int:
        return len(self._memo)

    def solve(self, state: SmallState) -> OracleLabel:
        if state.walls_remaining != (0, 0):
            raise ValueError("no-wall endgame tablebase requires both wall counters to be zero")
        value, best_action = self._search(state)
        outcome = SolverOutcome.WIN if value == 1 else SolverOutcome.LOSS if value == -1 else SolverOutcome.DRAW
        return OracleLabel(
            method="no-wall-tablebase",
            board_size=self.game.board_size,
            walls_per_player=self.game.spec.walls_per_player,
            state_key=self.game.state_key(state).hex(),
            ply=state.ply,
            current_player=state.current_player,
            outcome=outcome,
            value=value,
            exact=True,
            exhausted=False,
            best_action=best_action,
            depth=self.game.spec.max_plies - state.ply,
            nodes=len(self._memo),
            legal_action_count=int(self.game.legal_actions(state).sum()),
            distances=(
                self.game.shortest_path_distance(state, 0),
                self.game.shortest_path_distance(state, 1),
            ),
        )

    def _search(self, current: SmallState) -> tuple[int, int | None]:
        key = self.game.state_key(current)
        if key in self._memo:
            return self._memo[key]

        status = self.game.is_terminal(current)
        if status is not TerminalStatus.NOT_TERMINAL:
            value = int(self.game.terminal_value(current))
            result = (value, None)
            self._memo[key] = result
            return result

        best_value = -2
        best_action: int | None = None
        for action in _pawn_actions(self.game, current):
            child = self.game.next_state(current, action)
            child_value, _ = self._search(child)
            value = -child_value
            if value > best_value or (value == best_value and (best_action is None or action < best_action)):
                best_value = value
                best_action = action
            if best_value == 1:
                break
        if best_action is None:
            best_value = 0
        result = (best_value, best_action)
        self._memo[key] = result
        return result


def solve_no_wall_endgame(game: SmallGame, state: SmallState) -> OracleLabel:
    return NoWallTablebase(game).solve(state)


class LowWallEndgameSolver:
    """Exact finite-horizon solver for states with few unplaced walls remaining.

    The solver is deliberately conservative under budget pressure: it returns
    UNKNOWN instead of inferring losses/draws unless every relevant child has
    been proven, except that a single proven winning action is enough to label
    the root as an exact win.
    """

    def __init__(self, game: SmallGame, config: LowWallEndgameConfig | None = None) -> None:
        self.game = game
        self.config = config or LowWallEndgameConfig()
        self._memo: dict[bytes, _EndgameResult] = {}

    @property
    def cache_size(self) -> int:
        return len(self._memo)

    def can_solve(self, state: SmallState) -> bool:
        self.game.state_key(state)
        return sum(state.walls_remaining) <= self.config.max_walls_remaining

    def solve(self, state: SmallState) -> OracleLabel:
        if not self.can_solve(state):
            raise ValueError("low-wall endgame solver requires the remaining wall count to fit the configured budget")

        nodes = 0

        def search(current: SmallState) -> _EndgameResult:
            nonlocal nodes
            key = self.game.state_key(current)
            if key in self._memo:
                return self._memo[key]
            if nodes >= self.config.max_nodes:
                return _EndgameResult(SolverOutcome.UNKNOWN, None, None, True)
            nodes += 1

            status = self.game.is_terminal(current)
            if status is not TerminalStatus.NOT_TERMINAL:
                value = int(self.game.terminal_value(current))
                result = _EndgameResult(_outcome_from_value(value), value, None, False)
                self._memo[key] = result
                return result

            legal_actions = _ordered_actions(self.game, current)
            if not legal_actions:
                result = _EndgameResult(SolverOutcome.DRAW, 0, None, False)
                self._memo[key] = result
                return result

            first_draw: int | None = None
            first_unknown: int | None = None
            exhausted = False
            fallback_loss_action = int(legal_actions[0])

            for action in legal_actions:
                child = self.game.next_state(current, action)
                child_result = search(child)
                if child_result.value is None:
                    if first_unknown is None:
                        first_unknown = int(action)
                    exhausted = exhausted or child_result.exhausted
                    if child_result.exhausted:
                        break
                    continue

                value = -child_result.value
                if value == 1:
                    result = _EndgameResult(SolverOutcome.WIN, 1, int(action), False)
                    self._memo[key] = result
                    return result
                if value == 0 and first_draw is None:
                    first_draw = int(action)

            if first_unknown is not None:
                return _EndgameResult(SolverOutcome.UNKNOWN, None, first_unknown, exhausted)
            if first_draw is not None:
                result = _EndgameResult(SolverOutcome.DRAW, 0, first_draw, False)
            else:
                result = _EndgameResult(SolverOutcome.LOSS, -1, fallback_loss_action, False)
            self._memo[key] = result
            return result

        result = search(state)
        exact = result.outcome is not SolverOutcome.UNKNOWN and not result.exhausted
        return OracleLabel(
            method="low-wall-endgame",
            board_size=self.game.board_size,
            walls_per_player=self.game.spec.walls_per_player,
            state_key=self.game.state_key(state).hex(),
            ply=state.ply,
            current_player=state.current_player,
            outcome=result.outcome,
            value=result.value,
            exact=exact,
            exhausted=result.exhausted,
            best_action=result.best_action,
            depth=self.game.spec.max_plies - state.ply,
            nodes=min(nodes, self.config.max_nodes),
            legal_action_count=int(self.game.legal_actions(state).sum()),
            distances=(
                self.game.shortest_path_distance(state, 0),
                self.game.shortest_path_distance(state, 1),
            ),
        )


def hybrid_oracle_label(
    game: SmallGame,
    state: SmallState,
    *,
    proof: ProofSearchConfig | None = None,
    proof_number: ProofNumberConfig | None = None,
    tablebase: NoWallTablebase | None = None,
    low_wall_solver: LowWallEndgameSolver | None = None,
    proof_cache: ProofNumberSolvedCache | None = None,
) -> OracleLabel:
    if state.walls_remaining == (0, 0):
        return (tablebase or NoWallTablebase(game)).solve(state)
    if low_wall_solver is not None and low_wall_solver.can_solve(state):
        return low_wall_solver.solve(state)
    return proof_number_search(game, state, proof_number or ProofNumberConfig(), solved_cache=proof_cache)


def generate_oracle_corpus(
    path: str | Path,
    *,
    config_path: str | Path,
    positions: int,
    seed: int,
    random_plies: int,
    proof: ProofSearchConfig | None = None,
    method: str = "minimax",
    proof_number: ProofNumberConfig | None = None,
    low_wall: LowWallEndgameConfig | None = None,
    sampling: str = "random",
    proof_cache_in: str | Path | None = None,
    proof_cache_out: str | Path | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> CorpusSummary:
    if positions < 1:
        raise ValueError("positions must be positive")
    if random_plies < 0:
        raise ValueError("random_plies must be non-negative")
    _validate_shard(shard_index, shard_count)
    if method not in {"minimax", "proof-number", "hybrid"}:
        raise ValueError("method must be 'minimax', 'proof-number', or 'hybrid'")
    if method == "minimax" and (proof_cache_in is not None or proof_cache_out is not None):
        raise ValueError("proof cache paths require method 'proof-number' or 'hybrid'")
    if sampling not in {"random", "no-wall", "low-wall"}:
        raise ValueError("sampling must be 'random', 'no-wall', or 'low-wall'")
    proof = proof or ProofSearchConfig()
    proof_number = proof_number or ProofNumberConfig()
    if sampling == "low-wall" and low_wall is None:
        low_wall = LowWallEndgameConfig()
    config = load_config(config_path)
    game = small_game_from_config(config)
    rng = np.random.default_rng(seed)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels: list[OracleLabel] = []
    tablebase = NoWallTablebase(game) if method == "hybrid" else None
    low_wall_solver = LowWallEndgameSolver(game, low_wall) if method == "hybrid" and low_wall is not None else None
    proof_cache = ProofNumberSolvedCache(game) if method in {"proof-number", "hybrid"} else None
    if proof_cache is not None and proof_cache_in is not None:
        proof_cache.load_jsonl(proof_cache_in)

    with output.open("w", encoding="utf-8") as handle:
        for index in range(positions):
            state = _sample_state(
                game,
                rng,
                random_plies=random_plies,
                sampling=sampling,
                low_wall_max_remaining=low_wall.max_walls_remaining if low_wall is not None else 0,
            )
            if index % shard_count != shard_index:
                continue
            label = _label_state(
                game,
                state,
                method=method,
                proof=proof,
                proof_number=proof_number,
                tablebase=tablebase,
                low_wall_solver=low_wall_solver,
                proof_cache=proof_cache,
            )
            payload = label.to_dict()
            payload["record_index"] = index
            payload["config_hash"] = config_hash(config)
            payload["proof"] = asdict(proof)
            if method in {"proof-number", "hybrid"}:
                payload["proof_number_config"] = asdict(proof_number)
            if low_wall is not None:
                payload["low_wall_config"] = asdict(low_wall)
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            labels.append(label)

    if proof_cache is not None and proof_cache_out is not None:
        proof_cache.save_jsonl(proof_cache_out)

    return CorpusSummary(
        path=str(output),
        records=len(labels),
        exact_records=sum(1 for label in labels if label.exact),
        unknown_records=sum(1 for label in labels if label.outcome is SolverOutcome.UNKNOWN),
        exhausted_records=sum(1 for label in labels if label.exhausted),
        config_path=str(config_path),
        config_hash=config_hash(config),
        proof=proof,
        seed=seed,
        random_plies=random_plies,
        sampling=sampling,
        method=method,
        proof_number=proof_number if method in {"proof-number", "hybrid"} else None,
        low_wall=low_wall,
        proof_cache_records=proof_cache.cache_size if proof_cache is not None else 0,
        proof_cache_hits=proof_cache.hits if proof_cache is not None else 0,
        proof_cache_path=str(proof_cache_out or proof_cache_in) if proof_cache is not None and (proof_cache_out or proof_cache_in) else None,
        shard_index=shard_index,
        shard_count=shard_count,
    )


def load_oracle_corpus(path: str | Path) -> list[OracleLabel]:
    labels = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        labels.append(OracleLabel.from_dict(json.loads(line)))
    return labels


def merge_oracle_corpora(inputs: Sequence[str | Path], output_path: str | Path) -> CorpusMergeSummary:
    if not inputs:
        raise ValueError("at least one input corpus is required")

    by_record_index: dict[int, dict] = {}
    duplicate_records = 0
    for input_path in inputs:
        source = Path(input_path)
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if "record_index" not in payload:
                raise ValueError(f"{source}:{line_number} is missing record_index")
            record_index = int(payload["record_index"])
            if record_index in by_record_index:
                duplicate_records += 1
                raise ValueError(f"duplicate record_index {record_index} while merging {source}")
            by_record_index[record_index] = payload

    ordered = [by_record_index[index] for index in sorted(by_record_index)]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(payload, sort_keys=True) + "\n" for payload in ordered),
        encoding="utf-8",
    )

    labels = [OracleLabel.from_dict(payload) for payload in ordered]
    indices = sorted(by_record_index)
    config_hashes = sorted(
        {
            str(payload["config_hash"])
            for payload in ordered
            if payload.get("config_hash") is not None
        }
    )
    return CorpusMergeSummary(
        path=str(output),
        inputs=tuple(str(Path(path)) for path in inputs),
        records=len(ordered),
        duplicate_records=duplicate_records,
        exact_records=sum(1 for label in labels if label.exact),
        unknown_records=sum(1 for label in labels if label.outcome is SolverOutcome.UNKNOWN),
        exhausted_records=sum(1 for label in labels if label.exhausted),
        record_index_min=indices[0] if indices else None,
        record_index_max=indices[-1] if indices else None,
        config_hashes=tuple(config_hashes),
        methods=tuple(sorted({label.method for label in labels})),
    )


def compact_exact_oracle_corpora(
    inputs: Sequence[str | Path],
    output_path: str | Path,
    *,
    config_path: str | Path,
    records: int = 5000,
) -> CorpusCompactionSummary:
    """Build a balanced exact validation corpus from independent sampling runs."""
    if not inputs:
        raise ValueError("at least one input corpus is required")
    if records < 3:
        raise ValueError("records must be at least three for phase balancing")
    config = load_config(config_path)
    expected_hash = config_hash(config)
    game = small_game_from_config(config)
    requested = _balanced_phase_quotas(records)
    candidates: dict[str, list[tuple[dict, Path, int | None]]] = _empty_phase_lists()
    seen_state_keys: set[str] = set()
    input_records = 0
    non_exact_skipped = 0
    terminal_records_skipped = 0
    duplicate_states_skipped = 0

    for input_path in inputs:
        source = Path(input_path)
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            input_records += 1
            payload = json.loads(line)
            payload_hash = payload.get("config_hash")
            if payload_hash != expected_hash:
                raise ValueError(
                    f"{source}:{line_number} config hash {payload_hash!r} does not match {expected_hash}"
                )
            label = OracleLabel.from_dict(payload)
            if not label.exact:
                non_exact_skipped += 1
                continue
            if label.legal_action_count <= 0:
                terminal_records_skipped += 1
                continue
            if label.state_key in seen_state_keys:
                duplicate_states_skipped += 1
                continue
            seen_state_keys.add(label.state_key)
            phase = _phase_bucket(label.ply, game.spec.max_plies)
            candidates[phase].append((payload, source, payload.get("record_index")))

    candidate_counts = {phase: len(values) for phase, values in candidates.items()}
    for phase, required in requested.items():
        available = candidate_counts[phase]
        if available < required:
            raise ValueError(
                f"exact {phase} candidates {available} are below required compaction quota {required}"
            )

    selected: list[dict] = []
    selected_counts = _empty_phase_buckets()
    for phase in ("opening", "midgame", "endgame"):
        for payload, source, source_index in candidates[phase][: requested[phase]]:
            compacted = dict(payload)
            compacted["source_corpus"] = str(source)
            compacted["source_record_index"] = source_index
            compacted["record_index"] = len(selected)
            selected.append(compacted)
            selected_counts[phase] += 1

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(payload, sort_keys=True) + "\n" for payload in selected),
        encoding="utf-8",
    )
    labels = [OracleLabel.from_dict(payload) for payload in selected]
    return CorpusCompactionSummary(
        path=str(output),
        inputs=tuple(str(Path(path)) for path in inputs),
        requested_records=records,
        input_records=input_records,
        exact_candidates=sum(candidate_counts.values()),
        non_exact_skipped=non_exact_skipped,
        terminal_records_skipped=terminal_records_skipped,
        duplicate_states_skipped=duplicate_states_skipped,
        records=len(selected),
        requested_phase_buckets=requested,
        candidate_phase_buckets=candidate_counts,
        selected_phase_buckets=selected_counts,
        config_hash=expected_hash,
        methods=tuple(sorted({label.method for label in labels})),
    )


def audit_oracle_corpus(
    path: str | Path,
    *,
    config_path: str | Path | None = None,
    min_records: int = 5000,
    min_exact_fraction: float = 1.0,
    min_phase_records: int = 1,
) -> CorpusAuditSummary:
    if min_records < 1:
        raise ValueError("min_records must be positive")
    if not 0.0 <= min_exact_fraction <= 1.0:
        raise ValueError("min_exact_fraction must be between 0 and 1")
    if min_phase_records < 0:
        raise ValueError("min_phase_records must be non-negative")

    expected_config_hash: str | None = None
    max_plies = SmallGame().spec.max_plies
    if config_path is not None:
        config = load_config(config_path)
        expected_config_hash = config_hash(config)
        max_plies = small_game_from_config(config).spec.max_plies

    source = Path(path)
    labels: list[OracleLabel] = []
    config_hashes: set[str] = set()
    config_hash_mismatches = 0
    duplicate_state_keys = 0
    duplicate_record_indices = 0
    seen_state_keys: set[str] = set()
    seen_record_indices: set[int] = set()
    methods: dict[str, int] = {}
    outcomes: dict[str, int] = {}
    phase_buckets = _empty_phase_buckets()
    exact_phase_buckets = _empty_phase_buckets()

    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        label = OracleLabel.from_dict(payload)
        labels.append(label)

        state_key = label.state_key
        if state_key in seen_state_keys:
            duplicate_state_keys += 1
        else:
            seen_state_keys.add(state_key)

        if payload.get("record_index") is not None:
            record_index = int(payload["record_index"])
            if record_index in seen_record_indices:
                duplicate_record_indices += 1
            else:
                seen_record_indices.add(record_index)

        payload_hash = payload.get("config_hash")
        if payload_hash is not None:
            config_hashes.add(str(payload_hash))
        if expected_config_hash is not None and payload_hash != expected_config_hash:
            config_hash_mismatches += 1

        methods[label.method] = methods.get(label.method, 0) + 1
        outcomes[label.outcome.value] = outcomes.get(label.outcome.value, 0) + 1
        phase = _phase_bucket(label.ply, max_plies)
        phase_buckets[phase] += 1
        if label.exact and label.legal_action_count > 0:
            exact_phase_buckets[phase] += 1

    records = len(labels)
    exact_records = sum(1 for label in labels if label.exact)
    unknown_records = sum(1 for label in labels if label.outcome is SolverOutcome.UNKNOWN)
    exhausted_records = sum(1 for label in labels if label.exhausted)
    terminal_records = sum(1 for label in labels if label.legal_action_count <= 0)
    exact_fraction = exact_records / records if records else 0.0

    failures: list[str] = []
    if records < min_records:
        failures.append(f"record count {records} is below required minimum {min_records}")
    if exact_fraction < min_exact_fraction:
        failures.append(f"exact fraction {exact_fraction:.6f} is below required minimum {min_exact_fraction:.6f}")
    if duplicate_state_keys:
        failures.append(f"duplicate state_key count is {duplicate_state_keys}")
    if duplicate_record_indices:
        failures.append(f"duplicate record_index count is {duplicate_record_indices}")
    if terminal_records:
        failures.append(f"terminal record count is {terminal_records}")
    if config_hash_mismatches:
        failures.append(f"config hash mismatch count is {config_hash_mismatches}")
    if min_phase_records:
        for phase, count in exact_phase_buckets.items():
            if count < min_phase_records:
                failures.append(
                    f"exact {phase} phase count {count} is below required minimum {min_phase_records}"
                )

    return CorpusAuditSummary(
        path=str(source),
        records=records,
        min_records=min_records,
        exact_records=exact_records,
        exact_fraction=exact_fraction,
        min_exact_fraction=min_exact_fraction,
        unknown_records=unknown_records,
        exhausted_records=exhausted_records,
        terminal_records=terminal_records,
        duplicate_state_keys=duplicate_state_keys,
        duplicate_record_indices=duplicate_record_indices,
        config_path=None if config_path is None else str(config_path),
        expected_config_hash=expected_config_hash,
        config_hash_mismatches=config_hash_mismatches,
        config_hashes=tuple(sorted(config_hashes)),
        methods=dict(sorted(methods.items())),
        outcomes=dict(sorted(outcomes.items())),
        phase_buckets=phase_buckets,
        exact_phase_buckets=exact_phase_buckets,
        min_phase_records=min_phase_records,
        passed=not failures,
        failures=tuple(failures),
    )


def _label_state(
    game: SmallGame,
    state: SmallState,
    *,
    method: str,
    proof: ProofSearchConfig,
    proof_number: ProofNumberConfig,
    tablebase: NoWallTablebase | None = None,
    low_wall_solver: LowWallEndgameSolver | None = None,
    proof_cache: ProofNumberSolvedCache | None = None,
) -> OracleLabel:
    if method == "hybrid":
        return hybrid_oracle_label(
            game,
            state,
            proof=proof,
            proof_number=proof_number,
            tablebase=tablebase,
            low_wall_solver=low_wall_solver,
            proof_cache=proof_cache,
        )
    if method == "proof-number":
        return proof_number_search(game, state, proof_number, solved_cache=proof_cache)
    if method == "minimax":
        return prove_state(game, state, proof)
    raise ValueError("method must be 'minimax', 'proof-number', or 'hybrid'")


def refine_oracle_corpus(
    input_path: str | Path,
    output_path: str | Path,
    *,
    config_path: str | Path,
    method: str = "hybrid",
    proof: ProofSearchConfig | None = None,
    proof_number: ProofNumberConfig | None = None,
    low_wall: LowWallEndgameConfig | None = None,
    proof_cache_in: str | Path | None = None,
    proof_cache_out: str | Path | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
    force: bool = False,
) -> CorpusSummary:
    if method not in {"minimax", "proof-number", "hybrid"}:
        raise ValueError("method must be 'minimax', 'proof-number', or 'hybrid'")
    if method == "minimax" and (proof_cache_in is not None or proof_cache_out is not None):
        raise ValueError("proof cache paths require method 'proof-number' or 'hybrid'")
    _validate_shard(shard_index, shard_count)
    proof = proof or ProofSearchConfig()
    proof_number = proof_number or ProofNumberConfig()
    config = load_config(config_path)
    game = small_game_from_config(config)
    source = Path(input_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tablebase = NoWallTablebase(game) if method == "hybrid" else None
    low_wall_solver = LowWallEndgameSolver(game, low_wall) if method == "hybrid" and low_wall is not None else None
    proof_cache = ProofNumberSolvedCache(game) if method in {"proof-number", "hybrid"} else None
    if proof_cache is not None and proof_cache_in is not None:
        proof_cache.load_jsonl(proof_cache_in)

    records = 0
    refined_records = 0
    labels: list[OracleLabel] = []
    output_lines: list[str] = []
    source_index = 0
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            source_index += 1
            continue
        payload = json.loads(line)
        record_index = int(payload.get("record_index", source_index))
        source_index += 1
        label = OracleLabel.from_dict(payload)
        if proof_cache is not None and label.method == "proof-number" and label.exact and label.value in (-1, 1):
            state = SmallState.from_key(game.spec, bytes.fromhex(label.state_key))
            proof_cache.store(state, value=int(label.value), best_action=label.best_action)
        if record_index % shard_count != shard_index:
            continue
        records += 1
        if label.exact and not force:
            output_lines.append(line)
            labels.append(label)
            continue

        state = SmallState.from_key(game.spec, bytes.fromhex(label.state_key))
        new_label = _label_state(
            game,
            state,
            method=method,
            proof=proof,
            proof_number=proof_number,
            tablebase=tablebase,
            low_wall_solver=low_wall_solver,
            proof_cache=proof_cache,
        )
        new_payload = new_label.to_dict()
        new_payload["record_index"] = record_index
        new_payload["config_hash"] = config_hash(config)
        new_payload["proof"] = asdict(proof)
        if method in {"proof-number", "hybrid"}:
            new_payload["proof_number_config"] = asdict(proof_number)
        if low_wall is not None:
            new_payload["low_wall_config"] = asdict(low_wall)
        new_payload["refined_from"] = {
            "method": label.method,
            "outcome": label.outcome.value,
            "exact": label.exact,
            "exhausted": label.exhausted,
            "nodes": label.nodes,
        }
        output_lines.append(json.dumps(new_payload, sort_keys=True))
        labels.append(new_label)
        refined_records += 1

    output.write_text("\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8")
    if proof_cache is not None and proof_cache_out is not None:
        proof_cache.save_jsonl(proof_cache_out)

    return CorpusSummary(
        path=str(output),
        records=records,
        exact_records=sum(1 for label in labels if label.exact),
        unknown_records=sum(1 for label in labels if label.outcome is SolverOutcome.UNKNOWN),
        exhausted_records=sum(1 for label in labels if label.exhausted),
        config_path=str(config_path),
        config_hash=config_hash(config),
        proof=proof,
        seed=0,
        random_plies=0,
        sampling="refine",
        method=method,
        proof_number=proof_number if method in {"proof-number", "hybrid"} else None,
        low_wall=low_wall,
        proof_cache_records=proof_cache.cache_size if proof_cache is not None else 0,
        proof_cache_hits=proof_cache.hits if proof_cache is not None else 0,
        proof_cache_path=str(proof_cache_out or proof_cache_in) if proof_cache is not None and (proof_cache_out or proof_cache_in) else None,
        shard_index=shard_index,
        shard_count=shard_count,
        refined_records=refined_records,
    )


def _sample_state(
    game: SmallGame,
    rng: np.random.Generator,
    *,
    random_plies: int,
    sampling: str = "random",
    low_wall_max_remaining: int = 0,
) -> SmallState:
    state = game.initial_state()
    if sampling == "no-wall":
        state = _sample_no_wall_state(game, rng)
    elif sampling == "low-wall":
        state = _sample_low_wall_state(game, rng, max_walls_remaining=low_wall_max_remaining)
    for _ in range(random_plies):
        if game.is_terminal(state) is not TerminalStatus.NOT_TERMINAL:
            break
        legal = np.flatnonzero(game.legal_actions(state))
        if not legal.size:
            break
        state = game.next_state(state, int(rng.choice(legal)))
    return state


def _sample_no_wall_state(game: SmallGame, rng: np.random.Generator) -> SmallState:
    state = game.initial_state()
    while state.walls_remaining != (0, 0) and game.is_terminal(state) is TerminalStatus.NOT_TERMINAL:
        legal = np.flatnonzero(game.legal_actions(state))
        wall_actions = legal[legal >= 12]
        if wall_actions.size:
            action = int(rng.choice(wall_actions))
        else:
            action = int(rng.choice(legal))
        state = game.next_state(state, action)
    return state


def _sample_low_wall_state(game: SmallGame, rng: np.random.Generator, *, max_walls_remaining: int) -> SmallState:
    state = game.initial_state()
    while sum(state.walls_remaining) > max_walls_remaining and game.is_terminal(state) is TerminalStatus.NOT_TERMINAL:
        legal = np.flatnonzero(game.legal_actions(state))
        if not legal.size:
            break
        wall_actions = legal[legal >= 12]
        if wall_actions.size:
            action = int(rng.choice(wall_actions))
        else:
            action = int(rng.choice(legal))
        state = game.next_state(state, action)
    return state


def _ordered_actions(game: SmallGame, state: SmallState) -> list[int]:
    actions = [int(action) for action in np.flatnonzero(game.legal_actions(state))]

    def score(action: int) -> tuple[float, int, int]:
        child = game.next_state(state, action)
        if game.is_terminal(child) is TerminalStatus.MOVER_LOST:
            return (1_000_000.0, 0, -action)
        mover = state.current_player
        mover_distance = game.shortest_path_distance(child, mover)
        opponent_distance = game.shortest_path_distance(child, 1 - mover)
        distance_score = 0.0
        if mover_distance is not None and opponent_distance is not None:
            distance_score = float(opponent_distance - mover_distance)
        action_bias = 1 if action < 12 else 0
        return (distance_score, action_bias, -action)

    return sorted(actions, key=score, reverse=True)


def _pawn_actions(game: SmallGame, state: SmallState) -> list[int]:
    mask = game.legal_actions(state)
    return [int(action) for action in np.flatnonzero(mask[:12])]


def _capped_sum(values: Iterable[int]) -> int:
    total = 0
    for value in values:
        total += value
        if total >= INFINITY:
            return INFINITY
    return total


def _public_number(value: int) -> int:
    return INFINITY if value >= INFINITY else int(value)


def _validate_shard(shard_index: int, shard_count: int) -> None:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count")


def _empty_phase_buckets() -> dict[str, int]:
    return {"opening": 0, "midgame": 0, "endgame": 0}


def _empty_phase_lists() -> dict[str, list]:
    return {"opening": [], "midgame": [], "endgame": []}


def _balanced_phase_quotas(records: int) -> dict[str, int]:
    quotient, remainder = divmod(records, 3)
    return {
        "opening": quotient + (1 if remainder >= 1 else 0),
        "midgame": quotient + (1 if remainder >= 2 else 0),
        "endgame": quotient,
    }


def _phase_bucket(ply: int, max_plies: int) -> str:
    first_cut = max_plies // 3
    second_cut = (2 * max_plies) // 3
    if ply < first_cut:
        return "opening"
    if ply < second_cut:
        return "midgame"
    return "endgame"


def _proof_number_label_from_cached(
    game: SmallGame,
    state: SmallState,
    cached: _CachedProofNumberResult,
) -> OracleLabel:
    proof_number = 0 if cached.value == 1 else INFINITY
    disproof_number = INFINITY if cached.value == 1 else 0
    return OracleLabel(
        method="proof-number",
        board_size=game.board_size,
        walls_per_player=game.spec.walls_per_player,
        state_key=game.state_key(state).hex(),
        ply=state.ply,
        current_player=state.current_player,
        outcome=_outcome_from_value(cached.value),
        value=cached.value,
        exact=True,
        exhausted=False,
        best_action=cached.best_action,
        depth=0,
        nodes=0,
        legal_action_count=int(game.legal_actions(state).sum()),
        distances=(
            game.shortest_path_distance(state, 0),
            game.shortest_path_distance(state, 1),
        ),
        proof_number=_public_number(proof_number),
        disproof_number=_public_number(disproof_number),
    )


def _apply_cached_proof_number_result(
    node: _PNNode,
    cached: _CachedProofNumberResult,
    root_player: int,
) -> None:
    node.terminal = True
    node.expanded = True
    node.best_action = cached.best_action
    winner = node.state.current_player if cached.value == 1 else 1 - node.state.current_player
    if winner == root_player:
        node.proof_number = 0
        node.disproof_number = INFINITY
        node.disproof_is_loss = False
    else:
        node.proof_number = INFINITY
        node.disproof_number = 0
        node.disproof_is_loss = True


def _node_value_for_mover(node: _PNNode, root_player: int) -> int | None:
    if node.proof_number == 0:
        return 1 if node.state.current_player == root_player else -1
    if node.disproof_number == 0 and node.disproof_is_loss:
        return -1 if node.state.current_player == root_player else 1
    return None


def _outcome_from_value(value: int) -> SolverOutcome:
    if value > 0:
        return SolverOutcome.WIN
    if value < 0:
        return SolverOutcome.LOSS
    return SolverOutcome.DRAW


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a budgeted 5x5 oracle corpus for M2.")
    parser.add_argument("--config", type=Path, default=Path("configs/m2_5x5.json"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--audit-corpus", type=Path, default=None, help="audit an existing JSONL corpus instead of sampling")
    parser.add_argument("--audit-min-records", type=int, default=5000)
    parser.add_argument("--audit-min-exact-fraction", type=float, default=1.0)
    parser.add_argument("--audit-min-phase-records", type=int, default=1)
    parser.add_argument("--merge-from", type=Path, nargs="+", default=None, help="merge sharded JSONL corpora instead of sampling")
    parser.add_argument(
        "--compact-exact-from",
        type=Path,
        nargs="+",
        default=None,
        help="build a balanced, deduplicated exact corpus from independent sampling runs",
    )
    parser.add_argument("--compact-records", type=int, default=5000)
    parser.add_argument("--refine-from", type=Path, default=None, help="refine an existing JSONL corpus instead of sampling new states")
    parser.add_argument("--force", action="store_true", help="relabel exact records too when used with --refine-from")
    parser.add_argument("--positions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-plies", type=int, default=12)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--sampling", choices=("random", "no-wall", "low-wall"), default="random")
    parser.add_argument("--method", choices=("minimax", "proof-number", "hybrid"), default="minimax")
    parser.add_argument("--max-depth", type=int, default=DEFAULT_PROOF_CONFIG.max_depth)
    parser.add_argument("--max-nodes", type=int, default=DEFAULT_PROOF_CONFIG.max_nodes)
    parser.add_argument("--proof-cache-in", type=Path, default=None)
    parser.add_argument("--proof-cache-out", type=Path, default=None)
    parser.add_argument(
        "--low-wall-max-remaining",
        type=int,
        default=None,
        help="enable the hybrid exact low-wall solver for states with at most this many unplaced walls",
    )
    parser.add_argument("--low-wall-max-nodes", type=int, default=DEFAULT_LOW_WALL_CONFIG.max_nodes)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.audit_corpus is not None:
        if args.merge_from is not None or args.compact_exact_from is not None or args.refine_from is not None:
            raise ValueError(
                "--audit-corpus cannot be combined with --merge-from, "
                "--compact-exact-from, or --refine-from"
            )
        summary = audit_oracle_corpus(
            args.audit_corpus,
            config_path=args.config,
            min_records=args.audit_min_records,
            min_exact_fraction=args.audit_min_exact_fraction,
            min_phase_records=args.audit_min_phase_records,
        )
        print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
        return 0 if summary.passed else 1

    if args.compact_exact_from is not None:
        if args.merge_from is not None or args.refine_from is not None:
            raise ValueError(
                "--compact-exact-from cannot be combined with --merge-from or --refine-from"
            )
        if args.output is None:
            raise ValueError("--output is required with --compact-exact-from")
        summary = compact_exact_oracle_corpora(
            args.compact_exact_from,
            args.output,
            config_path=args.config,
            records=args.compact_records,
        )
        print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
        return 0

    if args.merge_from is not None:
        if args.refine_from is not None:
            raise ValueError("--merge-from cannot be combined with --refine-from")
        if args.output is None:
            raise ValueError("--output is required with --merge-from")
        summary = merge_oracle_corpora(args.merge_from, args.output)
        print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
        return 0

    if args.output is None:
        raise ValueError("--output is required unless --audit-corpus is used")

    proof = ProofSearchConfig(max_depth=args.max_depth, max_nodes=args.max_nodes)
    proof_number = ProofNumberConfig(max_nodes=args.max_nodes)
    low_wall = None
    if args.low_wall_max_remaining is not None:
        low_wall = LowWallEndgameConfig(
            max_walls_remaining=args.low_wall_max_remaining,
            max_nodes=args.low_wall_max_nodes,
        )
    elif args.sampling == "low-wall":
        low_wall = LowWallEndgameConfig(max_nodes=args.low_wall_max_nodes)
    if args.refine_from is not None:
        summary = refine_oracle_corpus(
            args.refine_from,
            args.output,
            config_path=args.config,
            method=args.method,
            proof=proof,
            proof_number=proof_number,
            low_wall=low_wall,
            proof_cache_in=args.proof_cache_in,
            proof_cache_out=args.proof_cache_out,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            force=args.force,
        )
    else:
        summary = generate_oracle_corpus(
            args.output,
            config_path=args.config,
            positions=args.positions,
            seed=args.seed,
            random_plies=args.random_plies,
            proof=proof,
            method=args.method,
            proof_number=proof_number,
            low_wall=low_wall,
            sampling=args.sampling,
            proof_cache_in=args.proof_cache_in,
            proof_cache_out=args.proof_cache_out,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
