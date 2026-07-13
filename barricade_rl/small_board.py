from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from heapq import heappop, heappush
from typing import Iterable

import numpy as np

from .game import TerminalStatus

MOVE_DELTAS = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)


@dataclass(frozen=True, slots=True)
class SmallBoardSpec:
    size: int = 5
    walls_per_player: int = 3
    max_plies: int = 80

    def __post_init__(self) -> None:
        if self.size < 3 or self.size % 2 == 0:
            raise ValueError("small-board size must be an odd integer >= 3")
        if self.size > 9:
            raise ValueError("small-board state key currently supports sizes up to 9")
        if not 0 <= self.walls_per_player <= 7:
            raise ValueError("walls_per_player must fit in three bits")
        if not 1 <= self.max_plies <= 255:
            raise ValueError("max_plies must be between 1 and 255")

    @property
    def wall_grid_size(self) -> int:
        return self.size - 1

    @property
    def action_count(self) -> int:
        return 12 + 2 * self.wall_grid_size * self.wall_grid_size


DEFAULT_5X5_SPEC = SmallBoardSpec()


def small_wall_action(spec: SmallBoardSpec, orientation: str, row: int, col: int) -> int:
    if orientation not in ("h", "v"):
        raise ValueError("wall orientation must be 'h' or 'v'")
    if not (0 <= row < spec.wall_grid_size and 0 <= col < spec.wall_grid_size):
        raise ValueError("wall anchor is outside the anchor grid")
    offset = row * spec.wall_grid_size + col
    return (12 if orientation == "h" else 12 + spec.wall_grid_size**2) + offset


def decode_small_wall_action(spec: SmallBoardSpec, action: int) -> tuple[str, int, int]:
    wall_actions = spec.wall_grid_size**2
    if 12 <= action < 12 + wall_actions:
        index = action - 12
        return "h", index // spec.wall_grid_size, index % spec.wall_grid_size
    if 12 + wall_actions <= action < spec.action_count:
        index = action - 12 - wall_actions
        return "v", index // spec.wall_grid_size, index % spec.wall_grid_size
    raise ValueError(f"action {action} is not a wall placement")


@dataclass(frozen=True, slots=True)
class SmallState:
    spec: SmallBoardSpec
    horizontal_bits: int
    vertical_bits: int
    pawns: tuple[tuple[int, int], tuple[int, int]]
    walls_remaining: tuple[int, int]
    current_player: int = 0
    ply: int = 0

    def __post_init__(self) -> None:
        if self.current_player not in (0, 1):
            raise ValueError("current_player must be 0 or 1")
        if not (0 <= self.ply <= 255):
            raise ValueError("ply must fit in one byte")
        if self.pawns[0] == self.pawns[1]:
            raise ValueError("pawns cannot occupy the same cell")
        for pawn in self.pawns:
            _cell_index(self.spec, pawn)
        if any(not 0 <= count <= self.spec.walls_per_player for count in self.walls_remaining):
            raise ValueError("walls remaining are outside the board spec")
        if self.horizontal_bits & self.vertical_bits:
            raise ValueError("perpendicular walls cannot cross at the same anchor")
        max_bits = 1 << (self.spec.wall_grid_size * self.spec.wall_grid_size)
        if not 0 <= self.horizontal_bits < max_bits or not 0 <= self.vertical_bits < max_bits:
            raise ValueError("wall bits are outside the board spec")

    @classmethod
    def from_components(
        cls,
        *,
        spec: SmallBoardSpec = DEFAULT_5X5_SPEC,
        pawns: tuple[tuple[int, int], tuple[int, int]] | None = None,
        horizontal_walls: Iterable[tuple[int, int]] = (),
        vertical_walls: Iterable[tuple[int, int]] = (),
        walls_remaining: tuple[int, int] | None = None,
        current_player: int = 0,
        ply: int = 0,
    ) -> SmallState:
        center = spec.size // 2
        return cls(
            spec=spec,
            horizontal_bits=_wall_bits(spec, horizontal_walls),
            vertical_bits=_wall_bits(spec, vertical_walls),
            pawns=pawns or ((0, center), (spec.size - 1, center)),
            walls_remaining=walls_remaining or (spec.walls_per_player, spec.walls_per_player),
            current_player=current_player,
            ply=ply,
        )

    @classmethod
    def from_key(cls, spec: SmallBoardSpec, key: bytes) -> SmallState:
        if len(key) != 12:
            raise ValueError("small-board state key must contain exactly 12 bytes")
        h_bits = int.from_bytes(key[0:4], "little")
        v_bits = int.from_bytes(key[4:8], "little")
        p0 = int(key[8])
        p1 = int(key[9])
        packed = int(key[10])
        walls_remaining = packed & 0x07, (packed >> 3) & 0x07
        current_player = (packed >> 6) & 1
        return cls(
            spec=spec,
            horizontal_bits=h_bits,
            vertical_bits=v_bits,
            pawns=(divmod(p0, spec.size), divmod(p1, spec.size)),
            walls_remaining=walls_remaining,
            current_player=current_player,
            ply=int(key[11]),
        )

    @property
    def horizontal_walls(self) -> tuple[tuple[int, int], ...]:
        return _anchors(self.spec, self.horizontal_bits)

    @property
    def vertical_walls(self) -> tuple[tuple[int, int], ...]:
        return _anchors(self.spec, self.vertical_bits)


class SmallGame:
    """Parameterized Python rules surface for M2's solved 5x5 curriculum."""

    def __init__(self, spec: SmallBoardSpec = DEFAULT_5X5_SPEC) -> None:
        self.spec = spec

    @property
    def board_size(self) -> int:
        return self.spec.size

    @property
    def wall_grid_size(self) -> int:
        return self.spec.wall_grid_size

    @property
    def action_count(self) -> int:
        return self.spec.action_count

    def initial_state(self) -> SmallState:
        return SmallState.from_components(spec=self.spec)

    def legal_actions(self, state: SmallState) -> np.ndarray:
        self._assert_state_spec(state)
        if self.is_terminal(state) is not TerminalStatus.NOT_TERMINAL:
            return np.zeros(self.action_count, dtype=np.bool_)
        return np.asarray([self._legal(state, action) for action in range(self.action_count)], dtype=np.bool_)

    def next_state(self, state: SmallState, action: int) -> SmallState:
        self._assert_state_spec(state)
        action = int(action)
        if self.is_terminal(state) is not TerminalStatus.NOT_TERMINAL:
            raise ValueError("cannot act in a terminal state")
        if not 0 <= action < self.action_count or not self._legal(state, action):
            raise ValueError(f"illegal action {action}")

        player = state.current_player
        pawns = [state.pawns[0], state.pawns[1]]
        h_bits = state.horizontal_bits
        v_bits = state.vertical_bits
        walls = [state.walls_remaining[0], state.walls_remaining[1]]

        if action < 12:
            destination = self._move_destination(state, action)
            assert destination is not None
            pawns[player] = destination
        else:
            orientation, row, col = decode_small_wall_action(self.spec, action)
            if player == 1:
                row = self.wall_grid_size - 1 - row
            if orientation == "h":
                h_bits |= _bit(self.spec, row, col)
            else:
                v_bits |= _bit(self.spec, row, col)
            walls[player] -= 1

        return SmallState(
            spec=self.spec,
            horizontal_bits=h_bits,
            vertical_bits=v_bits,
            pawns=(pawns[0], pawns[1]),
            walls_remaining=(walls[0], walls[1]),
            current_player=1 - player,
            ply=state.ply + 1,
        )

    def is_terminal(self, state: SmallState) -> TerminalStatus:
        self._assert_state_spec(state)
        previous = 1 - state.current_player
        row, _ = state.pawns[previous]
        if row == (self.board_size - 1 if previous == 0 else 0):
            return TerminalStatus.MOVER_LOST
        if state.ply >= self.spec.max_plies:
            return TerminalStatus.CAPPED
        return TerminalStatus.NOT_TERMINAL

    def terminal_value(self, state: SmallState) -> float:
        status = self.is_terminal(state)
        if status is TerminalStatus.NOT_TERMINAL:
            raise ValueError("terminal_value is undefined for a non-terminal state")
        return -1.0 if status is TerminalStatus.MOVER_LOST else 0.0

    def shortest_path_distance(self, state: SmallState, player: int) -> int | None:
        self._assert_state_spec(state)
        if player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        return self._shortest_path(state, player)

    def perft(self, state: SmallState, depth: int) -> int:
        self._assert_state_spec(state)
        if not 0 <= depth <= 255:
            raise ValueError("depth must be between 0 and 255")
        if depth == 0:
            return 1
        mask = self.legal_actions(state)
        return sum(
            self.perft(self.next_state(state, int(action)), depth - 1)
            for action in np.flatnonzero(mask)
        )

    def canonical_observation(self, state: SmallState) -> np.ndarray:
        self._assert_state_spec(state)
        mover = state.current_player
        opponent = 1 - mover
        observation = np.zeros((6, self.board_size, self.board_size), dtype=np.float32)

        def cell(position: tuple[int, int]) -> tuple[int, int]:
            row, col = position
            return (self.board_size - 1 - row, col) if mover == 1 else (row, col)

        mover_pos = cell(state.pawns[mover])
        opponent_pos = cell(state.pawns[opponent])
        observation[0, mover_pos[0], mover_pos[1]] = 1
        observation[1, opponent_pos[0], opponent_pos[1]] = 1
        for row, col in state.horizontal_walls:
            observation[2, self.wall_grid_size - 1 - row if mover == 1 else row, col] = 1
        for row, col in state.vertical_walls:
            observation[3, self.wall_grid_size - 1 - row if mover == 1 else row, col] = 1
        denominator = max(1, self.spec.walls_per_player)
        observation[4].fill(state.walls_remaining[mover] / denominator)
        observation[5].fill(state.walls_remaining[opponent] / denominator)
        return observation

    def mirror(self, value: SmallState | np.ndarray) -> SmallState | np.ndarray:
        if isinstance(value, SmallState):
            self._assert_state_spec(value)
            return SmallState.from_components(
                spec=self.spec,
                pawns=tuple((row, self.board_size - 1 - col) for row, col in value.pawns),  # type: ignore[arg-type]
                horizontal_walls=((row, self.wall_grid_size - 1 - col) for row, col in value.horizontal_walls),
                vertical_walls=((row, self.wall_grid_size - 1 - col) for row, col in value.vertical_walls),
                walls_remaining=value.walls_remaining,
                current_player=value.current_player,
                ply=value.ply,
            )
        policy = np.asarray(value)
        if policy.shape != (self.action_count,):
            raise ValueError(f"policy must have shape ({self.action_count},)")
        return policy[np.asarray([self.mirror_action(action) for action in range(self.action_count)])]

    def mirror_action(self, action: int) -> int:
        move_mirror = (0, 1, 3, 2, 4, 5, 7, 6, 9, 8, 11, 10)
        if not 0 <= action < self.action_count:
            raise ValueError(f"invalid action {action}")
        if action < 12:
            return move_mirror[action]
        orientation, row, col = decode_small_wall_action(self.spec, action)
        return small_wall_action(self.spec, orientation, row, self.wall_grid_size - 1 - col)

    def state_key(self, state: SmallState) -> bytes:
        self._assert_state_spec(state)
        p0 = _cell_index(self.spec, state.pawns[0])
        p1 = _cell_index(self.spec, state.pawns[1])
        packed = (
            state.walls_remaining[0]
            | (state.walls_remaining[1] << 3)
            | (state.current_player << 6)
        )
        return b"".join(
            (
                state.horizontal_bits.to_bytes(4, "little"),
                state.vertical_bits.to_bytes(4, "little"),
                bytes((p0, p1, packed, state.ply)),
            )
        )

    def transposition_key(self, state: SmallState) -> bytes:
        mirrored = self.mirror(state)
        assert isinstance(mirrored, SmallState)
        return min(self.state_key(state), self.state_key(mirrored))

    def render(self, state: SmallState) -> str:
        self._assert_state_spec(state)
        canvas = [[" " for _ in range(4 * self.board_size - 1)] for _ in range(2 * self.board_size - 1)]
        for row in range(self.board_size):
            for col in range(self.board_size):
                canvas[2 * row][4 * col] = "·"
        for player, label in enumerate(("A", "B")):
            row, col = state.pawns[player]
            canvas[2 * row][4 * col] = label
        for row, col in state.horizontal_walls:
            groove = 2 * row + 1
            start = 4 * col
            for offset in range(7):
                canvas[groove][start + offset] = "─"
        for row, col in state.vertical_walls:
            groove = 4 * col + 2
            for cell_row in (2 * row, 2 * row + 1, 2 * row + 2):
                canvas[cell_row][groove] = "│"
        return "\n".join("".join(row).rstrip() for row in canvas)

    def _assert_state_spec(self, state: SmallState) -> None:
        if state.spec != self.spec:
            raise ValueError("state was created for a different board spec")

    def _legal(self, state: SmallState, action: int) -> bool:
        if action < 12:
            return self._move_destination(state, action) is not None
        if action >= self.action_count:
            return False
        orientation, row, col = decode_small_wall_action(self.spec, action)
        if state.current_player == 1:
            row = self.wall_grid_size - 1 - row
        return self._wall_legal(state, orientation == "h", row, col)

    def _move_destination(self, state: SmallState, action: int) -> tuple[int, int] | None:
        player = state.current_player
        pawn, opponent = state.pawns[player], state.pawns[1 - player]
        dr, dc = self._delta(action, player)
        if action < 4:
            target = pawn[0] + dr, pawn[1] + dc
            return target if target != opponent and self._can_cross(state, pawn, target) else None
        if action < 8:
            adjacent = pawn[0] + dr, pawn[1] + dc
            landing = adjacent[0] + dr, adjacent[1] + dc
            if adjacent == opponent and self._can_cross(state, pawn, adjacent) and self._can_cross(state, adjacent, landing):
                return landing
            return None

        adjacent_delta = opponent[0] - pawn[0], opponent[1] - pawn[1]
        if abs(adjacent_delta[0]) + abs(adjacent_delta[1]) != 1 or not self._can_cross(state, pawn, opponent):
            return None
        straight = opponent[0] + adjacent_delta[0], opponent[1] + adjacent_delta[1]
        if self._can_cross(state, opponent, straight):
            return None
        target = pawn[0] + dr, pawn[1] + dc
        if (
            self._in_board(target)
            and abs(target[0] - opponent[0]) + abs(target[1] - opponent[1]) == 1
            and self._can_cross(state, opponent, target)
        ):
            return target
        return None

    def _delta(self, action: int, player: int) -> tuple[int, int]:
        dr, dc = MOVE_DELTAS[action]
        return (-dr if player else dr), dc

    def _can_cross(self, state: SmallState, start: tuple[int, int], end: tuple[int, int]) -> bool:
        if not self._in_board(start) or not self._in_board(end):
            return False
        dr, dc = end[0] - start[0], end[1] - start[1]
        if abs(dr) + abs(dc) != 1:
            return False
        if dr:
            row, col = min(start[0], end[0]), start[1]
            return not (
                (col > 0 and state.horizontal_bits & _bit(self.spec, row, col - 1))
                or (col < self.wall_grid_size and state.horizontal_bits & _bit(self.spec, row, col))
            )
        row, col = start[0], min(start[1], end[1])
        return not (
            (row > 0 and state.vertical_bits & _bit(self.spec, row - 1, col))
            or (row < self.wall_grid_size and state.vertical_bits & _bit(self.spec, row, col))
        )

    def _in_board(self, position: tuple[int, int]) -> bool:
        return 0 <= position[0] < self.board_size and 0 <= position[1] < self.board_size

    def _wall_legal(self, state: SmallState, horizontal: bool, row: int, col: int) -> bool:
        if state.walls_remaining[state.current_player] == 0:
            return False
        candidate = _bit(self.spec, row, col)
        same = state.horizontal_bits if horizontal else state.vertical_bits
        crossing = state.vertical_bits if horizontal else state.horizontal_bits
        if same & candidate or crossing & candidate:
            return False
        if horizontal:
            if (col > 0 and same & _bit(self.spec, row, col - 1)) or (
                col < self.wall_grid_size - 1 and same & _bit(self.spec, row, col + 1)
            ):
                return False
        elif (row > 0 and same & _bit(self.spec, row - 1, col)) or (
            row < self.wall_grid_size - 1 and same & _bit(self.spec, row + 1, col)
        ):
            return False

        trial = SmallState(
            spec=self.spec,
            horizontal_bits=state.horizontal_bits | candidate if horizontal else state.horizontal_bits,
            vertical_bits=state.vertical_bits if horizontal else state.vertical_bits | candidate,
            pawns=state.pawns,
            walls_remaining=state.walls_remaining,
            current_player=state.current_player,
            ply=state.ply,
        )
        return self._path_exists(trial, 0) and self._path_exists(trial, 1)

    def _shortest_path(self, state: SmallState, player: int) -> int | None:
        start = state.pawns[player]
        goal = self.board_size - 1 if player == 0 else 0
        queue = deque(((start, 0),))
        seen = {start}
        while queue:
            position, distance = queue.popleft()
            if position[0] == goal:
                return distance
            for dr, dc in MOVE_DELTAS[:4]:
                neighbor = position[0] + dr, position[1] + dc
                if neighbor not in seen and self._can_cross(state, position, neighbor):
                    seen.add(neighbor)
                    queue.append((neighbor, distance + 1))
        return None

    def _path_exists(self, state: SmallState, player: int) -> bool:
        start = state.pawns[player]
        goal = self.board_size - 1 if player == 0 else 0
        best = {start: 0}
        queue = [(abs(goal - start[0]), 0, start)]
        while queue:
            _, distance, position = heappop(queue)
            if distance != best[position]:
                continue
            if position[0] == goal:
                return True
            for dr, dc in MOVE_DELTAS[:4]:
                neighbor = position[0] + dr, position[1] + dc
                next_distance = distance + 1
                if self._can_cross(state, position, neighbor) and next_distance < best.get(neighbor, 256):
                    best[neighbor] = next_distance
                    heappush(queue, (next_distance + abs(goal - neighbor[0]), next_distance, neighbor))
        return False


class SolverOutcome(Enum):
    WIN = "win"
    LOSS = "loss"
    DRAW = "draw"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SolverResult:
    outcome: SolverOutcome
    value: int
    best_action: int | None
    depth: int
    nodes: int


def solve_state(game: SmallGame, state: SmallState, *, max_depth: int) -> SolverResult:
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    memo: dict[tuple[bytes, int], SolverResult] = {}

    def search(current: SmallState, depth: int) -> SolverResult:
        key = (game.transposition_key(current), depth)
        if key in memo:
            return memo[key]

        status = game.is_terminal(current)
        if status is TerminalStatus.MOVER_LOST:
            result = SolverResult(SolverOutcome.LOSS, -1, None, depth, 1)
            memo[key] = result
            return result
        if status is TerminalStatus.CAPPED:
            result = SolverResult(SolverOutcome.DRAW, 0, None, depth, 1)
            memo[key] = result
            return result
        if depth == 0:
            result = SolverResult(SolverOutcome.UNKNOWN, 0, None, depth, 1)
            memo[key] = result
            return result

        legal_actions = np.flatnonzero(game.legal_actions(current))
        nodes = 1
        first_unknown: int | None = None
        first_draw: int | None = None
        for action in legal_actions:
            child = game.next_state(current, int(action))
            child_result = search(child, depth - 1)
            nodes += child_result.nodes
            if child_result.outcome is SolverOutcome.LOSS:
                result = SolverResult(SolverOutcome.WIN, 1, int(action), depth, nodes)
                memo[key] = result
                return result
            if child_result.outcome is SolverOutcome.DRAW and first_draw is None:
                first_draw = int(action)
            if child_result.outcome is SolverOutcome.UNKNOWN and first_unknown is None:
                first_unknown = int(action)

        if first_unknown is not None:
            result = SolverResult(SolverOutcome.UNKNOWN, 0, first_unknown, depth, nodes)
        elif first_draw is not None:
            result = SolverResult(SolverOutcome.DRAW, 0, first_draw, depth, nodes)
        else:
            result = SolverResult(SolverOutcome.LOSS, -1, int(legal_actions[0]) if legal_actions.size else None, depth, nodes)
        memo[key] = result
        return result

    return search(state, max_depth)


def _cell_index(spec: SmallBoardSpec, position: tuple[int, int]) -> int:
    row, col = position
    if not (0 <= row < spec.size and 0 <= col < spec.size):
        raise ValueError(f"cell is outside the board: {position}")
    return row * spec.size + col


def _wall_bits(spec: SmallBoardSpec, anchors: Iterable[tuple[int, int]]) -> int:
    result = 0
    for row, col in anchors:
        if not (0 <= row < spec.wall_grid_size and 0 <= col < spec.wall_grid_size):
            raise ValueError(f"wall anchor is outside the board: {(row, col)}")
        result |= _bit(spec, row, col)
    return result


def _bit(spec: SmallBoardSpec, row: int, col: int) -> int:
    return 1 << (row * spec.wall_grid_size + col)


def _anchors(spec: SmallBoardSpec, bits: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        divmod(index, spec.wall_grid_size)
        for index in range(spec.wall_grid_size * spec.wall_grid_size)
        if bits & (1 << index)
    )
