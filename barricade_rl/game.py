from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable

import numpy as np

try:
    from . import _native as _engine
except ImportError:  # source-tree correctness fallback; CI and production use Rust
    from . import _engine

BOARD_SIZE = 9
WALL_GRID_SIZE = 8
WALLS_PER_PLAYER = 10
ACTION_COUNT = 140
OBSERVATION_VERSION = 1
DEFAULT_MAX_PLIES = 200

MOVE_NAMES = ("N", "S", "E", "W", "NN", "SS", "EE", "WW", "NE", "NW", "SE", "SW")
MOVE_ACTIONS = tuple(range(12))


class TerminalStatus(IntEnum):
    NOT_TERMINAL = 0
    MOVER_LOST = 1
    CAPPED = 2


def _cell_index(position: tuple[int, int]) -> int:
    row, col = position
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        raise ValueError(f"cell is outside the board: {position}")
    return row * BOARD_SIZE + col


def _wall_bits(anchors: Iterable[tuple[int, int]]) -> int:
    result = 0
    for row, col in anchors:
        if not (0 <= row < WALL_GRID_SIZE and 0 <= col < WALL_GRID_SIZE):
            raise ValueError(f"wall anchor is outside the board: {(row, col)}")
        result |= 1 << (row * WALL_GRID_SIZE + col)
    return result


@dataclass(frozen=True, slots=True)
class State:
    """Immutable, packed Quoridor state. ``data`` is the 20-byte transposition key."""

    data: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes) or len(self.data) != 20:
            raise ValueError("state data must be exactly 20 bytes")
        if self.pawns[0] == self.pawns[1]:
            raise ValueError("pawns cannot occupy the same cell")
        if any(count > WALLS_PER_PLAYER for count in self.walls_remaining):
            raise ValueError("walls remaining must be between 0 and 10")

    @classmethod
    def from_components(
        cls,
        *,
        pawns: tuple[tuple[int, int], tuple[int, int]],
        horizontal_walls: Iterable[tuple[int, int]] = (),
        vertical_walls: Iterable[tuple[int, int]] = (),
        walls_remaining: tuple[int, int] = (10, 10),
        current_player: int = 0,
        ply: int = 0,
    ) -> "State":
        if current_player not in (0, 1):
            raise ValueError("current_player must be 0 or 1")
        if not (0 <= ply <= 255):
            raise ValueError("ply must fit in one byte")
        if any(not 0 <= count <= WALLS_PER_PLAYER for count in walls_remaining):
            raise ValueError("walls remaining must be between 0 and 10")
        h_bits = _wall_bits(horizontal_walls)
        v_bits = _wall_bits(vertical_walls)
        if h_bits & v_bits:
            raise ValueError("perpendicular walls cannot cross at the same anchor")
        p0, p1 = (_cell_index(position) for position in pawns)
        if p0 == p1:
            raise ValueError("pawns cannot occupy the same cell")
        packed_pawns = p0 | (p1 << 7) | (current_player << 14)
        data = b"".join(
            (
                h_bits.to_bytes(8, "little"),
                v_bits.to_bytes(8, "little"),
                packed_pawns.to_bytes(2, "little"),
                bytes((walls_remaining[0] | (walls_remaining[1] << 4), ply)),
            )
        )
        return cls(data)

    @property
    def horizontal_bits(self) -> int:
        return int.from_bytes(self.data[0:8], "little")

    @property
    def vertical_bits(self) -> int:
        return int.from_bytes(self.data[8:16], "little")

    @property
    def pawns(self) -> tuple[tuple[int, int], tuple[int, int]]:
        packed = int.from_bytes(self.data[16:18], "little")
        return tuple(divmod((packed >> shift) & 0x7F, BOARD_SIZE) for shift in (0, 7))  # type: ignore[return-value]

    @property
    def current_player(self) -> int:
        return (int.from_bytes(self.data[16:18], "little") >> 14) & 1

    @property
    def walls_remaining(self) -> tuple[int, int]:
        return self.data[18] & 0x0F, self.data[18] >> 4

    @property
    def ply(self) -> int:
        return self.data[19]

    @property
    def horizontal_walls(self) -> tuple[tuple[int, int], ...]:
        return _anchors(self.horizontal_bits)

    @property
    def vertical_walls(self) -> tuple[tuple[int, int], ...]:
        return _anchors(self.vertical_bits)


def _anchors(bits: int) -> tuple[tuple[int, int], ...]:
    return tuple(divmod(index, WALL_GRID_SIZE) for index in range(64) if bits & (1 << index))


def wall_action(orientation: str, row: int, col: int) -> int:
    if orientation not in ("h", "v"):
        raise ValueError("wall orientation must be 'h' or 'v'")
    if not (0 <= row < WALL_GRID_SIZE and 0 <= col < WALL_GRID_SIZE):
        raise ValueError("wall anchor is outside the 8x8 anchor grid")
    return (12 if orientation == "h" else 76) + row * WALL_GRID_SIZE + col


def decode_wall_action(action: int) -> tuple[str, int, int]:
    if 12 <= action < 76:
        index = action - 12
        return "h", index // 8, index % 8
    if 76 <= action < ACTION_COUNT:
        index = action - 76
        return "v", index // 8, index % 8
    raise ValueError(f"action {action} is not a wall placement")


def _canonical_to_actual(action: int, player: int) -> int:
    if not 0 <= action < ACTION_COUNT or player not in (0, 1):
        raise ValueError("invalid action or player")
    if player == 0:
        return action
    move_flip = (1, 0, 2, 3, 5, 4, 6, 7, 10, 11, 8, 9)
    if action < 12:
        return move_flip[action]
    orientation, row, col = decode_wall_action(action)
    return wall_action(orientation, 7 - row, col)


CANONICAL_TO_ACTUAL = tuple(tuple(_canonical_to_actual(action, player) for action in range(ACTION_COUNT)) for player in (0, 1))
ACTUAL_TO_CANONICAL = tuple(
    tuple(sorted(range(ACTION_COUNT), key=CANONICAL_TO_ACTUAL[player].__getitem__)) for player in (0, 1)
)


def canonical_to_actual_action(action: int, player: int) -> int:
    return CANONICAL_TO_ACTUAL[player][action]


def actual_to_canonical_action(action: int, player: int) -> int:
    return ACTUAL_TO_CANONICAL[player][action]


def mirror_action(action: int) -> int:
    move_mirror = (0, 1, 3, 2, 4, 5, 7, 6, 9, 8, 11, 10)
    if action < 0 or action >= ACTION_COUNT:
        raise ValueError(f"invalid action {action}")
    if action < 12:
        return move_mirror[action]
    orientation, row, col = decode_wall_action(action)
    return wall_action(orientation, row, 7 - col)


MIRROR_ACTION = tuple(mirror_action(action) for action in range(ACTION_COUNT))


class Game:
    """Stateless functional interface consumed by search and training."""

    def __init__(self, max_plies: int = DEFAULT_MAX_PLIES):
        if not 1 <= max_plies <= 255:
            raise ValueError("max_plies must be between 1 and 255")
        self.max_plies = max_plies

    def initial_state(self) -> State:
        return State.from_components(pawns=((0, 4), (8, 4)))

    def legal_actions(self, state: State) -> np.ndarray:
        return np.asarray(_engine.legal_actions(state.data, self.max_plies), dtype=np.bool_)

    def next_state(self, state: State, action: int) -> State:
        return State(bytes(_engine.next_state(state.data, int(action), self.max_plies)))

    def is_terminal(self, state: State) -> TerminalStatus:
        return TerminalStatus(_engine.terminal_status(state.data, self.max_plies))

    def terminal_value(self, state: State) -> float:
        status = self.is_terminal(state)
        if status is TerminalStatus.NOT_TERMINAL:
            raise ValueError("terminal_value is undefined for a non-terminal state")
        return -1.0 if status is TerminalStatus.MOVER_LOST else 0.0

    def shortest_path_distance(self, state: State, player: int) -> int | None:
        return _engine.shortest_path_distance(state.data, player)

    def perft(self, state: State, depth: int) -> int:
        if not 0 <= depth <= 255:
            raise ValueError("depth must be between 0 and 255")
        return _engine.perft(state.data, depth, self.max_plies)

    def canonical_observation(self, state: State) -> np.ndarray:
        mover = state.current_player
        opponent = 1 - mover
        observation = np.zeros((6, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

        def cell(position: tuple[int, int]) -> tuple[int, int]:
            row, col = position
            return (8 - row, col) if mover == 1 else (row, col)

        mover_pos = cell(state.pawns[mover])
        opponent_pos = cell(state.pawns[opponent])
        observation[0, mover_pos[0], mover_pos[1]] = 1
        observation[1, opponent_pos[0], opponent_pos[1]] = 1
        for row, col in state.horizontal_walls:
            observation[2, 7 - row if mover == 1 else row, col] = 1
        for row, col in state.vertical_walls:
            observation[3, 7 - row if mover == 1 else row, col] = 1
        observation[4].fill(state.walls_remaining[mover] / WALLS_PER_PLAYER)
        observation[5].fill(state.walls_remaining[opponent] / WALLS_PER_PLAYER)
        return observation

    def mirror(self, value: State | np.ndarray) -> State | np.ndarray:
        if isinstance(value, State):
            return State.from_components(
                pawns=tuple((row, 8 - col) for row, col in value.pawns),  # type: ignore[arg-type]
                horizontal_walls=((row, 7 - col) for row, col in value.horizontal_walls),
                vertical_walls=((row, 7 - col) for row, col in value.vertical_walls),
                walls_remaining=value.walls_remaining,
                current_player=value.current_player,
                ply=value.ply,
            )
        policy = np.asarray(value)
        if policy.shape != (ACTION_COUNT,):
            raise ValueError("policy must have shape (140,)")
        return policy[np.asarray(MIRROR_ACTION)]

    def state_key(self, state: State) -> bytes:
        return state.data

    def transposition_key(self, state: State) -> bytes:
        mirrored = self.mirror(state)
        assert isinstance(mirrored, State)
        return min(state.data, mirrored.data)

    def render(self, state: State) -> str:
        canvas = [[" " for _ in range(4 * BOARD_SIZE - 1)] for _ in range(2 * BOARD_SIZE - 1)]
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
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
