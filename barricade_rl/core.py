from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

BOARD_SIZE = 9
WALL_GRID_SIZE = BOARD_SIZE - 1
WALLS_PER_PLAYER = 10
ACTION_COUNT = 132

MOVE_ACTIONS = {
    0: (-1, 0),  # up
    1: (1, 0),  # down
    2: (0, -1),  # left
    3: (0, 1),  # right
}
MOVE_NAMES = ("up", "down", "left", "right")


def in_board(pos: tuple[int, int]) -> bool:
    row, col = pos
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def opponent_of(player: int) -> int:
    return 1 - player


def goal_row(player: int) -> int:
    return 0 if player == 0 else BOARD_SIZE - 1


def start_pos(player: int) -> tuple[int, int]:
    return (BOARD_SIZE - 1, BOARD_SIZE // 2) if player == 0 else (0, BOARD_SIZE // 2)


def wall_action(orientation: str, row: int, col: int) -> int:
    offset = 4 if orientation == "h" else 68
    return offset + row * WALL_GRID_SIZE + col


def decode_wall_action(action: int) -> tuple[str, int, int]:
    if 4 <= action < 68:
        idx = action - 4
        return "h", idx // WALL_GRID_SIZE, idx % WALL_GRID_SIZE
    if 68 <= action < ACTION_COUNT:
        idx = action - 68
        return "v", idx // WALL_GRID_SIZE, idx % WALL_GRID_SIZE
    raise ValueError(f"Action {action} is not a wall placement")


def canonical_action_to_absolute(action: int, player: int) -> int:
    if player == 0:
        return action
    if action in MOVE_ACTIONS:
        return {0: 1, 1: 0, 2: 3, 3: 2}[action]
    orientation, row, col = decode_wall_action(action)
    return wall_action(orientation, WALL_GRID_SIZE - 1 - row, WALL_GRID_SIZE - 1 - col)


@dataclass(slots=True)
class BarricadeState:
    pawns: list[tuple[int, int]] = field(default_factory=lambda: [start_pos(0), start_pos(1)])
    h_walls: np.ndarray = field(default_factory=lambda: np.zeros((WALL_GRID_SIZE, WALL_GRID_SIZE), dtype=bool))
    v_walls: np.ndarray = field(default_factory=lambda: np.zeros((WALL_GRID_SIZE, WALL_GRID_SIZE), dtype=bool))
    walls_remaining: list[int] = field(default_factory=lambda: [WALLS_PER_PLAYER, WALLS_PER_PLAYER])
    current_player: int = 0
    winner: int | None = None
    move_count: int = 0

    def copy(self) -> "BarricadeState":
        return BarricadeState(
            pawns=list(self.pawns),
            h_walls=self.h_walls.copy(),
            v_walls=self.v_walls.copy(),
            walls_remaining=list(self.walls_remaining),
            current_player=self.current_player,
            winner=self.winner,
            move_count=self.move_count,
        )


class BarricadeGame:
    """Minimal game engine shared by training and UI code."""

    def __init__(self, max_moves: int = 500):
        self.max_moves = max_moves
        self.state = BarricadeState()
        self._legal_actions_cache_key = None
        self._legal_actions_cache = None
        self._shortest_path_cache = {}

    def reset(self) -> BarricadeState:
        self.state = BarricadeState()
        self._legal_actions_cache_key = None
        self._legal_actions_cache = None
        self._shortest_path_cache = {}
        return self.state

    def set_state(self, state: BarricadeState) -> BarricadeState:
        self.state = state.copy()
        self._legal_actions_cache_key = None
        self._legal_actions_cache = None
        self._shortest_path_cache = {}
        return self.state

    @property
    def terminated(self) -> bool:
        return self.state.winner is not None

    def truncated(self) -> bool:
        return self.state.move_count >= self.max_moves and not self.terminated

    def can_cross(self, a: tuple[int, int], b: tuple[int, int]) -> bool:
        if not in_board(a) or not in_board(b):
            return False
        ar, ac = a
        br, bc = b
        dr, dc = br - ar, bc - ac
        if abs(dr) + abs(dc) != 1:
            return False
        if dr == -1:
            row = br
            col = ac
            return not ((col > 0 and self.state.h_walls[row, col - 1]) or (col < WALL_GRID_SIZE and self.state.h_walls[row, col]))
        if dr == 1:
            row = ar
            col = ac
            return not ((col > 0 and self.state.h_walls[row, col - 1]) or (col < WALL_GRID_SIZE and self.state.h_walls[row, col]))
        if dc == -1:
            row = ar
            col = bc
            return not ((row > 0 and self.state.v_walls[row - 1, col]) or (row < WALL_GRID_SIZE and self.state.v_walls[row, col]))
        row = ar
        col = ac
        return not ((row > 0 and self.state.v_walls[row - 1, col]) or (row < WALL_GRID_SIZE and self.state.v_walls[row, col]))

    def legal_move_destination(self, action: int, player: int | None = None) -> tuple[int, int] | None:
        if action not in MOVE_ACTIONS:
            return None
        player = self.state.current_player if player is None else player
        pawn = self.state.pawns[player]
        opp = self.state.pawns[opponent_of(player)]
        dr, dc = MOVE_ACTIONS[action]
        target = (pawn[0] + dr, pawn[1] + dc)
        if not in_board(target) or not self.can_cross(pawn, target):
            return None
        if target != opp:
            return target

        behind = (opp[0] + dr, opp[1] + dc)
        if in_board(behind) and self.can_cross(opp, behind):
            return behind
        return None

    def legal_side_jump_destination(self, action: int, player: int | None = None) -> tuple[int, int] | None:
        if action not in MOVE_ACTIONS:
            return None
        player = self.state.current_player if player is None else player
        pawn = self.state.pawns[player]
        opp = self.state.pawns[opponent_of(player)]
        adj = (opp[0] - pawn[0], opp[1] - pawn[1])
        if abs(adj[0]) + abs(adj[1]) != 1 or not self.can_cross(pawn, opp):
            return None
        straight_behind = (opp[0] + adj[0], opp[1] + adj[1])
        straight_blocked = not in_board(straight_behind) or not self.can_cross(opp, straight_behind)
        if not straight_blocked:
            return None
        dr, dc = MOVE_ACTIONS[action]
        if adj[0] != 0 and dr != 0:
            return None
        if adj[1] != 0 and dc != 0:
            return None
        dest = (opp[0] + dr, opp[1] + dc)
        if in_board(dest) and self.can_cross(opp, dest):
            return dest
        return None

    def move_destination(self, action: int, player: int | None = None) -> tuple[int, int] | None:
        return self.legal_side_jump_destination(action, player) or self.legal_move_destination(action, player)

    def is_wall_legal(self, orientation: str, row: int, col: int, player: int | None = None) -> bool:
        player = self.state.current_player if player is None else player
        if self.state.walls_remaining[player] <= 0:
            return False
        if not (0 <= row < WALL_GRID_SIZE and 0 <= col < WALL_GRID_SIZE):
            return False
        walls = self.state.h_walls if orientation == "h" else self.state.v_walls
        crossing = self.state.v_walls if orientation == "h" else self.state.h_walls
        if walls[row, col] or crossing[row, col]:
            return False
        if orientation == "h":
            if (col > 0 and walls[row, col - 1]) or (col < WALL_GRID_SIZE - 1 and walls[row, col + 1]):
                return False
        elif (row > 0 and walls[row - 1, col]) or (row < WALL_GRID_SIZE - 1 and walls[row + 1, col]):
            return False

        walls[row, col] = True
        try:
            return self.has_path(0) and self.has_path(1)
        finally:
            walls[row, col] = False

    def legal_actions_mask(self, canonical: bool = False) -> np.ndarray:
        key = self.state_key()
        if self._legal_actions_cache_key == key and self._legal_actions_cache is not None:
            mask = self._legal_actions_cache.copy()
        else:
            mask = np.zeros(ACTION_COUNT, dtype=bool)
            for action in MOVE_ACTIONS:
                mask[action] = self.move_destination(action) is not None
            for row in range(WALL_GRID_SIZE):
                for col in range(WALL_GRID_SIZE):
                    if self.is_wall_legal("h", row, col):
                        mask[wall_action("h", row, col)] = True
                    if self.is_wall_legal("v", row, col):
                        mask[wall_action("v", row, col)] = True
            self._legal_actions_cache_key = key
            self._legal_actions_cache = mask.copy()
        if canonical:
            return self.to_canonical_action_mask(mask, self.state.current_player)
        return mask

    def to_canonical_action_mask(self, absolute_mask: np.ndarray, player: int) -> np.ndarray:
        if player == 0:
            return absolute_mask.copy()
        canonical_mask = np.zeros_like(absolute_mask)
        for absolute_action in np.flatnonzero(absolute_mask):
            canonical_mask[canonical_action_to_absolute(int(absolute_action), player)] = True
        return canonical_mask

    def state_key(self):
        return (
            tuple(self.state.pawns),
            self.state.h_walls.tobytes(),
            self.state.v_walls.tobytes(),
            tuple(self.state.walls_remaining),
            self.state.current_player,
            self.state.winner,
            self.state.move_count,
        )

    def cached_action_is_legal(self, action: int) -> bool | None:
        key = self.state_key()
        if self._legal_actions_cache_key == key and self._legal_actions_cache is not None:
            if 0 <= action < ACTION_COUNT:
                return bool(self._legal_actions_cache[action])
            return False
        return None

    def apply_action(self, action: int) -> bool:
        if self.terminated:
            raise RuntimeError("Cannot act after the game has ended")
        player = self.state.current_player
        if action in MOVE_ACTIONS:
            dest = self.move_destination(action)
            if dest is None:
                return False
            self.state.pawns[player] = dest
            if dest[0] == goal_row(player):
                self.state.winner = player
        elif 4 <= action < ACTION_COUNT:
            orientation, row, col = decode_wall_action(action)
            cached_legal = self.cached_action_is_legal(action)
            if cached_legal is False:
                return False
            if cached_legal is None and not self.is_wall_legal(orientation, row, col, player):
                return False
            walls = self.state.h_walls if orientation == "h" else self.state.v_walls
            walls[row, col] = True
            self.state.walls_remaining[player] -= 1
        else:
            return False

        self.state.move_count += 1
        if not self.terminated:
            self.state.current_player = opponent_of(player)
        return True

    def has_path(self, player: int) -> bool:
        return self.shortest_path_length(player) is not None

    def shortest_path_length(self, player: int) -> int | None:
        cache_key = (player, self.state_key())
        if cache_key in self._shortest_path_cache:
            return self._shortest_path_cache[cache_key]
        start = self.state.pawns[player]
        target_row = goal_row(player)
        seen = {start}
        queue: deque[tuple[tuple[int, int], int]] = deque([(start, 0)])
        while queue:
            pos, dist = queue.popleft()
            if pos[0] == target_row:
                self._shortest_path_cache[cache_key] = dist
                return dist
            for nxt in self.neighbors(pos):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, dist + 1))
        self._shortest_path_cache[cache_key] = None
        return None

    def neighbors(self, pos: tuple[int, int]) -> Iterable[tuple[int, int]]:
        for dr, dc in MOVE_ACTIONS.values():
            nxt = (pos[0] + dr, pos[1] + dc)
            if in_board(nxt) and self.can_cross(pos, nxt):
                yield nxt

    def observation(self, canonical: bool = True) -> np.ndarray:
        player = self.state.current_player
        opponent = opponent_of(player)
        obs = np.zeros((6, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        cur = self.state.pawns[player]
        opp = self.state.pawns[opponent]
        h = self.state.h_walls.astype(np.float32)
        v = self.state.v_walls.astype(np.float32)
        cur_walls = self.state.walls_remaining[player] / WALLS_PER_PLAYER
        opp_walls = self.state.walls_remaining[opponent] / WALLS_PER_PLAYER

        if canonical and player == 1:
            cur = (BOARD_SIZE - 1 - cur[0], BOARD_SIZE - 1 - cur[1])
            opp = (BOARD_SIZE - 1 - opp[0], BOARD_SIZE - 1 - opp[1])
            h = np.flip(h, axis=(0, 1))
            v = np.flip(v, axis=(0, 1))

        obs[0, cur[0], cur[1]] = 1.0
        obs[1, opp[0], opp[1]] = 1.0
        obs[2, :WALL_GRID_SIZE, :WALL_GRID_SIZE] = h
        obs[3, :WALL_GRID_SIZE, :WALL_GRID_SIZE] = v
        obs[4, :, :] = cur_walls
        obs[5, :, :] = opp_walls
        return obs
