"""Correctness-first Python oracle for the native rules engine.

The compiled ``barricade_rl._engine`` extension shadows this module in normal
installs. Keeping the same implementation in Python makes differential testing
and source-tree development possible without weakening the public contract.
"""

from __future__ import annotations

from collections import deque
from heapq import heappop, heappush

BOARD = 9
ANCHORS = 8
ACTIONS = 140


def _decode(data: bytes) -> dict:
    if len(data) != 20:
        raise ValueError("state key must contain exactly 20 bytes")
    packed = int.from_bytes(data[16:18], "little")
    state = {
        "h": int.from_bytes(data[:8], "little"),
        "v": int.from_bytes(data[8:16], "little"),
        "pawns": [packed & 0x7F, (packed >> 7) & 0x7F],
        "current": (packed >> 14) & 1,
        "walls": [data[18] & 0x0F, data[18] >> 4],
        "ply": data[19],
    }
    if max(state["pawns"]) >= 81 or state["pawns"][0] == state["pawns"][1]:
        raise ValueError("state contains invalid pawn positions")
    return state


def _encode(state: dict) -> bytes:
    packed = state["pawns"][0] | state["pawns"][1] << 7 | state["current"] << 14
    return b"".join(
        (
            state["h"].to_bytes(8, "little"),
            state["v"].to_bytes(8, "little"),
            packed.to_bytes(2, "little"),
            bytes((state["walls"][0] | state["walls"][1] << 4, state["ply"])),
        )
    )


def _position(state: dict, player: int) -> tuple[int, int]:
    return divmod(state["pawns"][player], BOARD)


def _bit(row: int, col: int) -> int:
    return 1 << (row * ANCHORS + col)


def _in_board(position: tuple[int, int]) -> bool:
    return 0 <= position[0] < BOARD and 0 <= position[1] < BOARD


def _can_cross(state: dict, start: tuple[int, int], end: tuple[int, int]) -> bool:
    if not _in_board(start) or not _in_board(end):
        return False
    dr, dc = end[0] - start[0], end[1] - start[1]
    if abs(dr) + abs(dc) != 1:
        return False
    if dr:
        row, col = min(start[0], end[0]), start[1]
        return not ((col > 0 and state["h"] & _bit(row, col - 1)) or (col < 8 and state["h"] & _bit(row, col)))
    row, col = start[0], min(start[1], end[1])
    return not ((row > 0 and state["v"] & _bit(row - 1, col)) or (row < 8 and state["v"] & _bit(row, col)))


def _delta(action: int, player: int) -> tuple[int, int]:
    dr, dc = (
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    )[action]
    return (-dr if player else dr), dc


def _move_destination(state: dict, action: int) -> tuple[int, int] | None:
    player = state["current"]
    pawn, opponent = _position(state, player), _position(state, 1 - player)
    dr, dc = _delta(action, player)
    if action < 4:
        target = pawn[0] + dr, pawn[1] + dc
        return target if target != opponent and _can_cross(state, pawn, target) else None
    if action < 8:
        adjacent = pawn[0] + dr, pawn[1] + dc
        landing = adjacent[0] + dr, adjacent[1] + dc
        return landing if adjacent == opponent and _can_cross(state, pawn, adjacent) and _can_cross(state, adjacent, landing) else None

    adjacent_delta = opponent[0] - pawn[0], opponent[1] - pawn[1]
    if abs(adjacent_delta[0]) + abs(adjacent_delta[1]) != 1 or not _can_cross(state, pawn, opponent):
        return None
    straight = opponent[0] + adjacent_delta[0], opponent[1] + adjacent_delta[1]
    if _can_cross(state, opponent, straight):
        return None
    target = pawn[0] + dr, pawn[1] + dc
    if _in_board(target) and abs(target[0] - opponent[0]) + abs(target[1] - opponent[1]) == 1 and _can_cross(state, opponent, target):
        return target
    return None


def _shortest_path(state: dict, player: int) -> int | None:
    start = _position(state, player)
    goal = 8 if player == 0 else 0
    queue = deque(((start, 0),))
    seen = {start}
    while queue:
        position, distance = queue.popleft()
        if position[0] == goal:
            return distance
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = position[0] + dr, position[1] + dc
            if neighbor not in seen and _can_cross(state, position, neighbor):
                seen.add(neighbor)
                queue.append((neighbor, distance + 1))
    return None


def _path_exists(state: dict, player: int) -> bool:
    start = _position(state, player)
    goal = 8 if player == 0 else 0
    best = {start: 0}
    queue = [(abs(goal - start[0]), 0, start)]
    while queue:
        _, distance, position = heappop(queue)
        if distance != best[position]:
            continue
        if position[0] == goal:
            return True
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = position[0] + dr, position[1] + dc
            next_distance = distance + 1
            if _can_cross(state, position, neighbor) and next_distance < best.get(neighbor, 256):
                best[neighbor] = next_distance
                heappush(queue, (next_distance + abs(goal - neighbor[0]), next_distance, neighbor))
    return False


def _wall_legal(state: dict, horizontal: bool, row: int, col: int) -> bool:
    if state["walls"][state["current"]] == 0:
        return False
    candidate = _bit(row, col)
    same, crossing = (state["h"], state["v"]) if horizontal else (state["v"], state["h"])
    if same & candidate or crossing & candidate:
        return False
    if horizontal:
        if (col > 0 and same & _bit(row, col - 1)) or (col < 7 and same & _bit(row, col + 1)):
            return False
    elif (row > 0 and same & _bit(row - 1, col)) or (row < 7 and same & _bit(row + 1, col)):
        return False
    trial = {**state, "pawns": state["pawns"].copy(), "walls": state["walls"].copy()}
    trial["h" if horizontal else "v"] |= candidate
    return _path_exists(trial, 0) and _path_exists(trial, 1)


def _terminal(state: dict, max_plies: int) -> int:
    previous = 1 - state["current"]
    row, _ = _position(state, previous)
    if row == (8 if previous == 0 else 0):
        return 1
    return 2 if state["ply"] >= max_plies else 0


def _legal(state: dict, action: int) -> bool:
    if action < 12:
        return _move_destination(state, action) is not None
    if action < 76:
        index = action - 12
        row, col = divmod(index, 8)
        return _wall_legal(state, True, 7 - row if state["current"] else row, col)
    if action < 140:
        index = action - 76
        row, col = divmod(index, 8)
        return _wall_legal(state, False, 7 - row if state["current"] else row, col)
    return False


def legal_actions(data: bytes, max_plies: int) -> list[bool]:
    state = _decode(data)
    return [False] * ACTIONS if _terminal(state, max_plies) else [_legal(state, action) for action in range(ACTIONS)]


def next_state(data: bytes, action: int, max_plies: int) -> bytes:
    state = _decode(data)
    if _terminal(state, max_plies):
        raise ValueError("cannot act in a terminal state")
    if not _legal(state, action):
        raise ValueError(f"illegal action {action}")
    player = state["current"]
    if action < 12:
        row, col = _move_destination(state, action)
        state["pawns"][player] = row * 9 + col
    else:
        key, canonical_index = ("h", action - 12) if action < 76 else ("v", action - 76)
        row, col = divmod(canonical_index, 8)
        if player == 1:
            row = 7 - row
        index = row * 8 + col
        state[key] |= 1 << index
        state["walls"][player] -= 1
    state["current"] = 1 - player
    state["ply"] += 1
    return _encode(state)


def terminal_status(data: bytes, max_plies: int) -> int:
    return _terminal(_decode(data), max_plies)


def shortest_path_distance(data: bytes, player: int) -> int | None:
    if player not in (0, 1):
        raise ValueError("player must be 0 or 1")
    return _shortest_path(_decode(data), player)


def perft(data: bytes, depth: int, max_plies: int) -> int:
    if depth == 0:
        return 1
    actions = legal_actions(data, max_plies)
    return sum(
        perft(next_state(data, action, max_plies), depth - 1, max_plies)
        for action, legal in enumerate(actions)
        if legal
    )
