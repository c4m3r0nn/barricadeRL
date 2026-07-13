from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .game import Game, State, TerminalStatus

LADDER_VERSION = 1
WIN_SCORE = 1_000_000.0


class OpponentPolicy(Protocol):
    name: str

    def select_action(self, game: Game, state: State, rng: np.random.Generator) -> int: ...


def _legal(game: Game, state: State) -> list[int]:
    return [int(action) for action in np.flatnonzero(game.legal_actions(state))]


def _evaluation(game: Game, state: State, player: int) -> float:
    status = game.is_terminal(state)
    if status is TerminalStatus.MOVER_LOST:
        return WIN_SCORE if 1 - state.current_player == player else -WIN_SCORE
    if status is TerminalStatus.CAPPED:
        return 0.0
    own_distance = game.shortest_path_distance(state, player)
    opponent_distance = game.shortest_path_distance(state, 1 - player)
    if own_distance is None or opponent_distance is None:
        raise AssertionError("legal state has no path to goal")
    wall_difference = state.walls_remaining[player] - state.walls_remaining[1 - player]
    return float(opponent_distance - own_distance) + 0.7 * wall_difference


@dataclass(frozen=True, slots=True)
class RandomOpponent:
    name: str = "random"

    def select_action(self, game: Game, state: State, rng: np.random.Generator) -> int:
        legal = np.flatnonzero(game.legal_actions(state))
        if not legal.size:
            raise ValueError("cannot select an action in a terminal state")
        return int(rng.choice(legal))


@dataclass(frozen=True, slots=True)
class GreedyRacer:
    name: str = "greedy-racer"

    def select_action(self, game: Game, state: State, rng: np.random.Generator) -> int:
        del rng
        player = state.current_player
        pawn_actions = [action for action in _legal(game, state) if action < 12]
        if not pawn_actions:
            raise AssertionError("reachable state has no legal pawn move")
        scored = []
        for action in pawn_actions:
            child = game.next_state(state, action)
            distance = game.shortest_path_distance(child, player)
            scored.append((WIN_SCORE if distance == 0 else -float(distance), -action, action))
        return max(scored)[2]


@dataclass(frozen=True, slots=True)
class HeuristicOne:
    name: str = "heuristic-1"

    def select_action(self, game: Game, state: State, rng: np.random.Generator) -> int:
        del rng
        player = state.current_player
        actions = _legal(game, state)
        if not actions:
            raise ValueError("cannot select an action in a terminal state")
        return max(actions, key=lambda action: (_evaluation(game, game.next_state(state, action), player), -action))


@dataclass(frozen=True, slots=True)
class AlphaBetaOpponent:
    depth: int

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError("depth must be positive")

    @property
    def name(self) -> str:
        return f"alpha-beta-d{self.depth}"

    def select_action(self, game: Game, state: State, rng: np.random.Generator) -> int:
        del rng
        root_player = state.current_player
        actions = _legal(game, state)
        if not actions:
            raise ValueError("cannot select an action in a terminal state")

        for action in actions:
            child = game.next_state(state, action)
            if game.is_terminal(child) is TerminalStatus.MOVER_LOST:
                return action

        best_action = actions[0]
        table: dict[tuple[bytes, int], float] = {}
        for depth in range(1, self.depth + 1):
            best_value = -float("inf")
            ordered = self._ordered_children(game, state, actions, root_player, maximizing=True)
            for action, child in ordered:
                value = self._search(
                    game,
                    child,
                    depth - 1,
                    root_player,
                    -float("inf"),
                    float("inf"),
                    table,
                )
                if value > best_value or (value == best_value and action < best_action):
                    best_value = value
                    best_action = action
        return best_action

    def _search(
        self,
        game: Game,
        state: State,
        depth: int,
        root_player: int,
        alpha: float,
        beta: float,
        table: dict[tuple[bytes, int], float],
    ) -> float:
        if depth == 0 or game.is_terminal(state) is not TerminalStatus.NOT_TERMINAL:
            return _evaluation(game, state, root_player)
        key = (state.data, depth)
        if key in table:
            return table[key]

        actions = _legal(game, state)
        maximizing = state.current_player == root_player
        children = self._ordered_children(game, state, actions, root_player, maximizing)
        cutoff = False
        if maximizing:
            value = -float("inf")
            for _, child in children:
                value = max(value, self._search(game, child, depth - 1, root_player, alpha, beta, table))
                alpha = max(alpha, value)
                if alpha >= beta:
                    cutoff = True
                    break
        else:
            value = float("inf")
            for _, child in children:
                value = min(value, self._search(game, child, depth - 1, root_player, alpha, beta, table))
                beta = min(beta, value)
                if alpha >= beta:
                    cutoff = True
                    break
        if not cutoff:
            table[key] = value
        return value

    @staticmethod
    def _ordered_children(
        game: Game,
        state: State,
        actions: list[int],
        root_player: int,
        maximizing: bool,
    ) -> list[tuple[int, State]]:
        children = [(action, game.next_state(state, action)) for action in actions]
        children.sort(
            key=lambda item: (_evaluation(game, item[1], root_player), -item[0]),
            reverse=maximizing,
        )
        return children


FROZEN_LADDER: tuple[OpponentPolicy, ...] = (
    RandomOpponent(),
    GreedyRacer(),
    HeuristicOne(),
    AlphaBetaOpponent(depth=3),
    AlphaBetaOpponent(depth=5),
)
