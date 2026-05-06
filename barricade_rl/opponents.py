from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from barricade_rl.core import BarricadeGame, MOVE_ACTIONS


class OpponentPolicy(Protocol):
    def select_action(self, game: BarricadeGame, rng: np.random.Generator) -> int:
        """Choose a legal action for the game's current player."""


def legal_actions(game: BarricadeGame) -> np.ndarray:
    return np.flatnonzero(game.legal_actions_mask())


@dataclass(slots=True)
class RandomOpponent:
    def select_action(self, game: BarricadeGame, rng: np.random.Generator) -> int:
        actions = legal_actions(game)
        if len(actions) == 0:
            raise RuntimeError("No legal opponent actions available")
        return int(rng.choice(actions))


@dataclass(slots=True)
class GreedyOpponent:
    wall_probability: float = 0.05

    def select_action(self, game: BarricadeGame, rng: np.random.Generator) -> int:
        mask = game.legal_actions_mask()
        move_actions = [action for action in MOVE_ACTIONS if mask[action]]
        if move_actions and rng.random() >= self.wall_probability:
            player = game.state.current_player
            scored = []
            original = game.state.pawns[player]
            for action in move_actions:
                dest = game.move_destination(action, player)
                if dest is None:
                    continue
                game.state.pawns[player] = dest
                distance = game.shortest_path_length(player)
                game.state.pawns[player] = original
                scored.append((distance if distance is not None else 999, action))
            if scored:
                best_distance = min(distance for distance, _ in scored)
                best_actions = [action for distance, action in scored if distance == best_distance]
                return int(rng.choice(best_actions))

        actions = np.flatnonzero(mask)
        if len(actions) == 0:
            raise RuntimeError("No legal opponent actions available")
        return int(rng.choice(actions))


def make_opponent(name: str) -> OpponentPolicy:
    normalized = name.lower().strip()
    if normalized == "random":
        return RandomOpponent()
    if normalized == "greedy":
        return GreedyOpponent()
    raise ValueError(f"Unknown opponent '{name}'. Expected 'random' or 'greedy'.")
