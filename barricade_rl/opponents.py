from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from barricade_rl.core import BarricadeGame, MOVE_ACTIONS, canonical_action_to_absolute


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


@dataclass(slots=True)
class MixedOpponent:
    random_probability: float = 0.5
    random_opponent: RandomOpponent = field(default_factory=RandomOpponent)
    greedy_opponent: GreedyOpponent = field(default_factory=GreedyOpponent)

    def select_action(self, game: BarricadeGame, rng: np.random.Generator) -> int:
        if rng.random() < self.random_probability:
            return self.random_opponent.select_action(game, rng)
        return self.greedy_opponent.select_action(game, rng)


@dataclass(slots=True)
class CheckpointPoolOpponent:
    models: list = field(default_factory=list)
    model_paths: list[Path] = field(default_factory=list)
    deterministic: bool = True
    _loaded_models: list = field(default_factory=list, init=False)

    @classmethod
    def from_paths(cls, paths: list[str | Path], deterministic: bool = True) -> "CheckpointPoolOpponent":
        return cls(model_paths=[Path(path) for path in paths], deterministic=deterministic)

    def select_action(self, game: BarricadeGame, rng: np.random.Generator) -> int:
        models = self._models()
        if not models:
            raise RuntimeError("CheckpointPoolOpponent requires at least one model or model path")
        model = models[int(rng.integers(len(models)))]
        player = game.state.current_player
        mask = game.legal_actions_mask(canonical=True)
        action, _ = model.predict(game.observation(canonical=True), deterministic=self.deterministic, action_masks=mask)
        action = int(action)
        if not mask[action]:
            actions = np.flatnonzero(mask)
            action = int(rng.choice(actions))
        return canonical_action_to_absolute(action, player)

    def _models(self):
        if self.models:
            return self.models
        if not self._loaded_models and self.model_paths:
            try:
                from sb3_contrib import MaskablePPO
            except ImportError as exc:
                raise RuntimeError("Install RL dependencies before using checkpoint opponents") from exc
            self._loaded_models = [MaskablePPO.load(path) for path in self.model_paths]
        return self._loaded_models


def make_opponent(name: str) -> OpponentPolicy:
    normalized = name.lower().strip()
    if normalized == "random":
        return RandomOpponent()
    if normalized == "greedy":
        return GreedyOpponent()
    if normalized == "mixed":
        return MixedOpponent()
    raise ValueError(f"Unknown opponent '{name}'. Expected 'random', 'greedy', or 'mixed'.")
