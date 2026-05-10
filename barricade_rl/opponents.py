from __future__ import annotations

import glob
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from barricade_rl.core import BarricadeGame, MOVE_ACTIONS, canonical_action_to_absolute, decode_wall_action


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
class AntiRushOpponent:
    fallback: GreedyOpponent = field(default_factory=lambda: GreedyOpponent(wall_probability=0.0))
    min_path_gain: int = 1
    wall_bias_distance: int = 5

    def select_action(self, game: BarricadeGame, rng: np.random.Generator) -> int:
        player = game.state.current_player
        target = 1 - player
        if game.state.walls_remaining[player] <= 0:
            return self.fallback.select_action(game, rng)

        mask = game.legal_actions_mask()
        wall_actions = [int(action) for action in np.flatnonzero(mask) if action >= 4]
        if not wall_actions:
            return self.fallback.select_action(game, rng)

        current_target_path = game.shortest_path_length(target)
        current_player_path = game.shortest_path_length(player)
        if current_target_path is None or current_player_path is None:
            return self.fallback.select_action(game, rng)

        scored = []
        for action in wall_actions:
            orientation, row, col = decode_wall_action(action)
            walls = game.state.h_walls if orientation == "h" else game.state.v_walls
            walls[row, col] = True
            try:
                target_path = game.shortest_path_length(target)
                player_path = game.shortest_path_length(player)
            finally:
                walls[row, col] = False
            if target_path is None or player_path is None:
                continue
            target_gain = target_path - current_target_path
            self_cost = player_path - current_player_path
            urgency = max(0, self.wall_bias_distance - current_target_path)
            score = target_gain * 4 - self_cost + urgency
            if target_gain >= self.min_path_gain:
                scored.append((score, action))

        if not scored:
            return self.fallback.select_action(game, rng)
        best_score = max(score for score, _ in scored)
        best_actions = [action for score, action in scored if score == best_score]
        return int(rng.choice(best_actions))


@dataclass(slots=True)
class CurriculumOpponent:
    random_weight: float = 0.10
    greedy_weight: float = 0.30
    anti_rush_weight: float = 0.45
    mixed_weight: float = 0.15
    random_opponent: RandomOpponent = field(default_factory=RandomOpponent)
    greedy_opponent: GreedyOpponent = field(default_factory=lambda: GreedyOpponent(wall_probability=0.0))
    anti_rush_opponent: AntiRushOpponent = field(default_factory=AntiRushOpponent)
    mixed_opponent: MixedOpponent = field(default_factory=MixedOpponent)

    def select_action(self, game: BarricadeGame, rng: np.random.Generator) -> int:
        opponents = [self.random_opponent, self.greedy_opponent, self.anti_rush_opponent, self.mixed_opponent]
        weights = np.array([self.random_weight, self.greedy_weight, self.anti_rush_weight, self.mixed_weight], dtype=np.float64)
        weights = weights / weights.sum()
        opponent = opponents[int(rng.choice(len(opponents), p=weights))]
        return opponent.select_action(game, rng)


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


@dataclass(slots=True)
class RefreshingCheckpointPoolOpponent:
    patterns: list[str] = field(default_factory=list)
    fallback: OpponentPolicy = field(default_factory=MixedOpponent)
    deterministic: bool = True
    checkpoint_probability: float = 1.0
    max_loaded_models: int = 8
    refresh_every_calls: int = 25
    _known_paths: list[Path] = field(default_factory=list, init=False)
    _model_cache: dict[Path, object] = field(default_factory=dict, init=False)
    _calls_since_refresh: int = field(default=0, init=False)

    def select_action(self, game: BarricadeGame, rng: np.random.Generator) -> int:
        paths = self._paths()
        if not paths or rng.random() >= self.checkpoint_probability:
            return self.fallback.select_action(game, rng)
        path = paths[int(rng.integers(len(paths)))]
        model = self._model(path)
        player = game.state.current_player
        mask = game.legal_actions_mask(canonical=True)
        action, _ = model.predict(game.observation(canonical=True), deterministic=self.deterministic, action_masks=mask)
        action = int(action)
        if not mask[action]:
            actions = np.flatnonzero(mask)
            action = int(rng.choice(actions))
        return canonical_action_to_absolute(action, player)

    def _paths(self) -> list[Path]:
        self._calls_since_refresh += 1
        if self._calls_since_refresh >= self.refresh_every_calls or not self._known_paths:
            self.refresh()
        return self._known_paths

    def refresh(self) -> None:
        paths: list[Path] = []
        for pattern in self.patterns:
            paths.extend(Path(match) for match in glob.glob(pattern))
        self._known_paths = sorted(set(paths))
        known = set(self._known_paths)
        for path in list(self._model_cache):
            if path not in known:
                del self._model_cache[path]
        self._calls_since_refresh = 0

    def _model(self, path: Path):
        if path in self._model_cache:
            return self._model_cache[path]
        try:
            from sb3_contrib import MaskablePPO
        except ImportError as exc:
            raise RuntimeError("Install RL dependencies before using checkpoint opponents") from exc
        model = MaskablePPO.load(path)
        self._model_cache[path] = model
        while len(self._model_cache) > self.max_loaded_models:
            oldest_path = next(iter(self._model_cache))
            if oldest_path == path and len(self._model_cache) > 1:
                oldest_path = next(iter(p for p in self._model_cache if p != path))
            del self._model_cache[oldest_path]
        return model


def make_opponent(name: str) -> OpponentPolicy:
    normalized = name.lower().strip()
    if normalized == "random":
        return RandomOpponent()
    if normalized == "greedy":
        return GreedyOpponent()
    if normalized == "mixed":
        return MixedOpponent()
    if normalized in {"anti_rush", "anti-rush"}:
        return AntiRushOpponent()
    if normalized in {"curriculum", "balanced"}:
        return CurriculumOpponent()
    raise ValueError(f"Unknown opponent '{name}'. Expected 'random', 'greedy', 'mixed', 'anti_rush', or 'curriculum'.")
