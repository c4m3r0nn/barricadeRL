from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .small_board import SmallBoardSpec, SmallGame


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    _validate_config(config, source)
    return config


def config_hash(config: Mapping[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def small_game_from_config(config: Mapping[str, Any]) -> SmallGame:
    board = config.get("board")
    if not isinstance(board, Mapping):
        raise ValueError("config must contain a board mapping")
    spec = SmallBoardSpec(
        size=int(board["size"]),
        walls_per_player=int(board["walls_per_player"]),
        max_plies=int(board["max_plies"]),
    )
    expected_actions = int(config.get("action_space", {}).get("action_count", spec.action_count))  # type: ignore[union-attr]
    if expected_actions != spec.action_count:
        raise ValueError(f"config action_count {expected_actions} does not match board spec {spec.action_count}")
    return SmallGame(spec)


def _validate_config(config: Any, source: Path) -> None:
    if not isinstance(config, dict):
        raise ValueError(f"{source} must contain a JSON object")
    for key in ("schema_version", "milestone", "board", "observation", "action_space"):
        if key not in config:
            raise ValueError(f"{source} is missing required key {key!r}")
    if config["schema_version"] != 1:
        raise ValueError(f"{source} has unsupported schema_version {config['schema_version']!r}")
    observation = config["observation"]
    if not isinstance(observation, Mapping) or int(observation.get("version", 0)) < 1:
        raise ValueError(f"{source} must define a positive observation version")
    small_game_from_config(config)
