from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from barricade_rl.core import BarricadeGame
from barricade_rl.single_agent import BarricadeSingleAgentEnv
from barricade_rl.opponents import make_opponent


def state_to_frame(game: BarricadeGame, **extra: Any) -> dict[str, Any]:
    frame = {
        "pawns": [list(pos) for pos in game.state.pawns],
        "h_walls": game.state.h_walls.astype(int).tolist(),
        "v_walls": game.state.v_walls.astype(int).tolist(),
        "walls_remaining": list(game.state.walls_remaining),
        "current_player": game.state.current_player,
        "winner": game.state.winner,
        "move_count": game.state.move_count,
    }
    frame.update(extra)
    return frame


def apply_frame(game: BarricadeGame, frame: dict[str, Any]) -> None:
    game.state.pawns = [tuple(pos) for pos in frame["pawns"]]
    game.state.h_walls = np.array(frame["h_walls"], dtype=bool)
    game.state.v_walls = np.array(frame["v_walls"], dtype=bool)
    game.state.walls_remaining = list(frame["walls_remaining"])
    game.state.current_player = int(frame["current_player"])
    game.state.winner = frame["winner"]
    game.state.move_count = int(frame["move_count"])


def save_replay(path: Path | str, frames: list[dict[str, Any]], metadata: dict[str, Any] | None = None) -> None:
    replay_path = Path(path)
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata or {},
        "frames": frames,
    }
    replay_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_replay(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def record_model_game(model, opponent_name: str = "random", seed: int = 0, max_steps: int = 500) -> list[dict[str, Any]]:
    env = BarricadeSingleAgentEnv(opponent=make_opponent(opponent_name), invalid_action="raise", max_moves=max_steps * 2)
    obs, info = env.reset(seed=seed)
    frames = [state_to_frame(env.game, label="start")]
    terminated = False
    truncated = False
    steps = 0
    while not (terminated or truncated) and steps < max_steps:
        mask = env.action_masks()
        action, _ = model.predict(obs, deterministic=True, action_masks=mask)
        obs, reward, terminated, truncated, info = env.step(int(action))
        frames.append(
            state_to_frame(
                env.game,
                label=f"step-{steps + 1}",
                learner_action=int(action),
                opponent_action=info.get("opponent_action"),
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
            )
        )
        steps += 1
    return frames
