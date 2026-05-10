from __future__ import annotations

import json
import argparse
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


def replay_summary(frames: list[dict[str, Any]], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    if not frames:
        raise ValueError("Cannot summarize an empty replay")
    metadata = metadata or {}
    final = frames[-1]
    walls_remaining = final["walls_remaining"]
    return {
        "timesteps": metadata.get("timesteps"),
        "opponent": metadata.get("opponent"),
        "seed": metadata.get("seed"),
        "frames": len(frames),
        "winner": final["winner"],
        "move_count": final["move_count"],
        "learner_walls_placed": 10 - walls_remaining[0],
        "opponent_walls_placed": 10 - walls_remaining[1],
    }


def summarize_replay_file(path: Path | str) -> dict[str, Any]:
    replay = load_replay(path)
    summary = replay_summary(replay["frames"], replay.get("metadata", {}))
    summary["path"] = str(path)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Summarize saved Barricade replay JSON files.")
    parser.add_argument("replays", nargs="+", type=Path)
    args = parser.parse_args()
    headers = ["path", "timesteps", "winner", "move_count", "learner_walls_placed", "opponent_walls_placed"]
    print("\t".join(headers))
    for replay_path in args.replays:
        summary = summarize_replay_file(replay_path)
        print("\t".join(str(summary.get(header)) for header in headers))


def record_model_game(
    model,
    opponent_name: str = "random",
    seed: int = 0,
    max_steps: int = 500,
    learner_side: int = 0,
) -> list[dict[str, Any]]:
    env = BarricadeSingleAgentEnv(
        opponent=make_opponent(opponent_name),
        invalid_action="raise",
        max_moves=max_steps * 2,
        learner_side=learner_side,
    )
    obs, info = env.reset(seed=seed)
    frames = [state_to_frame(env.game, label="start", learner_side=learner_side, opponent_opening_action=info.get("opponent_opening_action"))]
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
                learner_side=learner_side,
                opponent_action=info.get("opponent_action"),
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
            )
        )
        steps += 1
    return frames


def record_model_replay(
    model,
    path: Path | str,
    opponent_name: str = "random",
    seed: int = 0,
    max_steps: int = 500,
    learner_side: int = 0,
) -> None:
    frames = record_model_game(model, opponent_name=opponent_name, seed=seed, max_steps=max_steps, learner_side=learner_side)
    save_replay(
        path,
        frames,
        metadata={
            "opponent": opponent_name,
            "seed": seed,
            "max_steps": max_steps,
            "learner_side": learner_side,
        },
    )


def record_model_main():
    parser = argparse.ArgumentParser(description="Record one trained model game as a Barricade replay JSON.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--opponent", choices=["random", "greedy", "mixed"], default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--learner-side", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise SystemExit("Install RL dependencies first: .venv/bin/python -m pip install -e '.[dev,rl]'") from exc
    record_model_replay(
        MaskablePPO.load(args.model),
        args.out,
        opponent_name=args.opponent,
        seed=args.seed,
        max_steps=args.max_steps,
        learner_side=args.learner_side,
    )
    print(f"Saved replay to {args.out}")
