from pathlib import Path

from barricade_rl.core import BarricadeGame, wall_action
from barricade_rl.replay import (
    apply_frame,
    load_replay,
    record_model_game,
    record_model_replay,
    replay_summary,
    save_replay,
    state_to_frame,
)


def test_state_frame_round_trip():
    source = BarricadeGame()
    assert source.apply_action(wall_action("h", 3, 2))
    frame = state_to_frame(source, label="after-wall")

    target = BarricadeGame()
    apply_frame(target, frame)

    assert target.state.pawns == source.state.pawns
    assert target.state.walls_remaining == source.state.walls_remaining
    assert target.state.current_player == source.state.current_player
    assert target.state.winner == source.state.winner
    assert target.state.h_walls.tolist() == source.state.h_walls.tolist()
    assert target.state.v_walls.tolist() == source.state.v_walls.tolist()


def test_save_and_load_replay(tmp_path: Path):
    game = BarricadeGame()
    frames = [state_to_frame(game, label="start")]
    path = tmp_path / "replay.json"

    save_replay(path, frames, metadata={"timesteps": 100})
    replay = load_replay(path)

    assert replay["metadata"]["timesteps"] == 100
    assert replay["frames"][0]["label"] == "start"


class FirstLegalModel:
    def predict(self, obs, deterministic=True, action_masks=None):
        return int(action_masks.nonzero()[0][0]), None


def test_record_model_game_returns_replay_frames():
    frames = record_model_game(FirstLegalModel(), opponent_name="random", seed=0, max_steps=2)

    assert frames[0]["label"] == "start"
    assert len(frames) >= 2
    assert "learner_action" in frames[1]


def test_replay_summary_reports_game_outcome():
    game = BarricadeGame()
    frames = [state_to_frame(game, label="start")]
    game.state.pawns[0] = (0, 4)
    game.state.winner = 0
    game.state.move_count = 12
    game.state.walls_remaining = [7, 6]
    frames.append(state_to_frame(game, label="final"))

    summary = replay_summary(frames, metadata={"timesteps": 1000})

    assert summary["timesteps"] == 1000
    assert summary["winner"] == 0
    assert summary["frames"] == 2
    assert summary["move_count"] == 12
    assert summary["learner_walls_placed"] == 3
    assert summary["opponent_walls_placed"] == 4


def test_record_model_replay_saves_file_with_metadata(tmp_path: Path):
    path = tmp_path / "model_replay.json"
    record_model_replay(FirstLegalModel(), path, opponent_name="random", seed=3, max_steps=2)

    replay = load_replay(path)
    assert replay["metadata"]["opponent"] == "random"
    assert replay["metadata"]["seed"] == 3
    assert replay["frames"]
