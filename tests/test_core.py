import numpy as np
import pytest

from barricade_rl.core import ACTION_COUNT, BarricadeGame, canonical_action_to_absolute, wall_action
from barricade_rl.env import BarricadeEnv


def test_reset_action_mask_and_observation_shape():
    env = BarricadeEnv()
    obs, info = env.reset()
    assert obs.shape == (6, 9, 9)
    assert info["action_mask"].shape == (ACTION_COUNT,)
    assert info["action_mask"].dtype == np.bool_
    assert info["action_mask"][:4].tolist() == [True, False, True, True]


def test_pawn_cannot_move_through_wall():
    game = BarricadeGame()
    assert game.apply_action(wall_action("h", 7, 4))
    assert game.state.current_player == 1
    game.state.current_player = 0
    assert game.move_destination(0) is None


def test_wall_count_decreases_and_overlap_is_illegal():
    game = BarricadeGame()
    action = wall_action("h", 3, 3)
    assert game.apply_action(action)
    assert game.state.walls_remaining[0] == 9
    game.state.current_player = 1
    assert not game.apply_action(action)


def test_crossing_wall_is_illegal():
    game = BarricadeGame()
    assert game.apply_action(wall_action("h", 2, 2))
    game.state.current_player = 1
    assert not game.apply_action(wall_action("v", 2, 2))


def test_same_direction_walls_cannot_overlap_edges():
    game = BarricadeGame()
    assert game.apply_action(wall_action("h", 2, 2))
    game.state.current_player = 1
    assert not game.apply_action(wall_action("h", 2, 3))


def test_perpendicular_wall_can_fit_between_end_to_end_walls():
    game = BarricadeGame()
    assert game.apply_action(wall_action("h", 2, 0))
    game.state.current_player = 1
    assert game.apply_action(wall_action("h", 2, 2))
    game.state.current_player = 0
    assert game.apply_action(wall_action("v", 2, 1))


def test_jump_over_adjacent_opponent():
    game = BarricadeGame()
    game.state.pawns = [(4, 4), (3, 4)]
    assert game.move_destination(0) == (2, 4)


def test_side_jump_when_straight_jump_blocked():
    game = BarricadeGame()
    game.state.pawns = [(4, 4), (3, 4)]
    game.state.h_walls[2, 4] = True
    assert game.move_destination(2) == (3, 3)
    assert game.move_destination(3) == (3, 5)
    assert game.move_destination(0) is None


def test_wall_cannot_eliminate_all_paths():
    game = BarricadeGame()
    for col in range(7):
        game.state.h_walls[7, col] = True
    assert not game.is_wall_legal("h", 7, 7, player=0)


def test_win_terminates_immediately():
    env = BarricadeEnv()
    env.reset()
    env.game.state.pawns[0] = (1, 4)
    env.game.state.pawns[1] = (0, 0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert terminated
    assert not truncated
    assert reward == 1.0
    assert info["winner"] == 0


def test_invalid_action_raises_by_default():
    env = BarricadeEnv()
    env.reset()
    with pytest.raises(ValueError):
        env.step(1)


def test_legal_action_mask_is_cached_until_state_changes():
    game = BarricadeGame()
    original = game.is_wall_legal
    calls = {"count": 0}

    def counted(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    game.is_wall_legal = counted
    first = game.legal_actions_mask()
    first_calls = calls["count"]
    second = game.legal_actions_mask()
    assert np.array_equal(first, second)
    assert calls["count"] == first_calls

    game.state.pawns[0] = (7, 4)
    game.legal_actions_mask()
    assert calls["count"] > first_calls


def test_player_one_canonical_actions_rotate_to_absolute_board():
    assert canonical_action_to_absolute(0, player=1) == 1
    assert canonical_action_to_absolute(1, player=1) == 0
    assert canonical_action_to_absolute(2, player=1) == 3
    assert canonical_action_to_absolute(3, player=1) == 2
    assert canonical_action_to_absolute(wall_action("h", 0, 1), player=1) == wall_action("h", 7, 6)
