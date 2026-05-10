import numpy as np
import pytest

from barricade_rl.core import MOVE_ACTIONS, BarricadeGame
from barricade_rl.opponents import (
    AntiRushOpponent,
    CheckpointPoolOpponent,
    CurriculumOpponent,
    GreedyOpponent,
    MixedOpponent,
    RandomOpponent,
    RefreshingCheckpointPoolOpponent,
    make_opponent,
)
from barricade_rl.single_agent import BarricadeSingleAgentEnv


class FixedOpponent:
    def __init__(self, action):
        self.action = action

    def select_action(self, game, rng):
        return self.action


class FakeModel:
    def predict(self, obs, deterministic=True, action_masks=None):
        return int(action_masks.nonzero()[0][0]), None


def test_random_opponent_selects_legal_action():
    game = BarricadeGame()
    game.state.current_player = 1
    action = RandomOpponent().select_action(game, np.random.default_rng(0))
    assert game.legal_actions_mask()[action]


def test_greedy_opponent_prefers_shorter_path_move():
    game = BarricadeGame()
    game.state.current_player = 1
    action = GreedyOpponent(wall_probability=0.0).select_action(game, np.random.default_rng(0))
    assert action == 1


def test_make_opponent_rejects_unknown_name():
    with pytest.raises(ValueError):
        make_opponent("missing")


def test_mixed_opponent_selects_legal_action():
    game = BarricadeGame()
    game.state.current_player = 1
    action = MixedOpponent(random_probability=0.5).select_action(game, np.random.default_rng(0))
    assert game.legal_actions_mask()[action]


def test_make_opponent_supports_mixed():
    assert isinstance(make_opponent("mixed"), MixedOpponent)


def test_make_opponent_supports_curriculum_and_anti_rush():
    assert isinstance(make_opponent("anti_rush"), AntiRushOpponent)
    assert isinstance(make_opponent("curriculum"), CurriculumOpponent)


def test_anti_rush_opponent_prefers_wall_when_runner_is_close():
    game = BarricadeGame()
    game.state.pawns[0] = (2, 4)
    game.state.current_player = 1

    action = AntiRushOpponent().select_action(game, np.random.default_rng(0))

    assert action >= 4
    assert game.legal_actions_mask()[action]


def test_checkpoint_pool_opponent_selects_legal_action():
    game = BarricadeGame()
    game.state.current_player = 1
    opponent = CheckpointPoolOpponent(models=[FakeModel()])
    action = opponent.select_action(game, np.random.default_rng(0))
    assert game.legal_actions_mask()[action]


def test_refreshing_checkpoint_pool_falls_back_without_checkpoints(tmp_path):
    game = BarricadeGame()
    game.state.current_player = 1
    opponent = RefreshingCheckpointPoolOpponent(patterns=[str(tmp_path / "*.zip")], fallback=RandomOpponent())

    action = opponent.select_action(game, np.random.default_rng(0))

    assert game.legal_actions_mask()[action]


def test_refreshing_checkpoint_pool_can_keep_sampling_fallback_with_checkpoints(tmp_path):
    game = BarricadeGame()
    game.state.current_player = 1
    path = tmp_path / "checkpoint.zip"
    path.write_text("placeholder", encoding="utf-8")
    opponent = RefreshingCheckpointPoolOpponent(
        patterns=[str(path)],
        fallback=GreedyOpponent(wall_probability=0.0),
        checkpoint_probability=0.0,
    )

    action = opponent.select_action(game, np.random.default_rng(0))

    assert action == 1


def test_single_agent_step_returns_to_learner_turn_after_opponent():
    env = BarricadeSingleAgentEnv(opponent=FixedOpponent(1))
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    assert not terminated
    assert not truncated
    assert reward == 0.0
    assert info["current_player"] == 0
    assert info["opponent_action"] == 1


def test_single_agent_learner_win_reward():
    env = BarricadeSingleAgentEnv(opponent=FixedOpponent(1))
    env.reset()
    env.game.state.pawns[0] = (1, 4)
    env.game.state.pawns[1] = (0, 0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert terminated
    assert reward == 1.0
    assert info["winner"] == 0


def test_single_agent_opponent_win_reward():
    env = BarricadeSingleAgentEnv(opponent=FixedOpponent(1))
    env.reset()
    env.game.state.pawns[0] = (8, 0)
    env.game.state.pawns[1] = (7, 4)
    obs, reward, terminated, truncated, info = env.step(3)
    assert terminated
    assert reward == -1.0
    assert info["winner"] == 1


def test_single_agent_action_mask_is_for_learner_turn():
    env = BarricadeSingleAgentEnv(opponent=FixedOpponent(1))
    env.reset()
    mask = env.action_masks()
    assert mask.shape == (132,)
    assert mask[: len(MOVE_ACTIONS)].tolist() == [True, False, True, True]


def test_single_agent_shaped_reward_adds_path_delta():
    env = BarricadeSingleAgentEnv(opponent=FixedOpponent(1), shaped_reward=True)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    assert reward > 0
    assert info["shaped_reward"] > 0


def test_single_agent_can_train_as_player_one_after_opponent_opening():
    env = BarricadeSingleAgentEnv(opponent=FixedOpponent(0), learner_side=1)

    obs, info = env.reset(seed=0)

    assert info["learner_side"] == 1
    assert info["current_player"] == 1
    assert info["opponent_opening_action"] == 0
    assert env.game.state.pawns[0] == (7, 4)
    assert env.action_masks()[0]


def test_single_agent_player_one_win_gets_positive_reward():
    env = BarricadeSingleAgentEnv(opponent=FixedOpponent(0), learner_side=1)
    env.reset(seed=0)
    env.game.state.pawns[1] = (7, 4)
    env.game.state.current_player = 1

    obs, reward, terminated, truncated, info = env.step(0)

    assert terminated
    assert reward == 1.0
    assert info["winner"] == 1
    assert info["learner_side"] == 1
