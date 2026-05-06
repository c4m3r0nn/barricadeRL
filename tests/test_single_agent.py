import numpy as np
import pytest

from barricade_rl.core import MOVE_ACTIONS, BarricadeGame
from barricade_rl.opponents import GreedyOpponent, RandomOpponent, make_opponent
from barricade_rl.single_agent import BarricadeSingleAgentEnv


class FixedOpponent:
    def __init__(self, action):
        self.action = action

    def select_action(self, game, rng):
        return self.action


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
