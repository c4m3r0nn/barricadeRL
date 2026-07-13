import numpy as np
import pytest
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from barricade_rl import ENV_ID
from barricade_rl.env import QuoridorEnv
from barricade_rl.game import State


class ForwardOpponent:
    name = "forward-test"

    def select_action(self, game, state, rng):
        assert game.legal_actions(state)[0]
        return 0


def test_reset_exposes_canonical_observation_and_mask():
    env = QuoridorEnv(opponent_policy=ForwardOpponent())
    observation, info = env.reset(seed=7)

    assert observation.shape == (6, 9, 9)
    assert observation.dtype == np.float32
    assert info["action_mask"].shape == (140,)
    assert info["action_mask"].dtype == np.bool_
    assert info["current_player"] == 0


def test_step_plays_opponent_reply_before_returning():
    env = QuoridorEnv(opponent_policy=ForwardOpponent())
    env.reset(seed=7)
    _, reward, terminated, truncated, info = env.step(0)

    assert reward == 0.0
    assert not terminated and not truncated
    assert env.state.pawns == ((1, 4), (7, 4))
    assert env.state.current_player == env.learner_player == 0
    assert info["opponent_action"] == 0


def test_terminal_rewards_are_strictly_win_loss_or_cap():
    env = QuoridorEnv(opponent_policy=ForwardOpponent())
    env.reset()
    env.state = State.from_components(pawns=((7, 4), (8, 8)), current_player=0)
    _, reward, terminated, truncated, _ = env.step(0)
    assert (reward, terminated, truncated) == (1.0, True, False)

    env.reset()
    env.state = State.from_components(pawns=((4, 4), (1, 0)), current_player=0)
    _, reward, terminated, truncated, info = env.step(2)
    assert (reward, terminated, truncated) == (-1.0, True, False)
    assert info["winner"] == 1


def test_player_one_environment_plays_opening_opponent_move():
    env = QuoridorEnv(opponent_policy=ForwardOpponent(), learner_player=1)
    _, info = env.reset(seed=3)

    assert env.state.pawns[0] == (1, 4)
    assert env.state.current_player == 1
    assert info["opponent_opening_action"] == 0


def test_illegal_action_raises_without_mutating_state():
    env = QuoridorEnv(opponent_policy=ForwardOpponent())
    env.reset()
    before = env.state
    with pytest.raises(ValueError, match="illegal action"):
        env.step(1)
    assert env.state == before


def test_gymnasium_api_contract():
    wrapped = gym.make(ENV_ID, opponent_policy=ForwardOpponent())
    check_env(wrapped.unwrapped, skip_render_check=False)
    wrapped.close()
