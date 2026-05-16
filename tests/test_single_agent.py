import numpy as np
import pytest

from barricade_rl.core import BOARD_SIZE, MOVE_ACTIONS, BarricadeGame, goal_row
from barricade_rl.opponents import (
    AntiRushLiteOpponent,
    AntiRushMediumOpponent,
    AntiRushOpponent,
    CheckpointPoolOpponent,
    CurriculumOpponent,
    GreedyOpponent,
    MixedOpponent,
    RandomOpponent,
    RefreshingCheckpointPoolOpponent,
    StageThreeBridgeCurriculumOpponent,
    StageThreeGentleCurriculumOpponent,
    StageThreeCurriculumOpponent,
    StageTwoCurriculumOpponent,
    make_opponent,
)
from barricade_rl.single_agent import BarricadeSingleAgentEnv, EndgameStartSampler


class FixedOpponent:
    def __init__(self, action):
        self.action = action

    def select_action(self, game, rng):
        return self.action


class PathDamagingWallOpponent:
    def select_action(self, game, rng):
        target = 1 - game.state.current_player
        current_path = game.shortest_path_length(target)
        for action in np.flatnonzero(game.legal_actions_mask()):
            if action < 4:
                continue
            original_h = game.state.h_walls.copy()
            original_v = game.state.v_walls.copy()
            original_walls_remaining = list(game.state.walls_remaining)
            original_move_count = game.state.move_count
            try:
                if game.apply_action(int(action)):
                    new_path = game.shortest_path_length(target)
                    if current_path is not None and new_path is not None and new_path > current_path:
                        return int(action)
            finally:
                game.state.h_walls = original_h
                game.state.v_walls = original_v
                game.state.walls_remaining = original_walls_remaining
                game.state.move_count = original_move_count
                game.state.current_player = 1 - target
                game.state.winner = None
        raise AssertionError("Expected a legal wall that lengthens the learner path")


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
    assert isinstance(make_opponent("anti_rush_lite"), AntiRushLiteOpponent)
    assert isinstance(make_opponent("anti_rush_medium"), AntiRushMediumOpponent)
    assert isinstance(make_opponent("curriculum"), CurriculumOpponent)
    assert isinstance(make_opponent("curriculum_stage2"), StageTwoCurriculumOpponent)
    assert isinstance(make_opponent("curriculum_stage3_bridge"), StageThreeBridgeCurriculumOpponent)
    assert isinstance(make_opponent("curriculum_stage3_gentle"), StageThreeGentleCurriculumOpponent)
    assert isinstance(make_opponent("curriculum_stage3"), StageThreeCurriculumOpponent)


def test_anti_rush_opponent_prefers_wall_when_runner_is_close():
    game = BarricadeGame()
    game.state.pawns[0] = (2, 4)
    game.state.current_player = 1

    action = AntiRushOpponent().select_action(game, np.random.default_rng(0))

    assert action >= 4
    assert game.legal_actions_mask()[action]


def test_anti_rush_opponent_waits_until_runner_is_close():
    game = BarricadeGame()
    game.state.current_player = 1

    action = AntiRushOpponent().select_action(game, np.random.default_rng(0))

    assert action in MOVE_ACTIONS


def test_anti_rush_lite_is_less_aggressive_than_full_anti_rush():
    lite = AntiRushLiteOpponent()

    assert lite.wall_probability < 1.0
    assert lite.wall_bias_distance < AntiRushOpponent().wall_bias_distance


def test_anti_rush_medium_keeps_full_trigger_but_reduces_frequency():
    medium = AntiRushMediumOpponent()
    full = AntiRushOpponent()

    assert medium.wall_bias_distance == full.wall_bias_distance
    assert medium.max_self_cost == full.max_self_cost
    assert AntiRushLiteOpponent().wall_probability < medium.wall_probability < full.wall_probability


def test_curriculum_opponent_defaults_to_gentler_weights():
    opponent = CurriculumOpponent()

    assert opponent.random_weight == pytest.approx(0.25)
    assert opponent.greedy_weight == pytest.approx(0.70)
    assert opponent.anti_rush_weight == pytest.approx(0.0)
    assert opponent.mixed_weight == pytest.approx(0.05)


def test_stage_two_curriculum_introduces_light_anti_rush():
    opponent = StageTwoCurriculumOpponent()

    assert opponent.random_weight == pytest.approx(0.15)
    assert opponent.greedy_weight == pytest.approx(0.60)
    assert opponent.anti_rush_weight == pytest.approx(0.15)
    assert opponent.mixed_weight == pytest.approx(0.10)
    assert isinstance(opponent.anti_rush_opponent, AntiRushLiteOpponent)


def test_stage_three_curriculum_increases_anti_rush_pressure():
    opponent = StageThreeCurriculumOpponent()

    assert opponent.random_weight == pytest.approx(0.10)
    assert opponent.greedy_weight == pytest.approx(0.55)
    assert opponent.anti_rush_weight == pytest.approx(0.20)
    assert opponent.mixed_weight == pytest.approx(0.15)


def test_stage_three_gentle_curriculum_introduces_full_anti_rush_cautiously():
    opponent = StageThreeGentleCurriculumOpponent()

    assert opponent.random_weight == pytest.approx(0.15)
    assert opponent.greedy_weight == pytest.approx(0.67)
    assert opponent.anti_rush_weight == pytest.approx(0.08)
    assert opponent.mixed_weight == pytest.approx(0.10)
    assert isinstance(opponent.anti_rush_opponent, AntiRushOpponent)


def test_stage_three_bridge_curriculum_uses_medium_anti_rush_drills():
    opponent = StageThreeBridgeCurriculumOpponent()

    assert opponent.random_weight == pytest.approx(0.12)
    assert opponent.greedy_weight == pytest.approx(0.58)
    assert opponent.anti_rush_weight == pytest.approx(0.20)
    assert opponent.mixed_weight == pytest.approx(0.10)
    assert isinstance(opponent.anti_rush_opponent, AntiRushMediumOpponent)


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


def test_endgame_start_sampler_creates_legal_near_goal_state():
    sampler = EndgameStartSampler()
    game = BarricadeGame()

    state = sampler.sample(np.random.default_rng(0), learner_side=0)
    game.set_state(state)

    assert state.current_player == 0
    assert state.winner is None
    assert 2 <= abs(state.pawns[0][0] - goal_row(0)) <= 4
    assert state.pawns[0] != state.pawns[1]
    assert game.has_path(0)
    assert game.has_path(1)
    assert game.legal_actions_mask().any()


def test_single_agent_can_reset_from_endgame_start():
    env = BarricadeSingleAgentEnv(endgame_start_probability=1.0, learner_side=1)

    obs, info = env.reset(seed=0)

    assert info["endgame_start"] is True
    assert info["learner_side"] == 1
    assert info["current_player"] == 1
    assert 2 <= abs(info["pawns"][1][0] - goal_row(1)) <= 4
    assert env.game.has_path(0)
    assert env.game.has_path(1)
    assert obs.shape == (6, BOARD_SIZE, BOARD_SIZE)


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


def test_single_agent_wall_penalty_discourages_wall_spam():
    env = BarricadeSingleAgentEnv(opponent=FixedOpponent(1), wall_penalty=0.03)
    env.reset()

    obs, reward, terminated, truncated, info = env.step(4)

    assert reward == pytest.approx(-0.03)
    assert info["action_shaping_reward"] == pytest.approx(-0.03)


def test_single_agent_reverse_move_penalty_discourages_oscillation():
    env = BarricadeSingleAgentEnv(opponent=FixedOpponent(1), reverse_move_penalty=0.04)
    env.reset()
    env.step(0)

    obs, reward, terminated, truncated, info = env.step(1)

    assert reward == pytest.approx(-0.04)
    assert info["action_shaping_reward"] == pytest.approx(-0.04)


def test_single_agent_progress_reward_encourages_shorter_path_moves():
    env = BarricadeSingleAgentEnv(opponent=FixedOpponent(1), progress_reward_scale=0.05)
    env.reset()

    obs, reward, terminated, truncated, info = env.step(0)

    assert reward == pytest.approx(0.05)
    assert info["action_shaping_reward"] == pytest.approx(0.05)


def test_single_agent_survival_reward_adds_small_nonterminal_bonus():
    env = BarricadeSingleAgentEnv(opponent=FixedOpponent(1), survival_reward=0.02)
    env.reset()

    obs, reward, terminated, truncated, info = env.step(0)

    assert not terminated
    assert reward == pytest.approx(0.02)
    assert info["survival_reward"] == pytest.approx(0.02)


def test_single_agent_opponent_wall_value_penalty_tracks_path_damage():
    env = BarricadeSingleAgentEnv(opponent=PathDamagingWallOpponent(), opponent_wall_value_penalty_scale=0.03)
    env.reset()
    env.game.state.pawns[0] = (2, 4)
    env.game.state.pawns[1] = (8, 0)

    obs, reward, terminated, truncated, info = env.step(0)

    assert not terminated
    assert info["opponent_wall_value_delta"] > 0
    assert reward == pytest.approx(-0.03 * info["opponent_wall_value_delta"])
    assert info["opponent_wall_value_reward"] == pytest.approx(reward)


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
