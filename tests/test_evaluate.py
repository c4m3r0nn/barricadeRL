import numpy as np

from barricade_rl.evaluate import EvaluationResult, evaluate_model, select_model_action, select_policy_action
from barricade_rl.single_agent import BarricadeSingleAgentEnv


def test_random_eval_policy_selects_legal_action():
    env = BarricadeSingleAgentEnv()
    env.reset(seed=0)
    action = select_policy_action("random", env, np.random.default_rng(0))
    assert env.action_masks()[action]


def test_greedy_eval_policy_selects_legal_action():
    env = BarricadeSingleAgentEnv()
    env.reset(seed=0)
    action = select_policy_action("greedy", env, np.random.default_rng(0))
    assert env.action_masks()[action]


class FakeMaskedModel:
    def predict(self, obs, deterministic=True, action_masks=None):
        assert deterministic
        return int(action_masks.nonzero()[0][0]), None


def test_model_eval_policy_selects_legal_masked_action():
    env = BarricadeSingleAgentEnv()
    obs, info = env.reset(seed=0)
    action = select_model_action(FakeMaskedModel(), obs, env)
    assert env.action_masks()[action]


def test_evaluate_model_returns_summary():
    result = evaluate_model(FakeMaskedModel(), episodes=2, opponent_name="random", seed=0)
    assert isinstance(result, EvaluationResult)
    assert result.episodes == 2
    assert result.wins + result.losses + result.truncations == 2
    assert result.loss_rate == result.losses / result.episodes
    assert result.truncation_rate == result.truncations / result.episodes
    assert len(result.episode_lengths) == result.episodes
    assert result.min_learner_steps == min(result.episode_lengths)
    assert result.max_learner_steps == max(result.episode_lengths)
    assert result.avg_walls_placed >= 0
    assert result.avg_learner_walls_placed >= 0
    assert result.avg_opponent_walls_placed >= 0


def test_evaluation_result_rates_and_wall_metrics():
    result = EvaluationResult(
        episodes=4,
        wins=2,
        losses=1,
        truncations=1,
        total_steps=40,
        episode_lengths=[5, 10, 10, 15],
        learner_walls_placed=[1, 2, 3, 4],
        opponent_walls_placed=[0, 1, 1, 2],
    )
    assert result.win_rate == 0.5
    assert result.loss_rate == 0.25
    assert result.truncation_rate == 0.25
    assert result.avg_learner_steps == 10
    assert result.min_learner_steps == 5
    assert result.max_learner_steps == 15
    assert result.avg_learner_walls_placed == 2.5
    assert result.avg_opponent_walls_placed == 1.0
    assert result.avg_walls_placed == 3.5
