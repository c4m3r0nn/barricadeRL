import json

import numpy as np

from barricade_rl.baseline_dqn import (
    MaskedDQNConfig,
    MaskedDQNPolicy,
    load_masked_dqn,
    train_masked_dqn,
)
from barricade_rl.evaluate import play_match
from barricade_rl.game import ACTION_COUNT, Game
from barricade_rl.opponents import RandomOpponent


def test_masked_dqn_policy_never_selects_illegal_action():
    policy = MaskedDQNPolicy.initialized(seed=1, hidden_size=8, name="test-dqn")
    policy.b2[:] = 0.0
    policy.b2[1] = 100.0

    game = Game()
    state = game.initial_state()
    assert not game.legal_actions(state)[1]

    action = policy.select_action(game, state, np.random.default_rng(2))
    assert action != 1
    assert game.legal_actions(state)[action]


def test_masked_dqn_save_load_roundtrip(tmp_path):
    path = tmp_path / "policy.npz"
    policy = MaskedDQNPolicy.initialized(seed=3, hidden_size=8, name="roundtrip")
    policy.save(path, metadata={"episodes": 4})

    loaded = load_masked_dqn(path)
    assert loaded.name == "roundtrip"
    assert loaded.metadata["episodes"] == 4
    np.testing.assert_allclose(loaded.W1, policy.W1)
    np.testing.assert_allclose(loaded.b2, policy.b2)


def test_masked_dqn_smoke_training_beats_random_and_writes_artifact(tmp_path):
    artifact = tmp_path / "smoke_dqn.npz"
    config = MaskedDQNConfig(
        episodes=30,
        expert_episodes=30,
        hidden_size=32,
        batch_size=32,
        updates_per_episode=12,
        evaluation_games_per_color=5,
        seed=12,
    )

    result = train_masked_dqn(config, output_path=artifact)
    assert artifact.exists()
    assert result.evaluation.score_rate >= 0.8
    assert result.episodes == 30
    assert result.env_steps > 0
    assert result.loss is not None

    loaded = load_masked_dqn(artifact)
    match = play_match(loaded, RandomOpponent(), games_per_color=5, seed=99)
    assert match.score_rate >= 0.8

    metadata = json.loads(str(loaded.metadata["training_result_json"]))
    assert metadata["evaluation"]["opponent"] == "random"
    assert metadata["config"]["episodes"] == 30
    assert loaded.W2.shape == (32, ACTION_COUNT)
