import copy

import numpy as np

from barricade_rl.az_network import AlphaZeroNetwork
from barricade_rl.config import load_config, small_game_from_config
from barricade_rl.mcts import MCTS, MCTSConfig


def _tiny_config():
    config = copy.deepcopy(load_config("configs/m2_5x5.json"))
    config["network"]["blocks"] = 2
    config["network"]["filters"] = 8
    config["network"]["global_pool_blocks"] = [0]
    return config


def test_alpha_zero_network_forward_matches_m2_head_contract():
    config = _tiny_config()
    game = small_game_from_config(config)
    network = AlphaZeroNetwork.from_config(config, seed=123)
    observation = game.canonical_observation(game.initial_state())

    output = network.forward(observation[None, ...])

    assert output.policy_logits.shape == (1, game.action_count)
    assert output.opponent_policy_logits.shape == (1, game.action_count)
    assert output.value.shape == (1,)
    assert output.auxiliary_distances.shape == (1, 2)
    assert np.isfinite(output.policy_logits).all()
    assert np.isfinite(output.opponent_policy_logits).all()
    assert np.isfinite(output.auxiliary_distances).all()
    assert -1.0 <= float(output.value[0]) <= 1.0
    assert network.config.policy_head == "flat-board-size-dependent"
    assert network.config.global_pool_blocks == (0,)
    assert network.config.observation_version == 1
    assert network.config.batch_norm_epsilon == 1e-5
    assert "stem_bn_scale" in network.params
    assert "stem_bn_mean" in network.params


def test_alpha_zero_network_checkpoint_round_trips_outputs(tmp_path):
    config = _tiny_config()
    game = small_game_from_config(config)
    network = AlphaZeroNetwork.from_config(config, seed=77)
    observation = game.canonical_observation(game.initial_state())[None, ...]
    before = network.forward(observation)
    path = tmp_path / "az_network.npz"

    network.save_checkpoint(path, step=5, config_hash="hash")
    loaded = AlphaZeroNetwork.load_checkpoint(path)
    after = loaded.forward(observation)

    assert loaded.metadata["step"] == 5
    assert loaded.metadata["config_hash"] == "hash"
    assert loaded.config.observation_version == 1
    np.testing.assert_allclose(after.policy_logits, before.policy_logits)
    np.testing.assert_allclose(after.opponent_policy_logits, before.opponent_policy_logits)
    np.testing.assert_allclose(after.value, before.value)
    np.testing.assert_allclose(after.auxiliary_distances, before.auxiliary_distances)


def test_alpha_zero_network_is_mcts_evaluator_compatible():
    config = _tiny_config()
    game = small_game_from_config(config)
    network = AlphaZeroNetwork.from_config(config, seed=5)

    result = MCTS(MCTSConfig(simulations=2), network).run(game, game.initial_state())

    assert 0 <= result.action < game.action_count
    assert result.policy.shape == (game.action_count,)
    assert abs(float(result.policy.sum()) - 1.0) < 1e-6
