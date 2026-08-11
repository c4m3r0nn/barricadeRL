import copy
import hashlib
import json

import numpy as np
import torch

from barricade_rl.az_learner import (
    AlphaZeroLearner,
    LearnerConfig,
    main,
    mirror_replay_batch,
    recalibration_main,
)
from barricade_rl.az_network import AlphaZeroNetwork
from barricade_rl.az_replay import AlphaZeroReplayBuffer, make_replay_sample
from barricade_rl.config import config_hash as calculate_config_hash
from barricade_rl.config import load_config
from barricade_rl.small_board import SmallGame


def _tiny_config():
    config = copy.deepcopy(load_config("configs/m2_5x5.json"))
    config["network"]["blocks"] = 1
    config["network"]["filters"] = 4
    config["network"]["global_pool_blocks"] = []
    config["network"]["value_hidden"] = 4
    config["training"]["batch_size"] = 2
    config["training"]["mirror_augmentation"] = False
    config["training"]["initial_learning_rate"] = 0.01
    config["training"]["learning_rate_drop_steps"] = [2, 4]
    return config


def _replay(game, *, count=2):
    state = game.initial_state()
    mask = game.legal_actions(state)
    policy = mask.astype(np.float32)
    policy /= policy.sum()
    buffer = AlphaZeroReplayBuffer(
        capacity=16,
        observation_shape=game.canonical_observation(state).shape,
        action_count=game.action_count,
    )
    for index in range(count):
        buffer.add(
            make_replay_sample(
                game,
                state,
                policy=policy,
                value=1.0 if index % 2 == 0 else -1.0,
                opponent_policy=policy,
                opponent_action_mask=mask,
                config_hash="hash",
                game_id=f"game-{index}",
                run_id="run",
                git_commit="commit",
            )
        )
    return buffer


def test_learner_config_matches_handover_and_fixed_schedule():
    config = LearnerConfig.from_project_config(load_config("configs/m2_5x5.json"))

    assert config.optimizer == "sgd"
    assert config.momentum == 0.9
    assert config.batch_size == 512
    assert config.weight_decay == 1e-4
    assert config.auxiliary_loss_weight == 0.1
    assert config.opponent_policy_loss_weight == 0.15
    assert config.learning_rate(0) == 0.02
    assert config.learning_rate(config.learning_rate_drop_steps[0]) == 0.002
    assert config.learning_rate(config.learning_rate_drop_steps[1]) == 0.0002


def test_gradient_step_updates_raw_and_ema_weights_and_consumption_ratio():
    project_config = _tiny_config()
    game = SmallGame()
    network = AlphaZeroNetwork.from_config(project_config, seed=4)
    learner = AlphaZeroLearner(network, LearnerConfig.from_project_config(project_config), game, seed=7)
    replay = _replay(game)
    raw_before = {name: value.copy() for name, value in network.params.items()}
    ema_before = {name: value.copy() for name, value in network.ema_params.items()}
    running_mean_before = network.params["stem_bn_mean"].copy()

    metrics = learner.train_step(replay)

    assert metrics.step == 1
    assert metrics.batch_size == 2
    assert metrics.learning_rate == 0.01
    assert np.isfinite(metrics.total_loss)
    assert metrics.policy_loss > 0
    assert metrics.value_loss >= 0
    assert any(not np.array_equal(raw_before[name], network.params[name]) for name in raw_before)
    assert any(not np.array_equal(ema_before[name], network.ema_params[name]) for name in ema_before)
    assert not np.array_equal(running_mean_before, network.params["stem_bn_mean"])
    assert replay.samples_per_position_ratio == 1.0


def test_differentiable_training_model_matches_numpy_raw_inference():
    project_config = _tiny_config()
    game = SmallGame()
    network = AlphaZeroNetwork.from_config(project_config, seed=11)
    learner = AlphaZeroLearner(
        network, LearnerConfig.from_project_config(project_config), game, seed=5
    )
    observation = game.canonical_observation(game.initial_state())[None, ...]
    expected = network.forward(observation, use_ema=False)

    learner.model.eval()
    with torch.no_grad():
        policy, value, distances, opponent = learner.model(
            torch.from_numpy(observation)
        )

    np.testing.assert_allclose(policy.numpy(), expected.policy_logits, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(value.numpy(), expected.value, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        distances.numpy(), expected.auxiliary_distances, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        opponent.numpy(), expected.opponent_policy_logits, rtol=1e-5, atol=1e-5
    )


def test_differentiable_model_registers_batch_norm_statistics_as_buffers():
    project_config = _tiny_config()
    learner = AlphaZeroLearner(
        AlphaZeroNetwork.from_config(project_config, seed=14),
        LearnerConfig.from_project_config(project_config),
        SmallGame(),
    )

    parameter_names = {name for name, _ in learner.model.named_parameters()}
    buffer_names = {name for name, _ in learner.model.named_buffers()}

    assert not any(name.endswith(("_bn_mean", "_bn_var")) for name in parameter_names)
    assert any(name.endswith("stem_bn_mean") for name in buffer_names)
    assert any(name.endswith("stem_bn_var") for name in buffer_names)


def test_learner_refuses_to_overconsume_stale_replay():
    project_config = _tiny_config()
    game = SmallGame()
    learner = AlphaZeroLearner(
        AlphaZeroNetwork.from_config(project_config, seed=6),
        LearnerConfig.from_project_config(project_config),
        game,
    )
    replay = _replay(game)

    for _ in range(4):
        learner.train_step(replay)

    try:
        learner.train_step(replay)
    except RuntimeError as exc:
        assert "samples-per-position" in str(exc)
    else:
        raise AssertionError("learner must stop before overtraining stale replay")


def test_mirror_augmentation_flips_observation_and_both_policy_targets():
    game = SmallGame()
    replay = _replay(game)
    batch = replay.sample(2, rng=np.random.default_rng(1))

    mirrored = mirror_replay_batch(batch, game, np.asarray([True, False]))

    np.testing.assert_array_equal(mirrored.observations[0], batch.observations[0, :, :, ::-1])
    np.testing.assert_array_equal(mirrored.policies[0], game.mirror(batch.policies[0]))
    np.testing.assert_array_equal(
        mirrored.opponent_policies[0], game.mirror(batch.opponent_policies[0])
    )
    np.testing.assert_array_equal(mirrored.observations[1], batch.observations[1])
    np.testing.assert_array_equal(mirrored.auxiliary_distances, batch.auxiliary_distances)


def test_learner_checkpoint_restores_network_momentum_and_step(tmp_path):
    project_config = _tiny_config()
    game = SmallGame()
    learner = AlphaZeroLearner(
        AlphaZeroNetwork.from_config(project_config, seed=2),
        LearnerConfig.from_project_config(project_config),
        game,
        seed=3,
    )
    replay = _replay(game)
    learner.train_step(replay)
    path = tmp_path / "learner.npz"

    learner.save_checkpoint(
        path,
        run_id="run",
        git_commit="commit",
        config_hash="hash",
    )
    restored = AlphaZeroLearner.load_checkpoint(path, game)

    assert restored.step == learner.step == 1
    assert restored.network.metadata["run_id"] == "run"
    assert restored.network.metadata["git_commit"] == "commit"
    for name in learner.network.params:
        np.testing.assert_allclose(restored.network.params[name], learner.network.params[name])
        np.testing.assert_allclose(restored.momentum_buffers[name], learner.momentum_buffers[name])


def test_ema_batch_norm_recalibration_changes_only_ema_running_stats():
    project_config = _tiny_config()
    game = SmallGame()
    learner = AlphaZeroLearner(
        AlphaZeroNetwork.from_config(project_config, seed=12),
        LearnerConfig.from_project_config(project_config),
        game,
        seed=13,
    )
    replay = _replay(game, count=4)
    learner.train_step(replay)
    raw_before = {
        name: value.copy() for name, value in learner.network.params.items()
    }
    ema_before = {
        name: value.copy() for name, value in learner.network.ema_params.items()
    }
    momentum_before = {
        name: value.copy() for name, value in learner.momentum_buffers.items()
    }
    rng_before = json.dumps(learner.rng.bit_generator.state, sort_keys=True)
    replay_before = (
        replay.total_positions_added,
        replay.gradient_samples_consumed,
        replay.samples_per_position_ratio,
    )

    result = learner.recalibrate_ema_batch_norm(replay, batch_size=2)

    assert result.protocol == "replay-cumulative-v1"
    assert result.positions == replay.size == 4
    assert result.batches == 2
    assert result.batch_size == 2
    assert result.step == learner.step == 1
    assert result.device == "cpu"
    for name, value in raw_before.items():
        np.testing.assert_array_equal(learner.network.params[name], value)
    for name, value in ema_before.items():
        if name.endswith(("_bn_mean", "_bn_var")):
            continue
        np.testing.assert_array_equal(learner.network.ema_params[name], value)
    assert any(
        not np.array_equal(learner.network.ema_params[name], value)
        for name, value in ema_before.items()
        if name.endswith(("_bn_mean", "_bn_var"))
    )
    for name, value in momentum_before.items():
        np.testing.assert_array_equal(learner.momentum_buffers[name], value)
    assert json.dumps(learner.rng.bit_generator.state, sort_keys=True) == rng_before
    assert (
        replay.total_positions_added,
        replay.gradient_samples_consumed,
        replay.samples_per_position_ratio,
    ) == replay_before
    assert learner.network.metadata["ema_batch_norm"]["protocol"] == result.protocol

    calibrated = {
        name: value.copy()
        for name, value in learner.network.ema_params.items()
        if name.endswith(("_bn_mean", "_bn_var"))
    }
    learner.recalibrate_ema_batch_norm(replay, batch_size=2)
    for name, value in calibrated.items():
        np.testing.assert_array_equal(learner.network.ema_params[name], value)


def test_zero_step_cli_initializes_reproducible_incumbent(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_tiny_config()))
    output = tmp_path / "incumbent.npz"

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--output",
            str(output),
            "--steps",
            "0",
            "--run-id",
            "run",
            "--git-commit",
            "commit",
        ]
    )
    restored = AlphaZeroLearner.load_checkpoint(output, SmallGame())

    assert exit_code == 0
    assert restored.step == 0
    assert restored.network.metadata["run_id"] == "run"


def test_recalibration_cli_writes_provenanced_derivative_without_touching_source(
    tmp_path,
):
    project_config = _tiny_config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(project_config))
    game = SmallGame()
    replay = _replay(game, count=4)
    learner = AlphaZeroLearner(
        AlphaZeroNetwork.from_config(project_config, seed=21),
        LearnerConfig.from_project_config(project_config),
        game,
        seed=22,
    )
    learner.train_step(replay)
    source = tmp_path / "candidate.npz"
    replay_path = tmp_path / "replay.npz"
    output = tmp_path / "candidate-ema-bn-recalibrated.npz"
    learner.save_checkpoint(
        source,
        run_id="run",
        git_commit="training-commit",
        config_hash=calculate_config_hash(project_config),
    )
    replay.save_npz(replay_path)
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    replay_digest = hashlib.sha256(replay_path.read_bytes()).hexdigest()

    exit_code = recalibration_main(
        [
            "--config",
            str(config_path),
            "--checkpoint",
            str(source),
            "--replay",
            str(replay_path),
            "--output",
            str(output),
            "--batch-size",
            "2",
            "--git-commit",
            "recalibration-commit",
        ]
    )

    assert exit_code == 0
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_digest
    derived = AlphaZeroLearner.load_checkpoint(output, game)
    original = AlphaZeroLearner.load_checkpoint(source, game)
    assert derived.step == original.step
    assert derived.network.metadata["git_commit"] == "training-commit"
    provenance = derived.network.metadata["ema_batch_norm"]
    assert provenance["source_checkpoint"] == str(source)
    assert provenance["source_sha256"] == source_digest
    assert provenance["source_replay"] == str(replay_path)
    assert provenance["source_replay_sha256"] == replay_digest
    assert provenance["recalibration_git_commit"] == "recalibration-commit"
    assert provenance["elapsed_seconds"] >= 0.0
    for name in original.network.params:
        np.testing.assert_array_equal(
            derived.network.params[name], original.network.params[name]
        )
        np.testing.assert_array_equal(
            derived.momentum_buffers[name], original.momentum_buffers[name]
        )
        if not name.endswith(("_bn_mean", "_bn_var")):
            np.testing.assert_array_equal(
                derived.network.ema_params[name], original.network.ema_params[name]
            )
