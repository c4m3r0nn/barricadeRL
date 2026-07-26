import copy
import hashlib

import numpy as np
import pytest

from barricade_rl.apple_benchmark import (
    benchmark_inference,
    benchmark_learner,
    benchmark_mcts_process_scaling,
    parse_positive_int_csv,
)
from barricade_rl.az_learner import AlphaZeroLearner, LearnerConfig
from barricade_rl.az_network import AlphaZeroNetwork
from barricade_rl.az_replay import AlphaZeroReplayBuffer, make_replay_sample
from barricade_rl.config import config_hash, load_config
from barricade_rl.small_board import SmallGame


def _tiny_config():
    config = copy.deepcopy(load_config("configs/m2_5x5.json"))
    config["network"]["blocks"] = 1
    config["network"]["filters"] = 4
    config["network"]["global_pool_blocks"] = []
    config["network"]["value_hidden"] = 4
    config["training"]["batch_size"] = 2
    config["training"]["mirror_augmentation"] = False
    return config


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_parse_positive_int_csv_rejects_invalid_values():
    assert parse_positive_int_csv("1, 4,8") == (1, 4, 8)
    with pytest.raises(ValueError, match="positive"):
        parse_positive_int_csv("1,0")
    with pytest.raises(ValueError, match="at least one"):
        parse_positive_int_csv("")


def test_inference_benchmark_checks_numpy_torch_cpu_parity():
    config = _tiny_config()
    game = SmallGame()
    network = AlphaZeroNetwork.from_config(config, seed=7)
    state = game.initial_state()
    observations = np.stack(
        [
            game.canonical_observation(state),
            game.canonical_observation(game.next_state(state, 0)),
        ]
    )

    result = benchmark_inference(
        network,
        observations,
        batch_sizes=(1, 2),
        iterations=1,
        warmup=0,
        devices=("cpu",),
    )

    assert [entry["batch_size"] for entry in result["numpy"]] == [1, 2]
    assert [entry["batch_size"] for entry in result["torch"]["cpu"]] == [1, 2]
    assert result["parity"]["cpu"]["policy_logits_max_abs_error"] < 1e-5
    assert result["parity"]["cpu"]["value_max_abs_error"] < 1e-5
    assert all(entry["positions_per_second"] > 0 for entry in result["numpy"])
    assert all(
        entry["round_trip_positions_per_second"] > 0
        for entry in result["torch"]["cpu"]
    )


def test_learner_benchmark_does_not_modify_checkpoint_or_replay(tmp_path):
    config = _tiny_config()
    game = SmallGame()
    learner = AlphaZeroLearner(
        AlphaZeroNetwork.from_config(config, seed=11),
        LearnerConfig.from_project_config(config),
        game,
        seed=13,
    )
    checkpoint = tmp_path / "checkpoint.npz"
    learner.save_checkpoint(
        checkpoint,
        run_id="benchmark-test",
        git_commit="commit",
        config_hash=config_hash(config),
    )
    state = game.initial_state()
    replay = AlphaZeroReplayBuffer(
        capacity=8,
        observation_shape=game.canonical_observation(state).shape,
        action_count=game.action_count,
    )
    mask = game.legal_actions(state)
    policy = mask.astype(np.float32) / mask.sum()
    for value in (-1.0, 1.0):
        replay.add(
            make_replay_sample(
                game,
                state,
                policy=policy,
                value=value,
                opponent_policy=policy,
                opponent_action_mask=mask,
            )
        )
    replay_path = tmp_path / "replay.npz"
    replay.save_npz(replay_path)
    before = (_sha256(checkpoint), _sha256(replay_path))

    result = benchmark_learner(
        checkpoint,
        replay_path,
        game,
        devices=("cpu",),
        steps=1,
    )

    assert result["cpu"]["completed_steps"] == 1
    assert result["cpu"]["steps_per_second"] > 0
    assert len(result["cpu"]["step_seconds"]) == 1
    assert result["cpu"]["cold_step_seconds"] == result["cpu"]["step_seconds"][0]
    assert result["cpu"]["steady_state_steps_per_second"] is None
    assert result["cpu"]["final_metrics"]["step"] == 1
    assert before == (_sha256(checkpoint), _sha256(replay_path))


def test_mcts_process_benchmark_preserves_deterministic_actions(tmp_path):
    config = _tiny_config()
    game = SmallGame()
    checkpoint = tmp_path / "checkpoint.npz"
    AlphaZeroLearner(
        AlphaZeroNetwork.from_config(config, seed=17),
        LearnerConfig.from_project_config(config),
        game,
        seed=19,
    ).save_checkpoint(
        checkpoint,
        run_id="benchmark-test",
        git_commit="commit",
        config_hash=config_hash(config),
    )
    initial = game.initial_state()
    next_state = game.next_state(initial, int(np.flatnonzero(game.legal_actions(initial))[0]))

    result = benchmark_mcts_process_scaling(
        "configs/m2_5x5.json",
        checkpoint,
        (game.state_key(initial), game.state_key(next_state)),
        simulations=2,
        worker_counts=(1, 2),
        tasks=2,
    )

    assert result["startup_excluded"] is True
    assert result["deterministic"] is True
    assert [record["workers"] for record in result["records"]] == [1, 2]
    assert all(record["actions_match_first"] for record in result["records"])
    assert all(record["tasks_per_second"] > 0 for record in result["records"])
