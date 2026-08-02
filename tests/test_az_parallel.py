import copy

import numpy as np

from barricade_rl.az_gating import (
    GatingConfig,
    NetworkMCTSPolicy,
    gate_candidate,
    sample_gating_start_states,
)
from barricade_rl.az_learner import AlphaZeroLearner, LearnerConfig
from barricade_rl.az_network import AlphaZeroNetwork
from barricade_rl.az_parallel import (
    gate_checkpoints_parallel,
    generate_self_play_games_parallel,
)
from barricade_rl.az_replay import AlphaZeroReplayBuffer
from barricade_rl.az_self_play import SelfPlayConfig
from barricade_rl.config import config_hash, load_config, small_game_from_config


def _tiny_config():
    config = copy.deepcopy(load_config("configs/m2_5x5.json"))
    config["board"]["max_plies"] = 8
    config["network"]["blocks"] = 1
    config["network"]["filters"] = 4
    config["network"]["global_pool_blocks"] = []
    config["network"]["value_hidden"] = 4
    config["training"]["batch_size"] = 2
    config["training"]["mirror_augmentation"] = False
    return config


def _checkpoint(tmp_path, config, *, seed, name):
    game = small_game_from_config(config)
    learner = AlphaZeroLearner(
        AlphaZeroNetwork.from_config(config, seed=seed),
        LearnerConfig.from_project_config(config),
        game,
        seed=seed + 1,
    )
    path = tmp_path / f"{name}.npz"
    learner.save_checkpoint(
        path,
        run_id="parallel-test",
        git_commit="commit",
        config_hash=config_hash(config),
    )
    return path


def _replay(config):
    game = small_game_from_config(config)
    state = game.initial_state()
    return AlphaZeroReplayBuffer(
        capacity=128,
        observation_shape=game.canonical_observation(state).shape,
        action_count=game.action_count,
    )


def _record_signature(record):
    return (
        record.game_id,
        record.actions,
        record.terminal_status,
        record.winner,
        tuple(
            (
                sample.state_key,
                sample.value,
                sample.policy.tobytes(),
                sample.opponent_policy.tobytes(),
            )
            for sample in record.samples
        ),
    )


def test_parallel_self_play_is_identical_across_worker_counts(tmp_path):
    config = _tiny_config()
    checkpoint = _checkpoint(tmp_path, config, seed=7, name="incumbent")
    self_play = SelfPlayConfig(
        full_simulations=2,
        fast_simulations=1,
        full_search_probability=1.0,
        raw_policy_injection_probability=0.0,
    )
    one_replay = _replay(config)
    two_replay = _replay(config)
    kwargs = dict(
        project_config=config,
        checkpoint_path=checkpoint,
        config=self_play,
        games=4,
        run_id="run",
        config_hash="hash",
        git_commit="commit",
        seed=23,
        game_id_prefix="run-cycle-000013",
    )

    one = generate_self_play_games_parallel(
        **kwargs,
        replay_buffer=one_replay,
        workers=1,
    )
    two = generate_self_play_games_parallel(
        **kwargs,
        replay_buffer=two_replay,
        workers=2,
    )

    assert tuple(map(_record_signature, one)) == tuple(map(_record_signature, two))
    assert tuple(sample.game_id for sample in one_replay.samples) == tuple(
        sample.game_id for sample in two_replay.samples
    )


def test_parallel_checkpoint_gate_is_identical_across_worker_counts(tmp_path):
    config = _tiny_config()
    candidate = _checkpoint(tmp_path, config, seed=11, name="candidate")
    incumbent = _checkpoint(tmp_path, config, seed=13, name="incumbent")
    game = small_game_from_config(config)
    gating = GatingConfig(
        games=4,
        promotion_threshold=0.55,
        evaluation_simulations=2,
        start_min_plies=1,
        start_max_plies=3,
    )
    starts = sample_gating_start_states(game, count=2, seed=29, max_plies=3)
    kwargs = dict(
        project_config=config,
        candidate_checkpoint=candidate,
        incumbent_checkpoint=incumbent,
        game=game,
        config=gating,
        seed=31,
        initial_states=starts,
        start_seed=29,
    )

    one = gate_checkpoints_parallel(**kwargs, workers=1)
    two = gate_checkpoints_parallel(**kwargs, workers=2)
    candidate_network = AlphaZeroNetwork.load_checkpoint(candidate)
    incumbent_network = AlphaZeroNetwork.load_checkpoint(incumbent)
    serial = gate_candidate(
        NetworkMCTSPolicy(
            candidate_network,
            simulations=2,
            cpuct=1.6,
            name="candidate-0",
        ),
        NetworkMCTSPolicy(
            incumbent_network,
            simulations=2,
            cpuct=1.6,
            name="incumbent-0",
        ),
        game=game,
        config=gating,
        seed=31,
        initial_states=starts,
        start_seed=29,
    )

    assert one.to_dict() == two.to_dict()
    assert one.to_dict() == serial.to_dict()
    assert one.games == 4
    assert one.start_state_keys == tuple(game.state_key(state).hex() for state in starts)
