import copy

import numpy as np

from barricade_rl.az_gating import GatingResult
from barricade_rl.az_learner import AlphaZeroLearner, LearnerConfig
from barricade_rl.az_network import AlphaZeroNetwork
from barricade_rl.az_pipeline import AlphaZeroCoordinator
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


def _coordinator(tmp_path):
    project_config = _tiny_config()
    game = SmallGame()
    learner = AlphaZeroLearner(
        AlphaZeroNetwork.from_config(project_config, seed=2),
        LearnerConfig.from_project_config(project_config),
        game,
        seed=3,
    )
    incumbent = tmp_path / "incumbent.npz"
    learner.save_checkpoint(
        incumbent,
        run_id="run",
        git_commit="commit",
        config_hash=config_hash(project_config),
    )
    state = game.initial_state()
    replay = AlphaZeroReplayBuffer(
        capacity=16,
        observation_shape=game.canonical_observation(state).shape,
        action_count=game.action_count,
    )
    return AlphaZeroCoordinator(
        project_config=project_config,
        game=game,
        learner=learner,
        replay_buffer=replay,
        incumbent_checkpoint=incumbent,
        output_directory=tmp_path / "run",
        run_id="run",
        git_commit="commit",
        seed=4,
    )


def _fake_self_play(game, evaluator, config, *, games, replay_buffer, **kwargs):
    del evaluator, config, kwargs
    state = game.initial_state()
    mask = game.legal_actions(state)
    policy = mask.astype(np.float32) / mask.sum()
    for index in range(2):
        replay_buffer.add(
            make_replay_sample(
                game,
                state,
                policy=policy,
                value=1.0 if index == 0 else -1.0,
                opponent_policy=policy,
                opponent_action_mask=mask,
            )
        )
    return tuple(object() for _ in range(games))


def _gate_result(promoted):
    return GatingResult(
        candidate="candidate",
        incumbent="incumbent",
        games=200,
        games_per_color=100,
        candidate_wins=110 if promoted else 100,
        incumbent_wins=90 if promoted else 100,
        draws=0,
        candidate_score=110.0 if promoted else 100.0,
        score_rate=0.55 if promoted else 0.5,
        promotion_threshold=0.55,
        promoted=promoted,
        seed=4,
        avg_plies=10.0,
        cap_fraction=0.0,
        start_positions=100,
        start_state_keys=tuple(f"state-{index}" for index in range(100)),
        start_sampling="paired-random-legal-prefixes-v1",
        start_ply_range=(1, 16),
        start_seed=4,
    )


def test_training_cycle_promotes_and_advances_incumbent(tmp_path):
    coordinator = _coordinator(tmp_path)

    result = coordinator.run_cycle(
        self_play_games=2,
        learner_steps=1,
        self_play_runner=_fake_self_play,
        gate_runner=lambda *args, **kwargs: _gate_result(True),
    )

    assert result.promoted
    assert result.learner_step == 1
    assert result.generated_positions == 2
    assert result.candidate_checkpoint.is_file()
    assert result.incumbent_checkpoint.is_file()
    assert result.incumbent_checkpoint.parent.name == "gated"
    assert coordinator.incumbent_checkpoint == result.incumbent_checkpoint
    assert (tmp_path / "run" / "replay.npz").is_file()


def test_training_cycle_rejection_rolls_learner_back_to_incumbent(tmp_path):
    coordinator = _coordinator(tmp_path)
    original_incumbent = coordinator.incumbent_checkpoint

    result = coordinator.run_cycle(
        self_play_games=2,
        learner_steps=1,
        self_play_runner=_fake_self_play,
        gate_runner=lambda *args, **kwargs: _gate_result(False),
    )

    assert not result.promoted
    assert result.learner_step == 1
    assert coordinator.learner.step == 0
    assert coordinator.incumbent_checkpoint == original_incumbent
    assert result.candidate_checkpoint.is_file()


def test_training_cycle_supplies_one_diverse_start_per_colour_pair(tmp_path):
    coordinator = _coordinator(tmp_path)
    calls = []

    def gate_runner(*args, **kwargs):
        del args
        calls.append(kwargs)
        return _gate_result(False)

    coordinator.run_cycle(
        self_play_games=2,
        learner_steps=1,
        self_play_runner=_fake_self_play,
        gate_runner=gate_runner,
    )

    starts = calls[0]["initial_states"]
    assert len(starts) == coordinator.gating_config.games_per_color
    assert len({coordinator.game.state_key(state) for state in starts}) == len(starts)
    assert calls[0]["start_seed"] == coordinator.seed
