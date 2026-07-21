import copy

import numpy as np

from barricade_rl.az_gating import GatingResult
from barricade_rl.az_learner import AlphaZeroLearner, LearnerConfig
from barricade_rl.az_network import AlphaZeroNetwork
from barricade_rl.az_pipeline import AlphaZeroCoordinator, _parse_args
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


def _coordinator(tmp_path, *, cycle_index=None):
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
        cycle_index=cycle_index,
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
    assert result.learner_metrics["step"] == 1
    assert result.learner_metrics["policy_loss"] >= 0.0
    assert result.learner_metrics["value_loss"] >= 0.0
    assert result.learner_metrics["auxiliary_loss"] >= 0.0
    assert result.learner_metrics["root_policy_entropy"] >= 0.0
    assert result.candidate_checkpoint.is_file()
    assert result.incumbent_checkpoint.is_file()
    assert result.incumbent_checkpoint.parent.name == "gated"
    assert coordinator.incumbent_checkpoint == result.incumbent_checkpoint
    assert (tmp_path / "run" / "replay.npz").is_file()


def test_training_cycle_rejection_keeps_continuous_learner_and_gated_incumbent(tmp_path):
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
    assert coordinator.learner.step == 1
    assert coordinator.incumbent_checkpoint == original_incumbent
    assert result.candidate_checkpoint.is_file()


def test_cycle_cli_accepts_separate_continuous_learner_checkpoint(tmp_path):
    learner_checkpoint = tmp_path / "candidate.npz"

    args = _parse_args(
        [
            "--oracle-corpus",
            str(tmp_path / "oracle.jsonl"),
            "--incumbent",
            str(tmp_path / "incumbent.npz"),
            "--learner-checkpoint",
            str(learner_checkpoint),
            "--output-directory",
            str(tmp_path / "run"),
            "--self-play-games",
            "2",
            "--learner-steps",
            "1",
            "--run-id",
            "run",
            "--git-commit",
            "commit",
        ]
    )

    assert args.learner_checkpoint == learner_checkpoint


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
    assert calls[0]["start_seed"] == coordinator.gating_start_seed


def test_consecutive_cycles_have_distinct_seeds_ids_and_artifact_paths(tmp_path):
    calls = []

    def self_play_runner(game, evaluator, config, *, replay_buffer, games, **kwargs):
        calls.append(kwargs)
        return _fake_self_play(
            game,
            evaluator,
            config,
            replay_buffer=replay_buffer,
            games=games,
        )

    first = _coordinator(tmp_path, cycle_index=0)
    first_result = first.run_cycle(
        self_play_games=2,
        learner_steps=1,
        self_play_runner=self_play_runner,
        gate_runner=lambda *args, **kwargs: _gate_result(False),
    )
    second = _coordinator(tmp_path, cycle_index=1)
    second_result = second.run_cycle(
        self_play_games=2,
        learner_steps=1,
        self_play_runner=self_play_runner,
        gate_runner=lambda *args, **kwargs: _gate_result(False),
    )

    assert first_result.cycle_index == 0
    assert second_result.cycle_index == 1
    assert first_result.self_play_seed != second_result.self_play_seed
    assert calls[0]["game_id_prefix"] != calls[1]["game_id_prefix"]
    assert first_result.candidate_checkpoint != second_result.candidate_checkpoint


def test_three_high_cap_cycles_activate_adjudication_for_the_next_cycle(tmp_path):
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    (run_directory / "cycles.jsonl").write_text(
        "".join(
            '{"cycle_index": %d, "self_play_cap_fraction": 0.1}\n' % index
            for index in range(3)
        )
    )

    coordinator = _coordinator(tmp_path, cycle_index=3)

    assert coordinator.adjudication_active
    assert (
        coordinator.self_play_config.scoring_scheme
        == "terminal-win-loss-cap-shortest-path-adjudicated"
    )


def test_adjudication_stays_active_after_cap_fraction_recovers(tmp_path):
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    (run_directory / "cycles.jsonl").write_text(
        "".join(
            [
                '{"cycle_index": 0, "self_play_cap_fraction": 0.1, '
                '"adjudication_active": false}\n',
                '{"cycle_index": 1, "self_play_cap_fraction": 0.1, '
                '"adjudication_active": false}\n',
                '{"cycle_index": 2, "self_play_cap_fraction": 0.1, '
                '"adjudication_active": false}\n',
                '{"cycle_index": 3, "self_play_cap_fraction": 0.0, '
                '"adjudication_active": true, '
                '"scoring_scheme": '
                '"terminal-win-loss-cap-shortest-path-adjudicated"}\n',
            ]
        )
    )

    coordinator = _coordinator(tmp_path, cycle_index=4)

    assert coordinator.prior_high_cap_streak == 0
    assert coordinator.adjudication_active
    assert (
        coordinator.self_play_config.scoring_scheme
        == "terminal-win-loss-cap-shortest-path-adjudicated"
    )
