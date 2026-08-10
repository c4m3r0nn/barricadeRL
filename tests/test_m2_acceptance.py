import copy
import json

import numpy as np

from barricade_rl.az_learner import AlphaZeroLearner, LearnerConfig
from barricade_rl.az_network import AlphaZeroNetwork
from barricade_rl.config import config_hash, load_config, small_game_from_config
from barricade_rl.m2_acceptance import (
    _parse_args,
    _parse_validation_args,
    _select_position_indices,
    complete_optimal_actions,
    compute_value_metrics,
    evaluate_checkpoint_weight_variants,
    evaluate_m2_checkpoint,
    evaluate_weight_variants,
)
from barricade_rl.oracle5x5 import OracleLabel
from barricade_rl.oracle5x5 import NoWallTablebase
from barricade_rl.small_board import SmallState


def _tiny_config():
    config = copy.deepcopy(load_config("configs/m2_5x5.json"))
    config["board"]["max_plies"] = 12
    config["network"]["blocks"] = 1
    config["network"]["filters"] = 4
    config["network"]["global_pool_blocks"] = []
    config["network"]["value_hidden"] = 4
    config["training"]["batch_size"] = 2
    config["training"]["mirror_augmentation"] = False
    return config


def _artifacts(tmp_path):
    config = _tiny_config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    game = small_game_from_config(config)
    learner = AlphaZeroLearner(
        AlphaZeroNetwork.from_config(config, seed=5),
        LearnerConfig.from_project_config(config),
        game,
        seed=7,
    )
    checkpoint = tmp_path / "checkpoint.npz"
    learner.save_checkpoint(
        checkpoint,
        run_id="acceptance-test",
        git_commit="commit",
        config_hash=config_hash(config),
    )
    states = (
        SmallState.from_components(
            spec=game.spec,
            pawns=((3, 2), (4, 4)),
            walls_remaining=(0, 0),
            current_player=0,
            ply=2,
        ),
        SmallState.from_components(
            spec=game.spec,
            pawns=((1, 1), (3, 3)),
            walls_remaining=(0, 0),
            current_player=1,
            ply=6,
        ),
    )
    tablebase = NoWallTablebase(game)
    corpus = tmp_path / "oracle.jsonl"
    with corpus.open("w", encoding="utf-8") as handle:
        for index, state in enumerate(states):
            payload = tablebase.solve(state).to_dict()
            payload["record_index"] = index
            payload["config_hash"] = config_hash(config)
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return config_path, checkpoint, corpus, game, states


def test_complete_optimal_actions_includes_every_exact_winning_move():
    game = small_game_from_config(_tiny_config())
    state = SmallState.from_components(
        spec=game.spec,
        pawns=((3, 2), (4, 4)),
        walls_remaining=(0, 0),
        current_player=0,
        ply=2,
    )
    tablebase = NoWallTablebase(game)
    label = tablebase.solve(state)

    actions, complete = complete_optimal_actions(game, state, label, tablebase)

    assert complete
    assert actions
    assert label.best_action in actions
    assert all(
        -tablebase.solve(game.next_state(state, action)).value == label.value
        for action in actions
    )


def test_value_metrics_report_phase_accuracy_bias_and_calibration():
    metrics = compute_value_metrics(
        predictions=np.asarray([0.8, -0.2, -0.7, 0.4], dtype=np.float32),
        targets=np.asarray([1, 1, -1, -1], dtype=np.int8),
        phases=("opening", "opening", "midgame", "endgame"),
    )

    assert metrics["sign_accuracy"] == 0.5
    assert metrics["by_phase"]["opening"]["sign_accuracy"] == 0.5
    assert metrics["by_phase"]["midgame"]["sign_accuracy"] == 1.0
    assert metrics["predicted_positive_fraction"] == 0.5
    assert metrics["target_positive_fraction"] == 0.5
    assert sum(item["count"] for item in metrics["calibration_bins"]) == 4


def test_acceptance_evaluation_resumes_completed_mcts_positions(tmp_path):
    config_path, checkpoint, corpus, _, _ = _artifacts(tmp_path)
    output = tmp_path / "evaluation"
    kwargs = dict(
        config_path=config_path,
        oracle_corpus=corpus,
        checkpoint=checkpoint,
        output_directory=output,
        workers=1,
        simulations=2,
        position_limit=2,
    )

    first = evaluate_m2_checkpoint(**kwargs)
    records_before = (output / "positions.jsonl").read_text()
    second = evaluate_m2_checkpoint(**kwargs)

    assert first["mcts"]["completed_positions"] == 2
    assert second["mcts"]["resumed_positions"] == 2
    assert second["mcts"]["new_positions"] == 0
    assert (output / "positions.jsonl").read_text() == records_before
    assert second["optimal_action_sets"]["complete_fraction"] == 1.0
    assert second["criteria"]["initial_second_player"]["status"] == "blocked"
    assert second["criteria"]["monotone_ladder_elo"]["status"] == "blocked"


def test_acceptance_cli_exposes_parallel_resume_and_value_only(tmp_path):
    args = _parse_args(
        [
            "--oracle-corpus",
            str(tmp_path / "oracle.jsonl"),
            "--checkpoint",
            str(tmp_path / "checkpoint.npz"),
            "--output-directory",
            str(tmp_path / "evaluation"),
            "--workers",
            "8",
            "--position-limit",
            "32",
            "--value-only",
        ]
    )

    assert args.workers == 8
    assert args.position_limit == 32
    assert args.value_only


def test_limited_acceptance_positions_span_every_phase_deterministically():
    phases = ("opening",) * 5 + ("midgame",) * 5 + ("endgame",) * 5

    selected = _select_position_indices(phases, 8)

    assert selected == (0, 5, 10, 1, 6, 11, 2, 7)
    assert {phases[index] for index in selected} == {
        "opening",
        "midgame",
        "endgame",
    }


def test_weight_variant_diagnostics_expose_raw_and_ema_value_drift():
    config = _tiny_config()
    game = small_game_from_config(config)
    state = SmallState.from_components(
        spec=game.spec,
        pawns=((3, 2), (4, 4)),
        walls_remaining=(0, 0),
        current_player=0,
        ply=2,
    )
    tablebase = NoWallTablebase(game)
    label = tablebase.solve(state)
    actions, complete = complete_optimal_actions(game, state, label, tablebase)
    network = AlphaZeroNetwork.from_config(config, seed=3)
    network.params["value_fc2_b"].fill(3.0)
    network.ema_params["value_fc2_b"].fill(-3.0)

    result = evaluate_weight_variants(
        network=network,
        game=game,
        labels=(OracleLabel.from_dict(label.to_dict()),),
        optimal_sets=(actions,),
        optimal_complete=(complete,),
        batch_size=1,
    )

    assert result["raw"]["value"]["predicted_positive_fraction"] == 1.0
    assert result["ema"]["value"]["predicted_positive_fraction"] == 0.0
    assert result["delta_raw_minus_ema"]["value_sign_accuracy"] in (-1.0, 1.0)
    assert result["raw"]["policy"]["complete_optimal_set_positions"] == 1


def test_checkpoint_weight_diagnostics_reuse_complete_optimal_action_cache(tmp_path):
    config_path, checkpoint, corpus, game, _ = _artifacts(tmp_path)
    config = load_config(config_path)
    network = AlphaZeroNetwork.load_checkpoint(checkpoint)
    cache = tmp_path / "validation"
    kwargs = dict(
        project_config=config,
        game=game,
        network=network,
        oracle_corpus=corpus,
        cache_directory=cache,
        workers=1,
        batch_size=2,
    )

    first = evaluate_checkpoint_weight_variants(**kwargs)
    cache_before = (cache / "optimal_actions.jsonl").read_text()
    second = evaluate_checkpoint_weight_variants(**kwargs)

    assert first == second
    assert (cache / "optimal_actions.jsonl").read_text() == cache_before
    assert first["checkpoint_step"] == 0
    assert first["ema_initialization_weight"] == 1.0
    assert first["optimal_action_sets"]["complete_fraction"] == 1.0


def test_weight_validation_cli_exposes_cache_and_parallel_controls(tmp_path):
    args = _parse_validation_args(
        [
            "--oracle-corpus",
            str(tmp_path / "oracle.jsonl"),
            "--checkpoint",
            str(tmp_path / "checkpoint.npz"),
            "--cache-directory",
            str(tmp_path / "cache"),
            "--output",
            str(tmp_path / "validation.json"),
            "--workers",
            "8",
            "--batch-size",
            "1024",
        ]
    )

    assert args.workers == 8
    assert args.batch_size == 1024
    assert args.output == tmp_path / "validation.json"
