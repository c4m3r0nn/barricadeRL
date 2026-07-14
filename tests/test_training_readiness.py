import json
from pathlib import Path

from barricade_rl.config import config_hash, load_config
from barricade_rl.oracle5x5 import solve_no_wall_endgame
from barricade_rl.small_board import SmallGame, SmallState
from barricade_rl.training_readiness import check_training_readiness


def test_training_readiness_blocks_without_oracle_corpus():
    summary = check_training_readiness(Path("configs/m2_5x5.json"))

    assert not summary.ready
    assert not summary.oracle_ready
    assert summary.replay_ready
    assert summary.network_ready
    assert summary.self_play_ready
    assert summary.learner_ready
    assert summary.gating_ready
    assert summary.pipeline_ready
    assert any("oracle corpus" in blocker for blocker in summary.blockers)
    assert not any("network implementation" in blocker for blocker in summary.blockers)
    assert not any("self-play actor" in blocker for blocker in summary.blockers)
    assert not any("learner implementation" in blocker for blocker in summary.blockers)


def test_training_readiness_passes_with_an_audited_corpus_and_all_training_contracts(tmp_path):
    config_path = Path("configs/m2_5x5.json")
    config = load_config(config_path)
    game = SmallGame()
    states = (
        SmallState.from_components(walls_remaining=(0, 0), current_player=0, ply=0),
        SmallState.from_components(
            pawns=((1, 2), (3, 2)),
            walls_remaining=(0, 0),
            current_player=0,
            ply=80,
        ),
        SmallState.from_components(
            pawns=((2, 1), (3, 3)),
            walls_remaining=(0, 0),
            current_player=1,
            ply=160,
        ),
    )
    corpus = tmp_path / "oracle.jsonl"
    with corpus.open("w", encoding="utf-8") as handle:
        for index, state in enumerate(states):
            payload = solve_no_wall_endgame(game, state).to_dict()
            payload["record_index"] = index
            payload["config_hash"] = config_hash(config)
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    summary = check_training_readiness(
        config_path,
        oracle_corpus=corpus,
        min_records=3,
        min_exact_fraction=1.0,
        min_phase_records=1,
    )

    assert summary.ready
    assert summary.oracle_ready
    assert summary.replay_ready
    assert summary.network_ready
    assert summary.self_play_ready
    assert summary.learner_ready
    assert summary.gating_ready
    assert summary.pipeline_ready
    assert summary.oracle_audit is not None
    assert summary.oracle_audit["passed"] is True
    assert "network implementation is missing" not in summary.blockers
    assert "self-play actor implementation is missing" not in summary.blockers
    assert "learner implementation is missing" not in summary.blockers
