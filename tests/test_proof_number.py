import json
from pathlib import Path

from barricade_rl.config import load_config
from barricade_rl.oracle5x5 import (
    ProofNumberConfig,
    ProofNumberSolvedCache,
    ProofSearchConfig,
    generate_oracle_corpus,
    load_oracle_corpus,
    proof_number_search,
)
from barricade_rl.small_board import SmallGame, SmallState, SolverOutcome


def test_proof_number_search_proves_immediate_win():
    game = SmallGame()
    state = SmallState.from_components(
        pawns=((3, 2), (4, 4)),
        walls_remaining=(0, 0),
        current_player=0,
    )

    label = proof_number_search(game, state, ProofNumberConfig(max_nodes=64))

    assert label.outcome is SolverOutcome.WIN
    assert label.value == 1
    assert label.exact
    assert label.best_action == 0
    assert label.proof_number == 0
    assert label.disproof_number > 0


def test_proof_number_search_reuses_cached_exact_root_label():
    game = SmallGame()
    cache = ProofNumberSolvedCache(game)
    state = SmallState.from_components(
        pawns=((3, 2), (4, 4)),
        walls_remaining=(0, 0),
        current_player=0,
    )

    first = proof_number_search(game, state, ProofNumberConfig(max_nodes=64), solved_cache=cache)
    second = proof_number_search(game, state, ProofNumberConfig(max_nodes=1), solved_cache=cache)

    assert first.exact
    assert cache.cache_size >= 1
    assert second.method == "proof-number"
    assert second.outcome is SolverOutcome.WIN
    assert second.value == 1
    assert second.exact
    assert second.best_action == first.best_action
    assert second.nodes == 0
    assert cache.hits == 1


def test_proof_number_solved_cache_round_trips_jsonl(tmp_path):
    game = SmallGame()
    cache = ProofNumberSolvedCache(game)
    state = SmallState.from_components(
        pawns=((3, 2), (4, 4)),
        walls_remaining=(0, 0),
        current_player=0,
    )

    proved = proof_number_search(game, state, ProofNumberConfig(max_nodes=64), solved_cache=cache)
    path = tmp_path / "proof_cache.jsonl"
    cache.save_jsonl(path)

    fresh = ProofNumberSolvedCache(game)
    fresh.load_jsonl(path)
    reused = proof_number_search(game, state, ProofNumberConfig(max_nodes=1), solved_cache=fresh)
    payload = json.loads(path.read_text().splitlines()[0])

    assert proved.exact
    assert fresh.cache_size == cache.cache_size
    assert reused.outcome is SolverOutcome.WIN
    assert reused.exact
    assert reused.nodes == 0
    assert payload["board_size"] == 5
    assert payload["state_key"] == game.state_key(state).hex()
    assert payload["value"] == 1


def test_proof_number_search_proves_forced_loss():
    game = SmallGame()
    state = SmallState.from_components(
        pawns=((0, 0), (1, 2)),
        walls_remaining=(0, 0),
        current_player=0,
    )

    label = proof_number_search(game, state, ProofNumberConfig(max_nodes=128))

    assert label.outcome is SolverOutcome.LOSS
    assert label.value == -1
    assert label.exact
    assert label.proof_number > 0
    assert label.disproof_number == 0


def test_proof_number_search_reports_exhaustion_without_guessing():
    game = SmallGame()
    label = proof_number_search(game, game.initial_state(), ProofNumberConfig(max_nodes=1))

    assert label.outcome is SolverOutcome.UNKNOWN
    assert label.value is None
    assert not label.exact
    assert label.exhausted
    assert label.nodes == 1


def test_oracle_corpus_can_use_proof_number_method(tmp_path):
    config_path = Path("configs/m2_5x5.json")
    assert load_config(config_path)["board"]["size"] == 5
    output = tmp_path / "pn_corpus.jsonl"

    summary = generate_oracle_corpus(
        output,
        config_path=config_path,
        positions=4,
        seed=7,
        random_plies=2,
        proof=ProofSearchConfig(max_depth=3, max_nodes=128),
        method="proof-number",
        proof_number=ProofNumberConfig(max_nodes=128),
    )

    assert summary.records == 4
    labels = load_oracle_corpus(output)
    assert len(labels) == 4
    assert {label.method for label in labels} == {"proof-number"}
    assert all(label.proof_number is not None for label in labels)
    assert all(label.disproof_number is not None for label in labels)


def test_oracle_corpus_can_use_hybrid_method(tmp_path):
    config_path = Path("configs/m2_5x5.json")
    output = tmp_path / "hybrid_corpus.jsonl"

    summary = generate_oracle_corpus(
        output,
        config_path=config_path,
        positions=4,
        seed=11,
        random_plies=0,
        proof=ProofSearchConfig(max_depth=3, max_nodes=64),
        method="hybrid",
        proof_number=ProofNumberConfig(max_nodes=64),
    )

    assert summary.method == "hybrid"
    labels = load_oracle_corpus(output)
    assert len(labels) == 4
    assert all(label.method in {"proof-number", "no-wall-tablebase"} for label in labels)
