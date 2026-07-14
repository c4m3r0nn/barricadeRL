import numpy as np

from barricade_rl.az_gating import (
    GatingConfig,
    NetworkMCTSPolicy,
    archive_promoted_checkpoint,
    gate_candidate,
    sample_gating_start_states,
)
from barricade_rl.az_network import AlphaZeroNetwork
from barricade_rl.config import load_config
from barricade_rl.evaluate import MatchResult
from barricade_rl.small_board import SmallGame


class NamedStub:
    def __init__(self, name):
        self.name = name


def _match(candidate, opponent, *, games_per_color, seed, game, wins, initial_states=None):
    del seed, game, initial_states
    total = 2 * games_per_color
    return MatchResult(
        candidate=candidate.name,
        opponent=opponent.name,
        candidate_wins=wins,
        opponent_wins=total - wins,
        draws=0,
        games_per_color=games_per_color,
        avg_plies=10.0,
        records=(),
    )


def test_gating_config_matches_handover():
    config = GatingConfig.from_project_config(load_config("configs/m2_5x5.json"))

    assert config.games == 200
    assert config.games_per_color == 100
    assert config.promotion_threshold == 0.55
    assert config.evaluation_simulations == 800
    assert config.start_min_plies == 1
    assert config.start_max_plies == 16


def test_gate_promotes_at_exact_threshold_with_balanced_colours():
    config = GatingConfig(games=200, promotion_threshold=0.55, evaluation_simulations=8)
    calls = []

    def runner(candidate, opponent, **kwargs):
        calls.append(kwargs)
        return _match(candidate, opponent, wins=110, **kwargs)

    result = gate_candidate(
        NamedStub("candidate"),
        NamedStub("incumbent"),
        game=SmallGame(),
        config=config,
        seed=9,
        match_runner=runner,
    )

    assert result.promoted
    assert result.score_rate == 0.55
    assert result.candidate_wins == 110
    assert calls[0]["games_per_color"] == 100
    assert calls[0]["seed"] == 9


def test_diverse_gating_starts_are_deterministic_unique_and_nonterminal():
    game = SmallGame()

    first = sample_gating_start_states(game, count=100, seed=17)
    second = sample_gating_start_states(game, count=100, seed=17)

    keys = tuple(game.state_key(state) for state in first)
    assert keys == tuple(game.state_key(state) for state in second)
    assert len(set(keys)) == 100
    assert all(state.ply > 0 for state in first)
    assert all(game.legal_actions(state).any() for state in first)


def test_gate_passes_paired_starts_and_records_their_keys():
    game = SmallGame()
    starts = sample_gating_start_states(game, count=2, seed=3)
    calls = []

    def runner(candidate, opponent, **kwargs):
        calls.append(kwargs)
        return _match(candidate, opponent, wins=2, **kwargs)

    result = gate_candidate(
        NamedStub("candidate"),
        NamedStub("incumbent"),
        game=game,
        config=GatingConfig(games=4, promotion_threshold=0.5, evaluation_simulations=8),
        seed=9,
        initial_states=starts,
        start_seed=3,
        match_runner=runner,
    )

    assert calls[0]["initial_states"] == starts
    assert result.start_positions == 2
    assert result.start_state_keys == tuple(game.state_key(state).hex() for state in starts)
    assert result.start_sampling == "paired-random-legal-prefixes-v1"
    assert result.start_ply_range == (1, 16)
    assert result.start_seed == 3


def test_gate_rejects_candidate_below_threshold():
    config = GatingConfig(games=200, promotion_threshold=0.55, evaluation_simulations=8)

    result = gate_candidate(
        NamedStub("candidate"),
        NamedStub("incumbent"),
        game=SmallGame(),
        config=config,
        seed=1,
        match_runner=lambda candidate, opponent, **kwargs: _match(
            candidate, opponent, wins=109, **kwargs
        ),
    )

    assert not result.promoted
    assert result.score_rate == 0.545


def test_network_gate_policy_uses_deterministic_evaluation_search():
    project_config = load_config("configs/m2_5x5.json")
    project_config["network"]["blocks"] = 1
    project_config["network"]["filters"] = 4
    project_config["network"]["global_pool_blocks"] = []
    network = AlphaZeroNetwork.from_config(project_config, seed=2)
    policy = NetworkMCTSPolicy(network, simulations=8, cpuct=1.6, name="candidate")
    game = SmallGame()
    state = game.initial_state()

    assert policy.search_config.temperature == 0.0
    assert policy.search_config.root_noise_fraction == 0.0
    assert not policy.search_config.forced_playouts
    action = policy.select_action(game, state, np.random.default_rng(4))
    assert game.legal_actions(state)[action]


def test_promoted_checkpoint_is_archived_with_manifest(tmp_path):
    candidate = tmp_path / "candidate.npz"
    candidate.write_bytes(b"checkpoint")
    result = gate_candidate(
        NamedStub("candidate"),
        NamedStub("incumbent"),
        game=SmallGame(),
        config=GatingConfig(games=2, promotion_threshold=0.5, evaluation_simulations=1),
        seed=0,
        match_runner=lambda candidate, opponent, **kwargs: _match(
            candidate, opponent, wins=2, **kwargs
        ),
    )

    archived = archive_promoted_checkpoint(
        candidate,
        tmp_path / "gated",
        result,
        run_id="run-1",
        git_commit="deadbeef",
        config_hash="hash",
        step=12,
    )

    assert archived.read_bytes() == b"checkpoint"
    assert archived.name == "gated-step-000000012.npz"
    manifest = (tmp_path / "gated" / "manifest.jsonl").read_text()
    assert '"run_id": "run-1"' in manifest
    assert '"promoted": true' in manifest
