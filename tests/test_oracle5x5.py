import json
from pathlib import Path

from barricade_rl.config import config_hash, load_config
from barricade_rl.oracle5x5 import (
    LowWallEndgameConfig,
    LowWallEndgameSolver,
    NoWallTablebase,
    ProofNumberConfig,
    ProofNumberSolvedCache,
    ProofSearchConfig,
    audit_oracle_corpus,
    compact_exact_oracle_corpora,
    generate_oracle_corpus,
    hybrid_oracle_label,
    load_oracle_corpus,
    main as oracle_main,
    merge_oracle_corpora,
    proof_number_search,
    prove_state,
    solve_no_wall_endgame,
)
from barricade_rl.small_board import SmallGame, SmallState, SolverOutcome


def test_budgeted_proof_labels_terminal_and_immediate_win_exactly():
    game = SmallGame()
    terminal = SmallState.from_components(
        pawns=((4, 2), (0, 0)),
        current_player=1,
        ply=7,
    )
    terminal_label = prove_state(game, terminal, ProofSearchConfig(max_depth=0, max_nodes=8))
    assert terminal_label.outcome is SolverOutcome.LOSS
    assert terminal_label.value == -1
    assert terminal_label.exact
    assert terminal_label.best_action is None

    winning = SmallState.from_components(
        pawns=((3, 2), (4, 4)),
        walls_remaining=(0, 0),
        current_player=0,
    )
    win_label = prove_state(game, winning, ProofSearchConfig(max_depth=1, max_nodes=64))
    assert win_label.outcome is SolverOutcome.WIN
    assert win_label.value == 1
    assert win_label.exact
    assert win_label.best_action == 0


def test_budgeted_proof_reports_unknown_when_node_budget_is_exhausted():
    game = SmallGame()
    label = prove_state(game, game.initial_state(), ProofSearchConfig(max_depth=8, max_nodes=1))

    assert label.outcome is SolverOutcome.UNKNOWN
    assert label.value is None
    assert not label.exact
    assert label.nodes == 1
    assert label.exhausted


def test_oracle_corpus_generation_round_trips_jsonl(tmp_path):
    config_path = Path("configs/m2_5x5.json")
    config = load_config(config_path)
    output = tmp_path / "corpus.jsonl"

    summary = generate_oracle_corpus(
        output,
        config_path=config_path,
        positions=8,
        seed=123,
        random_plies=4,
        proof=ProofSearchConfig(max_depth=4, max_nodes=512),
    )

    assert summary.records == 8
    assert summary.config_hash == config_hash(config)
    assert output.exists()

    labels = load_oracle_corpus(output)
    assert len(labels) == 8
    assert all(label.board_size == 5 for label in labels)
    assert all(label.walls_per_player == 3 for label in labels)
    assert all(label.legal_action_count > 0 for label in labels)

    first_payload = json.loads(output.read_text().splitlines()[0])
    state = SmallState.from_key(SmallGame().spec, bytes.fromhex(first_payload["state_key"]))
    replayed = prove_state(SmallGame(), state, ProofSearchConfig(max_depth=4, max_nodes=512))
    assert replayed.outcome.value == first_payload["outcome"]


def test_oracle_corpus_shards_reconstruct_unsharded_sample_sequence(tmp_path):
    config_path = Path("configs/m2_5x5.json")
    proof = ProofSearchConfig(max_depth=1, max_nodes=64)
    full = tmp_path / "full.jsonl"
    shard0 = tmp_path / "shard0.jsonl"
    shard1 = tmp_path / "shard1.jsonl"

    full_summary = generate_oracle_corpus(
        full,
        config_path=config_path,
        positions=6,
        seed=222,
        random_plies=2,
        proof=proof,
    )
    shard0_summary = generate_oracle_corpus(
        shard0,
        config_path=config_path,
        positions=6,
        seed=222,
        random_plies=2,
        proof=proof,
        shard_index=0,
        shard_count=2,
    )
    shard1_summary = generate_oracle_corpus(
        shard1,
        config_path=config_path,
        positions=6,
        seed=222,
        random_plies=2,
        proof=proof,
        shard_index=1,
        shard_count=2,
    )

    full_payloads = [json.loads(line) for line in full.read_text().splitlines()]
    shard_payloads = [
        json.loads(line)
        for path in (shard0, shard1)
        for line in path.read_text().splitlines()
    ]
    combined = sorted(shard_payloads, key=lambda payload: payload["record_index"])

    assert full_summary.records == 6
    assert shard0_summary.records == 3
    assert shard1_summary.records == 3
    assert shard0_summary.shard_index == 0
    assert shard0_summary.shard_count == 2
    assert [payload["record_index"] for payload in shard_payloads] == [0, 2, 4, 1, 3, 5]
    assert [
        (payload["record_index"], payload["state_key"])
        for payload in combined
    ] == [
        (payload["record_index"], payload["state_key"])
        for payload in full_payloads
    ]


def test_merge_oracle_corpora_reconstructs_sharded_output(tmp_path):
    config_path = Path("configs/m2_5x5.json")
    proof = ProofSearchConfig(max_depth=1, max_nodes=64)
    full = tmp_path / "full.jsonl"
    shard0 = tmp_path / "shard0.jsonl"
    shard1 = tmp_path / "shard1.jsonl"
    merged = tmp_path / "merged.jsonl"

    generate_oracle_corpus(
        full,
        config_path=config_path,
        positions=6,
        seed=333,
        random_plies=2,
        proof=proof,
    )
    generate_oracle_corpus(
        shard0,
        config_path=config_path,
        positions=6,
        seed=333,
        random_plies=2,
        proof=proof,
        shard_index=0,
        shard_count=2,
    )
    generate_oracle_corpus(
        shard1,
        config_path=config_path,
        positions=6,
        seed=333,
        random_plies=2,
        proof=proof,
        shard_index=1,
        shard_count=2,
    )

    summary = merge_oracle_corpora([shard1, shard0], merged)
    full_payloads = [json.loads(line) for line in full.read_text().splitlines()]
    merged_payloads = [json.loads(line) for line in merged.read_text().splitlines()]

    assert summary.records == 6
    assert summary.duplicate_records == 0
    assert summary.record_index_min == 0
    assert summary.record_index_max == 5
    assert [payload["record_index"] for payload in merged_payloads] == list(range(6))
    assert [
        (payload["record_index"], payload["state_key"])
        for payload in merged_payloads
    ] == [
        (payload["record_index"], payload["state_key"])
        for payload in full_payloads
    ]


def test_merge_oracle_corpora_rejects_duplicate_record_indices(tmp_path):
    config_path = Path("configs/m2_5x5.json")
    shard = tmp_path / "shard.jsonl"
    merged = tmp_path / "merged.jsonl"

    generate_oracle_corpus(
        shard,
        config_path=config_path,
        positions=2,
        seed=444,
        random_plies=1,
        proof=ProofSearchConfig(max_depth=1, max_nodes=64),
    )

    try:
        merge_oracle_corpora([shard, shard], merged)
    except ValueError as exc:
        assert "duplicate record_index" in str(exc)
    else:
        raise AssertionError("duplicate record indices must be rejected")


def test_audit_oracle_corpus_accepts_exact_multi_phase_corpus(tmp_path):
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
    corpus = tmp_path / "audit_pass.jsonl"
    lines = []
    for index, state in enumerate(states):
        payload = solve_no_wall_endgame(game, state).to_dict()
        payload["record_index"] = index
        payload["config_hash"] = config_hash(config)
        lines.append(json.dumps(payload, sort_keys=True))
    corpus.write_text("\n".join(lines) + "\n")

    summary = audit_oracle_corpus(
        corpus,
        config_path=config_path,
        min_records=3,
        min_exact_fraction=1.0,
        min_phase_records=1,
    )

    assert summary.passed
    assert summary.records == 3
    assert summary.exact_records == 3
    assert summary.duplicate_state_keys == 0
    assert summary.config_hash_mismatches == 0
    assert summary.methods == {"no-wall-tablebase": 3}
    assert summary.exact_phase_buckets == {"opening": 1, "midgame": 1, "endgame": 1}
    assert summary.failures == ()


def test_exact_compactor_combines_independent_phase_runs_and_reindexes(tmp_path):
    config_path = Path("configs/m2_5x5.json")
    config = load_config(config_path)
    game = SmallGame()
    states = (
        SmallState.from_components(walls_remaining=(0, 0), current_player=0, ply=8),
        SmallState.from_components(
            pawns=((1, 2), (3, 2)), walls_remaining=(0, 0), current_player=0, ply=100
        ),
        SmallState.from_components(
            pawns=((2, 1), (3, 3)), walls_remaining=(0, 0), current_player=1, ply=160
        ),
    )
    inputs = []
    for phase, state in zip(("opening", "midgame", "endgame"), states):
        payload = solve_no_wall_endgame(game, state).to_dict()
        payload["record_index"] = 0  # Independent runs are allowed to collide here.
        payload["config_hash"] = config_hash(config)
        path = tmp_path / f"{phase}.jsonl"
        path.write_text(json.dumps(payload, sort_keys=True) + "\n")
        inputs.append(path)
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(inputs[0].read_text())
    inputs.append(duplicate)
    terminal_state = SmallState.from_components(
        pawns=((4, 2), (3, 3)), walls_remaining=(0, 0), current_player=1, ply=160
    )
    terminal_payload = solve_no_wall_endgame(game, terminal_state).to_dict()
    terminal_payload.update(record_index=0, config_hash=config_hash(config))
    terminal = tmp_path / "terminal.jsonl"
    terminal.write_text(json.dumps(terminal_payload) + "\n")
    inputs.append(terminal)
    output = tmp_path / "compacted.jsonl"

    summary = compact_exact_oracle_corpora(
        inputs,
        output,
        config_path=config_path,
        records=3,
    )
    payloads = [json.loads(line) for line in output.read_text().splitlines()]

    assert summary.records == 3
    assert summary.duplicate_states_skipped == 1
    assert summary.terminal_records_skipped == 1
    assert summary.selected_phase_buckets == {"opening": 1, "midgame": 1, "endgame": 1}
    assert [payload["record_index"] for payload in payloads] == [0, 1, 2]
    assert all(payload["exact"] for payload in payloads)
    assert all("source_record_index" in payload for payload in payloads)


def test_exact_compactor_fails_when_a_phase_quota_is_not_available(tmp_path):
    config_path = Path("configs/m2_5x5.json")
    config = load_config(config_path)
    game = SmallGame()
    state = SmallState.from_components(walls_remaining=(0, 0), current_player=0, ply=8)
    payload = solve_no_wall_endgame(game, state).to_dict()
    payload["record_index"] = 0
    payload["config_hash"] = config_hash(config)
    source = tmp_path / "opening_only.jsonl"
    source.write_text(json.dumps(payload) + "\n")

    try:
        compact_exact_oracle_corpora(
            [source],
            tmp_path / "output.jsonl",
            config_path=config_path,
            records=3,
        )
    except ValueError as exc:
        assert "midgame" in str(exc)
    else:
        raise AssertionError("compaction must enforce every phase quota")


def test_cli_compacts_independent_exact_corpora(tmp_path, capsys):
    config_path = Path("configs/m2_5x5.json")
    config = load_config(config_path)
    game = SmallGame()
    states = (
        SmallState.from_components(walls_remaining=(0, 0), current_player=0, ply=8),
        SmallState.from_components(
            pawns=((1, 2), (3, 2)), walls_remaining=(0, 0), current_player=0, ply=100
        ),
        SmallState.from_components(
            pawns=((2, 1), (3, 3)), walls_remaining=(0, 0), current_player=1, ply=160
        ),
    )
    inputs = []
    for index, state in enumerate(states):
        payload = solve_no_wall_endgame(game, state).to_dict()
        payload.update(record_index=0, config_hash=config_hash(config))
        source = tmp_path / f"phase_{index}.jsonl"
        source.write_text(json.dumps(payload) + "\n")
        inputs.append(source)
    output = tmp_path / "exact.jsonl"

    exit_code = oracle_main(
        [
            "--config",
            str(config_path),
            "--output",
            str(output),
            "--compact-exact-from",
            *(str(path) for path in inputs),
            "--compact-records",
            "3",
        ]
    )

    assert exit_code == 0
    assert len(output.read_text().splitlines()) == 3
    assert json.loads(capsys.readouterr().out)["records"] == 3


def test_audit_oracle_corpus_rejects_duplicate_states_and_low_exact_coverage(tmp_path):
    config_path = Path("configs/m2_5x5.json")
    config = load_config(config_path)
    game = SmallGame()
    state = SmallState.from_components(walls_remaining=(0, 0), current_player=0, ply=0)
    exact_payload = solve_no_wall_endgame(game, state).to_dict()
    exact_payload["record_index"] = 0
    exact_payload["config_hash"] = config_hash(config)
    unknown_payload = dict(exact_payload)
    unknown_payload["record_index"] = 1
    unknown_payload["method"] = "proof-number"
    unknown_payload["outcome"] = "unknown"
    unknown_payload["value"] = None
    unknown_payload["exact"] = False
    unknown_payload["exhausted"] = True
    corpus = tmp_path / "audit_fail.jsonl"
    corpus.write_text(
        json.dumps(exact_payload, sort_keys=True) + "\n" + json.dumps(unknown_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = audit_oracle_corpus(
        corpus,
        config_path=config_path,
        min_records=2,
        min_exact_fraction=1.0,
        min_phase_records=0,
    )

    assert not summary.passed
    assert summary.records == 2
    assert summary.exact_records == 1
    assert summary.duplicate_state_keys == 1
    assert summary.exhausted_records == 1
    assert any("exact fraction" in failure for failure in summary.failures)
    assert any("duplicate state_key" in failure for failure in summary.failures)


def test_audit_oracle_corpus_rejects_terminal_validation_positions(tmp_path):
    config_path = Path("configs/m2_5x5.json")
    config = load_config(config_path)
    game = SmallGame()
    terminal = SmallState.from_components(
        pawns=((4, 2), (3, 3)), walls_remaining=(0, 0), current_player=1, ply=64
    )
    payload = solve_no_wall_endgame(game, terminal).to_dict()
    payload.update(record_index=0, config_hash=config_hash(config))
    corpus = tmp_path / "terminal.jsonl"
    corpus.write_text(json.dumps(payload) + "\n")

    summary = audit_oracle_corpus(
        corpus,
        config_path=config_path,
        min_records=1,
        min_exact_fraction=1.0,
        min_phase_records=0,
    )

    assert not summary.passed
    assert summary.terminal_records == 1
    assert any("terminal record" in failure for failure in summary.failures)


def test_oracle_corpus_preserves_persistent_proof_cache(tmp_path):
    config_path = Path("configs/m2_5x5.json")
    game = SmallGame()
    state = SmallState.from_components(
        pawns=((3, 2), (4, 4)),
        walls_remaining=(1, 0),
        current_player=0,
    )
    cache_in = tmp_path / "cache_in.jsonl"
    cache_out = tmp_path / "cache_out.jsonl"
    output = tmp_path / "corpus.jsonl"

    cache = ProofNumberSolvedCache(game)
    proved = proof_number_search(game, state, ProofNumberConfig(max_nodes=64), solved_cache=cache)
    cache.save_jsonl(cache_in)
    summary = generate_oracle_corpus(
        output,
        config_path=config_path,
        positions=1,
        seed=99,
        random_plies=0,
        method="proof-number",
        proof_number=ProofNumberConfig(max_nodes=1),
        proof_cache_in=cache_in,
        proof_cache_out=cache_out,
    )

    fresh = ProofNumberSolvedCache(game)
    fresh.load_jsonl(cache_out)
    reused = proof_number_search(game, state, ProofNumberConfig(max_nodes=1), solved_cache=fresh)

    assert proved.exact
    assert summary.proof_cache_path == str(cache_out)
    assert fresh.cache_size >= cache.cache_size
    assert reused.exact
    assert reused.nodes == 0


def test_oracle_corpus_can_sample_no_wall_endgames(tmp_path):
    config_path = Path("configs/m2_5x5.json")
    output = tmp_path / "no_wall.jsonl"

    summary = generate_oracle_corpus(
        output,
        config_path=config_path,
        positions=8,
        seed=321,
        random_plies=6,
        proof=ProofSearchConfig(max_depth=4, max_nodes=512),
        method="hybrid",
        sampling="no-wall",
    )

    labels = load_oracle_corpus(output)
    assert summary.records == 8
    assert summary.exact_records == 8
    assert {label.method for label in labels} == {"no-wall-tablebase"}


def test_oracle_corpus_can_sample_low_wall_endgames(tmp_path):
    config_path = Path("configs/m2_5x5.json")
    output = tmp_path / "low_wall.jsonl"

    low_wall = LowWallEndgameConfig(max_walls_remaining=1, max_nodes=1)
    summary = generate_oracle_corpus(
        output,
        config_path=config_path,
        positions=4,
        seed=654,
        random_plies=0,
        proof=ProofSearchConfig(max_depth=1, max_nodes=1),
        method="hybrid",
        proof_number=ProofNumberConfig(max_nodes=1),
        low_wall=low_wall,
        sampling="low-wall",
    )

    labels = load_oracle_corpus(output)
    assert summary.records == 4
    assert summary.low_wall == low_wall
    assert {label.method for label in labels} == {"low-wall-endgame"}
    for label in labels:
        state = SmallState.from_key(SmallGame().spec, bytes.fromhex(label.state_key))
        assert sum(state.walls_remaining) <= low_wall.max_walls_remaining


def test_no_wall_endgame_tablebase_solves_initial_race_exactly():
    game = SmallGame()
    state = SmallState.from_components(walls_remaining=(0, 0), current_player=0)

    label = solve_no_wall_endgame(game, state)

    assert label.method == "no-wall-tablebase"
    assert label.outcome is SolverOutcome.LOSS
    assert label.value == -1
    assert label.exact
    assert label.best_action in {0, 2, 3}


def test_no_wall_tablebase_reuses_cache_across_queries():
    game = SmallGame()
    tablebase = NoWallTablebase(game)
    first = SmallState.from_components(walls_remaining=(0, 0), current_player=0)
    second = SmallState.from_components(
        pawns=((3, 2), (4, 4)),
        walls_remaining=(0, 0),
        current_player=0,
    )

    first_label = tablebase.solve(first)
    cache_after_first = tablebase.cache_size
    second_label = tablebase.solve(second)
    cache_after_second = tablebase.cache_size
    repeated = tablebase.solve(first)

    assert first_label.outcome is SolverOutcome.LOSS
    assert second_label.outcome is SolverOutcome.WIN
    assert cache_after_first > 0
    assert cache_after_second >= cache_after_first
    assert tablebase.cache_size == cache_after_second
    assert repeated.outcome is first_label.outcome


def test_no_wall_endgame_tablebase_solves_immediate_win():
    game = SmallGame()
    state = SmallState.from_components(
        pawns=((3, 2), (4, 4)),
        walls_remaining=(0, 0),
        current_player=0,
    )

    label = solve_no_wall_endgame(game, state)

    assert label.outcome is SolverOutcome.WIN
    assert label.value == 1
    assert label.best_action == 0


def test_low_wall_endgame_solver_proves_immediate_one_wall_win():
    game = SmallGame()
    solver = LowWallEndgameSolver(game, LowWallEndgameConfig(max_walls_remaining=1, max_nodes=64))
    state = SmallState.from_components(
        pawns=((3, 2), (4, 4)),
        walls_remaining=(1, 0),
        current_player=0,
    )

    label = solver.solve(state)

    assert label.method == "low-wall-endgame"
    assert label.outcome is SolverOutcome.WIN
    assert label.value == 1
    assert label.exact
    assert not label.exhausted
    assert label.best_action == 0


def test_low_wall_endgame_solver_reports_budget_exhaustion_without_guessing():
    game = SmallGame()
    solver = LowWallEndgameSolver(game, LowWallEndgameConfig(max_walls_remaining=1, max_nodes=1))
    state = SmallState.from_components(walls_remaining=(1, 0), current_player=0)

    label = solver.solve(state)

    assert label.method == "low-wall-endgame"
    assert label.outcome is SolverOutcome.UNKNOWN
    assert label.value is None
    assert not label.exact
    assert label.exhausted
    assert label.nodes == 1


def test_hybrid_oracle_uses_no_wall_tablebase_before_search():
    game = SmallGame()
    state = SmallState.from_components(walls_remaining=(0, 0), current_player=0)

    label = hybrid_oracle_label(
        game,
        state,
        proof=ProofSearchConfig(max_depth=0, max_nodes=1),
        proof_number=ProofNumberConfig(max_nodes=1),
    )

    assert label.method == "no-wall-tablebase"
    assert label.outcome is SolverOutcome.LOSS
    assert label.exact
    assert not label.exhausted


def test_hybrid_oracle_uses_low_wall_solver_before_proof_number():
    game = SmallGame()
    solver = LowWallEndgameSolver(game, LowWallEndgameConfig(max_walls_remaining=1, max_nodes=64))
    state = SmallState.from_components(
        pawns=((3, 2), (4, 4)),
        walls_remaining=(1, 0),
        current_player=0,
    )

    label = hybrid_oracle_label(
        game,
        state,
        proof=ProofSearchConfig(max_depth=0, max_nodes=1),
        proof_number=ProofNumberConfig(max_nodes=1),
        low_wall_solver=solver,
    )

    assert label.method == "low-wall-endgame"
    assert label.outcome is SolverOutcome.WIN
    assert label.exact
    assert label.best_action == 0


def test_hybrid_oracle_uses_proof_number_for_wall_positions():
    game = SmallGame()
    state = SmallState.from_components(
        pawns=((3, 2), (4, 4)),
        walls_remaining=(1, 0),
        current_player=0,
    )

    label = hybrid_oracle_label(
        game,
        state,
        proof=ProofSearchConfig(max_depth=0, max_nodes=1),
        proof_number=ProofNumberConfig(max_nodes=64),
    )

    assert label.method == "proof-number"
    assert label.outcome is SolverOutcome.WIN
    assert label.best_action == 0


def test_hybrid_oracle_reuses_proof_number_cache_for_wall_positions():
    game = SmallGame()
    cache = ProofNumberSolvedCache(game)
    state = SmallState.from_components(
        pawns=((3, 2), (4, 4)),
        walls_remaining=(1, 0),
        current_player=0,
    )

    first = hybrid_oracle_label(
        game,
        state,
        proof=ProofSearchConfig(max_depth=0, max_nodes=1),
        proof_number=ProofNumberConfig(max_nodes=64),
        proof_cache=cache,
    )
    second = hybrid_oracle_label(
        game,
        state,
        proof=ProofSearchConfig(max_depth=0, max_nodes=1),
        proof_number=ProofNumberConfig(max_nodes=1),
        proof_cache=cache,
    )

    assert first.exact
    assert second.method == "proof-number"
    assert second.outcome is SolverOutcome.WIN
    assert second.exact
    assert second.nodes == 0
