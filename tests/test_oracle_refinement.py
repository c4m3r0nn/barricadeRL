from pathlib import Path

from barricade_rl.oracle5x5 import (
    ProofNumberConfig,
    ProofSearchConfig,
    generate_oracle_corpus,
    load_oracle_corpus,
    refine_oracle_corpus,
)


def test_refine_oracle_corpus_relabels_unknown_records_without_changing_count(tmp_path):
    config_path = Path("configs/m2_5x5.json")
    raw = tmp_path / "raw.jsonl"
    refined = tmp_path / "refined.jsonl"

    raw_summary = generate_oracle_corpus(
        raw,
        config_path=config_path,
        positions=8,
        seed=5,
        random_plies=3,
        method="proof-number",
        sampling="no-wall",
        proof=ProofSearchConfig(max_depth=1, max_nodes=1),
        proof_number=ProofNumberConfig(max_nodes=1),
    )
    assert raw_summary.records == 8
    assert raw_summary.exact_records == 0

    refined_summary = refine_oracle_corpus(
        raw,
        refined,
        config_path=config_path,
        method="hybrid",
        proof=ProofSearchConfig(max_depth=4, max_nodes=512),
        proof_number=ProofNumberConfig(max_nodes=512),
    )

    assert refined_summary.records == raw_summary.records
    assert refined_summary.exact_records == raw_summary.records
    assert refined_summary.refined_records == raw_summary.records

    labels = load_oracle_corpus(refined)
    assert len(labels) == raw_summary.records
    assert {label.method for label in labels} == {"no-wall-tablebase"}
    assert all(label.exact for label in labels)


def test_refine_oracle_corpus_keeps_existing_exact_records_by_default(tmp_path):
    config_path = Path("configs/m2_5x5.json")
    raw = tmp_path / "raw_exact.jsonl"
    refined = tmp_path / "refined_exact.jsonl"

    raw_summary = generate_oracle_corpus(
        raw,
        config_path=config_path,
        positions=4,
        seed=6,
        random_plies=3,
        method="hybrid",
        sampling="no-wall",
        proof=ProofSearchConfig(max_depth=4, max_nodes=512),
        proof_number=ProofNumberConfig(max_nodes=512),
    )
    assert raw_summary.exact_records == 4

    refined_summary = refine_oracle_corpus(
        raw,
        refined,
        config_path=config_path,
        method="proof-number",
        proof=ProofSearchConfig(max_depth=1, max_nodes=1),
        proof_number=ProofNumberConfig(max_nodes=1),
    )

    assert refined_summary.records == 4
    assert refined_summary.refined_records == 0
    assert refined.read_text() == raw.read_text()
