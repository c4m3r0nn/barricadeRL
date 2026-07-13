from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .az_replay import AlphaZeroReplayBuffer
from .config import config_hash, load_config, small_game_from_config
from .oracle5x5 import audit_oracle_corpus


@dataclass(frozen=True, slots=True)
class TrainingReadinessSummary:
    config_path: str
    config_hash: str
    oracle_corpus: str | None
    oracle_ready: bool
    oracle_audit: dict | None
    replay_ready: bool
    mcts_ready: bool
    network_ready: bool
    self_play_ready: bool
    learner_ready: bool
    gating_ready: bool
    pipeline_ready: bool
    ready: bool
    blockers: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def check_training_readiness(
    config_path: str | Path,
    *,
    oracle_corpus: str | Path | None = None,
    min_records: int | None = None,
    min_exact_fraction: float = 1.0,
    min_phase_records: int = 1,
) -> TrainingReadinessSummary:
    config = load_config(config_path)
    game = small_game_from_config(config)
    cfg_hash = config_hash(config)
    acceptance = config.get("acceptance", {})
    required_records = int(min_records or acceptance.get("solver_labelled_positions", 5000))
    blockers: list[str] = []

    oracle_ready = False
    oracle_audit: dict | None = None
    if oracle_corpus is None:
        blockers.append("oracle corpus is required before proper training")
    else:
        audit = audit_oracle_corpus(
            oracle_corpus,
            config_path=config_path,
            min_records=required_records,
            min_exact_fraction=min_exact_fraction,
            min_phase_records=min_phase_records,
        )
        oracle_audit = audit.to_dict()
        oracle_ready = audit.passed
        if not audit.passed:
            blockers.append("oracle corpus audit failed")
            blockers.extend(audit.failures)

    replay_ready = _replay_contract_ready(game)
    if not replay_ready:
        blockers.append("AlphaZero replay contract is not ready")

    mcts_ready = importlib.util.find_spec("barricade_rl.mcts") is not None
    if not mcts_ready:
        blockers.append("MCTS implementation is missing")

    network_ready = importlib.util.find_spec("barricade_rl.az_network") is not None
    if not network_ready:
        blockers.append("network implementation is missing")

    self_play_ready = _self_play_contract_ready(config)
    if not self_play_ready:
        blockers.append("self-play actor implementation is missing")

    learner_ready = _learner_contract_ready(config)
    if not learner_ready:
        blockers.append("learner implementation is missing")

    gating_ready = _gating_contract_ready(config)
    if not gating_ready:
        blockers.append("checkpoint gating implementation is missing or invalid")

    pipeline_ready = importlib.util.find_spec("barricade_rl.az_pipeline") is not None
    if not pipeline_ready:
        blockers.append("gated training coordinator is missing")

    ready = not blockers
    return TrainingReadinessSummary(
        config_path=str(config_path),
        config_hash=cfg_hash,
        oracle_corpus=None if oracle_corpus is None else str(oracle_corpus),
        oracle_ready=oracle_ready,
        oracle_audit=oracle_audit,
        replay_ready=replay_ready,
        mcts_ready=mcts_ready,
        network_ready=network_ready,
        self_play_ready=self_play_ready,
        learner_ready=learner_ready,
        gating_ready=gating_ready,
        pipeline_ready=pipeline_ready,
        ready=ready,
        blockers=tuple(blockers),
    )


def _replay_contract_ready(game) -> bool:
    try:
        state = game.initial_state()
        observation_shape = tuple(game.canonical_observation(state).shape)
        AlphaZeroReplayBuffer(
            capacity=1,
            observation_shape=observation_shape,
            action_count=len(game.legal_actions(state)),
        )
    except Exception:
        return False
    return True


def _self_play_contract_ready(config) -> bool:
    try:
        from .az_self_play import SelfPlayConfig

        SelfPlayConfig.from_project_config(config)
    except Exception:
        return False
    return True


def _learner_contract_ready(config) -> bool:
    try:
        from .az_learner import LearnerConfig

        LearnerConfig.from_project_config(config)
    except Exception:
        return False
    return True


def _gating_contract_ready(config) -> bool:
    try:
        from .az_gating import GatingConfig

        GatingConfig.from_project_config(config)
    except Exception:
        return False
    return True


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether M2 proper AlphaZero training is unblocked.")
    parser.add_argument("--config", type=Path, default=Path("configs/m2_5x5.json"))
    parser.add_argument("--oracle-corpus", type=Path, default=None)
    parser.add_argument("--min-records", type=int, default=None)
    parser.add_argument("--min-exact-fraction", type=float, default=1.0)
    parser.add_argument("--min-phase-records", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = check_training_readiness(
        args.config,
        oracle_corpus=args.oracle_corpus,
        min_records=args.min_records,
        min_exact_fraction=args.min_exact_fraction,
        min_phase_records=args.min_phase_records,
    )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0 if summary.ready else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
