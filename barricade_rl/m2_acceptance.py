from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .az_network import AlphaZeroNetwork
from .az_parallel import PARALLEL_PROTOCOL, _single_threaded_child_environment
from .config import config_hash as calculate_config_hash
from .config import load_config, small_game_from_config
from .mcts import MCTS, MCTSConfig
from .oracle5x5 import NoWallTablebase, OracleLabel
from .small_board import SmallState


ACCEPTANCE_SCHEMA_VERSION = 1

_MCTS_GAME = None
_MCTS_NETWORK = None
_MCTS_CONFIG = None

_OPTIMAL_GAME = None
_OPTIMAL_TABLEBASE = None


def complete_optimal_actions(
    game,
    state: SmallState,
    label: OracleLabel,
    tablebase: NoWallTablebase,
) -> tuple[tuple[int, ...], bool]:
    """Return every solver-optimal action when the exact backend supports it."""
    if not label.exact or label.value not in (-1, 0, 1):
        return _recorded_action(label), False
    if state.walls_remaining != (0, 0):
        return _recorded_action(label), False
    optimal = []
    for action in np.flatnonzero(game.legal_actions(state)):
        child = game.next_state(state, int(action))
        child_label = tablebase.solve(child)
        if child_label.value is not None and -int(child_label.value) == int(label.value):
            optimal.append(int(action))
    if not optimal:
        return _recorded_action(label), False
    return tuple(optimal), True


def compute_value_metrics(
    *,
    predictions: np.ndarray,
    targets: np.ndarray,
    phases: Sequence[str],
) -> dict:
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.int8).reshape(-1)
    if predictions.shape != targets.shape or len(phases) != predictions.size:
        raise ValueError("predictions, targets, and phases must have equal lengths")
    if predictions.size < 1 or not np.isfinite(predictions).all():
        raise ValueError("value predictions must be non-empty and finite")
    if not np.isin(targets, (-1, 1)).all():
        raise ValueError("acceptance value targets must be binary -1/+1 labels")
    predicted_positive = predictions > 0.0
    target_positive = targets > 0
    result = {
        "positions": int(predictions.size),
        "sign_accuracy": float(np.mean(predicted_positive == target_positive)),
        "mean_squared_error": float(np.mean((predictions - targets) ** 2)),
        "mean_prediction": float(np.mean(predictions)),
        "mean_absolute_prediction": float(np.mean(np.abs(predictions))),
        "predicted_positive_fraction": float(np.mean(predicted_positive)),
        "target_positive_fraction": float(np.mean(target_positive)),
        "by_phase": {},
        "calibration_bins": [],
    }
    phase_array = np.asarray(tuple(phases), dtype=object)
    for phase in ("opening", "midgame", "endgame"):
        mask = phase_array == phase
        if not mask.any():
            result["by_phase"][phase] = {"count": 0, "sign_accuracy": None}
            continue
        phase_predictions = predictions[mask]
        phase_targets = targets[mask]
        result["by_phase"][phase] = {
            "count": int(mask.sum()),
            "sign_accuracy": float(
                np.mean((phase_predictions > 0.0) == (phase_targets > 0))
            ),
            "mean_squared_error": float(
                np.mean((phase_predictions - phase_targets) ** 2)
            ),
            "mean_prediction": float(np.mean(phase_predictions)),
            "predicted_positive_fraction": float(np.mean(phase_predictions > 0.0)),
            "target_positive_fraction": float(np.mean(phase_targets > 0)),
        }
    edges = np.linspace(-1.0, 1.0, 11)
    indices = np.clip(np.digitize(predictions, edges[1:-1], right=False), 0, 9)
    for index in range(10):
        mask = indices == index
        if not mask.any():
            continue
        result["calibration_bins"].append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": int(mask.sum()),
                "mean_prediction": float(np.mean(predictions[mask])),
                "mean_target": float(np.mean(targets[mask])),
                "positive_fraction": float(np.mean(targets[mask] > 0)),
            }
        )
    return result


def evaluate_m2_checkpoint(
    *,
    config_path: str | Path,
    oracle_corpus: str | Path,
    checkpoint: str | Path,
    output_directory: str | Path,
    workers: int = 8,
    simulations: int | None = None,
    position_limit: int | None = None,
    value_only: bool = False,
    batch_size: int = 256,
    progress_every: int = 100,
) -> dict:
    if workers < 1 or batch_size < 1 or progress_every < 1:
        raise ValueError("workers, batch size, and progress interval must be positive")
    config_path = Path(config_path)
    corpus_path = Path(oracle_corpus)
    checkpoint_path = Path(checkpoint)
    for path in (config_path, corpus_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    project_config = load_config(config_path)
    game = small_game_from_config(project_config)
    cfg_hash = calculate_config_hash(project_config)
    simulations = int(
        project_config["mcts"]["evaluation_simulations"]
        if simulations is None
        else simulations
    )
    if simulations < 1:
        raise ValueError("MCTS simulations must be positive")
    labels, corpus_payloads = _load_validated_corpus(corpus_path, cfg_hash)
    if position_limit is not None and position_limit < 1:
        raise ValueError("position limit must be positive")
    mcts_target = min(len(labels), position_limit or len(labels))
    network = AlphaZeroNetwork.load_checkpoint(checkpoint_path)
    _validate_network(game, network, cfg_hash)

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": ACCEPTANCE_SCHEMA_VERSION,
        "config_hash": cfg_hash,
        "corpus_sha256": _sha256(corpus_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_step": int(network.metadata.get("step", 0)),
        "mcts_simulations": simulations,
        "positions": len(labels),
        "parallel_protocol": PARALLEL_PROTOCOL,
    }
    _ensure_manifest(output / "manifest.json", manifest)

    states = tuple(
        SmallState.from_key(game.spec, bytes.fromhex(label.state_key))
        for label in labels
    )
    phases = tuple(_phase_bucket(label.ply, game.spec.max_plies) for label in labels)
    optimal_sets, optimal_complete = _load_or_compute_optimal_actions(
        output / "optimal_actions.jsonl",
        project_config,
        labels,
        workers=workers,
        progress_every=progress_every,
    )
    predictions, raw_actions = _network_predictions(
        network,
        game,
        states,
        batch_size=batch_size,
    )
    targets = np.asarray([int(label.value) for label in labels], dtype=np.int8)
    value_metrics = compute_value_metrics(
        predictions=predictions,
        targets=targets,
        phases=phases,
    )
    complete_mask = np.asarray(optimal_complete, dtype=np.bool_)
    raw_matches = np.asarray(
        [action in actions for action, actions in zip(raw_actions, optimal_sets, strict=True)],
        dtype=np.bool_,
    )
    raw_policy = {
        "positions": len(labels),
        "complete_optimal_set_positions": int(complete_mask.sum()),
        "optimal_action_accuracy": (
            float(np.mean(raw_matches[complete_mask])) if complete_mask.any() else None
        ),
        "recorded_best_action_accuracy": float(
            np.mean(
                raw_actions
                == np.asarray([int(label.best_action) for label in labels], dtype=np.int64)
            )
        ),
    }

    positions_path = output / "positions.jsonl"
    existing = _load_position_records(positions_path, labels)
    selected_indices = _select_position_indices(phases, mcts_target)
    resumed = sum(index in existing for index in selected_indices)
    new_positions = 0
    if not value_only:
        missing_tasks = tuple(
            (
                index,
                labels[index].state_key,
                phases[index],
                optimal_sets[index],
                optimal_complete[index],
            )
            for index in selected_indices
            if index not in existing
        )
        if missing_tasks:
            context = multiprocessing.get_context("spawn")
            with _single_threaded_child_environment():
                with ProcessPoolExecutor(
                    max_workers=min(workers, len(missing_tasks)),
                    mp_context=context,
                    initializer=_initialize_mcts_worker,
                    initargs=(dict(project_config), str(checkpoint_path.resolve()), simulations),
                ) as executor:
                    with positions_path.open("a", encoding="utf-8") as handle:
                        for record in executor.map(
                            _evaluate_mcts_task,
                            missing_tasks,
                            chunksize=1,
                        ):
                            handle.write(json.dumps(record, sort_keys=True) + "\n")
                            handle.flush()
                            existing[int(record["record_index"])] = record
                            new_positions += 1
                            if new_positions % progress_every == 0:
                                print(
                                    f"evaluated {new_positions}/{len(missing_tasks)} new MCTS positions",
                                    file=sys.stderr,
                                    flush=True,
                                )

    selected_records = [existing[index] for index in selected_indices if index in existing]
    mcts_metrics = _mcts_metrics(
        selected_records,
        target=mcts_target,
        corpus_positions=len(labels),
        resumed=resumed,
        new_positions=new_positions,
        requested=not value_only,
    )
    required_positions = int(
        project_config.get("acceptance", {}).get("solver_labelled_positions", 5000)
    )
    value_threshold = float(
        project_config.get("acceptance", {}).get("value_sign_accuracy_min", 0.99)
    )
    mcts_threshold = float(
        project_config.get("acceptance", {}).get(
            "mcts_optimal_move_accuracy_min",
            0.99,
        )
    )
    criteria = _acceptance_criteria(
        value_metrics=value_metrics,
        mcts_metrics=mcts_metrics,
        complete_optimal_fraction=float(np.mean(complete_mask)),
        corpus_positions=len(labels),
        required_positions=required_positions,
        value_threshold=value_threshold,
        mcts_threshold=mcts_threshold,
    )
    summary = {
        "schema_version": ACCEPTANCE_SCHEMA_VERSION,
        "manifest": manifest,
        "corpus": {
            "path": str(corpus_path),
            "positions": len(labels),
            "exact_positions": sum(bool(payload["exact"]) for payload in corpus_payloads),
            "phase_counts": {
                phase: phases.count(phase)
                for phase in ("opening", "midgame", "endgame")
            },
        },
        "optimal_action_sets": {
            "complete_positions": int(complete_mask.sum()),
            "complete_fraction": float(np.mean(complete_mask)),
        },
        "value": value_metrics,
        "raw_policy": raw_policy,
        "mcts": mcts_metrics,
        "criteria": criteria,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _recorded_action(label: OracleLabel) -> tuple[int, ...]:
    return () if label.best_action is None else (int(label.best_action),)


def _load_validated_corpus(
    path: Path,
    expected_config_hash: str,
) -> tuple[tuple[OracleLabel, ...], tuple[dict, ...]]:
    payloads = tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not payloads:
        raise ValueError("oracle corpus is empty")
    labels = tuple(OracleLabel.from_dict(payload) for payload in payloads)
    for index, (payload, label) in enumerate(zip(payloads, labels, strict=True)):
        if int(payload.get("record_index", index)) != index:
            raise ValueError("oracle corpus record indices must be contiguous and ordered")
        if payload.get("config_hash") != expected_config_hash:
            raise ValueError("oracle corpus config hash does not match evaluation config")
        if not label.exact or label.exhausted or label.value not in (-1, 1):
            raise ValueError("M2 acceptance corpus must contain exact binary labels")
        if label.best_action is None:
            raise ValueError("M2 acceptance labels must include a recorded best action")
    return labels, payloads


def _validate_network(game, network: AlphaZeroNetwork, expected_config_hash: str) -> None:
    if network.config.board_size != game.board_size:
        raise ValueError("checkpoint board size does not match evaluation game")
    if network.config.action_count != game.action_count:
        raise ValueError("checkpoint action count does not match evaluation game")
    checkpoint_hash = network.metadata.get("config_hash")
    if checkpoint_hash is not None and checkpoint_hash != expected_config_hash:
        raise ValueError("checkpoint config hash does not match evaluation config")


def _load_or_compute_optimal_actions(
    path: Path,
    project_config: Mapping,
    labels: Sequence[OracleLabel],
    *,
    workers: int,
    progress_every: int,
) -> tuple[tuple[tuple[int, ...], ...], tuple[bool, ...]]:
    cached: dict[int, tuple[tuple[int, ...], bool]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            index = int(payload["record_index"])
            if index in cached:
                raise ValueError("optimal-action cache contains duplicate indices")
            if not 0 <= index < len(labels):
                raise ValueError("optimal-action cache index is outside the corpus")
            if labels[index].state_key != payload["state_key"]:
                raise ValueError("optimal-action cache state key mismatch")
            cached[index] = (
                tuple(int(action) for action in payload["optimal_actions"]),
                bool(payload["complete"]),
            )
    missing = tuple(
        (index, label.to_dict())
        for index, label in enumerate(labels)
        if index not in cached
    )
    if missing:
        context = multiprocessing.get_context("spawn")
        with _single_threaded_child_environment():
            with ProcessPoolExecutor(
                max_workers=min(workers, len(missing)),
                mp_context=context,
                initializer=_initialize_optimal_worker,
                initargs=(dict(project_config),),
            ) as executor:
                with path.open("a", encoding="utf-8") as handle:
                    for completed, payload in enumerate(
                        executor.map(_compute_optimal_task, missing, chunksize=1),
                        start=1,
                    ):
                        handle.write(json.dumps(payload, sort_keys=True) + "\n")
                        handle.flush()
                        index = int(payload["record_index"])
                        cached[index] = (
                            tuple(int(action) for action in payload["optimal_actions"]),
                            bool(payload["complete"]),
                        )
                        if completed % progress_every == 0:
                            print(
                                f"derived {completed}/{len(missing)} new optimal-action sets",
                                file=sys.stderr,
                                flush=True,
                            )
    if len(cached) != len(labels):
        raise RuntimeError("optimal-action cache is incomplete")
    return (
        tuple(cached[index][0] for index in range(len(labels))),
        tuple(cached[index][1] for index in range(len(labels))),
    )


def _initialize_optimal_worker(project_config: Mapping) -> None:
    global _OPTIMAL_GAME, _OPTIMAL_TABLEBASE
    _OPTIMAL_GAME = small_game_from_config(project_config)
    _OPTIMAL_TABLEBASE = NoWallTablebase(_OPTIMAL_GAME)


def _compute_optimal_task(task: tuple[int, dict]) -> dict:
    index, label_payload = task
    label = OracleLabel.from_dict(label_payload)
    state = SmallState.from_key(_OPTIMAL_GAME.spec, bytes.fromhex(label.state_key))
    actions, complete = complete_optimal_actions(
        _OPTIMAL_GAME,
        state,
        label,
        _OPTIMAL_TABLEBASE,
    )
    return {
        "record_index": index,
        "state_key": label.state_key,
        "optimal_actions": list(actions),
        "complete": complete,
    }


def _network_predictions(
    network: AlphaZeroNetwork,
    game,
    states: Sequence[SmallState],
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = []
    actions = []
    for start in range(0, len(states), batch_size):
        batch_states = states[start : start + batch_size]
        observations = np.stack(
            [game.canonical_observation(state) for state in batch_states]
        )
        masks = np.stack([game.legal_actions(state) for state in batch_states])
        output = network.forward(observations, use_ema=True)
        values.append(output.value)
        actions.append(np.where(masks, output.policy_logits, -np.inf).argmax(axis=1))
    return np.concatenate(values), np.concatenate(actions).astype(np.int64)


def _initialize_mcts_worker(
    project_config: Mapping,
    checkpoint_path: str,
    simulations: int,
) -> None:
    global _MCTS_GAME, _MCTS_NETWORK, _MCTS_CONFIG
    _MCTS_GAME = small_game_from_config(project_config)
    _MCTS_NETWORK = AlphaZeroNetwork.load_checkpoint(checkpoint_path)
    _MCTS_CONFIG = MCTSConfig(
        simulations=simulations,
        cpuct=float(project_config["mcts"]["cpuct_init"]),
        temperature=0.0,
        root_noise_fraction=0.0,
        forced_playouts=False,
        policy_target_pruning=False,
    )


def _evaluate_mcts_task(task: tuple) -> dict:
    index, state_key, phase, optimal_actions, complete = task
    state = SmallState.from_key(_MCTS_GAME.spec, bytes.fromhex(state_key))
    result = MCTS(_MCTS_CONFIG, _MCTS_NETWORK).run(_MCTS_GAME, state)
    return {
        "record_index": int(index),
        "state_key": state_key,
        "phase": phase,
        "selected_action": int(result.action),
        "root_value": float(result.root_value),
        "optimal_actions": list(optimal_actions),
        "optimal_set_complete": bool(complete),
        "solver_optimal": bool(complete and int(result.action) in optimal_actions),
    }


def _load_position_records(
    path: Path,
    labels: Sequence[OracleLabel],
) -> dict[int, dict]:
    records = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        index = int(payload["record_index"])
        if index in records:
            raise ValueError("MCTS result file contains duplicate record indices")
        if not 0 <= index < len(labels) or labels[index].state_key != payload["state_key"]:
            raise ValueError("MCTS result state key does not match oracle corpus")
        records[index] = payload
    return records


def _mcts_metrics(
    records: Sequence[dict],
    *,
    target: int,
    corpus_positions: int,
    resumed: int,
    new_positions: int,
    requested: bool,
) -> dict:
    complete_records = [record for record in records if record["optimal_set_complete"]]
    result = {
        "requested": requested,
        "target_positions": target,
        "corpus_positions": corpus_positions,
        "completed_positions": len(records),
        "resumed_positions": resumed,
        "new_positions": new_positions,
        "complete_optimal_set_positions": len(complete_records),
        "optimal_action_accuracy": None,
        "by_phase": {},
    }
    if complete_records:
        result["optimal_action_accuracy"] = float(
            np.mean([bool(record["solver_optimal"]) for record in complete_records])
        )
    for phase in ("opening", "midgame", "endgame"):
        phase_records = [
            record
            for record in complete_records
            if record["phase"] == phase
        ]
        result["by_phase"][phase] = {
            "count": len(phase_records),
            "optimal_action_accuracy": (
                float(np.mean([record["solver_optimal"] for record in phase_records]))
                if phase_records
                else None
            ),
        }
    return result


def _acceptance_criteria(
    *,
    value_metrics: dict,
    mcts_metrics: dict,
    complete_optimal_fraction: float,
    corpus_positions: int,
    required_positions: int,
    value_threshold: float,
    mcts_threshold: float,
) -> dict:
    value_scope = corpus_positions >= required_positions
    value_passed = value_scope and value_metrics["sign_accuracy"] >= value_threshold
    if not mcts_metrics["requested"]:
        mcts_status = "not_run"
    elif (
        mcts_metrics["completed_positions"] < required_positions
        or mcts_metrics["target_positions"] < required_positions
    ):
        mcts_status = "incomplete"
    elif complete_optimal_fraction < 1.0:
        mcts_status = "blocked"
    else:
        mcts_status = (
            "pass"
            if mcts_metrics["optimal_action_accuracy"] is not None
            and mcts_metrics["optimal_action_accuracy"] >= mcts_threshold
            else "fail"
        )
    criteria = {
        "value_sign": {
            "status": "pass" if value_passed else "fail",
            "measured": value_metrics["sign_accuracy"],
            "required": value_threshold,
            "positions": corpus_positions,
            "required_positions": required_positions,
        },
        "mcts_optimal_move": {
            "status": mcts_status,
            "measured": mcts_metrics["optimal_action_accuracy"],
            "required": mcts_threshold,
            "completed_positions": mcts_metrics["completed_positions"],
            "required_positions": required_positions,
        },
        "initial_second_player": {
            "status": "blocked",
            "required": 1.0,
            "reason": (
                "the project does not yet contain a full-wall initial-state solver "
                "or exhaustive opponent proof"
            ),
        },
        "monotone_ladder_elo": {
            "status": "blocked",
            "reason": "a fixed-ladder Elo history has not yet been recorded for M2 cycles",
        },
    }
    statuses = tuple(item["status"] for item in criteria.values())
    criteria["overall_status"] = (
        "pass"
        if all(status == "pass" for status in statuses)
        else "fail"
        if "fail" in statuses
        else "blocked"
    )
    return criteria


def _phase_bucket(ply: int, max_plies: int) -> str:
    if ply < max_plies // 3:
        return "opening"
    if ply < (2 * max_plies) // 3:
        return "midgame"
    return "endgame"


def _select_position_indices(phases: Sequence[str], limit: int) -> tuple[int, ...]:
    """Select a deterministic phase-interleaved prefix for resumable evaluation."""
    if limit < 1:
        raise ValueError("position limit must be positive")
    buckets = {
        phase: [index for index, value in enumerate(phases) if value == phase]
        for phase in ("opening", "midgame", "endgame")
    }
    if sum(len(bucket) for bucket in buckets.values()) != len(phases):
        raise ValueError("all positions must belong to a known game phase")
    target = min(limit, len(phases))
    selected: list[int] = []
    offset = 0
    while len(selected) < target:
        for phase in ("opening", "midgame", "endgame"):
            bucket = buckets[phase]
            if offset < len(bucket):
                selected.append(bucket[offset])
                if len(selected) == target:
                    return tuple(selected)
        offset += 1
    return tuple(selected)


def _ensure_manifest(path: Path, expected: dict) -> None:
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError("existing evaluation manifest does not match requested inputs")
        return
    path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a 5x5 checkpoint against the supervisor's M2 gates."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/m2_5x5.json"))
    parser.add_argument("--oracle-corpus", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--mcts-simulations", type=int, default=None)
    parser.add_argument("--position-limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--value-only", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = evaluate_m2_checkpoint(
        config_path=args.config,
        oracle_corpus=args.oracle_corpus,
        checkpoint=args.checkpoint,
        output_directory=args.output_directory,
        workers=args.workers,
        simulations=args.mcts_simulations,
        position_limit=args.position_limit,
        value_only=args.value_only,
        batch_size=args.batch_size,
        progress_every=args.progress_every,
    )
    print(json.dumps(summary, sort_keys=True))
    if args.require_pass and summary["criteria"]["overall_status"] != "pass":
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
