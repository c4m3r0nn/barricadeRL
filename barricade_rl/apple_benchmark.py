from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import platform
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np
import torch

from .az_learner import AlphaZeroLearner, _DifferentiableNetwork
from .az_network import AlphaZeroNetwork
from .az_replay import AlphaZeroReplayBuffer
from .config import load_config, small_game_from_config
from .mcts import MCTS, MCTSConfig
from .small_board import SmallState


def parse_positive_int_csv(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("expected at least one integer")
    if any(item < 1 for item in values):
        raise ValueError("benchmark integers must be positive")
    return values


def parse_devices(value: str) -> tuple[str, ...]:
    devices = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not devices:
        raise ValueError("expected at least one device")
    unsupported = sorted(set(devices) - {"cpu", "mps"})
    if unsupported:
        raise ValueError(f"unsupported benchmark devices: {', '.join(unsupported)}")
    return tuple(dict.fromkeys(devices))


def benchmark_inference(
    network: AlphaZeroNetwork,
    observations: np.ndarray,
    *,
    batch_sizes: Sequence[int],
    iterations: int,
    warmup: int,
    devices: Sequence[str],
) -> dict:
    observations = np.asarray(observations, dtype=np.float32)
    if observations.ndim != 4 or observations.shape[0] < 1:
        raise ValueError("observations must be a non-empty NCHW batch")
    if iterations < 1 or warmup < 0:
        raise ValueError("iterations must be positive and warmup non-negative")
    sizes = tuple(int(size) for size in batch_sizes)
    if not sizes or any(size < 1 for size in sizes):
        raise ValueError("batch sizes must be positive")

    result: dict = {
        "numpy": [],
        "torch": {},
        "parity": {},
        "unavailable_devices": {},
    }
    for batch_size in sizes:
        batch = _observation_batch(observations, batch_size)
        for _ in range(warmup):
            network.forward(batch, use_ema=True)
        started = perf_counter()
        for _ in range(iterations):
            network.forward(batch, use_ema=True)
        seconds = perf_counter() - started
        result["numpy"].append(
            _throughput_record(
                batch_size=batch_size,
                iterations=iterations,
                seconds=seconds,
            )
        )

    parity_batch = _observation_batch(observations, min(max(sizes), 16))
    numpy_output = network.forward(parity_batch, use_ema=True)
    for device_name in devices:
        if not _device_available(device_name):
            result["unavailable_devices"][device_name] = _device_unavailable_reason(
                device_name
            )
            continue
        device = torch.device(device_name)
        model = _ema_torch_model(network, device)
        device_records = []
        for batch_size in sizes:
            batch = _observation_batch(observations, batch_size)
            tensor = torch.from_numpy(batch).to(device)
            with torch.inference_mode():
                for _ in range(warmup):
                    model(tensor)
                _synchronize(device)
                started = perf_counter()
                for _ in range(iterations):
                    model(tensor)
                _synchronize(device)
                compute_seconds = perf_counter() - started

                for _ in range(warmup):
                    _torch_round_trip(model, batch, device)
                _synchronize(device)
                started = perf_counter()
                for _ in range(iterations):
                    _torch_round_trip(model, batch, device)
                _synchronize(device)
                round_trip_seconds = perf_counter() - started
            record = _throughput_record(
                batch_size=batch_size,
                iterations=iterations,
                seconds=compute_seconds,
            )
            record["round_trip_seconds"] = round_trip_seconds
            record["round_trip_positions_per_second"] = (
                batch_size * iterations / max(round_trip_seconds, 1e-12)
            )
            device_records.append(record)
        result["torch"][device_name] = device_records

        parity_tensor = torch.from_numpy(parity_batch).to(device)
        with torch.inference_mode():
            torch_output = model(parity_tensor)
            torch_arrays = tuple(value.detach().cpu().numpy() for value in torch_output)
        result["parity"][device_name] = {
            "batch_size": int(parity_batch.shape[0]),
            "policy_logits_max_abs_error": _max_abs_error(
                numpy_output.policy_logits, torch_arrays[0]
            ),
            "value_max_abs_error": _max_abs_error(
                numpy_output.value, torch_arrays[1]
            ),
            "auxiliary_distances_max_abs_error": _max_abs_error(
                numpy_output.auxiliary_distances, torch_arrays[2]
            ),
            "opponent_policy_logits_max_abs_error": _max_abs_error(
                numpy_output.opponent_policy_logits, torch_arrays[3]
            ),
        }
    return result


def benchmark_learner(
    checkpoint: str | Path,
    replay_path: str | Path,
    game,
    *,
    devices: Sequence[str],
    steps: int,
) -> dict:
    if steps < 1:
        raise ValueError("learner benchmark steps must be positive")
    results = {}
    for device_name in devices:
        if not _device_available(device_name):
            results[device_name] = {
                "available": False,
                "reason": _device_unavailable_reason(device_name),
            }
            continue
        try:
            learner = AlphaZeroLearner.load_checkpoint(
                checkpoint,
                game,
                device=device_name,
            )
            replay = AlphaZeroReplayBuffer.load_npz(replay_path)
            initial_step = learner.step
            step_seconds = []
            metrics = None
            for _ in range(steps):
                _synchronize(learner.device)
                started = perf_counter()
                metrics = learner.train(replay, steps=1)
                _synchronize(learner.device)
                step_seconds.append(perf_counter() - started)
            seconds = sum(step_seconds)
            completed = learner.step - initial_step
            steady_seconds = sum(step_seconds[1:])
            results[device_name] = {
                "available": True,
                "requested_steps": steps,
                "completed_steps": completed,
                "seconds": seconds,
                "steps_per_second": completed / max(seconds, 1e-12),
                "step_seconds": step_seconds,
                "cold_step_seconds": step_seconds[0],
                "steady_state_steps_per_second": (
                    (completed - 1) / max(steady_seconds, 1e-12)
                    if completed > 1
                    else None
                ),
                "final_metrics": metrics.to_dict(),
            }
        except Exception as error:  # pragma: no cover - backend-specific failure path
            results[device_name] = {
                "available": True,
                "error": f"{type(error).__name__}: {error}",
            }
    return results


_WORKER_GAME = None
_WORKER_NETWORK = None
_WORKER_MCTS_CONFIG = None


def _initialize_mcts_worker(
    project_config_path: str,
    checkpoint_path: str,
    simulations: int,
) -> None:
    global _WORKER_GAME, _WORKER_NETWORK, _WORKER_MCTS_CONFIG
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    project_config = load_config(project_config_path)
    _WORKER_GAME = small_game_from_config(project_config)
    _WORKER_NETWORK = AlphaZeroNetwork.load_checkpoint(checkpoint_path)
    _WORKER_MCTS_CONFIG = MCTSConfig(
        simulations=simulations,
        cpuct=float(project_config["mcts"]["cpuct_init"]),
        temperature=0.0,
        root_noise_fraction=0.0,
        forced_playouts=False,
        policy_target_pruning=False,
    )


def _mcts_worker_action(state_key_hex: str) -> int:
    state = SmallState.from_key(_WORKER_GAME.spec, bytes.fromhex(state_key_hex))
    return MCTS(_WORKER_MCTS_CONFIG, _WORKER_NETWORK).run(
        _WORKER_GAME,
        state,
    ).action


def benchmark_mcts_process_scaling(
    project_config_path: str | Path,
    checkpoint_path: str | Path,
    state_keys: Sequence[bytes],
    *,
    simulations: int,
    worker_counts: Sequence[int],
    tasks: int,
) -> dict:
    if simulations < 1 or tasks < 1:
        raise ValueError("MCTS simulations and tasks must be positive")
    counts = tuple(int(count) for count in worker_counts)
    if not counts or any(count < 1 for count in counts):
        raise ValueError("MCTS worker counts must be positive")
    if not state_keys:
        raise ValueError("at least one state key is required")
    task_keys = [state_keys[index % len(state_keys)].hex() for index in range(tasks)]
    records = []
    baseline_throughput = None
    baseline_actions = None
    context = multiprocessing.get_context("spawn")
    for workers in counts:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_initialize_mcts_worker,
            initargs=(
                str(Path(project_config_path).resolve()),
                str(Path(checkpoint_path).resolve()),
                simulations,
            ),
        ) as executor:
            warmup_keys = task_keys[: min(tasks, workers)]
            tuple(executor.map(_mcts_worker_action, warmup_keys, chunksize=1))
            started = perf_counter()
            actions = tuple(
                executor.map(_mcts_worker_action, task_keys, chunksize=1)
            )
            seconds = perf_counter() - started
        throughput = tasks / max(seconds, 1e-12)
        if baseline_throughput is None:
            baseline_throughput = throughput
            baseline_actions = actions
        records.append(
            {
                "workers": workers,
                "tasks": tasks,
                "simulations_per_task": simulations,
                "seconds": seconds,
                "tasks_per_second": throughput,
                "speedup_vs_first": throughput / baseline_throughput,
                "actions_match_first": actions == baseline_actions,
                "action_checksum": int(sum(actions)),
            }
        )
    return {
        "records": records,
        "startup_excluded": True,
        "deterministic": all(record["actions_match_first"] for record in records),
    }


def _ema_torch_model(
    network: AlphaZeroNetwork,
    device: torch.device,
) -> _DifferentiableNetwork:
    ema_network = AlphaZeroNetwork(
        network.config,
        network.ema_params,
        ema_params=network.ema_params,
        metadata=network.metadata,
    )
    model = _DifferentiableNetwork(ema_network).to(device)
    model.eval()
    return model


def _torch_round_trip(
    model: _DifferentiableNetwork,
    batch: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, ...]:
    tensor = torch.from_numpy(batch).to(device)
    output = model(tensor)
    return tuple(value.detach().cpu().numpy() for value in output)


def _observation_batch(observations: np.ndarray, batch_size: int) -> np.ndarray:
    indices = np.arange(batch_size, dtype=np.int64) % observations.shape[0]
    return np.ascontiguousarray(observations[indices], dtype=np.float32)


def _throughput_record(
    *,
    batch_size: int,
    iterations: int,
    seconds: float,
) -> dict:
    return {
        "batch_size": batch_size,
        "iterations": iterations,
        "seconds": seconds,
        "milliseconds_per_batch": 1000.0 * seconds / iterations,
        "positions_per_second": batch_size * iterations / max(seconds, 1e-12),
    }


def _max_abs_error(reference: np.ndarray, actual: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(reference) - np.asarray(actual))))


def _device_available(device_name: str) -> bool:
    if device_name == "cpu":
        return True
    if device_name == "mps":
        return bool(torch.backends.mps.is_available())
    return False


def _device_unavailable_reason(device_name: str) -> str:
    if device_name == "mps":
        if not torch.backends.mps.is_built():
            return "PyTorch was not built with MPS support"
        return "MPS is not available on this host"
    return f"unsupported device {device_name}"


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark BarricadeRL CPU and Apple Metal acceleration paths."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/m2_5x5.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--devices", default="cpu,mps")
    parser.add_argument("--batch-sizes", default="1,16,64,256")
    parser.add_argument("--inference-iterations", type=int, default=10)
    parser.add_argument("--inference-warmup", type=int, default=2)
    parser.add_argument("--learner-steps", type=int, default=1)
    parser.add_argument("--mcts-workers", default="1,2,4,8")
    parser.add_argument("--mcts-tasks", type=int, default=16)
    parser.add_argument("--mcts-simulations", type=int, default=50)
    parser.add_argument("--skip-learner", action="store_true")
    parser.add_argument("--skip-mcts", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    for path in (args.config, args.checkpoint, args.replay):
        if not path.is_file():
            raise FileNotFoundError(path)
    devices = parse_devices(args.devices)
    batch_sizes = parse_positive_int_csv(args.batch_sizes)
    worker_counts = parse_positive_int_csv(args.mcts_workers)
    if args.inference_iterations < 1 or args.inference_warmup < 0:
        raise ValueError("inference iterations must be positive and warmup non-negative")

    hashes_before = {
        "checkpoint": _sha256(args.checkpoint),
        "replay": _sha256(args.replay),
    }
    project_config = load_config(args.config)
    game = small_game_from_config(project_config)
    network = AlphaZeroNetwork.load_checkpoint(args.checkpoint)
    replay = AlphaZeroReplayBuffer.load_npz(args.replay)
    observations = np.stack(
        [sample.observation for sample in replay.samples[: max(batch_sizes)]]
    )
    started = perf_counter()
    payload = {
        "schema_version": 1,
        "hardware": {
            "machine": platform.machine(),
            "macos": platform.mac_ver()[0],
            "logical_cpu_count": os.cpu_count(),
            "torch_version": torch.__version__,
            "mps_built": bool(torch.backends.mps.is_built()),
            "mps_available": bool(torch.backends.mps.is_available()),
            "torch_cpu_threads": torch.get_num_threads(),
        },
        "inputs": {
            "config": str(args.config),
            "checkpoint": str(args.checkpoint),
            "replay": str(args.replay),
            "checkpoint_step": int(network.metadata.get("step", 0)),
            "replay_size": replay.size,
        },
        "inference": benchmark_inference(
            network,
            observations,
            batch_sizes=batch_sizes,
            iterations=args.inference_iterations,
            warmup=args.inference_warmup,
            devices=devices,
        ),
    }
    if not args.skip_learner:
        payload["learner"] = benchmark_learner(
            args.checkpoint,
            args.replay,
            game,
            devices=devices,
            steps=args.learner_steps,
        )
    if not args.skip_mcts:
        payload["mcts_process_scaling"] = benchmark_mcts_process_scaling(
            args.config,
            args.checkpoint,
            [sample.state_key for sample in replay.samples],
            simulations=args.mcts_simulations,
            worker_counts=worker_counts,
            tasks=args.mcts_tasks,
        )
    payload["elapsed_seconds"] = perf_counter() - started
    hashes_after = {
        "checkpoint": _sha256(args.checkpoint),
        "replay": _sha256(args.replay),
    }
    payload["input_files_unchanged"] = hashes_before == hashes_after
    print(json.dumps(payload, indent=2 if args.pretty else None, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
