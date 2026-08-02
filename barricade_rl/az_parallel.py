from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .az_gating import GatingConfig, GatingResult, NetworkMCTSPolicy, gate_candidate
from .az_network import AlphaZeroNetwork
from .az_replay import AlphaZeroReplayBuffer
from .az_self_play import SelfPlayConfig, SelfPlayGameRecord, play_self_play_game
from .config import small_game_from_config
from .evaluate import GameRecord, MatchResult, play_game
from .small_board import SmallState


PARALLEL_PROTOCOL = "spawned-ordered-games-v1"

_SELF_PLAY_GAME = None
_SELF_PLAY_NETWORK = None
_SELF_PLAY_CONFIG = None

_GATE_GAME = None
_GATE_CANDIDATE = None
_GATE_INCUMBENT = None


@dataclass(frozen=True, slots=True)
class _NamedPolicy:
    name: str


def _limit_worker_threads() -> None:
    # Each game is deliberately single-threaded. Parallelism belongs at the game
    # level, otherwise Accelerate/OpenMP oversubscription erases process scaling.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"


@contextmanager
def _single_threaded_child_environment():
    names = ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    previous = {name: os.environ.get(name) for name in names}
    _limit_worker_threads()
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _initialize_self_play_worker(
    project_config: Mapping,
    checkpoint_path: str,
    config: SelfPlayConfig,
) -> None:
    global _SELF_PLAY_GAME, _SELF_PLAY_NETWORK, _SELF_PLAY_CONFIG
    _limit_worker_threads()
    _SELF_PLAY_GAME = small_game_from_config(project_config)
    _SELF_PLAY_NETWORK = AlphaZeroNetwork.load_checkpoint(checkpoint_path)
    _SELF_PLAY_CONFIG = config


def _play_self_play_task(task: tuple) -> SelfPlayGameRecord:
    index, seed, game_id_prefix, run_id, config_hash, git_commit = task
    return play_self_play_game(
        _SELF_PLAY_GAME,
        _SELF_PLAY_NETWORK,
        _SELF_PLAY_CONFIG,
        rng=np.random.default_rng(seed),
        game_id=f"{game_id_prefix}-{index:08d}",
        run_id=run_id,
        config_hash=config_hash,
        git_commit=git_commit,
    )


def generate_self_play_games_parallel(
    *,
    project_config: Mapping,
    checkpoint_path: str | Path,
    config: SelfPlayConfig,
    games: int,
    replay_buffer: AlphaZeroReplayBuffer,
    run_id: str,
    config_hash: str,
    git_commit: str,
    seed: int,
    workers: int,
    game_id_prefix: str | None = None,
) -> tuple[SelfPlayGameRecord, ...]:
    """Generate independent games in processes and merge them by game index."""
    if games < 1:
        raise ValueError("games must be positive")
    if workers < 1:
        raise ValueError("self-play workers must be positive")
    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    prefix = run_id if game_id_prefix is None else game_id_prefix
    seeds = _indexed_seeds(seed, games, stream=0)
    tasks = tuple(
        (index, seeds[index], prefix, run_id, config_hash, git_commit)
        for index in range(games)
    )
    context = multiprocessing.get_context("spawn")
    with _single_threaded_child_environment():
        with ProcessPoolExecutor(
            max_workers=min(workers, games),
            mp_context=context,
            initializer=_initialize_self_play_worker,
            initargs=(dict(project_config), str(checkpoint), config),
        ) as executor:
            records = tuple(executor.map(_play_self_play_task, tasks, chunksize=1))
    for record in records:
        replay_buffer.extend(record.samples)
    return records


def _initialize_gate_worker(
    project_config: Mapping,
    candidate_checkpoint: str,
    incumbent_checkpoint: str,
    gating_config: GatingConfig,
    candidate_name: str,
    incumbent_name: str,
) -> None:
    global _GATE_GAME, _GATE_CANDIDATE, _GATE_INCUMBENT
    _limit_worker_threads()
    _GATE_GAME = small_game_from_config(project_config)
    candidate_network = AlphaZeroNetwork.load_checkpoint(candidate_checkpoint)
    incumbent_network = AlphaZeroNetwork.load_checkpoint(incumbent_checkpoint)
    _GATE_CANDIDATE = NetworkMCTSPolicy(
        candidate_network,
        simulations=gating_config.evaluation_simulations,
        cpuct=gating_config.cpuct,
        name=candidate_name,
    )
    _GATE_INCUMBENT = NetworkMCTSPolicy(
        incumbent_network,
        simulations=gating_config.evaluation_simulations,
        cpuct=gating_config.cpuct,
        name=incumbent_name,
    )


def _play_gate_task(task: tuple[int, int, str]) -> GameRecord:
    candidate_player, game_seed, state_key_hex = task
    state = SmallState.from_key(_GATE_GAME.spec, bytes.fromhex(state_key_hex))
    if candidate_player == 0:
        return play_game(
            _GATE_CANDIDATE,
            _GATE_INCUMBENT,
            seed=game_seed,
            game=_GATE_GAME,
            initial_state=state,
        )
    return play_game(
        _GATE_INCUMBENT,
        _GATE_CANDIDATE,
        seed=game_seed,
        game=_GATE_GAME,
        initial_state=state,
    )


def gate_checkpoints_parallel(
    *,
    project_config: Mapping,
    candidate_checkpoint: str | Path,
    incumbent_checkpoint: str | Path,
    game,
    config: GatingConfig,
    seed: int,
    workers: int,
    initial_states: Sequence | None = None,
    start_seed: int | None = None,
) -> GatingResult:
    """Gate checkpoint files in parallel without changing paired-game ordering."""
    if workers < 1:
        raise ValueError("gating workers must be positive")
    candidate_path = Path(candidate_checkpoint).resolve()
    incumbent_path = Path(incumbent_checkpoint).resolve()
    for path in (candidate_path, incumbent_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if initial_states is None:
        raise ValueError("parallel gating requires pre-sampled initial states")
    starts = tuple(initial_states)
    if len(starts) != config.games_per_color:
        raise ValueError("gating requires exactly one start state per colour pair")

    candidate_network = AlphaZeroNetwork.load_checkpoint(candidate_path)
    incumbent_network = AlphaZeroNetwork.load_checkpoint(incumbent_path)
    candidate_name = f"candidate-{candidate_network.metadata.get('step', 0)}"
    incumbent_name = f"incumbent-{incumbent_network.metadata.get('step', 0)}"
    master_rng = np.random.default_rng(seed)
    tasks = tuple(
        (
            candidate_player,
            int(master_rng.integers(0, np.iinfo(np.int64).max)),
            game.state_key(starts[game_index]).hex(),
        )
        for candidate_player in (0, 1)
        for game_index in range(config.games_per_color)
    )
    context = multiprocessing.get_context("spawn")
    with _single_threaded_child_environment():
        with ProcessPoolExecutor(
            max_workers=min(workers, config.games),
            mp_context=context,
            initializer=_initialize_gate_worker,
            initargs=(
                dict(project_config),
                str(candidate_path),
                str(incumbent_path),
                config,
                candidate_name,
                incumbent_name,
            ),
        ) as executor:
            records = tuple(executor.map(_play_gate_task, tasks, chunksize=1))

    candidate_wins = 0
    incumbent_wins = 0
    draws = 0
    for task, record in zip(tasks, records, strict=True):
        candidate_player = task[0]
        if record.winner is None:
            draws += 1
        elif record.winner == candidate_player:
            candidate_wins += 1
        else:
            incumbent_wins += 1
    match = MatchResult(
        candidate=candidate_name,
        opponent=incumbent_name,
        candidate_wins=candidate_wins,
        opponent_wins=incumbent_wins,
        draws=draws,
        games_per_color=config.games_per_color,
        avg_plies=float(np.mean([record.plies for record in records])),
        records=records,
    )
    return gate_candidate(
        _NamedPolicy(candidate_name),
        _NamedPolicy(incumbent_name),
        game=game,
        config=config,
        seed=seed,
        initial_states=starts,
        start_seed=start_seed,
        match_runner=lambda *args, **kwargs: match,
    )


def _indexed_seeds(base_seed: int, count: int, *, stream: int) -> tuple[int, ...]:
    return tuple(
        int(
            np.random.SeedSequence([int(base_seed), int(index), int(stream)])
            .generate_state(1, dtype=np.uint64)[0]
            % np.iinfo(np.int64).max
        )
        for index in range(count)
    )
