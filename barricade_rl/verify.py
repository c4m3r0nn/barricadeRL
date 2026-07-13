from __future__ import annotations

import argparse
import importlib.util
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np

from . import _native as native_engine
from .game import Game, State, TerminalStatus


def assert_state_invariants(game: Game, state: State) -> None:
    assert game.shortest_path_distance(state, 0) is not None
    assert game.shortest_path_distance(state, 1) is not None
    assert state.horizontal_bits & state.vertical_bits == 0
    for row, col in state.horizontal_walls:
        assert (row, col + 1) not in state.horizontal_walls
    for row, col in state.vertical_walls:
        assert (row + 1, col) not in state.vertical_walls
    placed = state.horizontal_bits.bit_count() + state.vertical_bits.bit_count()
    assert placed + sum(state.walls_remaining) == 20


def random_game_verification(games: int, seed: int = 0) -> dict[str, float | int]:
    game = Game()
    rng = np.random.default_rng(seed)
    total_plies = 0
    started = perf_counter()
    for _ in range(games):
        state = game.initial_state()
        actions: list[int] = []
        while game.is_terminal(state) is TerminalStatus.NOT_TERMINAL:
            mask = game.legal_actions(state)
            legal = np.flatnonzero(mask)
            assert legal.size > 0
            action = int(rng.choice(legal))
            state = game.next_state(state, action)
            actions.append(action)
            assert_state_invariants(game, state)
        status = game.is_terminal(state)
        if status is TerminalStatus.MOVER_LOST:
            previous = 1 - state.current_player
            assert state.pawns[previous][0] == (8 if previous == 0 else 0)
        else:
            assert status is TerminalStatus.CAPPED and state.ply == game.max_plies

        replayed = game.initial_state()
        for action in actions:
            replayed = game.next_state(replayed, action)
        assert replayed == state
        total_plies += len(actions)
    elapsed = perf_counter() - started
    return {
        "games": games,
        "plies": total_plies,
        "seconds": elapsed,
        "plies_per_second": total_plies / elapsed,
    }


def differential_verification(states: int, seed: int = 0) -> int:
    """Compare the compiled engine with the in-repository Python oracle."""
    oracle_path = Path(__file__).with_name("_engine.py")
    spec = importlib.util.spec_from_file_location("barricade_python_oracle", oracle_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Python rules oracle")
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)

    game = Game()
    rng = np.random.default_rng(seed)
    state = game.initial_state()
    compared = 0
    while compared < states:
        native_mask = list(native_engine.legal_actions(state.data, game.max_plies))
        oracle_mask = oracle.legal_actions(state.data, game.max_plies)
        assert native_mask == oracle_mask
        for player in (0, 1):
            assert native_engine.shortest_path_distance(state.data, player) == oracle.shortest_path_distance(state.data, player)
        compared += 1
        legal = np.flatnonzero(native_mask)
        if not legal.size:
            state = game.initial_state()
            continue
        action = int(rng.choice(legal))
        native_next = bytes(native_engine.next_state(state.data, action, game.max_plies))
        oracle_next = oracle.next_state(state.data, action, game.max_plies)
        assert native_next == oracle_next
        state = State(native_next)
    return compared


def external_differential_verification(states: int, seed: int = 0) -> int:
    from .external_oracle import external_legal_actions

    game = Game()
    rng = np.random.default_rng(seed)
    state = game.initial_state()
    compared = 0
    while compared < states:
        native_mask = game.legal_actions(state)
        external_mask = external_legal_actions(state)
        if not np.array_equal(native_mask, external_mask):
            differences = np.flatnonzero(native_mask != external_mask).tolist()
            raise AssertionError(
                f"external oracle disagreement at state {state.data.hex()}: actions {differences}"
            )
        compared += 1
        legal = np.flatnonzero(native_mask)
        if not legal.size:
            state = game.initial_state()
        else:
            state = game.next_state(state, int(rng.choice(legal)))
    return compared


def _external_game_verification_chunk(games: int, seed: int) -> dict[str, int]:
    from .external_oracle import external_legal_actions

    game = Game()
    rng = np.random.default_rng(seed)
    external_cache: dict[bytes, np.ndarray] = {}
    compared_states = 0
    compared_plies = 0
    for _ in range(games):
        state = game.initial_state()
        while True:
            native_mask = game.legal_actions(state)
            external_mask = external_cache.get(state.data)
            if external_mask is None:
                external_mask = external_legal_actions(state)
                external_cache[state.data] = external_mask
            if not np.array_equal(native_mask, external_mask):
                differences = np.flatnonzero(native_mask != external_mask).tolist()
                raise AssertionError(
                    f"external oracle disagreement at state {state.data.hex()}: actions {differences}"
                )
            compared_states += 1
            legal = np.flatnonzero(native_mask)
            if not legal.size:
                break

            pawn_actions = legal[legal < 12]
            wall_actions = legal[legal >= 12]
            forward_actions = np.intersect1d(pawn_actions, np.array((0, 4, 8, 9)))
            choice = rng.random()
            if forward_actions.size and choice < 0.65:
                candidates = forward_actions
            elif wall_actions.size and choice < 0.85:
                candidates = wall_actions
            else:
                candidates = pawn_actions if pawn_actions.size else legal
            state = game.next_state(state, int(rng.choice(candidates)))
            compared_plies += 1
    return {"games": games, "states": compared_states, "plies": compared_plies}


def external_game_verification(games: int, seed: int = 0, workers: int | None = None) -> dict[str, int]:
    if games < 0:
        raise ValueError("games must be non-negative")
    if games == 0:
        return {"games": 0, "states": 0, "plies": 0}
    if workers is None:
        workers = 1 if games < 100 else min(8, os.cpu_count() or 1)
    workers = max(1, min(workers, games))
    if workers == 1:
        return _external_game_verification_chunk(games, seed)

    base, extra = divmod(games, workers)
    chunks = [base + (index < extra) for index in range(workers)]
    seeds = [seed + index * 1_000_003 for index in range(workers)]
    start_method = "spawn" if sys.platform == "win32" else "fork"
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context(start_method),
    ) as executor:
        results = executor.map(_external_game_verification_chunk, chunks, seeds)
    totals = {"games": 0, "states": 0, "plies": 0}
    for result in results:
        for key in totals:
            totals[key] += result[key]
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the blocking M0 rules-engine verification")
    parser.add_argument("--games", type=int, default=10_000)
    parser.add_argument("--differential-states", type=int, default=10_000)
    parser.add_argument("--external-games", type=int, default=10_000)
    parser.add_argument("--external-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    differential_verification(args.differential_states, args.seed)
    external_metrics = external_game_verification(args.external_games, args.seed, args.external_workers)
    metrics = random_game_verification(args.games, args.seed)
    print({"random": metrics, "external": external_metrics})


if __name__ == "__main__":
    main()
