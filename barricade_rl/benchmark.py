from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np

from .game import Game, TerminalStatus


def benchmark_random_playout_plies(plies: int = 100_000, seed: int = 0) -> float:
    game = Game()
    state = game.initial_state()
    rng = np.random.default_rng(seed)
    completed = 0
    started = perf_counter()
    while completed < plies:
        if game.is_terminal(state) is not TerminalStatus.NOT_TERMINAL:
            state = game.initial_state()
        legal = np.flatnonzero(game.legal_actions(state))
        state = game.next_state(state, int(rng.choice(legal)))
        completed += 1
    return completed / (perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the bare rules engine")
    parser.add_argument("--plies", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    throughput = benchmark_random_playout_plies(args.plies, args.seed)
    print(f"{throughput:,.0f} random-playout plies/second")
    if throughput < 20_000:
        raise SystemExit("M0 throughput gate failed: expected at least 20,000 plies/second")


if __name__ == "__main__":
    main()
