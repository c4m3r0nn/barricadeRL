from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .game import ACTION_COUNT, Game, TerminalStatus
from .opponents import (
    FROZEN_LADDER,
    LADDER_VERSION,
    AlphaBetaOpponent,
    GreedyRacer,
    HeuristicOne,
    OpponentPolicy,
    RandomOpponent,
)


def _policy_name(policy: OpponentPolicy) -> str:
    return str(getattr(policy, "name", policy.__class__.__name__))


@dataclass(frozen=True, slots=True)
class NamedPolicy:
    policy: OpponentPolicy
    name: str

    def select_action(self, game: Game, state, rng) -> int:
        return int(self.policy.select_action(game, state, rng))


@dataclass(frozen=True, slots=True)
class GameRecord:
    player0: str
    player1: str
    seed: int | None
    winner: int | None
    termination: str
    plies: int
    actions: tuple[int, ...]
    final_state_key: str
    walls_placed: tuple[int, int]

    def to_dict(self) -> dict:
        return {
            "player0": self.player0,
            "player1": self.player1,
            "seed": self.seed,
            "winner": self.winner,
            "termination": self.termination,
            "plies": self.plies,
            "actions": list(self.actions),
            "final_state_key": self.final_state_key,
            "walls_placed": list(self.walls_placed),
        }


@dataclass(frozen=True, slots=True)
class MatchResult:
    candidate: str
    opponent: str
    candidate_wins: int
    opponent_wins: int
    draws: int
    games_per_color: int
    avg_plies: float
    records: tuple[GameRecord, ...]

    @property
    def games(self) -> int:
        return self.candidate_wins + self.opponent_wins + self.draws

    @property
    def candidate_score(self) -> float:
        return float(self.candidate_wins) + 0.5 * float(self.draws)

    @property
    def score_rate(self) -> float:
        return self.candidate_score / self.games if self.games else 0.0

    @property
    def cap_fraction(self) -> float:
        if not self.games:
            return 0.0
        caps = sum(1 for record in self.records if record.termination == "cap")
        return caps / self.games

    def to_dict(self, *, include_records: bool = True) -> dict:
        payload = {
            "candidate": self.candidate,
            "opponent": self.opponent,
            "candidate_wins": self.candidate_wins,
            "opponent_wins": self.opponent_wins,
            "draws": self.draws,
            "games": self.games,
            "games_per_color": self.games_per_color,
            "candidate_score": self.candidate_score,
            "score_rate": self.score_rate,
            "avg_plies": self.avg_plies,
            "cap_fraction": self.cap_fraction,
        }
        if include_records:
            payload["records"] = [record.to_dict() for record in self.records]
        return payload


@dataclass(frozen=True, slots=True)
class LadderEvaluation:
    candidate: str
    run_id: str
    seed: int | None
    ladder_version: int
    matches: tuple[MatchResult, ...]
    elo_ratings: dict[str, float]

    @property
    def games(self) -> int:
        return sum(match.games for match in self.matches)

    @property
    def avg_plies(self) -> float:
        if not self.games:
            return 0.0
        return sum(match.avg_plies * match.games for match in self.matches) / self.games

    @property
    def cap_fraction(self) -> float:
        if not self.games:
            return 0.0
        caps = sum(
            1
            for match in self.matches
            for record in match.records
            if record.termination == "cap"
        )
        return caps / self.games

    @property
    def mean_walls_placed(self) -> float:
        if not self.games:
            return 0.0
        wall_total = sum(
            sum(record.walls_placed)
            for match in self.matches
            for record in match.records
        )
        return wall_total / self.games

    def to_dict(self, *, include_records: bool = True) -> dict:
        return {
            "candidate": self.candidate,
            "run_id": self.run_id,
            "seed": self.seed,
            "ladder_version": self.ladder_version,
            "games": self.games,
            "avg_plies": self.avg_plies,
            "cap_fraction": self.cap_fraction,
            "mean_walls_placed": self.mean_walls_placed,
            "elo_ratings": dict(sorted(self.elo_ratings.items())),
            "matches": [
                match.to_dict(include_records=include_records)
                for match in self.matches
            ],
        }


def play_game(
    player0: OpponentPolicy,
    player1: OpponentPolicy,
    *,
    seed: int | None = None,
    game: Game | None = None,
) -> GameRecord:
    game = game or Game()
    policies = (player0, player1)
    names = (_policy_name(player0), _policy_name(player1))
    rng = np.random.default_rng(seed)
    state = game.initial_state()
    actions: list[int] = []

    while game.is_terminal(state) is TerminalStatus.NOT_TERMINAL:
        policy = policies[state.current_player]
        action = int(policy.select_action(game, state, rng))
        legal_mask = game.legal_actions(state)
        if not 0 <= action < len(legal_mask) or not legal_mask[action]:
            raise RuntimeError(f"{_policy_name(policy)} selected illegal action {action}")
        actions.append(action)
        state = game.next_state(state, action)

    status = game.is_terminal(state)
    if status is TerminalStatus.MOVER_LOST:
        winner = 1 - state.current_player
        termination = "win"
    elif status is TerminalStatus.CAPPED:
        winner = None
        termination = "cap"
    else:  # pragma: no cover - loop exits only at terminal states
        raise AssertionError("game loop ended before terminal state")

    walls_remaining = state.walls_remaining
    walls_per_player = int(getattr(getattr(game, "spec", None), "walls_per_player", 10))
    return GameRecord(
        player0=names[0],
        player1=names[1],
        seed=seed,
        winner=winner,
        termination=termination,
        plies=state.ply,
        actions=tuple(actions),
        final_state_key=game.state_key(state).hex(),
        walls_placed=(
            walls_per_player - walls_remaining[0],
            walls_per_player - walls_remaining[1],
        ),
    )


def play_match(
    candidate: OpponentPolicy,
    opponent: OpponentPolicy,
    *,
    games_per_color: int,
    seed: int | None = None,
    game: Game | None = None,
) -> MatchResult:
    if games_per_color < 1:
        raise ValueError("games_per_color must be positive")
    game = game or Game()
    master_rng = np.random.default_rng(seed)
    candidate_name = _policy_name(candidate)
    opponent_name = _policy_name(opponent)
    candidate_wins = 0
    opponent_wins = 0
    draws = 0
    records: list[GameRecord] = []

    for candidate_player in (0, 1):
        for _ in range(games_per_color):
            game_seed = int(master_rng.integers(0, np.iinfo(np.int64).max))
            if candidate_player == 0:
                record = play_game(candidate, opponent, seed=game_seed, game=game)
            else:
                record = play_game(opponent, candidate, seed=game_seed, game=game)
            records.append(record)
            if record.winner is None:
                draws += 1
            elif record.winner == candidate_player:
                candidate_wins += 1
            else:
                opponent_wins += 1

    avg_plies = float(np.mean([record.plies for record in records])) if records else 0.0
    return MatchResult(
        candidate=candidate_name,
        opponent=opponent_name,
        candidate_wins=candidate_wins,
        opponent_wins=opponent_wins,
        draws=draws,
        games_per_color=games_per_color,
        avg_plies=avg_plies,
        records=tuple(records),
    )


def estimate_elos(
    matches: Iterable[MatchResult],
    *,
    anchor: str = "random",
    prior_draws: float = 1.0,
    iterations: int = 100,
) -> dict[str, float]:
    """Estimate Elo ratings from aggregate match scores.

    The random policy is fixed at 0 Elo. Each aggregate score receives one
    pseudo-draw by default so small smoke evaluations and shutouts produce
    finite ratings rather than infinities.
    """

    match_list = tuple(matches)
    players = sorted({name for match in match_list for name in (match.candidate, match.opponent)})
    if anchor not in players:
        raise ValueError(f"anchor player {anchor!r} is not present in the match results")
    if prior_draws < 0:
        raise ValueError("prior_draws must be non-negative")
    if iterations < 1:
        raise ValueError("iterations must be positive")

    variable_players = [player for player in players if player != anchor]
    ratings = {player: 0.0 for player in players}
    if not variable_players:
        return ratings

    index = {player: i for i, player in enumerate(variable_players)}
    k = np.log(10.0) / 400.0
    ridge = 1e-7

    for _ in range(iterations):
        gradient = np.zeros(len(variable_players), dtype=np.float64)
        hessian = np.zeros((len(variable_players), len(variable_players)), dtype=np.float64)
        for match in match_list:
            games = float(match.games) + prior_draws
            score = float(match.candidate_score) + 0.5 * prior_draws
            diff = ratings[match.candidate] - ratings[match.opponent]
            expected = 1.0 / (1.0 + np.exp(-k * diff))
            residual = k * (score - games * expected)
            weight = games * k * k * expected * (1.0 - expected)

            a = index.get(match.candidate)
            b = index.get(match.opponent)
            if a is not None:
                gradient[a] += residual
                hessian[a, a] -= weight
            if b is not None:
                gradient[b] -= residual
                hessian[b, b] -= weight
            if a is not None and b is not None:
                hessian[a, b] += weight
                hessian[b, a] += weight

        vector = np.array([ratings[player] for player in variable_players], dtype=np.float64)
        gradient -= ridge * vector
        hessian -= ridge * np.eye(len(variable_players), dtype=np.float64)
        step = np.linalg.solve(hessian, gradient)
        max_step = float(np.max(np.abs(step))) if step.size else 0.0
        if max_step > 100.0:
            step *= 100.0 / max_step
        for player, delta in zip(variable_players, step):
            ratings[player] -= float(delta)
        if max_step < 1e-7:
            break

    ratings[anchor] = 0.0
    return {player: round(float(rating), 3) for player, rating in sorted(ratings.items())}


def evaluate_ladder(
    candidate: OpponentPolicy,
    *,
    ladder: Sequence[OpponentPolicy] = FROZEN_LADDER,
    games_per_color: int = 10,
    seed: int | None = None,
    game: Game | None = None,
    run_id: str | None = None,
) -> LadderEvaluation:
    if games_per_color < 1:
        raise ValueError("games_per_color must be positive")
    candidate_name = _policy_name(candidate)
    ladder_names = [_policy_name(policy) for policy in ladder]
    if candidate_name in ladder_names:
        raise ValueError("candidate name must be distinct from frozen ladder names")
    game = game or Game()
    master_rng = np.random.default_rng(seed)
    matches = tuple(
        play_match(
            candidate,
            opponent,
            games_per_color=games_per_color,
            seed=int(master_rng.integers(0, np.iinfo(np.int64).max)),
            game=game,
        )
        for opponent in ladder
    )
    ratings = estimate_elos(matches)
    return LadderEvaluation(
        candidate=candidate_name,
        run_id=run_id or f"evaluation-{seed if seed is not None else 'unseeded'}",
        seed=seed,
        ladder_version=LADDER_VERSION,
        matches=matches,
        elo_ratings=ratings,
    )


def _builtin_policy(name: str) -> OpponentPolicy:
    if name == "random":
        return RandomOpponent()
    if name == "greedy-racer":
        return GreedyRacer()
    if name == "heuristic-1":
        return HeuristicOne()
    if name == "alpha-beta-d3":
        return AlphaBetaOpponent(depth=3)
    if name == "alpha-beta-d5":
        return AlphaBetaOpponent(depth=5)
    raise ValueError(f"unknown built-in policy {name!r}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a built-in policy against the frozen ladder.")
    parser.add_argument(
        "--candidate",
        default="greedy-racer",
        choices=("random", "greedy-racer", "heuristic-1", "alpha-beta-d3", "alpha-beta-d5"),
    )
    parser.add_argument("--games-per-color", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output", type=Path, default=None, help="write evaluation JSON to this path")
    parser.add_argument("--dashboard-events", type=Path, default=None, help="append a dashboard JSONL event")
    parser.add_argument(
        "--compact",
        action="store_true",
        help="omit per-game action records from stdout/output JSON",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    candidate = NamedPolicy(_builtin_policy(args.candidate), name=f"candidate-{args.candidate}")
    evaluation = evaluate_ladder(
        candidate,
        ladder=FROZEN_LADDER,
        games_per_color=args.games_per_color,
        seed=args.seed,
        run_id=args.run_id,
    )
    payload = evaluation.to_dict(include_records=not args.compact)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    if args.dashboard_events is not None:
        from .dashboard import evaluation_to_dashboard_event, write_dashboard_event

        write_dashboard_event(args.dashboard_events, evaluation_to_dashboard_event(evaluation))
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
