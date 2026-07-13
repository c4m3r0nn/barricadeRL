from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from .az_network import AlphaZeroNetwork
from .config import config_hash as calculate_config_hash
from .config import load_config, small_game_from_config
from .evaluate import MatchResult, play_match
from .mcts import MCTS, MCTSConfig


@dataclass(frozen=True, slots=True)
class GatingConfig:
    games: int = 200
    promotion_threshold: float = 0.55
    evaluation_simulations: int = 800
    cpuct: float = 1.6

    def __post_init__(self) -> None:
        if self.games < 2 or self.games % 2:
            raise ValueError("gating games must be a positive even number")
        if not 0.5 <= self.promotion_threshold <= 1.0:
            raise ValueError("promotion_threshold must be in [0.5, 1]")
        if self.evaluation_simulations < 1:
            raise ValueError("evaluation_simulations must be positive")
        if self.cpuct <= 0:
            raise ValueError("cpuct must be positive")

    @property
    def games_per_color(self) -> int:
        return self.games // 2

    @classmethod
    def from_project_config(cls, config: Mapping) -> "GatingConfig":
        gating = config["gating"]
        mcts = config["mcts"]
        if not bool(gating["enabled"]):
            raise ValueError("checkpoint gating is disabled in the project config")
        return cls(
            games=int(gating["games"]),
            promotion_threshold=float(gating["promotion_threshold"]),
            evaluation_simulations=int(mcts["evaluation_simulations"]),
            cpuct=float(mcts["cpuct_init"]),
        )


@dataclass(frozen=True, slots=True)
class GatingResult:
    candidate: str
    incumbent: str
    games: int
    games_per_color: int
    candidate_wins: int
    incumbent_wins: int
    draws: int
    candidate_score: float
    score_rate: float
    promotion_threshold: float
    promoted: bool
    seed: int
    avg_plies: float
    cap_fraction: float

    def to_dict(self) -> dict:
        return asdict(self)


class NetworkMCTSPolicy:
    def __init__(
        self,
        network: AlphaZeroNetwork,
        *,
        simulations: int,
        cpuct: float,
        name: str,
    ) -> None:
        self.network = network
        self.name = name
        self.search_config = MCTSConfig(
            simulations=simulations,
            cpuct=cpuct,
            temperature=0.0,
            root_noise_fraction=0.0,
            forced_playouts=False,
            policy_target_pruning=False,
        )

    def select_action(self, game, state, rng: np.random.Generator) -> int:
        del rng
        return MCTS(self.search_config, self.network).run(game, state).action


MatchRunner = Callable[..., MatchResult]


def gate_candidate(
    candidate,
    incumbent,
    *,
    game,
    config: GatingConfig,
    seed: int,
    match_runner: MatchRunner = play_match,
) -> GatingResult:
    match = match_runner(
        candidate,
        incumbent,
        games_per_color=config.games_per_color,
        seed=seed,
        game=game,
    )
    if match.games != config.games:
        raise RuntimeError(
            f"gating match returned {match.games} games; expected {config.games}"
        )
    promoted = match.score_rate >= config.promotion_threshold
    return GatingResult(
        candidate=match.candidate,
        incumbent=match.opponent,
        games=match.games,
        games_per_color=match.games_per_color,
        candidate_wins=match.candidate_wins,
        incumbent_wins=match.opponent_wins,
        draws=match.draws,
        candidate_score=match.candidate_score,
        score_rate=match.score_rate,
        promotion_threshold=config.promotion_threshold,
        promoted=promoted,
        seed=int(seed),
        avg_plies=match.avg_plies,
        cap_fraction=match.cap_fraction,
    )


def archive_promoted_checkpoint(
    candidate_path: str | Path,
    gated_directory: str | Path,
    result: GatingResult,
    *,
    run_id: str,
    git_commit: str,
    config_hash: str,
    step: int,
) -> Path:
    if not result.promoted:
        raise ValueError("a rejected checkpoint cannot be archived as gated")
    source = Path(candidate_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    destination_directory = Path(gated_directory)
    destination_directory.mkdir(parents=True, exist_ok=True)
    destination = destination_directory / f"gated-step-{step:09d}.npz"
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copy2(source, destination)
    manifest_record = {
        "schema_version": 1,
        "checkpoint": destination.name,
        "step": int(step),
        "run_id": run_id,
        "git_commit": git_commit,
        "config_hash": config_hash,
        **result.to_dict(),
    }
    with (destination_directory / "manifest.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest_record, sort_keys=True) + "\n")
    return destination


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate an AlphaZero candidate checkpoint.")
    parser.add_argument("--config", type=Path, default=Path("configs/m2_5x5.json"))
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--gated-directory", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    project_config = load_config(args.config)
    game = small_game_from_config(project_config)
    gating_config = GatingConfig.from_project_config(project_config)
    candidate_network = AlphaZeroNetwork.load_checkpoint(args.candidate)
    incumbent_network = AlphaZeroNetwork.load_checkpoint(args.incumbent)
    candidate = NetworkMCTSPolicy(
        candidate_network,
        simulations=gating_config.evaluation_simulations,
        cpuct=gating_config.cpuct,
        name=f"candidate-{candidate_network.metadata.get('step', 'unknown')}",
    )
    incumbent = NetworkMCTSPolicy(
        incumbent_network,
        simulations=gating_config.evaluation_simulations,
        cpuct=gating_config.cpuct,
        name=f"incumbent-{incumbent_network.metadata.get('step', 'unknown')}",
    )
    result = gate_candidate(
        candidate,
        incumbent,
        game=game,
        config=gating_config,
        seed=args.seed,
    )
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n")
    if result.promoted:
        archive_promoted_checkpoint(
            args.candidate,
            args.gated_directory,
            result,
            run_id=args.run_id,
            git_commit=args.git_commit,
            config_hash=calculate_config_hash(project_config),
            step=int(candidate_network.metadata.get("step", 0)),
        )
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0 if result.promoted else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
