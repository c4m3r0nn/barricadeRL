from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from .az_gating import (
    GatingConfig,
    GatingResult,
    NetworkMCTSPolicy,
    archive_promoted_checkpoint,
    gate_candidate,
    sample_gating_start_states,
)
from .az_learner import AlphaZeroLearner
from .az_network import AlphaZeroNetwork
from .az_replay import AlphaZeroReplayBuffer
from .az_self_play import SelfPlayConfig, generate_self_play_games
from .config import config_hash as calculate_config_hash
from .config import load_config, small_game_from_config
from .training_readiness import check_training_readiness


@dataclass(frozen=True, slots=True)
class TrainingCycleResult:
    run_id: str
    cycle_index: int
    self_play_seed: int
    learner_step: int
    requested_learner_steps: int
    completed_learner_steps: int
    learner_steps_clamped: bool
    self_play_games: int
    generated_positions: int
    replay_size: int
    samples_per_position: float
    learner_metrics: dict[str, int | float]
    self_play_cap_fraction: float
    high_cap_streak: int
    adjudication_active: bool
    scoring_scheme: str
    learner_input_checkpoint: Path
    candidate_checkpoint: Path
    incumbent_checkpoint: Path
    promoted: bool
    gating: GatingResult

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["learner_input_checkpoint"] = str(self.learner_input_checkpoint)
        payload["candidate_checkpoint"] = str(self.candidate_checkpoint)
        payload["incumbent_checkpoint"] = str(self.incumbent_checkpoint)
        return payload


@dataclass(frozen=True, slots=True)
class CapAdjudicationConfig:
    fraction_threshold: float = 0.05
    consecutive_cycles: int = 3
    scoring_scheme: str = "terminal-win-loss-cap-shortest-path-adjudicated"

    @classmethod
    def from_project_config(cls, config: Mapping) -> "CapAdjudicationConfig":
        payload = config.get("self_play", {}).get("cap_adjudication", {})
        result = cls(
            fraction_threshold=float(payload.get("fraction_threshold", 0.05)),
            consecutive_cycles=int(payload.get("consecutive_cycles", 3)),
            scoring_scheme=str(
                payload.get(
                    "scoring_scheme",
                    "terminal-win-loss-cap-shortest-path-adjudicated",
                )
            ),
        )
        if not 0.0 <= result.fraction_threshold <= 1.0:
            raise ValueError("cap adjudication fraction threshold must be in [0, 1]")
        if result.consecutive_cycles < 1:
            raise ValueError("cap adjudication consecutive cycles must be positive")
        return result


class AlphaZeroCoordinator:
    """Synchronous M2 cycle with a continuous learner and gated self-play.

    This is the correctness-first local coordinator. The later inference-server
    deployment can replace its runners without changing replay, learner, or gate
    contracts.
    """

    def __init__(
        self,
        *,
        project_config: Mapping,
        game,
        learner: AlphaZeroLearner,
        learner_checkpoint: str | Path | None = None,
        replay_buffer: AlphaZeroReplayBuffer,
        incumbent_checkpoint: str | Path,
        output_directory: str | Path,
        run_id: str,
        git_commit: str,
        seed: int,
        cycle_index: int | None = None,
    ) -> None:
        if not run_id or not git_commit:
            raise ValueError("run_id and git_commit are required")
        incumbent = Path(incumbent_checkpoint)
        if not incumbent.is_file():
            raise FileNotFoundError(incumbent)
        self.project_config = dict(project_config)
        self.game = game
        self.learner = learner
        self.learner_checkpoint = (
            incumbent if learner_checkpoint is None else Path(learner_checkpoint)
        )
        if not self.learner_checkpoint.is_file():
            raise FileNotFoundError(self.learner_checkpoint)
        self.replay_buffer = replay_buffer
        self.incumbent_checkpoint = incumbent
        self.output_directory = Path(output_directory)
        self.run_id = run_id
        self.git_commit = git_commit
        self.seed = int(seed)
        self.cycle_index = (
            _next_cycle_index(self.output_directory)
            if cycle_index is None
            else int(cycle_index)
        )
        if self.cycle_index < 0:
            raise ValueError("cycle_index must be non-negative")
        self.self_play_seed = _cycle_seed(self.seed, self.cycle_index, stream=0)
        self.gating_start_seed = _cycle_seed(self.seed, self.cycle_index, stream=1)
        self.gating_seed = _cycle_seed(self.seed, self.cycle_index, stream=2)
        self.rng = np.random.default_rng(self.self_play_seed)
        self.config_hash = calculate_config_hash(project_config)
        self.self_play_config = SelfPlayConfig.from_project_config(project_config)
        self.cap_adjudication_config = CapAdjudicationConfig.from_project_config(
            project_config
        )
        self.prior_high_cap_streak = _high_cap_streak(
            self.output_directory,
            threshold=self.cap_adjudication_config.fraction_threshold,
        )
        self.adjudication_active = (
            _adjudication_was_activated(
                self.output_directory,
                scoring_scheme=self.cap_adjudication_config.scoring_scheme,
            )
            or self.prior_high_cap_streak
            >= self.cap_adjudication_config.consecutive_cycles
        )
        if self.adjudication_active:
            self.self_play_config = replace(
                self.self_play_config,
                scoring_scheme=self.cap_adjudication_config.scoring_scheme,
            )
        self.gating_config = GatingConfig.from_project_config(project_config)
        self.gating_start_states = sample_gating_start_states(
            game,
            count=self.gating_config.games_per_color,
            seed=self.gating_start_seed,
            min_plies=self.gating_config.start_min_plies,
            max_plies=self.gating_config.start_max_plies,
        )

    def run_cycle(
        self,
        *,
        self_play_games: int,
        learner_steps: int,
        self_play_runner: Callable = generate_self_play_games,
        gate_runner: Callable = gate_candidate,
    ) -> TrainingCycleResult:
        if self_play_games < 1 or learner_steps < 1:
            raise ValueError("self_play_games and learner_steps must be positive")
        incumbent_network = AlphaZeroNetwork.load_checkpoint(self.incumbent_checkpoint)
        positions_before = self.replay_buffer.total_positions_added
        self_play_records = self_play_runner(
            self.game,
            incumbent_network,
            self.self_play_config,
            games=self_play_games,
            replay_buffer=self.replay_buffer,
            run_id=self.run_id,
            config_hash=self.config_hash,
            git_commit=self.git_commit,
            rng=self.rng,
            game_id_prefix=(
                f"{self.run_id}-cycle-{self.cycle_index:06d}"
            ),
        )
        self_play_cap_fraction = (
            sum(bool(getattr(record, "capped", False)) for record in self_play_records)
            / len(self_play_records)
            if self_play_records
            else 0.0
        )
        high_cap_streak = (
            self.prior_high_cap_streak + 1
            if self_play_cap_fraction > self.cap_adjudication_config.fraction_threshold
            else 0
        )
        generated_positions = self.replay_buffer.total_positions_added - positions_before
        if self.replay_buffer.size < self.learner.config.batch_size:
            raise RuntimeError(
                "self-play did not produce enough recorded positions for one learner batch"
            )

        remaining_gradient_samples = (
            self.learner.config.target_samples_per_position_max
            * self.replay_buffer.total_positions_added
            - self.replay_buffer.gradient_samples_consumed
        )
        maximum_learner_steps = max(
            0,
            int(remaining_gradient_samples // self.learner.config.batch_size),
        )
        completed_learner_steps = min(learner_steps, maximum_learner_steps)
        if completed_learner_steps < 1:
            raise RuntimeError(
                "self-play did not create enough replay headroom for one learner step"
            )
        latest_metrics = self.learner.train(
            self.replay_buffer,
            steps=completed_learner_steps,
        )
        step = self.learner.step
        candidate_directory = self.output_directory / "candidates"
        candidate_directory.mkdir(parents=True, exist_ok=True)
        candidate_path = candidate_directory / (
            f"candidate-cycle-{self.cycle_index:06d}-step-{step:09d}.npz"
        )
        self.learner.save_checkpoint(
            candidate_path,
            run_id=self.run_id,
            git_commit=self.git_commit,
            config_hash=self.config_hash,
        )

        candidate_policy = NetworkMCTSPolicy(
            self.learner.network,
            simulations=self.gating_config.evaluation_simulations,
            cpuct=self.gating_config.cpuct,
            name=f"candidate-{step}",
        )
        incumbent_policy = NetworkMCTSPolicy(
            incumbent_network,
            simulations=self.gating_config.evaluation_simulations,
            cpuct=self.gating_config.cpuct,
            name=f"incumbent-{incumbent_network.metadata.get('step', 0)}",
        )
        gating_result = gate_runner(
            candidate_policy,
            incumbent_policy,
            game=self.game,
            config=self.gating_config,
            seed=self.gating_seed,
            initial_states=self.gating_start_states,
            start_seed=self.gating_start_seed,
        )

        gating_directory = self.output_directory / "gating"
        gating_directory.mkdir(parents=True, exist_ok=True)
        gating_path = gating_directory / (
            f"gate-cycle-{self.cycle_index:06d}-step-{step:09d}.json"
        )
        gating_path.write_text(
            json.dumps(gating_result.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if gating_result.promoted:
            self.incumbent_checkpoint = archive_promoted_checkpoint(
                candidate_path,
                self.output_directory / "gated",
                gating_result,
                run_id=self.run_id,
                git_commit=self.git_commit,
                config_hash=self.config_hash,
                step=step,
            )

        replay_path = self.output_directory / "replay.npz"
        self.replay_buffer.save_npz(replay_path)
        result = TrainingCycleResult(
            run_id=self.run_id,
            cycle_index=self.cycle_index,
            self_play_seed=self.self_play_seed,
            learner_step=step,
            requested_learner_steps=learner_steps,
            completed_learner_steps=completed_learner_steps,
            learner_steps_clamped=completed_learner_steps < learner_steps,
            self_play_games=self_play_games,
            generated_positions=generated_positions,
            replay_size=self.replay_buffer.size,
            samples_per_position=latest_metrics.samples_per_position,
            learner_metrics=latest_metrics.to_dict(),
            self_play_cap_fraction=self_play_cap_fraction,
            high_cap_streak=high_cap_streak,
            adjudication_active=self.adjudication_active,
            scoring_scheme=self.self_play_config.scoring_scheme,
            learner_input_checkpoint=self.learner_checkpoint,
            candidate_checkpoint=candidate_path,
            incumbent_checkpoint=self.incumbent_checkpoint,
            promoted=gating_result.promoted,
            gating=gating_result,
        )
        with (self.output_directory / "cycles.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")
        return result


def _cycle_seed(base_seed: int, cycle_index: int, *, stream: int) -> int:
    sequence = np.random.SeedSequence([int(base_seed), int(cycle_index), int(stream)])
    return int(sequence.generate_state(1, dtype=np.uint64)[0] % np.iinfo(np.int64).max)


def _cycle_records(output_directory: str | Path) -> tuple[dict, ...]:
    path = Path(output_directory) / "cycles.jsonl"
    if not path.exists():
        return ()
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _next_cycle_index(output_directory: str | Path) -> int:
    records = _cycle_records(output_directory)
    if not records:
        return 0
    return max(int(record.get("cycle_index", index)) for index, record in enumerate(records)) + 1


def _high_cap_streak(output_directory: str | Path, *, threshold: float) -> int:
    streak = 0
    for record in reversed(_cycle_records(output_directory)):
        if float(record.get("self_play_cap_fraction", 0.0)) <= threshold:
            break
        streak += 1
    return streak


def _adjudication_was_activated(
    output_directory: str | Path,
    *,
    scoring_scheme: str,
) -> bool:
    return any(
        bool(record.get("adjudication_active", False))
        or record.get("scoring_scheme") == scoring_scheme
        for record in _cycle_records(output_directory)
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one gated M2 AlphaZero cycle.")
    parser.add_argument("--config", type=Path, default=Path("configs/m2_5x5.json"))
    parser.add_argument("--oracle-corpus", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--learner-checkpoint", type=Path, default=None)
    parser.add_argument("--replay", type=Path, default=None)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--self-play-games", type=int, required=True)
    parser.add_argument("--learner-steps", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    readiness = check_training_readiness(
        args.config,
        oracle_corpus=args.oracle_corpus,
    )
    if not readiness.ready:
        raise RuntimeError("training readiness failed: " + "; ".join(readiness.blockers))
    project_config = load_config(args.config)
    game = small_game_from_config(project_config)
    learner_checkpoint = args.learner_checkpoint or args.incumbent
    learner = AlphaZeroLearner.load_checkpoint(
        learner_checkpoint,
        game,
        device=args.device,
    )
    if args.replay is not None and args.replay.exists():
        replay = AlphaZeroReplayBuffer.load_npz(args.replay)
    else:
        state = game.initial_state()
        replay = AlphaZeroReplayBuffer(
            capacity=int(project_config["replay"]["positions"]),
            observation_shape=game.canonical_observation(state).shape,
            action_count=game.action_count,
        )
    coordinator = AlphaZeroCoordinator(
        project_config=project_config,
        game=game,
        learner=learner,
        learner_checkpoint=learner_checkpoint,
        replay_buffer=replay,
        incumbent_checkpoint=args.incumbent,
        output_directory=args.output_directory,
        run_id=args.run_id,
        git_commit=args.git_commit,
        seed=args.seed,
    )
    result = coordinator.run_cycle(
        self_play_games=args.self_play_games,
        learner_steps=args.learner_steps,
    )
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
