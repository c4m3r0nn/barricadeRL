from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from .az_gating import (
    GatingConfig,
    GatingResult,
    NetworkMCTSPolicy,
    archive_promoted_checkpoint,
    gate_candidate,
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
    learner_step: int
    self_play_games: int
    generated_positions: int
    replay_size: int
    samples_per_position: float
    candidate_checkpoint: Path
    incumbent_checkpoint: Path
    promoted: bool
    gating: GatingResult

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["candidate_checkpoint"] = str(self.candidate_checkpoint)
        payload["incumbent_checkpoint"] = str(self.incumbent_checkpoint)
        return payload


class AlphaZeroCoordinator:
    """Synchronous M2 cycle with gated rollback semantics.

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
        replay_buffer: AlphaZeroReplayBuffer,
        incumbent_checkpoint: str | Path,
        output_directory: str | Path,
        run_id: str,
        git_commit: str,
        seed: int,
    ) -> None:
        if not run_id or not git_commit:
            raise ValueError("run_id and git_commit are required")
        incumbent = Path(incumbent_checkpoint)
        if not incumbent.is_file():
            raise FileNotFoundError(incumbent)
        self.project_config = dict(project_config)
        self.game = game
        self.learner = learner
        self.replay_buffer = replay_buffer
        self.incumbent_checkpoint = incumbent
        self.output_directory = Path(output_directory)
        self.run_id = run_id
        self.git_commit = git_commit
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.config_hash = calculate_config_hash(project_config)
        self.self_play_config = SelfPlayConfig.from_project_config(project_config)
        self.gating_config = GatingConfig.from_project_config(project_config)

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
        self_play_runner(
            self.game,
            incumbent_network,
            self.self_play_config,
            games=self_play_games,
            replay_buffer=self.replay_buffer,
            run_id=self.run_id,
            config_hash=self.config_hash,
            git_commit=self.git_commit,
            rng=self.rng,
        )
        generated_positions = self.replay_buffer.total_positions_added - positions_before
        if self.replay_buffer.size < self.learner.config.batch_size:
            raise RuntimeError(
                "self-play did not produce enough recorded positions for one learner batch"
            )

        latest_metrics = self.learner.train(
            self.replay_buffer,
            steps=learner_steps,
        )
        step = self.learner.step
        candidate_directory = self.output_directory / "candidates"
        candidate_directory.mkdir(parents=True, exist_ok=True)
        candidate_path = candidate_directory / f"candidate-step-{step:09d}.npz"
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
        gating_seed = int(self.rng.integers(0, np.iinfo(np.int64).max))
        gating_result = gate_runner(
            candidate_policy,
            incumbent_policy,
            game=self.game,
            config=self.gating_config,
            seed=gating_seed,
        )

        gating_directory = self.output_directory / "gating"
        gating_directory.mkdir(parents=True, exist_ok=True)
        gating_path = gating_directory / f"gate-step-{step:09d}.json"
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
        else:
            self.learner = AlphaZeroLearner.load_checkpoint(
                self.incumbent_checkpoint,
                self.game,
                device=self.learner.device,
            )

        replay_path = self.output_directory / "replay.npz"
        self.replay_buffer.save_npz(replay_path)
        result = TrainingCycleResult(
            run_id=self.run_id,
            learner_step=step,
            self_play_games=self_play_games,
            generated_positions=generated_positions,
            replay_size=self.replay_buffer.size,
            samples_per_position=latest_metrics.samples_per_position,
            candidate_checkpoint=candidate_path,
            incumbent_checkpoint=self.incumbent_checkpoint,
            promoted=gating_result.promoted,
            gating=gating_result,
        )
        with (self.output_directory / "cycles.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")
        return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one gated M2 AlphaZero cycle.")
    parser.add_argument("--config", type=Path, default=Path("configs/m2_5x5.json"))
    parser.add_argument("--oracle-corpus", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
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
    learner = AlphaZeroLearner.load_checkpoint(args.incumbent, game, device=args.device)
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
