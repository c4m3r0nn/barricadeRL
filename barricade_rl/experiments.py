from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from barricade_rl.evaluate import EvaluationResult


@dataclass(slots=True)
class ExperimentSpec:
    name: str
    timesteps: int = 10_000
    opponent: str = "random"
    seed: int = 0
    shaped_reward: bool = False
    checkpoint_opponents: list[str] = field(default_factory=list)
    replay_freq: int = 1_000


def experiment_dir(root: Path, spec: ExperimentSpec) -> Path:
    return root / spec.name


def build_train_command(spec: ExperimentSpec, root: Path) -> list[str]:
    out_dir = experiment_dir(root, spec)
    command = [
        sys.executable,
        "-m",
        "barricade_rl.train_maskable_ppo",
        "--timesteps",
        str(spec.timesteps),
        "--opponent",
        spec.opponent,
        "--seed",
        str(spec.seed),
        "--out",
        str(out_dir),
        "--replay-freq",
        str(spec.replay_freq),
    ]
    if spec.shaped_reward:
        command.append("--shaped-reward")
    if spec.checkpoint_opponents:
        command.append("--checkpoint-opponents")
        command.extend(spec.checkpoint_opponents)
    return command


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def default_experiments(timesteps: int, seed: int) -> list[ExperimentSpec]:
    return [
        ExperimentSpec(name="random", timesteps=timesteps, opponent="random", seed=seed),
        ExperimentSpec(name="mixed", timesteps=timesteps, opponent="mixed", seed=seed + 1),
        ExperimentSpec(name="mixed_shaped", timesteps=timesteps, opponent="mixed", seed=seed + 2, shaped_reward=True),
    ]


def save_spec(out_dir: Path, spec: ExperimentSpec) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "experiment.json").write_text(json.dumps(asdict(spec), indent=2), encoding="utf-8")


def evaluation_to_dict(result: EvaluationResult) -> dict:
    return {
        "episodes": result.episodes,
        "wins": result.wins,
        "losses": result.losses,
        "truncations": result.truncations,
        "win_rate": result.win_rate,
        "loss_rate": result.loss_rate,
        "truncation_rate": result.truncation_rate,
        "avg_learner_steps": result.avg_learner_steps,
        "min_learner_steps": result.min_learner_steps,
        "max_learner_steps": result.max_learner_steps,
        "avg_walls_placed": result.avg_walls_placed,
        "avg_learner_walls_placed": result.avg_learner_walls_placed,
        "avg_opponent_walls_placed": result.avg_opponent_walls_placed,
    }


def build_report(spec: ExperimentSpec, metrics: list[dict], evaluation: EvaluationResult | None = None) -> dict:
    report = {
        "spec": asdict(spec),
        "latest_metrics": metrics[-1] if metrics else {},
        "metrics_count": len(metrics),
    }
    if evaluation is not None:
        report["evaluation"] = evaluation_to_dict(evaluation)
    return report


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def run_experiments(specs: list[ExperimentSpec], root: Path) -> int:
    for spec in specs:
        out_dir = experiment_dir(root, spec)
        save_spec(out_dir, spec)
        completed = subprocess.run(build_train_command(spec, root), check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


def main():
    parser = argparse.ArgumentParser(description="Run a small suite of Barricade RL experiments.")
    parser.add_argument("--root", type=Path, default=Path("runs/experiments"))
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    raise SystemExit(run_experiments(default_experiments(args.timesteps, args.seed), args.root))


if __name__ == "__main__":
    main()
