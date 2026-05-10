from __future__ import annotations

import argparse
import html
import json
import re
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
    policy: str = "mlp"
    self_play: bool = False
    self_play_save_freq: int = 10_000
    randomize_learner_side: bool = False
    checkpoint_probability: float = 0.60
    eval_opponents: list[str] = field(default_factory=lambda: ["random", "greedy", "mixed", "anti_rush"])
    eval_episodes: int = 10
    scripted_eval_freq: int | None = None


GRAPH_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b"]
REPLAY_NAME_PATTERN = re.compile(r"^replay_(\d+)(?:_p[01])?\.json$")


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
        "--policy",
        spec.policy,
    ]
    if spec.shaped_reward:
        command.append("--shaped-reward")
    if spec.checkpoint_opponents:
        command.append("--checkpoint-opponents")
        command.extend(spec.checkpoint_opponents)
    if spec.self_play:
        command.append("--self-play")
        command.extend(["--self-play-save-freq", str(spec.self_play_save_freq)])
    if spec.randomize_learner_side:
        command.append("--randomize-learner-side")
    if spec.self_play:
        command.extend(["--checkpoint-probability", str(spec.checkpoint_probability)])
    if spec.eval_opponents:
        command.append("--eval-opponents")
        command.extend(spec.eval_opponents)
    else:
        command.append("--eval-opponents")
    command.extend(["--eval-episodes", str(spec.eval_episodes)])
    if spec.scripted_eval_freq is not None:
        command.extend(["--scripted-eval-freq", str(spec.scripted_eval_freq)])
    return command


def experiment_presets(
    timesteps: int = 25_000,
    seed: int = 0,
    checkpoint_glob: str = "runs/maskable_ppo_barricade/best/*.zip",
) -> dict[str, ExperimentSpec]:
    return {
        "random": ExperimentSpec(name="random", timesteps=timesteps, opponent="random", seed=seed),
        "mixed": ExperimentSpec(name="mixed", timesteps=timesteps, opponent="mixed", seed=seed + 1),
        "mixed + shaped reward": ExperimentSpec(
            name="mixed_shaped",
            timesteps=timesteps,
            opponent="mixed",
            seed=seed + 2,
            shaped_reward=True,
        ),
        "checkpoint pool": ExperimentSpec(
            name="checkpoint_pool",
            timesteps=timesteps,
            opponent="mixed",
            seed=seed + 3,
            checkpoint_opponents=[checkpoint_glob],
        ),
        "cnn self-play": ExperimentSpec(
            name="cnn_self_play",
            timesteps=timesteps,
            opponent="curriculum",
            seed=seed + 4,
            checkpoint_opponents=[checkpoint_glob],
            policy="cnn",
            self_play=True,
            self_play_save_freq=10_000,
            randomize_learner_side=True,
            checkpoint_probability=0.60,
        ),
    }


def replay_sort_key(path: Path) -> tuple[int, str]:
    match = REPLAY_NAME_PATTERN.match(path.name)
    if match:
        return int(match.group(1)), path.name
    return sys.maxsize, path.name


def available_replays(run_dir: Path) -> list[Path]:
    replay_dir = run_dir / "replays"
    if not replay_dir.exists():
        return []
    replays = list(replay_dir.glob("*.json"))
    return sorted(replays, key=replay_sort_key)


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
    presets = experiment_presets(timesteps=timesteps, seed=seed)
    return [presets["random"], presets["mixed"], presets["mixed + shaped reward"]]


def save_graph_svg(run_dir: Path, rows: list[dict], metrics: list[str], filename: str = "metrics.svg") -> Path:
    graph_dir = run_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    path = graph_dir / filename
    width = 900
    height = 420
    pad_left = 64
    pad_right = 28
    pad_top = 58
    pad_bottom = 54
    selected = [metric for metric in metrics if any(metric in row for row in rows)]

    def write_empty(message: str) -> None:
        path.write_text(
            "\n".join(
                [
                    f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                    '<rect width="100%" height="100%" fill="#fbfaf7"/>',
                    f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="16" fill="#555">{html.escape(message)}</text>',
                    "</svg>",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    if len(rows) < 2 or not selected:
        write_empty("Waiting for metrics")
        return path

    all_xs = [float(row["timesteps"]) for row in rows if "timesteps" in row]
    if not all_xs:
        write_empty("Waiting for timesteps")
        return path
    min_x, max_x = min(all_xs), max(all_xs)
    if min_x == max_x:
        min_x -= 1
        max_x += 1

    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    def scale_x(value: float) -> float:
        return pad_left + (value - min_x) / (max_x - min_x) * plot_w

    def scale_y(value: float, min_y: float, max_y: float) -> float:
        return pad_top + (1 - (value - min_y) / (max_y - min_y)) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        '<g font-family="Helvetica, Arial, sans-serif" fill="#232323">',
        '<text x="20" y="28" font-size="18" font-weight="700">Barricade RL metrics</text>',
        '<text x="20" y="48" font-size="12" fill="#666">Series are scaled independently so reward, FPS, and losses can be compared together.</text>',
        "</g>",
        f'<line x1="{pad_left}" y1="{height - pad_bottom}" x2="{width - pad_right}" y2="{height - pad_bottom}" stroke="#c9c1b4"/>',
        f'<line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{height - pad_bottom}" stroke="#c9c1b4"/>',
        f'<text x="{width - pad_right}" y="{height - 18}" text-anchor="end" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#555">{int(max_x)}</text>',
        f'<text x="{pad_left}" y="{height - 18}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#555">{int(min_x)}</text>',
    ]

    for index, metric in enumerate(selected):
        points = [(float(row["timesteps"]), float(row[metric])) for row in rows if "timesteps" in row and metric in row]
        if len(points) < 2:
            continue
        ys = [point[1] for point in points]
        min_y, max_y = min(ys), max(ys)
        if min_y == max_y:
            min_y -= 1
            max_y += 1
        coords = " ".join(f"{scale_x(x):.1f},{scale_y(y, min_y, max_y):.1f}" for x, y in points)
        color = GRAPH_COLORS[index % len(GRAPH_COLORS)]
        safe_metric = html.escape(metric)
        legend_y = 76 + index * 20
        lines.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        lines.append(f'<line x1="20" y1="{legend_y - 4}" x2="42" y2="{legend_y - 4}" stroke="{color}" stroke-width="3"/>')
        lines.append(
            f'<text x="50" y="{legend_y}" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#333">{safe_metric} ({min_y:.3g} to {max_y:.3g})</text>'
        )

    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


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
