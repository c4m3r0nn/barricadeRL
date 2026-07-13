from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .evaluate import LadderEvaluation

DASHBOARD_SCHEMA_VERSION = 1

REQUIRED_DASHBOARD_METRICS = (
    "policy_loss",
    "value_loss",
    "auxiliary_loss",
    "root_policy_entropy",
    "value_calibration",
    "avg_game_length",
    "cap_fraction",
    "mean_walls_placed",
    "samples_per_position",
    "games_per_hour",
    "gpu_utilization",
    "ladder_elo",
)


def evaluation_to_dashboard_event(evaluation: LadderEvaluation) -> dict:
    candidate_elo = evaluation.elo_ratings.get(evaluation.candidate)
    return {
        "schema_version": DASHBOARD_SCHEMA_VERSION,
        "kind": "ladder_evaluation",
        "run_id": evaluation.run_id,
        "candidate": evaluation.candidate,
        "seed": evaluation.seed,
        "ladder_version": evaluation.ladder_version,
        "games": evaluation.games,
        "elo_ratings": dict(sorted(evaluation.elo_ratings.items())),
        "metrics": {
            "policy_loss": None,
            "value_loss": None,
            "auxiliary_loss": None,
            "root_policy_entropy": None,
            "value_calibration": None,
            "avg_game_length": evaluation.avg_plies,
            "cap_fraction": evaluation.cap_fraction,
            "mean_walls_placed": evaluation.mean_walls_placed,
            "samples_per_position": None,
            "games_per_hour": None,
            "gpu_utilization": None,
            "ladder_elo": candidate_elo,
        },
        "matches": [
            match.to_dict(include_records=False)
            for match in evaluation.matches
        ],
    }


def write_dashboard_event(path: str | Path, event: Mapping) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def load_dashboard_events(path: str | Path) -> list[dict]:
    source = Path(path)
    if not source.exists():
        return []
    events = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def render_dashboard_html(path_or_events: str | Path | Iterable[Mapping]) -> str:
    if isinstance(path_or_events, (str, Path)):
        events = load_dashboard_events(path_or_events)
    else:
        events = [dict(event) for event in path_or_events]

    rows = []
    for event in events:
        metrics = event.get("metrics", {})
        rows.append(
            "<tr>"
            f"<td>{_cell(event.get('run_id'))}</td>"
            f"<td>{_cell(event.get('candidate'))}</td>"
            f"<td>{_cell(event.get('games'))}</td>"
            f"<td>{_cell(metrics.get('ladder_elo'))}</td>"
            f"<td>{_cell(metrics.get('avg_game_length'))}</td>"
            f"<td>{_cell(metrics.get('cap_fraction'))}</td>"
            f"<td>{_cell(metrics.get('mean_walls_placed'))}</td>"
            "</tr>"
        )

    metric_items = "\n".join(f"<li>{html.escape(metric)}</li>" for metric in REQUIRED_DASHBOARD_METRICS)
    body = "\n".join(rows) if rows else "<tr><td colspan='7'>No evaluation events yet.</td></tr>"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>BarricadeRL training dashboard</title>
  <style>
    body {{ font-family: sans-serif; margin: 2rem; }}
    table {{ border-collapse: collapse; }}
    th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: right; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>BarricadeRL training dashboard</h1>
  <p>Schema version {DASHBOARD_SCHEMA_VERSION}. This M1 skeleton exposes ladder evaluations now and reserves the training metrics required by the handover.</p>
  <table>
    <thead>
      <tr>
        <th>run_id</th>
        <th>candidate</th>
        <th>games</th>
        <th>ladder_elo</th>
        <th>avg_game_length</th>
        <th>cap_fraction</th>
        <th>mean_walls_placed</th>
      </tr>
    </thead>
    <tbody>
      {body}
    </tbody>
  </table>
  <h2>Reserved metrics</h2>
  <ul>
    {metric_items}
  </ul>
</body>
</html>
"""


def write_dashboard_html(events_path: str | Path, output_path: str | Path) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_dashboard_html(events_path), encoding="utf-8")


def _cell(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return html.escape(f"{value:.3f}")
    return html.escape(str(value))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the BarricadeRL dashboard skeleton from JSONL events.")
    parser.add_argument("events", type=Path, help="dashboard JSONL events path")
    parser.add_argument("--output", type=Path, default=None, help="write HTML to this path instead of stdout")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    html_text = render_dashboard_html(args.events)
    if args.output is None:
        print(html_text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(html_text, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
