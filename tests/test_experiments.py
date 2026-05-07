from pathlib import Path

from barricade_rl.evaluate import EvaluationResult
from barricade_rl.experiments import ExperimentSpec, build_report, build_train_command, read_jsonl, write_jsonl


def test_build_train_command_includes_core_options(tmp_path: Path):
    spec = ExperimentSpec(
        name="mixed_shaped",
        timesteps=1024,
        opponent="mixed",
        seed=7,
        shaped_reward=True,
        checkpoint_opponents=["runs/base/best/*.zip"],
    )

    command = build_train_command(spec, tmp_path)

    assert "--timesteps" in command
    assert "1024" in command
    assert "--opponent" in command
    assert "mixed" in command
    assert "--shaped-reward" in command
    assert "--checkpoint-opponents" in command
    assert "runs/base/best/*.zip" in command


def test_jsonl_round_trip(tmp_path: Path):
    path = tmp_path / "metrics.jsonl"
    rows = [{"timesteps": 1, "reward": 0.5}, {"timesteps": 2, "reward": 1.0}]

    write_jsonl(path, rows)

    assert read_jsonl(path) == rows


def test_build_report_contains_config_metrics_and_eval(tmp_path: Path):
    spec = ExperimentSpec(name="run", timesteps=128, opponent="random", seed=1)
    metrics = [{"timesteps": 128, "ep_rew_mean": 0.25}]
    result = EvaluationResult(
        episodes=2,
        wins=1,
        losses=1,
        truncations=0,
        total_steps=20,
        episode_lengths=[8, 12],
        learner_walls_placed=[1, 2],
        opponent_walls_placed=[3, 4],
    )

    report = build_report(spec, metrics, result)

    assert report["spec"]["name"] == "run"
    assert report["latest_metrics"]["ep_rew_mean"] == 0.25
    assert report["evaluation"]["win_rate"] == 0.5
    assert report["evaluation"]["avg_walls_placed"] == 5.0
