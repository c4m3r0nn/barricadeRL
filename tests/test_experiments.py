from pathlib import Path

from barricade_rl.evaluate import EvaluationResult
from barricade_rl.experiments import (
    ExperimentSpec,
    available_replays,
    build_report,
    build_train_command,
    experiment_presets,
    read_jsonl,
    save_graph_svg,
    write_jsonl,
)


def test_build_train_command_includes_core_options(tmp_path: Path):
    spec = ExperimentSpec(
        name="mixed_shaped",
        timesteps=1024,
        opponent="mixed",
        seed=7,
        shaped_reward=True,
        checkpoint_opponents=["runs/base/best/*.zip"],
        policy="cnn",
        self_play=True,
        self_play_save_freq=2048,
        randomize_learner_side=True,
        checkpoint_probability=0.55,
        eval_opponents=["random", "mixed"],
        eval_episodes=12,
        scripted_eval_freq=4096,
    )

    command = build_train_command(spec, tmp_path)

    assert "--timesteps" in command
    assert "1024" in command
    assert "--opponent" in command
    assert "mixed" in command
    assert "--shaped-reward" in command
    assert "--checkpoint-opponents" in command
    assert "runs/base/best/*.zip" in command
    assert "--policy" in command
    assert "cnn" in command
    assert "--self-play" in command
    assert "--self-play-save-freq" in command
    assert "2048" in command
    assert "--randomize-learner-side" in command
    assert "--checkpoint-probability" in command
    assert "0.55" in command
    assert "--eval-opponents" in command
    assert "random" in command
    assert "--eval-episodes" in command
    assert "12" in command
    assert "--scripted-eval-freq" in command
    assert "4096" in command


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


def test_experiment_presets_include_named_training_modes():
    presets = experiment_presets(timesteps=25_000, seed=10, checkpoint_glob="runs/base/*.zip")

    assert list(presets) == ["random", "mixed", "mixed + shaped reward", "checkpoint pool", "cnn self-play"]
    assert presets["random"].opponent == "random"
    assert presets["mixed"].opponent == "mixed"
    assert presets["mixed + shaped reward"].shaped_reward is True
    assert presets["checkpoint pool"].checkpoint_opponents == ["runs/base/*.zip"]
    assert presets["checkpoint pool"].seed == 13
    assert presets["cnn self-play"].policy == "cnn"
    assert presets["cnn self-play"].opponent == "curriculum"
    assert presets["cnn self-play"].self_play is True
    assert presets["cnn self-play"].randomize_learner_side is True
    assert presets["cnn self-play"].checkpoint_probability == 0.60


def test_save_graph_svg_writes_selected_metrics(tmp_path: Path):
    rows = [
        {"timesteps": 1, "ep_rew_mean": -1.0, "train_loss": 0.7},
        {"timesteps": 2, "ep_rew_mean": 0.0, "train_loss": 0.4},
        {"timesteps": 3, "ep_rew_mean": 1.0, "train_loss": 0.2},
    ]

    graph_path = save_graph_svg(tmp_path, rows, ["ep_rew_mean", "train_loss"])

    assert graph_path == tmp_path / "graphs" / "metrics.svg"
    content = graph_path.read_text(encoding="utf-8")
    assert "<svg" in content
    assert "ep_rew_mean" in content
    assert "train_loss" in content


def test_available_replays_sorts_milestones_numerically(tmp_path: Path):
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    for name in ["replay_5000.json", "replay_10000.json", "manual_replay.json", "replay_25000.json"]:
        (replay_dir / name).write_text("{}", encoding="utf-8")

    replays = available_replays(tmp_path)

    assert [path.name for path in replays] == [
        "replay_5000.json",
        "replay_10000.json",
        "replay_25000.json",
        "manual_replay.json",
    ]
