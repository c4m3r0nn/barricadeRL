import pytest


pytest.importorskip("sb3_contrib")


def test_maskable_ppo_smoke_learns_a_few_steps(tmp_path):
    from barricade_rl.train_maskable_ppo import build_model, make_training_env

    env = make_training_env("random")
    model = build_model(env, seed=0, tensorboard_log=tmp_path, n_steps=8, batch_size=4, verbose=0)
    model.learn(total_timesteps=8, progress_bar=False)


def test_metrics_callback_writes_jsonl(tmp_path):
    from barricade_rl.experiments import read_jsonl
    from barricade_rl.train_maskable_ppo import MetricsJsonlCallback, build_model, make_training_env

    env = make_training_env("random")
    model = build_model(env, seed=0, tensorboard_log=tmp_path, n_steps=8, batch_size=4, verbose=0)
    metrics_path = tmp_path / "metrics.jsonl"
    model.learn(total_timesteps=8, callback=MetricsJsonlCallback(metrics_path), progress_bar=False)

    rows = read_jsonl(metrics_path)
    assert rows
    assert rows[-1]["timesteps"] >= 8
    assert "fps" in rows[-1]
