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


def test_periodic_scripted_eval_callback_writes_win_rates(tmp_path):
    from barricade_rl.experiments import read_jsonl
    from barricade_rl.train_maskable_ppo import PeriodicScriptedEvalCallback, build_model, make_training_env

    env = make_training_env("random")
    model = build_model(env, seed=0, tensorboard_log=tmp_path, n_steps=8, batch_size=4, verbose=0)
    metrics_path = tmp_path / "metrics.jsonl"
    callback = PeriodicScriptedEvalCallback(metrics_path, opponents=["random"], episodes=1, eval_freq=1, seed=0).callback

    model.learn(total_timesteps=8, callback=callback, progress_bar=False)

    rows = read_jsonl(metrics_path)
    assert rows
    assert "eval_random_p0_win_rate" in rows[-1]
    assert "eval_random_p1_win_rate" in rows[-1]
    assert "eval_balanced_win_rate" in rows[-1]
    assert rows[-1]["eval_random_p0_episodes"] == 1


def test_cnn_model_smoke_learns_a_few_steps(tmp_path):
    from barricade_rl.train_maskable_ppo import build_model, make_training_env

    env = make_training_env("random")
    model = build_model(env, seed=0, tensorboard_log=tmp_path, n_steps=8, batch_size=4, verbose=0, policy="cnn")
    model.learn(total_timesteps=8, progress_bar=False)


def test_build_self_play_training_env_uses_refreshing_pool(tmp_path):
    from barricade_rl.opponents import RefreshingCheckpointPoolOpponent
    from barricade_rl.train_maskable_ppo import build_training_opponent

    opponent = build_training_opponent(
        "curriculum",
        checkpoint_patterns=[],
        self_play_patterns=[str(tmp_path / "self_play_pool" / "*.zip")],
        checkpoint_probability=0.6,
    )

    assert isinstance(opponent, RefreshingCheckpointPoolOpponent)
    assert opponent.checkpoint_probability == 0.6


def test_randomize_learner_side_training_env_can_reset_on_both_sides():
    from barricade_rl.train_maskable_ppo import make_training_env

    env = make_training_env("random", randomize_learner_side=True)
    sides = set()
    for seed in range(10):
        obs, info = env.reset(seed=seed)
        sides.add(info["learner_side"])

    assert sides == {0, 1}
