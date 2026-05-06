import pytest


pytest.importorskip("sb3_contrib")


def test_maskable_ppo_smoke_learns_a_few_steps(tmp_path):
    from barricade_rl.train_maskable_ppo import build_model, make_training_env

    env = make_training_env("random")
    model = build_model(env, seed=0, tensorboard_log=tmp_path, n_steps=8, batch_size=4, verbose=0)
    model.learn(total_timesteps=8, progress_bar=False)
