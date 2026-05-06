from __future__ import annotations

import argparse
from pathlib import Path

from barricade_rl.opponents import make_opponent
from barricade_rl.single_agent import BarricadeSingleAgentEnv


def make_training_env(opponent_name: str):
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.monitor import Monitor

    def mask_fn(env):
        return env.action_masks()

    return Monitor(ActionMasker(BarricadeSingleAgentEnv(opponent=make_opponent(opponent_name)), mask_fn))


def build_model(env, seed: int, tensorboard_log: Path | str, n_steps: int = 512, batch_size: int = 128, verbose: int = 1):
    from sb3_contrib import MaskablePPO

    return MaskablePPO(
        "MlpPolicy",
        env,
        verbose=verbose,
        seed=seed,
        tensorboard_log=str(tensorboard_log),
        n_steps=n_steps,
        batch_size=batch_size,
    )


def main():
    parser = argparse.ArgumentParser(description="Smoke-train MaskablePPO on Barricade.")
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--opponent", choices=["random", "greedy"], default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/maskable_ppo_barricade"))
    args = parser.parse_args()

    try:
        from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
    except ImportError as exc:
        raise SystemExit("Install RL dependencies first: .venv/bin/python -m pip install -e '.[dev,rl]'") from exc

    args.out.mkdir(parents=True, exist_ok=True)
    train_env = make_training_env(args.opponent)
    eval_env = make_training_env(args.opponent)
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(args.out / "best"),
        log_path=str(args.out / "eval"),
        eval_freq=max(1_000, args.timesteps // 10),
        deterministic=True,
        render=False,
    )
    model = build_model(train_env, seed=args.seed, tensorboard_log=args.out / "tb")
    model.learn(total_timesteps=args.timesteps, callback=eval_callback, progress_bar=False)
    model.save(args.out / "final_model")
    print(f"Saved model to {args.out / 'final_model.zip'}")


if __name__ == "__main__":
    main()
