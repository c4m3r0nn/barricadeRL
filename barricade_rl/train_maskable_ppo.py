from __future__ import annotations

import argparse
from pathlib import Path

from barricade_rl.opponents import make_opponent
from barricade_rl.single_agent import BarricadeSingleAgentEnv


def main():
    parser = argparse.ArgumentParser(description="Smoke-train MaskablePPO on Barricade.")
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--opponent", choices=["random", "greedy"], default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/maskable_ppo_barricade"))
    args = parser.parse_args()

    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise SystemExit("Install RL dependencies first: .venv/bin/python -m pip install -e '.[dev,rl]'") from exc

    def mask_fn(env):
        return env.action_masks()

    args.out.mkdir(parents=True, exist_ok=True)
    train_env = Monitor(ActionMasker(BarricadeSingleAgentEnv(opponent=make_opponent(args.opponent)), mask_fn))
    eval_env = Monitor(ActionMasker(BarricadeSingleAgentEnv(opponent=make_opponent(args.opponent)), mask_fn))
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(args.out / "best"),
        log_path=str(args.out / "eval"),
        eval_freq=max(1_000, args.timesteps // 10),
        deterministic=True,
        render=False,
    )
    model = MaskablePPO(
        "CnnPolicy",
        train_env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(args.out / "tb"),
        n_steps=512,
        batch_size=128,
    )
    model.learn(total_timesteps=args.timesteps, callback=eval_callback, progress_bar=False)
    model.save(args.out / "final_model")
    print(f"Saved model to {args.out / 'final_model.zip'}")


if __name__ == "__main__":
    main()
