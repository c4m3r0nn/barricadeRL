from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

from barricade_rl.experiments import append_jsonl
from barricade_rl.opponents import CheckpointPoolOpponent, make_opponent
from barricade_rl.replay import record_model_game, save_replay
from barricade_rl.single_agent import BarricadeSingleAgentEnv


def expand_checkpoint_paths(patterns: list[str] | None) -> list[Path]:
    if not patterns:
        return []
    paths = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        paths.extend(Path(match) for match in matches)
    return sorted(set(paths))


def build_training_opponent(opponent_name: str, checkpoint_patterns: list[str] | None = None):
    checkpoint_paths = expand_checkpoint_paths(checkpoint_patterns)
    if checkpoint_paths:
        return CheckpointPoolOpponent.from_paths(checkpoint_paths)
    return make_opponent(opponent_name)


def make_training_env(opponent_name: str, checkpoint_patterns: list[str] | None = None, shaped_reward: bool = False):
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.monitor import Monitor

    def mask_fn(env):
        return env.action_masks()

    return Monitor(
        ActionMasker(
            BarricadeSingleAgentEnv(
                opponent=build_training_opponent(opponent_name, checkpoint_patterns),
                shaped_reward=shaped_reward,
            ),
            mask_fn,
        )
    )


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


class MilestoneReplayCallback:
    def __init__(self, out_dir: Path, opponent_name: str, replay_freq: int, seed: int):
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.next_replay_step = replay_freq

            def _on_step(self) -> bool:
                if replay_freq <= 0:
                    return True
                if self.num_timesteps < self.next_replay_step:
                    return True
                replay_dir = out_dir / "replays"
                frames = record_model_game(
                    self.model,
                    opponent_name=opponent_name,
                    seed=seed + self.num_timesteps,
                )
                save_replay(
                    replay_dir / f"replay_{self.num_timesteps}.json",
                    frames,
                    metadata={
                        "timesteps": self.num_timesteps,
                        "opponent": opponent_name,
                        "seed": seed + self.num_timesteps,
                    },
                )
                self.next_replay_step += replay_freq
                return True

        self.callback = _Callback()


class MetricsJsonlCallback:
    def __new__(cls, metrics_path: Path):
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.started_at = time.perf_counter()

            def _on_step(self) -> bool:
                return True

            def _on_rollout_end(self) -> None:
                elapsed = max(time.perf_counter() - self.started_at, 1e-9)
                ep_infos = list(self.model.ep_info_buffer)
                row = {
                    "timesteps": int(self.num_timesteps),
                    "fps": float(self.num_timesteps / elapsed),
                    "episodes": len(ep_infos),
                }
                if ep_infos:
                    rewards = [info["r"] for info in ep_infos if "r" in info]
                    lengths = [info["l"] for info in ep_infos if "l" in info]
                    if rewards:
                        row["ep_rew_mean"] = float(sum(rewards) / len(rewards))
                    if lengths:
                        row["ep_len_mean"] = float(sum(lengths) / len(lengths))
                # SB3 exposes recent train scalars through the logger after updates.
                for key, value in getattr(self.logger, "name_to_value", {}).items():
                    if key.startswith("train/"):
                        row[key.replace("/", "_")] = float(value)
                append_jsonl(metrics_path, row)

        return _Callback()


def train_maskable_ppo(
    timesteps: int,
    opponent: str,
    seed: int,
    out: Path,
    replay_freq: int = 1_000,
    checkpoint_opponents: list[str] | None = None,
    shaped_reward: bool = False,
):
    try:
        from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
        from stable_baselines3.common.callbacks import CallbackList
    except ImportError as exc:
        raise SystemExit("Install RL dependencies first: .venv/bin/python -m pip install -e '.[dev,rl]'") from exc

    out.mkdir(parents=True, exist_ok=True)
    train_env = make_training_env(opponent, checkpoint_opponents, shaped_reward=shaped_reward)
    eval_env = make_training_env(opponent, checkpoint_opponents, shaped_reward=shaped_reward)
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(out / "best"),
        log_path=str(out / "eval"),
        eval_freq=max(1_000, timesteps // 10),
        deterministic=True,
        render=False,
    )
    replay_opponent = opponent if not checkpoint_opponents else "mixed"
    replay_callback = MilestoneReplayCallback(out, replay_opponent, replay_freq, seed).callback
    metrics_callback = MetricsJsonlCallback(out / "metrics.jsonl")
    callbacks = CallbackList([eval_callback, replay_callback, metrics_callback])
    model = build_model(train_env, seed=seed, tensorboard_log=out / "tb")
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=False)
    model.save(out / "final_model")
    return out / "final_model.zip"


def main():
    parser = argparse.ArgumentParser(description="Smoke-train MaskablePPO on Barricade.")
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--opponent", choices=["random", "greedy", "mixed"], default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/maskable_ppo_barricade"))
    parser.add_argument("--replay-freq", type=int, default=1_000, help="Save one replay every N timesteps. Use 0 to disable.")
    parser.add_argument("--checkpoint-opponents", nargs="*", help="Checkpoint paths or glob patterns for a self-play opponent pool.")
    parser.add_argument("--shaped-reward", action="store_true", help="Add small shortest-path shaping to sparse rewards.")
    args = parser.parse_args()

    model_path = train_maskable_ppo(
        timesteps=args.timesteps,
        opponent=args.opponent,
        seed=args.seed,
        out=args.out,
        replay_freq=args.replay_freq,
        checkpoint_opponents=args.checkpoint_opponents,
        shaped_reward=args.shaped_reward,
    )
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
