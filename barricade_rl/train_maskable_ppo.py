from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

from barricade_rl.experiments import append_jsonl
from barricade_rl.evaluate import evaluate_model
from barricade_rl.opponents import CheckpointPoolOpponent, RefreshingCheckpointPoolOpponent, make_opponent
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


def build_training_opponent(
    opponent_name: str,
    checkpoint_patterns: list[str] | None = None,
    self_play_patterns: list[str] | None = None,
    checkpoint_probability: float = 1.0,
):
    if self_play_patterns:
        patterns = list(checkpoint_patterns or []) + list(self_play_patterns)
        return RefreshingCheckpointPoolOpponent(
            patterns=patterns,
            fallback=make_opponent(opponent_name),
            checkpoint_probability=checkpoint_probability,
        )
    checkpoint_paths = expand_checkpoint_paths(checkpoint_patterns)
    if checkpoint_paths:
        return CheckpointPoolOpponent.from_paths(checkpoint_paths)
    return make_opponent(opponent_name)


def make_training_env(
    opponent_name: str,
    checkpoint_patterns: list[str] | None = None,
    shaped_reward: bool = False,
    self_play_patterns: list[str] | None = None,
    randomize_learner_side: bool = False,
    checkpoint_probability: float = 1.0,
    wall_penalty: float = 0.0,
    reverse_move_penalty: float = 0.0,
    progress_reward_scale: float = 0.0,
    survival_reward: float = 0.0,
    opponent_wall_value_penalty_scale: float = 0.0,
    endgame_start_probability: float = 0.0,
):
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.monitor import Monitor

    def mask_fn(env):
        return env.action_masks()

    return Monitor(
        ActionMasker(
            BarricadeSingleAgentEnv(
                opponent=build_training_opponent(
                    opponent_name,
                    checkpoint_patterns,
                    self_play_patterns,
                    checkpoint_probability=checkpoint_probability,
                ),
                shaped_reward=shaped_reward,
                wall_penalty=wall_penalty,
                reverse_move_penalty=reverse_move_penalty,
                progress_reward_scale=progress_reward_scale,
                survival_reward=survival_reward,
                opponent_wall_value_penalty_scale=opponent_wall_value_penalty_scale,
                endgame_start_probability=endgame_start_probability,
                learner_side=None if randomize_learner_side else 0,
            ),
            mask_fn,
        ),
        info_keywords=("survival_reward", "opponent_wall_value_delta", "opponent_wall_value_reward", "endgame_start"),
    )


def build_model(
    env,
    seed: int,
    tensorboard_log: Path | str,
    n_steps: int = 512,
    batch_size: int = 128,
    verbose: int = 1,
    policy: str = "mlp",
):
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    import torch as th
    import torch.nn as nn

    class SmallBoardCNN(BaseFeaturesExtractor):
        def __init__(self, observation_space, features_dim: int = 128):
            super().__init__(observation_space, features_dim)
            channels = observation_space.shape[0]
            self.cnn = nn.Sequential(
                nn.Conv2d(channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            with th.no_grad():
                sample = th.as_tensor(observation_space.sample()[None]).float()
                n_flatten = self.cnn(sample).shape[1]
            self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

        def forward(self, observations):
            return self.linear(self.cnn(observations))

    normalized_policy = policy.lower().strip()
    if normalized_policy == "cnn":
        policy_name = "CnnPolicy"
        policy_kwargs = {
            "features_extractor_class": SmallBoardCNN,
            "features_extractor_kwargs": {"features_dim": 128},
            "normalize_images": False,
        }
    elif normalized_policy == "mlp":
        policy_name = "MlpPolicy"
        policy_kwargs = None
    else:
        raise ValueError("policy must be 'mlp' or 'cnn'")

    return MaskablePPO(
        policy_name,
        env,
        verbose=verbose,
        seed=seed,
        tensorboard_log=str(tensorboard_log),
        n_steps=n_steps,
        batch_size=batch_size,
        policy_kwargs=policy_kwargs,
    )


class SelfPlayCheckpointCallback:
    def __init__(self, out_dir: Path, save_freq: int):
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.next_save_step = save_freq

            def _on_step(self) -> bool:
                if save_freq <= 0:
                    return True
                if self.num_timesteps < self.next_save_step:
                    return True
                pool_dir = out_dir / "self_play_pool"
                pool_dir.mkdir(parents=True, exist_ok=True)
                self.model.save(pool_dir / f"checkpoint_{self.num_timesteps}")
                self.next_save_step += save_freq
                return True

        self.callback = _Callback()


class MilestoneReplayCallback:
    def __init__(self, out_dir: Path, opponent_name: str, replay_freq: int, seed: int, learner_sides: tuple[int, ...] = (0,)):
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
                for learner_side in learner_sides:
                    frames = record_model_game(
                        self.model,
                        opponent_name=opponent_name,
                        seed=seed + self.num_timesteps + learner_side * 10_000,
                        learner_side=learner_side,
                    )
                    side_suffix = "" if learner_sides == (0,) else f"_p{learner_side}"
                    save_replay(
                        replay_dir / f"replay_{self.num_timesteps}{side_suffix}.json",
                        frames,
                        metadata={
                            "timesteps": self.num_timesteps,
                            "opponent": opponent_name,
                            "seed": seed + self.num_timesteps + learner_side * 10_000,
                            "learner_side": learner_side,
                        },
                    )
                self.next_replay_step += replay_freq
                return True

        self.callback = _Callback()


class MetricsJsonlCallback:
    def __new__(cls, metrics_path: Path, self_play_pool_dir: Path | None = None):
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
                if self_play_pool_dir is not None:
                    row["self_play_pool_size"] = len(list(self_play_pool_dir.glob("*.zip")))
                if ep_infos:
                    rewards = [info["r"] for info in ep_infos if "r" in info]
                    lengths = [info["l"] for info in ep_infos if "l" in info]
                    if rewards:
                        row["ep_rew_mean"] = float(sum(rewards) / len(rewards))
                        row["train_env_ep_rew_mean"] = row["ep_rew_mean"]
                    if lengths:
                        row["ep_len_mean"] = float(sum(lengths) / len(lengths))
                    for key in ("survival_reward", "opponent_wall_value_delta", "opponent_wall_value_reward", "endgame_start"):
                        values = [info[key] for info in ep_infos if key in info]
                        if values:
                            row[f"train_{key}_mean"] = float(sum(values) / len(values))
                # SB3 exposes recent train scalars through the logger after updates.
                for key, value in getattr(self.logger, "name_to_value", {}).items():
                    if key.startswith("train/"):
                        row[key.replace("/", "_")] = float(value)
                append_jsonl(metrics_path, row)

        return _Callback()


class PeriodicScriptedEvalCallback:
    def __init__(self, metrics_path: Path, opponents: list[str], episodes: int, eval_freq: int, seed: int):
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.next_eval_step = eval_freq

            def _on_step(self) -> bool:
                if eval_freq <= 0 or episodes <= 0 or not opponents:
                    return True
                if self.num_timesteps < self.next_eval_step:
                    return True
                row = {"timesteps": int(self.num_timesteps)}
                win_rates = []
                for opponent_name in opponents:
                    for learner_side in (0, 1):
                        result = evaluate_model(
                            self.model,
                            episodes=episodes,
                            opponent_name=opponent_name,
                            seed=seed + self.num_timesteps + learner_side * 10_000,
                            deterministic=True,
                            learner_side=learner_side,
                        )
                        prefix = f"eval_{opponent_name}_p{learner_side}"
                        row[f"{prefix}_win_rate"] = result.win_rate
                        win_rates.append(result.win_rate)
                        row[f"{prefix}_loss_rate"] = result.loss_rate
                        row[f"{prefix}_truncation_rate"] = result.truncation_rate
                        row[f"{prefix}_episodes"] = result.episodes
                        row[f"{prefix}_avg_steps"] = result.avg_learner_steps
                if win_rates:
                    row["eval_balanced_win_rate"] = float(sum(win_rates) / len(win_rates))
                append_jsonl(metrics_path, row)
                self.next_eval_step += eval_freq
                return True

        self.callback = _Callback()


def train_maskable_ppo(
    timesteps: int,
    opponent: str,
    seed: int,
    out: Path,
    replay_freq: int = 1_000,
    checkpoint_opponents: list[str] | None = None,
    initial_model: str | Path | None = None,
    shaped_reward: bool = False,
    policy: str = "mlp",
    self_play: bool = False,
    self_play_save_freq: int = 10_000,
    randomize_learner_side: bool = False,
    checkpoint_probability: float = 0.60,
    wall_penalty: float = 0.0,
    reverse_move_penalty: float = 0.0,
    progress_reward_scale: float = 0.0,
    survival_reward: float = 0.0,
    opponent_wall_value_penalty_scale: float = 0.0,
    endgame_start_probability: float = 0.0,
    eval_opponents: list[str] | None = None,
    eval_episodes: int = 10,
    scripted_eval_freq: int | None = None,
):
    try:
        from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
        from stable_baselines3.common.callbacks import CallbackList
    except ImportError as exc:
        raise SystemExit("Install RL dependencies first: .venv/bin/python -m pip install -e '.[dev,rl]'") from exc

    out.mkdir(parents=True, exist_ok=True)
    self_play_patterns = [str(out / "self_play_pool" / "*.zip")] if self_play else None
    train_env = make_training_env(
        opponent,
        checkpoint_opponents,
        shaped_reward=shaped_reward,
        self_play_patterns=self_play_patterns,
        randomize_learner_side=randomize_learner_side,
        checkpoint_probability=checkpoint_probability if self_play else 1.0,
        wall_penalty=wall_penalty,
        reverse_move_penalty=reverse_move_penalty,
        progress_reward_scale=progress_reward_scale,
        survival_reward=survival_reward,
        opponent_wall_value_penalty_scale=opponent_wall_value_penalty_scale,
        endgame_start_probability=endgame_start_probability,
    )
    eval_env = make_training_env(
        opponent,
        checkpoint_opponents,
        shaped_reward=shaped_reward,
        self_play_patterns=self_play_patterns,
        checkpoint_probability=checkpoint_probability if self_play else 1.0,
        wall_penalty=wall_penalty,
        reverse_move_penalty=reverse_move_penalty,
        progress_reward_scale=progress_reward_scale,
        survival_reward=survival_reward,
        opponent_wall_value_penalty_scale=opponent_wall_value_penalty_scale,
        endgame_start_probability=endgame_start_probability,
    )
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(out / "best"),
        log_path=str(out / "eval"),
        eval_freq=max(1_000, timesteps // 10),
        deterministic=True,
        render=False,
    )
    replay_opponent = opponent
    replay_sides = (0, 1) if randomize_learner_side else (0,)
    replay_callback = MilestoneReplayCallback(out, replay_opponent, replay_freq, seed, learner_sides=replay_sides).callback
    metrics_callback = MetricsJsonlCallback(out / "metrics.jsonl", out / "self_play_pool" if self_play else None)
    callback_items = [eval_callback, replay_callback, metrics_callback]
    if eval_opponents:
        callback_items.append(
            PeriodicScriptedEvalCallback(
                out / "metrics.jsonl",
                opponents=eval_opponents,
                episodes=eval_episodes,
                eval_freq=scripted_eval_freq or max(5_000, timesteps // 20),
                seed=seed,
            ).callback
        )
    if self_play:
        callback_items.append(SelfPlayCheckpointCallback(out, self_play_save_freq).callback)
    callbacks = CallbackList(callback_items)
    if initial_model:
        from sb3_contrib import MaskablePPO

        model = MaskablePPO.load(initial_model, env=train_env, tensorboard_log=str(out / "tb"))
    else:
        model = build_model(train_env, seed=seed, tensorboard_log=out / "tb", policy=policy)
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=False)
    model.save(out / "final_model")
    return out / "final_model.zip"


def main():
    parser = argparse.ArgumentParser(description="Smoke-train MaskablePPO on Barricade.")
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument(
        "--opponent",
        choices=[
            "random",
            "greedy",
            "mixed",
            "anti_rush_lite",
            "anti_rush_medium",
            "anti_rush",
            "curriculum",
            "curriculum_stage2",
            "curriculum_stage3_bridge",
            "curriculum_stage3_gentle",
            "curriculum_stage3",
        ],
        default="random",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/maskable_ppo_barricade"))
    parser.add_argument("--replay-freq", type=int, default=1_000, help="Save one replay every N timesteps. Use 0 to disable.")
    parser.add_argument("--checkpoint-opponents", nargs="*", help="Checkpoint paths or glob patterns for a self-play opponent pool.")
    parser.add_argument("--initial-model", type=Path, help="Existing model checkpoint to continue training from.")
    parser.add_argument("--shaped-reward", action="store_true", help="Add small shortest-path shaping to sparse rewards.")
    parser.add_argument("--policy", choices=["mlp", "cnn"], default="mlp", help="Policy architecture to train.")
    parser.add_argument("--self-play", action="store_true", help="Save current-run checkpoints and train against a refreshing pool of older snapshots.")
    parser.add_argument("--self-play-save-freq", type=int, default=10_000, help="Save a self-play checkpoint every N timesteps.")
    parser.add_argument("--randomize-learner-side", action="store_true", help="Randomly train the learner as player 0 or player 1 each episode.")
    parser.add_argument("--checkpoint-probability", type=float, default=0.60, help="For self-play, probability of sampling a checkpoint opponent instead of the scripted fallback.")
    parser.add_argument("--wall-penalty", type=float, default=0.0, help="Small reward penalty for each learner wall placement.")
    parser.add_argument("--reverse-move-penalty", type=float, default=0.0, help="Small reward penalty when the learner immediately reverses its previous pawn move.")
    parser.add_argument("--progress-reward-scale", type=float, default=0.0, help="Small reward bonus when a learner pawn move shortens its path to goal.")
    parser.add_argument("--survival-reward", type=float, default=0.0, help="Small reward bonus for surviving a nonterminal opponent turn.")
    parser.add_argument(
        "--opponent-wall-value-penalty-scale",
        type=float,
        default=0.0,
        help="Penalty scale for opponent walls that increase the learner's shortest path.",
    )
    parser.add_argument("--endgame-start-probability", type=float, default=0.0, help="Probability of resetting episodes from a near-goal anti-rush practice state.")
    parser.add_argument(
        "--eval-opponents",
        nargs="*",
        choices=["random", "greedy", "mixed", "anti_rush_lite", "anti_rush"],
        default=["random", "greedy", "mixed", "anti_rush_lite", "anti_rush"],
    )
    parser.add_argument("--eval-episodes", type=int, default=10, help="Episodes per scripted evaluation opponent. Use 0 to disable.")
    parser.add_argument("--scripted-eval-freq", type=int, help="Evaluate against scripted opponents every N timesteps.")
    args = parser.parse_args()

    model_path = train_maskable_ppo(
        timesteps=args.timesteps,
        opponent=args.opponent,
        seed=args.seed,
        out=args.out,
        replay_freq=args.replay_freq,
        checkpoint_opponents=args.checkpoint_opponents,
        initial_model=args.initial_model,
        shaped_reward=args.shaped_reward,
        policy=args.policy,
        self_play=args.self_play,
        self_play_save_freq=args.self_play_save_freq,
        randomize_learner_side=args.randomize_learner_side,
        checkpoint_probability=args.checkpoint_probability,
        wall_penalty=args.wall_penalty,
        reverse_move_penalty=args.reverse_move_penalty,
        progress_reward_scale=args.progress_reward_scale,
        survival_reward=args.survival_reward,
        opponent_wall_value_penalty_scale=args.opponent_wall_value_penalty_scale,
        endgame_start_probability=args.endgame_start_probability,
        eval_opponents=args.eval_opponents,
        eval_episodes=args.eval_episodes,
        scripted_eval_freq=args.scripted_eval_freq,
    )
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
