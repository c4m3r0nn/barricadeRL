from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from barricade_rl.opponents import make_opponent
from barricade_rl.single_agent import BarricadeSingleAgentEnv


@dataclass(slots=True)
class EvaluationResult:
    episodes: int
    wins: int
    losses: int
    truncations: int
    total_steps: int
    episode_lengths: list[int]
    learner_walls_placed: list[int]
    opponent_walls_placed: list[int]

    @property
    def win_rate(self) -> float:
        return self.wins / self.episodes

    @property
    def loss_rate(self) -> float:
        return self.losses / self.episodes

    @property
    def truncation_rate(self) -> float:
        return self.truncations / self.episodes

    @property
    def avg_learner_steps(self) -> float:
        return self.total_steps / self.episodes

    @property
    def min_learner_steps(self) -> int:
        return min(self.episode_lengths)

    @property
    def max_learner_steps(self) -> int:
        return max(self.episode_lengths)

    @property
    def avg_learner_walls_placed(self) -> float:
        return sum(self.learner_walls_placed) / self.episodes

    @property
    def avg_opponent_walls_placed(self) -> float:
        return sum(self.opponent_walls_placed) / self.episodes

    @property
    def avg_walls_placed(self) -> float:
        return self.avg_learner_walls_placed + self.avg_opponent_walls_placed


def select_policy_action(policy_name: str, env: BarricadeSingleAgentEnv, rng: np.random.Generator) -> int:
    if policy_name == "random":
        actions = np.flatnonzero(env.action_masks())
        return int(rng.choice(actions))
    if policy_name == "greedy":
        return make_opponent("greedy").select_action(env.game, rng)
    raise ValueError(f"Unknown policy '{policy_name}'")


def select_model_action(model, obs, env: BarricadeSingleAgentEnv, deterministic: bool = True) -> int:
    action, _ = model.predict(obs, deterministic=deterministic, action_masks=env.action_masks())
    return int(action)


def evaluate_model(model, episodes: int, opponent_name: str, seed: int, deterministic: bool = True) -> EvaluationResult:
    env = BarricadeSingleAgentEnv(opponent=make_opponent(opponent_name), invalid_action="raise")
    wins = 0
    losses = 0
    truncations = 0
    total_steps = 0
    episode_lengths = []
    learner_walls_placed = []
    opponent_walls_placed = []
    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        episode_steps = 0
        while not (terminated or truncated):
            action = select_model_action(model, obs, env, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_steps += 1
            episode_steps += 1
        if truncated:
            truncations += 1
        elif info["winner"] == 0:
            wins += 1
        else:
            losses += 1
        episode_lengths.append(episode_steps)
        learner_walls_placed.append(10 - info["walls_remaining"][0])
        opponent_walls_placed.append(10 - info["walls_remaining"][1])
    return EvaluationResult(
        episodes=episodes,
        wins=wins,
        losses=losses,
        truncations=truncations,
        total_steps=total_steps,
        episode_lengths=episode_lengths,
        learner_walls_placed=learner_walls_placed,
        opponent_walls_placed=opponent_walls_placed,
    )


def print_result(result: EvaluationResult, policy: str, opponent_name: str):
    print(f"episodes={result.episodes}")
    print(f"policy={policy}")
    print(f"opponent={opponent_name}")
    print(f"wins={result.wins}")
    print(f"losses={result.losses}")
    print(f"truncations={result.truncations}")
    print(f"win_rate={result.win_rate:.3f}")
    print(f"loss_rate={result.loss_rate:.3f}")
    print(f"truncation_rate={result.truncation_rate:.3f}")
    print(f"avg_learner_steps={result.avg_learner_steps:.1f}")
    print(f"min_learner_steps={result.min_learner_steps}")
    print(f"max_learner_steps={result.max_learner_steps}")
    print(f"avg_walls_placed={result.avg_walls_placed:.1f}")
    print(f"avg_learner_walls_placed={result.avg_learner_walls_placed:.1f}")
    print(f"avg_opponent_walls_placed={result.avg_opponent_walls_placed:.1f}")


def run_evaluation(episodes: int, policy_name: str, opponent_name: str, seed: int):
    rng = np.random.default_rng(seed)
    env = BarricadeSingleAgentEnv(opponent=make_opponent(opponent_name), invalid_action="raise")
    wins = 0
    losses = 0
    truncations = 0
    total_steps = 0
    episode_lengths = []
    learner_walls_placed = []
    opponent_walls_placed = []
    for episode in range(episodes):
        env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        episode_steps = 0
        while not (terminated or truncated):
            action = select_policy_action(policy_name, env, rng)
            obs, reward, terminated, truncated, info = env.step(action)
            total_steps += 1
            episode_steps += 1
        if truncated:
            truncations += 1
        elif info["winner"] == 0:
            wins += 1
        else:
            losses += 1
        episode_lengths.append(episode_steps)
        learner_walls_placed.append(10 - info["walls_remaining"][0])
        opponent_walls_placed.append(10 - info["walls_remaining"][1])

    print_result(
        EvaluationResult(
            episodes=episodes,
            wins=wins,
            losses=losses,
            truncations=truncations,
            total_steps=total_steps,
            episode_lengths=episode_lengths,
            learner_walls_placed=learner_walls_placed,
            opponent_walls_placed=opponent_walls_placed,
        ),
        policy=policy_name,
        opponent_name=opponent_name,
    )


def run_model_evaluation(model_path: Path, episodes: int, opponent_name: str, seed: int, deterministic: bool):
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise SystemExit("Install RL dependencies first: .venv/bin/python -m pip install -e '.[dev,rl]'") from exc

    model = MaskablePPO.load(model_path)
    result = evaluate_model(model, episodes=episodes, opponent_name=opponent_name, seed=seed, deterministic=deterministic)
    print_result(result, policy=str(model_path), opponent_name=opponent_name)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a simple Barricade policy against a scripted opponent.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--policy", choices=["random", "greedy"], default="greedy")
    parser.add_argument("--opponent", choices=["random", "greedy", "mixed"], default="random")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_evaluation(args.episodes, args.policy, args.opponent, args.seed)


def model_main():
    parser = argparse.ArgumentParser(description="Evaluate a trained MaskablePPO Barricade model.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--opponent", choices=["random", "greedy", "mixed"], default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic model actions instead of deterministic actions.")
    args = parser.parse_args()
    run_model_evaluation(
        model_path=args.model,
        episodes=args.episodes,
        opponent_name=args.opponent,
        seed=args.seed,
        deterministic=not args.stochastic,
    )


if __name__ == "__main__":
    main()
