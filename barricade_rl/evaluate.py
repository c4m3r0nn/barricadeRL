from __future__ import annotations

import argparse

import numpy as np

from barricade_rl.opponents import make_opponent
from barricade_rl.single_agent import BarricadeSingleAgentEnv


def select_policy_action(policy_name: str, env: BarricadeSingleAgentEnv, rng: np.random.Generator) -> int:
    if policy_name == "random":
        actions = np.flatnonzero(env.action_masks())
        return int(rng.choice(actions))
    if policy_name == "greedy":
        return make_opponent("greedy").select_action(env.game, rng)
    raise ValueError(f"Unknown policy '{policy_name}'")


def run_evaluation(episodes: int, policy_name: str, opponent_name: str, seed: int):
    rng = np.random.default_rng(seed)
    env = BarricadeSingleAgentEnv(opponent=make_opponent(opponent_name), invalid_action="raise")
    wins = 0
    losses = 0
    truncations = 0
    total_steps = 0
    for episode in range(episodes):
        env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = select_policy_action(policy_name, env, rng)
            obs, reward, terminated, truncated, info = env.step(action)
            total_steps += 1
        if truncated:
            truncations += 1
        elif info["winner"] == 0:
            wins += 1
        else:
            losses += 1

    print(f"episodes={episodes}")
    print(f"policy={policy_name}")
    print(f"opponent={opponent_name}")
    print(f"wins={wins}")
    print(f"losses={losses}")
    print(f"truncations={truncations}")
    print(f"win_rate={wins / episodes:.3f}")
    print(f"avg_learner_steps={total_steps / episodes:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a simple Barricade policy against a scripted opponent.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--policy", choices=["random", "greedy"], default="greedy")
    parser.add_argument("--opponent", choices=["random", "greedy"], default="random")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_evaluation(args.episodes, args.policy, args.opponent, args.seed)


if __name__ == "__main__":
    main()
