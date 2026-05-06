from __future__ import annotations

import argparse
import time

from barricade_rl.opponents import make_opponent
from barricade_rl.single_agent import BarricadeSingleAgentEnv


def run_benchmark(episodes: int, opponent_name: str, seed: int):
    env = BarricadeSingleAgentEnv(opponent=make_opponent(opponent_name), invalid_action="raise")
    obs, info = env.reset(seed=seed)
    steps = 0
    started = time.perf_counter()
    for episode in range(episodes):
        if episode > 0:
            obs, info = env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            mask = env.action_masks()
            action = int(env.np_random.choice(mask.nonzero()[0]))
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
    elapsed = time.perf_counter() - started
    print(f"episodes={episodes}")
    print(f"learner_steps={steps}")
    print(f"seconds={elapsed:.3f}")
    print(f"learner_steps_per_second={steps / elapsed:.0f}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Barricade environment step speed.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--opponent", choices=["random", "greedy"], default="random")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_benchmark(args.episodes, args.opponent, args.seed)


if __name__ == "__main__":
    main()
