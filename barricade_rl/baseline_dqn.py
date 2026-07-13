from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .dashboard import evaluation_to_dashboard_event, write_dashboard_event
from .env import QuoridorEnv
from .evaluate import MatchResult, evaluate_ladder, play_match
from .game import ACTION_COUNT, BOARD_SIZE, Game, State
from .opponents import FROZEN_LADDER, GreedyRacer, RandomOpponent

INPUT_DIM = 6 * BOARD_SIZE * BOARD_SIZE


@dataclass(frozen=True, slots=True)
class MaskedDQNConfig:
    episodes: int = 200
    expert_episodes: int = 120
    hidden_size: int = 64
    learning_rate: float = 0.03
    gamma: float = 0.97
    monte_carlo_weight: float = 0.8
    batch_size: int = 64
    replay_capacity: int = 8192
    updates_per_episode: int = 16
    epsilon_start: float = 0.15
    epsilon_end: float = 0.02
    target_update_episodes: int = 10
    evaluation_games_per_color: int = 25
    min_random_score_rate: float = 0.8
    seed: int = 0
    alternate_colours: bool = True

    def __post_init__(self) -> None:
        if self.episodes < 1:
            raise ValueError("episodes must be positive")
        if not 0 <= self.expert_episodes <= self.episodes:
            raise ValueError("expert_episodes must be between 0 and episodes")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")
        if not 0 <= self.monte_carlo_weight <= 1:
            raise ValueError("monte_carlo_weight must be between 0 and 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be at least batch_size")
        if self.updates_per_episode < 1:
            raise ValueError("updates_per_episode must be positive")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("epsilon schedule must satisfy 0 <= end <= start <= 1")
        if self.target_update_episodes < 1:
            raise ValueError("target_update_episodes must be positive")
        if self.evaluation_games_per_color < 1:
            raise ValueError("evaluation_games_per_color must be positive")
        if not 0 <= self.min_random_score_rate <= 1:
            raise ValueError("min_random_score_rate must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class TrainingResult:
    episodes: int
    env_steps: int
    wins: int
    losses: int
    caps: int
    loss: float | None
    evaluation: MatchResult
    config: MaskedDQNConfig

    @property
    def passed(self) -> bool:
        return self.evaluation.score_rate >= self.config.min_random_score_rate

    def to_dict(self) -> dict:
        return {
            "episodes": self.episodes,
            "env_steps": self.env_steps,
            "wins": self.wins,
            "losses": self.losses,
            "caps": self.caps,
            "loss": self.loss,
            "passed": self.passed,
            "config": asdict(self.config),
            "evaluation": self.evaluation.to_dict(include_records=False),
        }


DEFAULT_CONFIG = MaskedDQNConfig()


class MaskedDQNPolicy:
    """Small dependency-free masked DQN used only as the M1 smoke baseline.

    This is intentionally not the production AlphaZero network. It consumes the
    Gym wrapper's observations, masks illegal actions before every argmax, and
    is just large enough to validate observation/mask/reward plumbing cheaply.
    """

    def __init__(
        self,
        *,
        W1: np.ndarray,
        b1: np.ndarray,
        W2: np.ndarray,
        b2: np.ndarray,
        name: str = "masked-dqn-smoke",
        metadata: dict | None = None,
    ) -> None:
        self.W1 = np.asarray(W1, dtype=np.float32)
        self.b1 = np.asarray(b1, dtype=np.float32)
        self.W2 = np.asarray(W2, dtype=np.float32)
        self.b2 = np.asarray(b2, dtype=np.float32)
        if self.W1.shape[0] != INPUT_DIM:
            raise ValueError(f"W1 must have input dimension {INPUT_DIM}")
        if self.W1.shape[1] != self.b1.shape[0]:
            raise ValueError("W1 and b1 shapes do not match")
        if self.W2.shape != (self.b1.shape[0], ACTION_COUNT):
            raise ValueError(f"W2 must have shape ({self.b1.shape[0]}, {ACTION_COUNT})")
        if self.b2.shape != (ACTION_COUNT,):
            raise ValueError(f"b2 must have shape ({ACTION_COUNT},)")
        self.name = name
        self.metadata = dict(metadata or {})

    @classmethod
    def initialized(cls, *, seed: int, hidden_size: int = 64, name: str = "masked-dqn-smoke") -> "MaskedDQNPolicy":
        rng = np.random.default_rng(seed)
        W1 = (rng.standard_normal((INPUT_DIM, hidden_size)) * np.sqrt(2.0 / INPUT_DIM)).astype(np.float32)
        b1 = np.zeros(hidden_size, dtype=np.float32)
        W2 = np.zeros((hidden_size, ACTION_COUNT), dtype=np.float32)
        b2 = np.zeros(ACTION_COUNT, dtype=np.float32)
        return cls(W1=W1, b1=b1, W2=W2, b2=b2, name=name, metadata={"seed": seed})

    def copy(self) -> "MaskedDQNPolicy":
        return MaskedDQNPolicy(
            W1=self.W1.copy(),
            b1=self.b1.copy(),
            W2=self.W2.copy(),
            b2=self.b2.copy(),
            name=self.name,
            metadata=self.metadata.copy(),
        )

    def copy_from(self, other: "MaskedDQNPolicy") -> None:
        self.W1[...] = other.W1
        self.b1[...] = other.b1
        self.W2[...] = other.W2
        self.b2[...] = other.b2

    def q_values(self, observation: np.ndarray) -> np.ndarray:
        x = _flatten_observation(observation)[None, :]
        _, hidden, q = _forward(self, x)
        del hidden
        return q[0]

    def select_action_from_observation(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        rng: np.random.Generator,
        *,
        epsilon: float = 0.0,
    ) -> int:
        legal = np.flatnonzero(np.asarray(action_mask, dtype=np.bool_))
        if not legal.size:
            raise ValueError("cannot select an action without legal actions")
        if epsilon > 0 and rng.random() < epsilon:
            return int(rng.choice(legal))
        q = self.q_values(observation)
        masked = np.full(ACTION_COUNT, -np.inf, dtype=np.float32)
        masked[legal] = q[legal]
        return int(np.argmax(masked))

    def select_action(self, game: Game, state: State, rng: np.random.Generator) -> int:
        return self.select_action_from_observation(
            game.canonical_observation(state),
            game.legal_actions(state),
            rng,
        )

    def save(self, path: str | Path, *, metadata: dict | None = None) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        stored_metadata = self.metadata.copy()
        if metadata:
            stored_metadata.update(metadata)
        np.savez_compressed(
            target,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            name=np.asarray(self.name),
            metadata_json=np.asarray(json.dumps(stored_metadata, sort_keys=True)),
        )


def load_masked_dqn(path: str | Path) -> MaskedDQNPolicy:
    with np.load(Path(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        return MaskedDQNPolicy(
            W1=data["W1"],
            b1=data["b1"],
            W2=data["W2"],
            b2=data["b2"],
            name=str(data["name"]),
            metadata=metadata,
        )


class _ReplayBuffer:
    def __init__(self, *, capacity: int, observation_shape: tuple[int, ...], rng: np.random.Generator) -> None:
        self.capacity = capacity
        self.rng = rng
        self.observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.next_observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.next_masks = np.zeros((capacity, ACTION_COUNT), dtype=np.bool_)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.position = 0

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        next_mask: np.ndarray,
        done: bool,
        return_target: float,
    ) -> None:
        self.observations[self.position] = observation
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_observations[self.position] = next_observation
        self.next_masks[self.position] = next_mask
        self.dones[self.position] = done
        self.returns[self.position] = return_target
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = self.rng.integers(0, self.size, size=batch_size)
        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.next_masks[indices],
            self.dones[indices],
            self.returns[indices],
        )


def train_masked_dqn(
    config: MaskedDQNConfig | None = None,
    *,
    output_path: str | Path | None = None,
) -> TrainingResult:
    config = config or MaskedDQNConfig()
    rng = np.random.default_rng(config.seed)
    policy = MaskedDQNPolicy.initialized(seed=config.seed, hidden_size=config.hidden_size)
    target_policy = policy.copy()
    teacher = GreedyRacer()
    replay = _ReplayBuffer(
        capacity=config.replay_capacity,
        observation_shape=(6, BOARD_SIZE, BOARD_SIZE),
        rng=rng,
    )
    wins = losses = caps = env_steps = 0
    last_loss: float | None = None

    for episode in range(config.episodes):
        learner_player = episode % 2 if config.alternate_colours else 0
        env = QuoridorEnv(opponent_policy=RandomOpponent(), learner_player=learner_player)
        episode_seed = int(rng.integers(0, np.iinfo(np.int64).max))
        observation, info = env.reset(seed=episode_seed)
        episode_transitions: list[tuple[np.ndarray, int, float, np.ndarray, np.ndarray, bool]] = []
        terminated = truncated = False
        reward = 0.0
        epsilon = _epsilon(config, episode)

        while not (terminated or truncated):
            if episode < config.expert_episodes:
                action = teacher.select_action(env.game, env.state, rng)
            else:
                action = policy.select_action_from_observation(
                    observation,
                    info["action_mask"],
                    rng,
                    epsilon=epsilon,
                )
            next_observation, reward, terminated, truncated, next_info = env.step(action)
            episode_transitions.append(
                (
                    observation.copy(),
                    int(action),
                    float(reward),
                    next_observation.copy(),
                    next_info["action_mask"].copy(),
                    bool(terminated or truncated),
                )
            )
            observation, info = next_observation, next_info
            env_steps += 1

        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            caps += 1

        for offset, (
            transition_observation,
            action,
            transition_reward,
            next_observation,
            next_mask,
            done,
        ) in enumerate(episode_transitions):
            steps_to_terminal = len(episode_transitions) - offset - 1
            return_target = float(reward) * (config.gamma ** steps_to_terminal)
            replay.add(
                transition_observation,
                action,
                transition_reward,
                next_observation,
                next_mask,
                done,
                return_target,
            )

        if replay.size >= config.batch_size:
            for _ in range(config.updates_per_episode):
                batch = replay.sample(config.batch_size)
                last_loss = _train_batch(
                    policy,
                    target_policy,
                    batch,
                    gamma=config.gamma,
                    monte_carlo_weight=config.monte_carlo_weight,
                    learning_rate=config.learning_rate,
                )
        if (episode + 1) % config.target_update_episodes == 0:
            target_policy.copy_from(policy)

    evaluation = play_match(
        policy,
        RandomOpponent(),
        games_per_color=config.evaluation_games_per_color,
        seed=config.seed + 1,
    )
    result = TrainingResult(
        episodes=config.episodes,
        env_steps=env_steps,
        wins=wins,
        losses=losses,
        caps=caps,
        loss=last_loss,
        evaluation=evaluation,
        config=config,
    )
    policy.metadata.update(
        {
            "training_result_json": json.dumps(result.to_dict(), sort_keys=True),
            "config_json": json.dumps(asdict(config), sort_keys=True),
        }
    )
    if output_path is not None:
        policy.save(output_path)
    return result


def _epsilon(config: MaskedDQNConfig, episode: int) -> float:
    if episode < config.expert_episodes:
        return 0.0
    remaining = max(1, config.episodes - config.expert_episodes - 1)
    progress = min(1.0, (episode - config.expert_episodes) / remaining)
    return config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)


def _flatten_observation(observation: np.ndarray) -> np.ndarray:
    array = np.asarray(observation, dtype=np.float32)
    if array.shape != (6, BOARD_SIZE, BOARD_SIZE):
        raise ValueError(f"observation must have shape (6, {BOARD_SIZE}, {BOARD_SIZE})")
    return array.reshape(-1)


def _forward(policy: MaskedDQNPolicy, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z1 = x @ policy.W1 + policy.b1
    hidden = np.maximum(z1, 0.0)
    q = hidden @ policy.W2 + policy.b2
    return z1, hidden, q


def _train_batch(
    policy: MaskedDQNPolicy,
    target_policy: MaskedDQNPolicy,
    batch: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    gamma: float,
    monte_carlo_weight: float,
    learning_rate: float,
) -> float:
    observations, actions, rewards, next_observations, next_masks, dones, returns = batch
    x = observations.reshape(observations.shape[0], -1).astype(np.float32)
    actions = actions.astype(np.int64)
    rewards = rewards.astype(np.float32)
    next_x = next_observations.reshape(next_observations.shape[0], -1).astype(np.float32)
    _, _, next_q = _forward(target_policy, next_x)
    masked_next_q = np.where(next_masks, next_q, -np.inf)
    max_next_q = np.max(masked_next_q, axis=1)
    max_next_q = np.where(np.isfinite(max_next_q), max_next_q, 0.0).astype(np.float32)
    td_targets = rewards + gamma * (1.0 - dones.astype(np.float32)) * max_next_q
    targets = (
        monte_carlo_weight * returns.astype(np.float32)
        + (1.0 - monte_carlo_weight) * td_targets.astype(np.float32)
    )
    z1, hidden, q = _forward(policy, x)
    row_indices = np.arange(x.shape[0])
    predictions = q[row_indices, actions]
    errors = predictions - targets
    loss = float(np.mean(errors * errors))
    errors = np.clip(errors, -2.0, 2.0)

    grad_q = np.zeros_like(q, dtype=np.float32)
    grad_q[row_indices, actions] = (2.0 / x.shape[0]) * errors
    grad_W2 = hidden.T @ grad_q
    grad_b2 = grad_q.sum(axis=0)
    grad_hidden = grad_q @ policy.W2.T
    grad_z1 = grad_hidden * (z1 > 0.0)
    grad_W1 = x.T @ grad_z1
    grad_b1 = grad_z1.sum(axis=0)

    _clip_in_place(grad_W1, 5.0)
    _clip_in_place(grad_b1, 5.0)
    _clip_in_place(grad_W2, 5.0)
    _clip_in_place(grad_b2, 5.0)

    policy.W1 -= learning_rate * grad_W1
    policy.b1 -= learning_rate * grad_b1
    policy.W2 -= learning_rate * grad_W2
    policy.b2 -= learning_rate * grad_b2
    return loss


def _clip_in_place(array: np.ndarray, threshold: float) -> None:
    norm = float(np.linalg.norm(array))
    if norm > threshold:
        array *= threshold / norm


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the M1 dependency-free masked DQN smoke baseline.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_CONFIG.episodes)
    parser.add_argument("--expert-episodes", type=int, default=DEFAULT_CONFIG.expert_episodes)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_CONFIG.hidden_size)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_CONFIG.learning_rate)
    parser.add_argument("--gamma", type=float, default=DEFAULT_CONFIG.gamma)
    parser.add_argument("--monte-carlo-weight", type=float, default=DEFAULT_CONFIG.monte_carlo_weight)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size)
    parser.add_argument("--replay-capacity", type=int, default=DEFAULT_CONFIG.replay_capacity)
    parser.add_argument("--updates-per-episode", type=int, default=DEFAULT_CONFIG.updates_per_episode)
    parser.add_argument("--epsilon-start", type=float, default=DEFAULT_CONFIG.epsilon_start)
    parser.add_argument("--epsilon-end", type=float, default=DEFAULT_CONFIG.epsilon_end)
    parser.add_argument("--evaluation-games-per-color", type=int, default=DEFAULT_CONFIG.evaluation_games_per_color)
    parser.add_argument("--min-random-score-rate", type=float, default=DEFAULT_CONFIG.min_random_score_rate)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, default=None)
    parser.add_argument("--dashboard-events", type=Path, default=None)
    parser.add_argument("--ladder", action="store_true", help="also evaluate the trained policy against the full frozen ladder")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = MaskedDQNConfig(
        episodes=args.episodes,
        expert_episodes=args.expert_episodes,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        monte_carlo_weight=args.monte_carlo_weight,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        updates_per_episode=args.updates_per_episode,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        evaluation_games_per_color=args.evaluation_games_per_color,
        min_random_score_rate=args.min_random_score_rate,
        seed=args.seed,
    )
    result = train_masked_dqn(config, output_path=args.output)
    payload = result.to_dict()

    if args.ladder or args.dashboard_events is not None:
        policy = load_masked_dqn(args.output)
        ladder = evaluate_ladder(
            policy,
            ladder=FROZEN_LADDER if args.ladder else (RandomOpponent(),),
            games_per_color=args.evaluation_games_per_color,
            seed=args.seed + 2,
            run_id=f"masked-dqn-smoke-{args.seed}",
        )
        payload["ladder_evaluation"] = ladder.to_dict(include_records=False)
        if args.dashboard_events is not None:
            write_dashboard_event(args.dashboard_events, evaluation_to_dashboard_event(ladder))

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.result_json is not None:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result.passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
