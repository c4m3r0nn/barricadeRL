from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class ReplaySample:
    state_key: bytes
    observation: np.ndarray
    policy: np.ndarray
    value: float
    action_mask: np.ndarray
    ply: int
    current_player: int
    source: str = "self-play"
    config_hash: str | None = None
    root_value: float | None = None
    observation_version: int = 1
    board_size: int | None = None
    scoring_scheme: str = "terminal-win-loss-cap-zero"
    game_id: str | None = None
    run_id: str | None = None
    git_commit: str | None = None
    target_origin: str = "original"
    auxiliary_distances: np.ndarray | None = None
    opponent_policy: np.ndarray | None = None
    opponent_action_mask: np.ndarray | None = None
    has_auxiliary_target: bool | None = None
    has_opponent_policy_target: bool | None = None

    def __post_init__(self) -> None:
        observation = np.asarray(self.observation, dtype=np.float32)
        policy = np.asarray(self.policy, dtype=np.float32)
        action_mask = np.asarray(self.action_mask, dtype=np.bool_)
        state_key = bytes(self.state_key)
        value = float(self.value)
        root_value = None if self.root_value is None else float(self.root_value)
        has_auxiliary_target = (
            self.auxiliary_distances is not None
            if self.has_auxiliary_target is None
            else bool(self.has_auxiliary_target)
        )
        auxiliary_distances = (
            np.full(2, np.nan, dtype=np.float32)
            if self.auxiliary_distances is None
            else np.asarray(self.auxiliary_distances, dtype=np.float32)
        )
        has_opponent_policy_target = (
            self.opponent_policy is not None
            if self.has_opponent_policy_target is None
            else bool(self.has_opponent_policy_target)
        )
        opponent_policy = (
            np.zeros_like(policy, dtype=np.float32)
            if self.opponent_policy is None
            else np.asarray(self.opponent_policy, dtype=np.float32)
        )
        opponent_action_mask = (
            opponent_policy > 0
            if self.opponent_action_mask is None
            else np.asarray(self.opponent_action_mask, dtype=np.bool_)
        )

        if not state_key:
            raise ValueError("state_key must be non-empty")
        if observation.ndim != 3:
            raise ValueError("observation must be a CHW tensor")
        board_size = observation.shape[-1] if self.board_size is None else int(self.board_size)
        if observation.shape[-2:] != (board_size, board_size):
            raise ValueError("board_size must match the square observation tensor")
        if policy.ndim != 1:
            raise ValueError("policy must be a 1D vector")
        if action_mask.shape != policy.shape:
            raise ValueError("action_mask shape must match policy shape")
        if not np.isfinite(policy).all():
            raise ValueError("policy contains non-finite values")
        if np.any(policy < -1e-8):
            raise ValueError("policy contains negative probabilities")
        if not np.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError("value must be finite and in [-1, 1]")
        if root_value is not None and (not np.isfinite(root_value) or not -1.0 <= root_value <= 1.0):
            raise ValueError("root_value must be finite and in [-1, 1]")
        if action_mask.any():
            illegal_mass = float(policy[~action_mask].sum())
            if illegal_mass > 1e-6:
                raise ValueError("policy assigns probability to an illegal action")
            policy_sum = float(policy.sum())
            if abs(policy_sum - 1.0) > 1e-5:
                raise ValueError("policy probabilities must sum to 1")
        elif float(policy.sum()) != 0.0:
            raise ValueError("terminal/no-action policies must sum to zero")
        if self.current_player not in (0, 1):
            raise ValueError("current_player must be 0 or 1")
        if self.ply < 0:
            raise ValueError("ply must be non-negative")
        if self.observation_version < 1:
            raise ValueError("observation_version must be positive")
        if not self.scoring_scheme:
            raise ValueError("scoring_scheme must be non-empty")
        if self.target_origin not in ("original", "reanalyzed"):
            raise ValueError("target_origin must be 'original' or 'reanalyzed'")
        if auxiliary_distances.shape != (2,):
            raise ValueError("auxiliary_distances must have shape (2,)")
        if has_auxiliary_target and not np.isfinite(auxiliary_distances).all():
            raise ValueError("active auxiliary distance targets must be finite")
        if opponent_policy.shape != policy.shape or opponent_action_mask.shape != policy.shape:
            raise ValueError("opponent policy target shapes must match policy")
        if has_opponent_policy_target:
            if not np.isfinite(opponent_policy).all() or np.any(opponent_policy < -1e-8):
                raise ValueError("opponent policy target is invalid")
            if float(opponent_policy[~opponent_action_mask].sum()) > 1e-6:
                raise ValueError("opponent policy assigns mass outside its legal mask")
            if abs(float(opponent_policy.sum()) - 1.0) > 1e-5:
                raise ValueError("opponent policy probabilities must sum to 1")

        object.__setattr__(self, "state_key", state_key)
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "action_mask", action_mask)
        object.__setattr__(self, "root_value", root_value)
        object.__setattr__(self, "board_size", board_size)
        object.__setattr__(self, "auxiliary_distances", auxiliary_distances)
        object.__setattr__(self, "opponent_policy", opponent_policy)
        object.__setattr__(self, "opponent_action_mask", opponent_action_mask)
        object.__setattr__(self, "has_auxiliary_target", has_auxiliary_target)
        object.__setattr__(self, "has_opponent_policy_target", has_opponent_policy_target)


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    observations: np.ndarray
    policies: np.ndarray
    values: np.ndarray
    action_masks: np.ndarray
    state_keys: tuple[bytes, ...]
    auxiliary_distances: np.ndarray
    auxiliary_target_mask: np.ndarray
    opponent_policies: np.ndarray
    opponent_action_masks: np.ndarray
    opponent_policy_target_mask: np.ndarray


class AlphaZeroReplayBuffer:
    def __init__(self, *, capacity: int, observation_shape: tuple[int, ...], action_count: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if action_count < 1:
            raise ValueError("action_count must be positive")
        if len(observation_shape) != 3:
            raise ValueError("observation_shape must be CHW")
        self.capacity = int(capacity)
        self.observation_shape = tuple(int(value) for value in observation_shape)
        self.action_count = int(action_count)
        self._samples: list[ReplaySample] = []
        self.total_positions_added = 0
        self.gradient_samples_consumed = 0

    @property
    def size(self) -> int:
        return len(self._samples)

    @property
    def samples(self) -> tuple[ReplaySample, ...]:
        return tuple(self._samples)

    @property
    def samples_per_position_ratio(self) -> float:
        if self.total_positions_added == 0:
            return 0.0
        return self.gradient_samples_consumed / self.total_positions_added

    def add(self, sample: ReplaySample) -> None:
        self._validate_sample_shape(sample)
        self._samples.append(sample)
        self.total_positions_added += 1
        overflow = len(self._samples) - self.capacity
        if overflow > 0:
            del self._samples[:overflow]

    def extend(self, samples: list[ReplaySample] | tuple[ReplaySample, ...]) -> None:
        for sample in samples:
            self.add(sample)

    def sample(self, batch_size: int, *, rng: np.random.Generator | None = None) -> ReplayBatch:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if batch_size > self.size:
            raise ValueError("batch_size cannot exceed replay size")
        rng = rng or np.random.default_rng()
        indices = rng.choice(self.size, size=batch_size, replace=False)
        samples = [self._samples[int(index)] for index in indices]
        self.gradient_samples_consumed += batch_size
        return ReplayBatch(
            observations=np.stack([sample.observation for sample in samples]).astype(np.float32),
            policies=np.stack([sample.policy for sample in samples]).astype(np.float32),
            values=np.asarray([sample.value for sample in samples], dtype=np.float32),
            action_masks=np.stack([sample.action_mask for sample in samples]).astype(np.bool_),
            state_keys=tuple(sample.state_key for sample in samples),
            auxiliary_distances=np.stack([sample.auxiliary_distances for sample in samples]).astype(np.float32),
            auxiliary_target_mask=np.asarray(
                [sample.has_auxiliary_target for sample in samples], dtype=np.bool_
            ),
            opponent_policies=np.stack([sample.opponent_policy for sample in samples]).astype(np.float32),
            opponent_action_masks=np.stack(
                [sample.opponent_action_mask for sample in samples]
            ).astype(np.bool_),
            opponent_policy_target_mask=np.asarray(
                [sample.has_opponent_policy_target for sample in samples], dtype=np.bool_
            ),
        )

    def save_npz(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        observations = np.stack([sample.observation for sample in self._samples]) if self._samples else np.zeros((0, *self.observation_shape), dtype=np.float32)
        policies = np.stack([sample.policy for sample in self._samples]) if self._samples else np.zeros((0, self.action_count), dtype=np.float32)
        masks = np.stack([sample.action_mask for sample in self._samples]) if self._samples else np.zeros((0, self.action_count), dtype=np.bool_)
        np.savez_compressed(
            output,
            capacity=np.asarray([self.capacity], dtype=np.int64),
            observation_shape=np.asarray(self.observation_shape, dtype=np.int64),
            action_count=np.asarray([self.action_count], dtype=np.int64),
            observations=observations.astype(np.float32),
            policies=policies.astype(np.float32),
            values=np.asarray([sample.value for sample in self._samples], dtype=np.float32),
            action_masks=masks.astype(np.bool_),
            state_keys=np.asarray([sample.state_key.hex() for sample in self._samples]),
            plies=np.asarray([sample.ply for sample in self._samples], dtype=np.int64),
            current_players=np.asarray([sample.current_player for sample in self._samples], dtype=np.int64),
            sources=np.asarray([sample.source for sample in self._samples]),
            config_hashes=np.asarray([sample.config_hash or "" for sample in self._samples]),
            root_values=np.asarray(
                [np.nan if sample.root_value is None else sample.root_value for sample in self._samples],
                dtype=np.float32,
            ),
            observation_versions=np.asarray(
                [sample.observation_version for sample in self._samples], dtype=np.int64
            ),
            board_sizes=np.asarray([sample.board_size for sample in self._samples], dtype=np.int64),
            scoring_schemes=np.asarray([sample.scoring_scheme for sample in self._samples]),
            game_ids=np.asarray([sample.game_id or "" for sample in self._samples]),
            run_ids=np.asarray([sample.run_id or "" for sample in self._samples]),
            git_commits=np.asarray([sample.git_commit or "" for sample in self._samples]),
            target_origins=np.asarray([sample.target_origin for sample in self._samples]),
            auxiliary_distances=np.stack(
                [sample.auxiliary_distances for sample in self._samples]
            ) if self._samples else np.zeros((0, 2), dtype=np.float32),
            auxiliary_target_masks=np.asarray(
                [sample.has_auxiliary_target for sample in self._samples], dtype=np.bool_
            ),
            opponent_policies=np.stack(
                [sample.opponent_policy for sample in self._samples]
            ) if self._samples else np.zeros((0, self.action_count), dtype=np.float32),
            opponent_action_masks=np.stack(
                [sample.opponent_action_mask for sample in self._samples]
            ) if self._samples else np.zeros((0, self.action_count), dtype=np.bool_),
            opponent_policy_target_masks=np.asarray(
                [sample.has_opponent_policy_target for sample in self._samples], dtype=np.bool_
            ),
            total_positions_added=np.asarray([self.total_positions_added], dtype=np.int64),
            gradient_samples_consumed=np.asarray([self.gradient_samples_consumed], dtype=np.int64),
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "AlphaZeroReplayBuffer":
        with np.load(Path(path), allow_pickle=False) as payload:
            capacity = int(payload["capacity"][0])
            observation_shape = tuple(int(value) for value in payload["observation_shape"])
            action_count = int(payload["action_count"][0])
            buffer = cls(capacity=capacity, observation_shape=observation_shape, action_count=action_count)
            observations = payload["observations"]
            policies = payload["policies"]
            values = payload["values"]
            masks = payload["action_masks"]
            state_keys = payload["state_keys"]
            plies = payload["plies"]
            current_players = payload["current_players"]
            sources = payload["sources"]
            config_hashes = payload["config_hashes"]
            root_values = payload["root_values"]
            count = int(values.shape[0])
            observation_versions = payload.get("observation_versions", np.ones(count, dtype=np.int64))
            board_sizes = payload.get(
                "board_sizes", np.full(count, observation_shape[-1], dtype=np.int64)
            )
            scoring_schemes = payload.get(
                "scoring_schemes", np.full(count, "terminal-win-loss-cap-zero")
            )
            game_ids = payload.get("game_ids", np.full(count, ""))
            run_ids = payload.get("run_ids", np.full(count, ""))
            git_commits = payload.get("git_commits", np.full(count, ""))
            target_origins = payload.get("target_origins", np.full(count, "original"))
            auxiliary_distances = payload.get(
                "auxiliary_distances", np.full((count, 2), np.nan, dtype=np.float32)
            )
            auxiliary_target_masks = payload.get(
                "auxiliary_target_masks", np.zeros(count, dtype=np.bool_)
            )
            opponent_policies = payload.get(
                "opponent_policies", np.zeros((count, action_count), dtype=np.float32)
            )
            opponent_action_masks = payload.get(
                "opponent_action_masks", np.zeros((count, action_count), dtype=np.bool_)
            )
            opponent_policy_target_masks = payload.get(
                "opponent_policy_target_masks", np.zeros(count, dtype=np.bool_)
            )
            stored_total_added = int(payload.get("total_positions_added", np.asarray([count]))[0])
            stored_samples_consumed = int(
                payload.get("gradient_samples_consumed", np.asarray([0]))[0]
            )

            for index in range(count):
                root_value = float(root_values[index])
                sample = ReplaySample(
                    state_key=bytes.fromhex(str(state_keys[index])),
                    observation=observations[index],
                    policy=policies[index],
                    value=float(values[index]),
                    action_mask=masks[index],
                    ply=int(plies[index]),
                    current_player=int(current_players[index]),
                    source=str(sources[index]),
                    config_hash=str(config_hashes[index]) or None,
                    root_value=None if np.isnan(root_value) else root_value,
                    observation_version=int(observation_versions[index]),
                    board_size=int(board_sizes[index]),
                    scoring_scheme=str(scoring_schemes[index]),
                    game_id=str(game_ids[index]) or None,
                    run_id=str(run_ids[index]) or None,
                    git_commit=str(git_commits[index]) or None,
                    target_origin=str(target_origins[index]),
                    auxiliary_distances=auxiliary_distances[index],
                    opponent_policy=opponent_policies[index],
                    opponent_action_mask=opponent_action_masks[index],
                    has_auxiliary_target=bool(auxiliary_target_masks[index]),
                    has_opponent_policy_target=bool(opponent_policy_target_masks[index]),
                )
                buffer.add(sample)
            buffer.total_positions_added = stored_total_added
            buffer.gradient_samples_consumed = stored_samples_consumed
        return buffer

    def _validate_sample_shape(self, sample: ReplaySample) -> None:
        if sample.observation.shape != self.observation_shape:
            raise ValueError(f"sample observation shape {sample.observation.shape} does not match {self.observation_shape}")
        if sample.policy.shape != (self.action_count,):
            raise ValueError(f"sample policy shape {sample.policy.shape} does not match ({self.action_count},)")
        if sample.action_mask.shape != (self.action_count,):
            raise ValueError(f"sample action mask shape {sample.action_mask.shape} does not match ({self.action_count},)")


def make_replay_sample(
    game,
    state,
    *,
    policy: np.ndarray,
    value: float,
    source: str = "self-play",
    config_hash: str | None = None,
    root_value: float | None = None,
    observation_version: int = 1,
    scoring_scheme: str = "terminal-win-loss-cap-zero",
    game_id: str | None = None,
    run_id: str | None = None,
    git_commit: str | None = None,
    target_origin: str = "original",
    auxiliary_distances: np.ndarray | None = None,
    opponent_policy: np.ndarray | None = None,
    opponent_action_mask: np.ndarray | None = None,
) -> ReplaySample:
    if auxiliary_distances is None and hasattr(game, "shortest_path_distance"):
        mover = int(state.current_player)
        own_distance = game.shortest_path_distance(state, mover)
        opponent_distance = game.shortest_path_distance(state, 1 - mover)
        if own_distance is not None and opponent_distance is not None:
            auxiliary_distances = np.asarray(
                [own_distance / 20.0, opponent_distance / 20.0], dtype=np.float32
            )
    return ReplaySample(
        state_key=game.state_key(state),
        observation=game.canonical_observation(state),
        policy=policy,
        value=value,
        action_mask=game.legal_actions(state),
        ply=int(state.ply),
        current_player=int(state.current_player),
        source=source,
        config_hash=config_hash,
        root_value=root_value,
        observation_version=observation_version,
        board_size=int(game.board_size),
        scoring_scheme=scoring_scheme,
        game_id=game_id,
        run_id=run_id,
        git_commit=git_commit,
        target_origin=target_origin,
        auxiliary_distances=auxiliary_distances,
        opponent_policy=opponent_policy,
        opponent_action_mask=opponent_action_mask,
    )
