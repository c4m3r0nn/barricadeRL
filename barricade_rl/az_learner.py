from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as torch_functional
from torch import nn

from .az_network import AlphaZeroNetwork, AlphaZeroNetworkConfig
from .az_replay import AlphaZeroReplayBuffer, ReplayBatch
from .config import config_hash as calculate_config_hash
from .config import load_config, small_game_from_config


@dataclass(frozen=True, slots=True)
class LearnerConfig:
    optimizer: str = "sgd"
    momentum: float = 0.9
    initial_learning_rate: float = 0.02
    learning_rate_drops: tuple[float, ...] = (0.002, 0.0002)
    learning_rate_drop_steps: tuple[int, ...] = (100_000, 200_000)
    planned_steps: int = 300_000
    batch_size: int = 512
    weight_decay: float = 1e-4
    mirror_augmentation: bool = True
    auxiliary_loss_weight: float = 0.1
    opponent_policy_loss_weight: float = 0.15
    ema_decay: float = 0.999
    target_samples_per_position_min: float = 1.0
    target_samples_per_position_max: float = 4.0

    def __post_init__(self) -> None:
        if self.optimizer != "sgd":
            raise ValueError("the AlphaZero learner requires SGD")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        rates = (self.initial_learning_rate, *self.learning_rate_drops)
        if any(rate <= 0 for rate in rates):
            raise ValueError("learning rates must be positive")
        if len(self.learning_rate_drops) != len(self.learning_rate_drop_steps):
            raise ValueError("each learning-rate drop requires a fixed step")
        if tuple(sorted(self.learning_rate_drop_steps)) != self.learning_rate_drop_steps:
            raise ValueError("learning-rate drop steps must be ordered")
        if self.learning_rate_drop_steps and self.learning_rate_drop_steps[-1] >= self.planned_steps:
            raise ValueError("learning-rate drops must occur before planned_steps")
        if self.batch_size < 1 or self.planned_steps < 1:
            raise ValueError("batch_size and planned_steps must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.auxiliary_loss_weight < 0 or self.opponent_policy_loss_weight < 0:
            raise ValueError("auxiliary loss weights must be non-negative")
        if not 0.0 <= self.ema_decay <= 1.0:
            raise ValueError("ema_decay must be in [0, 1]")
        if not 0 < self.target_samples_per_position_min <= self.target_samples_per_position_max:
            raise ValueError("samples-per-position bounds are invalid")

    @classmethod
    def from_project_config(cls, config: Mapping) -> "LearnerConfig":
        training = config["training"]
        network = config["network"]
        replay = config["replay"]
        return cls(
            optimizer=str(training["optimizer"]),
            momentum=float(training["momentum"]),
            initial_learning_rate=float(training["initial_learning_rate"]),
            learning_rate_drops=tuple(float(value) for value in training["learning_rate_drops"]),
            learning_rate_drop_steps=tuple(
                int(value) for value in training["learning_rate_drop_steps"]
            ),
            planned_steps=int(training["planned_steps"]),
            batch_size=int(training["batch_size"]),
            weight_decay=float(training["weight_decay"]),
            mirror_augmentation=bool(training["mirror_augmentation"]),
            auxiliary_loss_weight=float(network["auxiliary_loss_weight"]),
            opponent_policy_loss_weight=float(network["opponent_policy_head_loss_weight"]),
            ema_decay=float(network["ema_decay"]),
            target_samples_per_position_min=float(
                replay["target_samples_per_position_min"]
            ),
            target_samples_per_position_max=float(
                replay["target_samples_per_position_max"]
            ),
        )

    def learning_rate(self, step: int) -> float:
        if step < 0:
            raise ValueError("step must be non-negative")
        rate = self.initial_learning_rate
        for drop_step, drop_rate in zip(self.learning_rate_drop_steps, self.learning_rate_drops):
            if step >= drop_step:
                rate = drop_rate
        return rate

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["learning_rate_drops"] = list(self.learning_rate_drops)
        payload["learning_rate_drop_steps"] = list(self.learning_rate_drop_steps)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping) -> "LearnerConfig":
        data = dict(payload)
        data["learning_rate_drops"] = tuple(data["learning_rate_drops"])
        data["learning_rate_drop_steps"] = tuple(data["learning_rate_drop_steps"])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class LearnerMetrics:
    step: int
    batch_size: int
    learning_rate: float
    total_loss: float
    policy_loss: float
    value_loss: float
    auxiliary_loss: float
    opponent_policy_loss: float
    l2_loss: float
    root_policy_entropy: float
    samples_per_position: float

    def to_dict(self) -> dict:
        return asdict(self)


class _DifferentiableNetwork(nn.Module):
    def __init__(self, network: AlphaZeroNetwork) -> None:
        super().__init__()
        self.config = network.config
        self.parameters_by_name = nn.ParameterDict(
            {
                name: nn.Parameter(
                    torch.from_numpy(value.copy()),
                    requires_grad=not name.endswith(("_bn_mean", "_bn_var")),
                )
                for name, value in network.params.items()
            }
        )

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, ...]:
        parameters = self.parameters_by_name
        x = torch.relu(
            self._batch_norm(
                torch_functional.conv2d(
                observations, parameters["stem_w"], parameters["stem_b"], padding=1
                ),
                "stem",
            )
        )
        for block in range(self.config.blocks):
            residual = x
            y = torch.relu(
                self._batch_norm(
                    torch_functional.conv2d(
                    x,
                    parameters[f"block{block}_conv1_w"],
                    parameters[f"block{block}_conv1_b"],
                    padding=1,
                    ),
                    f"block{block}_conv1",
                )
            )
            if block in self.config.global_pool_blocks:
                pooled_channels = parameters[f"block{block}_global_w"].shape[1] // 2
                pooled = y[:, :pooled_channels]
                features = torch.cat(
                    (pooled.mean(dim=(2, 3)), pooled.amax(dim=(2, 3))), dim=1
                )
                bias = torch_functional.linear(
                    features,
                    parameters[f"block{block}_global_w"],
                    parameters[f"block{block}_global_b"],
                )
                y = y + bias[:, :, None, None]
            y = self._batch_norm(
                torch_functional.conv2d(
                    y,
                    parameters[f"block{block}_conv2_w"],
                    parameters[f"block{block}_conv2_b"],
                    padding=1,
                ),
                f"block{block}_conv2",
            )
            x = torch.relu(residual + y)

        policy = self._policy_head(x, "policy")
        opponent_policy = self._policy_head(x, "opponent_policy")
        value = self._value_head(x)
        distances = self._distance_head(x)
        return policy, value, distances, opponent_policy

    def _policy_head(self, trunk: torch.Tensor, prefix: str) -> torch.Tensor:
        parameters = self.parameters_by_name
        x = torch.relu(
            self._batch_norm(
                torch_functional.conv2d(
                    trunk,
                    parameters[f"{prefix}_conv_w"],
                    parameters[f"{prefix}_conv_b"],
                ),
                f"{prefix}_conv",
            )
        )
        return torch_functional.linear(
            x.flatten(1), parameters[f"{prefix}_fc_w"], parameters[f"{prefix}_fc_b"]
        )

    def _value_head(self, trunk: torch.Tensor) -> torch.Tensor:
        parameters = self.parameters_by_name
        x = torch.relu(
            self._batch_norm(
                torch_functional.conv2d(
                    trunk, parameters["value_conv_w"], parameters["value_conv_b"]
                ),
                "value_conv",
            )
        )
        hidden = torch.relu(
            torch_functional.linear(
                x.flatten(1), parameters["value_fc1_w"], parameters["value_fc1_b"]
            )
        )
        return torch.tanh(
            torch_functional.linear(
                hidden, parameters["value_fc2_w"], parameters["value_fc2_b"]
            )
        ).flatten()

    def _distance_head(self, trunk: torch.Tensor) -> torch.Tensor:
        parameters = self.parameters_by_name
        x = torch.relu(
            torch_functional.conv2d(
                trunk, parameters["distance_conv_w"], parameters["distance_conv_b"]
            )
        )
        return torch_functional.linear(
            x.flatten(1), parameters["distance_fc_w"], parameters["distance_fc_b"]
        )

    def _batch_norm(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        parameters = self.parameters_by_name
        return torch_functional.batch_norm(
            x,
            parameters[f"{prefix}_bn_mean"],
            parameters[f"{prefix}_bn_var"],
            parameters[f"{prefix}_bn_scale"],
            parameters[f"{prefix}_bn_bias"],
            training=self.training,
            momentum=self.config.batch_norm_momentum,
            eps=self.config.batch_norm_epsilon,
        )


class AlphaZeroLearner:
    def __init__(
        self,
        network: AlphaZeroNetwork,
        config: LearnerConfig,
        game,
        *,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        self.network = network
        self.config = config
        self.game = game
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.step = int(network.metadata.get("step", 0))
        self.model = _DifferentiableNetwork(network).to(self.device)
        self.momentum_buffers = {
            name: np.zeros_like(value, dtype=np.float32) for name, value in network.params.items()
        }

    def train_step(self, replay_buffer: AlphaZeroReplayBuffer) -> LearnerMetrics:
        projected_ratio = (
            replay_buffer.gradient_samples_consumed + self.config.batch_size
        ) / max(1, replay_buffer.total_positions_added)
        if projected_ratio > self.config.target_samples_per_position_max:
            raise RuntimeError(
                "training step would exceed the configured samples-per-position maximum"
            )
        batch = replay_buffer.sample(self.config.batch_size, rng=self.rng)
        if self.config.mirror_augmentation:
            flags = self.rng.random(self.config.batch_size) < 0.5
            batch = mirror_replay_batch(batch, self.game, flags)

        self.model.zero_grad(set_to_none=True)
        losses = self._losses(batch)
        losses["total"].backward()
        learning_rate = self.config.learning_rate(self.step)

        with torch.no_grad():
            for name, parameter in self.model.parameters_by_name.items():
                if parameter.grad is None:
                    self.network.params[name] = (
                        parameter.detach().cpu().numpy().astype(np.float32).copy()
                    )
                    continue
                gradient = parameter.grad.detach().cpu().numpy().astype(np.float32)
                velocity = self.momentum_buffers[name]
                velocity *= self.config.momentum
                velocity += gradient
                updated = parameter.detach().cpu().numpy() - learning_rate * velocity
                parameter.copy_(torch.from_numpy(updated).to(self.device))
                self.network.params[name] = updated.astype(np.float32).copy()

        self.network.update_ema(decay=self.config.ema_decay)
        self.step += 1
        self.network.metadata["step"] = self.step
        return LearnerMetrics(
            step=self.step,
            batch_size=self.config.batch_size,
            learning_rate=learning_rate,
            total_loss=float(losses["total"].detach().cpu()),
            policy_loss=float(losses["policy"].detach().cpu()),
            value_loss=float(losses["value"].detach().cpu()),
            auxiliary_loss=float(losses["auxiliary"].detach().cpu()),
            opponent_policy_loss=float(losses["opponent"].detach().cpu()),
            l2_loss=float(losses["l2"].detach().cpu()),
            root_policy_entropy=float(losses["entropy"].detach().cpu()),
            samples_per_position=replay_buffer.samples_per_position_ratio,
        )

    def train(
        self,
        replay_buffer: AlphaZeroReplayBuffer,
        *,
        steps: int,
        on_step=None,
    ) -> LearnerMetrics:
        if steps < 1:
            raise ValueError("steps must be positive")
        latest: LearnerMetrics | None = None
        for _ in range(steps):
            latest = self.train_step(replay_buffer)
            if on_step is not None:
                on_step(latest)
        assert latest is not None
        return latest

    def _losses(self, batch: ReplayBatch) -> dict[str, torch.Tensor]:
        observations = torch.from_numpy(np.ascontiguousarray(batch.observations)).to(self.device)
        policies = torch.from_numpy(np.ascontiguousarray(batch.policies)).to(self.device)
        values = torch.from_numpy(np.ascontiguousarray(batch.values)).to(self.device)
        action_masks = torch.from_numpy(np.ascontiguousarray(batch.action_masks)).to(self.device)
        distance_targets = torch.from_numpy(
            np.ascontiguousarray(batch.auxiliary_distances)
        ).to(self.device)
        distance_mask = torch.from_numpy(
            np.ascontiguousarray(batch.auxiliary_target_mask)
        ).to(self.device)
        opponent_targets = torch.from_numpy(
            np.ascontiguousarray(batch.opponent_policies)
        ).to(self.device)
        opponent_masks = torch.from_numpy(
            np.ascontiguousarray(batch.opponent_action_masks)
        ).to(self.device)
        opponent_target_mask = torch.from_numpy(
            np.ascontiguousarray(batch.opponent_policy_target_mask)
        ).to(self.device)

        policy_logits, value_predictions, distance_predictions, opponent_logits = self.model(
            observations
        )
        policy_loss, entropy = _masked_policy_loss(
            policy_logits, policies, action_masks
        )
        value_loss = torch.mean((value_predictions - values) ** 2)
        auxiliary_loss = _masked_mean_square(
            distance_predictions, distance_targets, distance_mask
        )
        opponent_loss, _ = _masked_policy_loss(
            opponent_logits,
            opponent_targets,
            opponent_masks,
            row_mask=opponent_target_mask,
        )
        l2_loss = 0.5 * self.config.weight_decay * sum(
            torch.sum(parameter**2)
            for name, parameter in self.model.parameters_by_name.items()
            if name.endswith("_w")
        )
        total = (
            policy_loss
            + value_loss
            + self.config.auxiliary_loss_weight * auxiliary_loss
            + self.config.opponent_policy_loss_weight * opponent_loss
            + l2_loss
        )
        return {
            "total": total,
            "policy": policy_loss,
            "value": value_loss,
            "auxiliary": auxiliary_loss,
            "opponent": opponent_loss,
            "l2": l2_loss,
            "entropy": entropy,
        }

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        run_id: str,
        git_commit: str,
        config_hash: str,
    ) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(self.network.metadata)
        metadata.update(
            {
                "schema_version": 1,
                "step": self.step,
                "run_id": run_id,
                "git_commit": git_commit,
                "config_hash": config_hash,
                "config": self.network.config.to_dict(),
                "learner_config": self.config.to_dict(),
                "rng_state": self.rng.bit_generator.state,
            }
        )
        self.network.metadata = dict(metadata)
        payload: dict[str, np.ndarray] = {
            "metadata_json": np.asarray([json.dumps(metadata, sort_keys=True)])
        }
        for name, value in sorted(self.network.params.items()):
            payload[f"raw__{name}"] = value.astype(np.float32)
            payload[f"ema__{name}"] = self.network.ema_params[name].astype(np.float32)
            payload[f"momentum__{name}"] = self.momentum_buffers[name].astype(np.float32)
        np.savez_compressed(output, **payload)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        game,
        *,
        device: str | torch.device = "cpu",
    ) -> "AlphaZeroLearner":
        with np.load(Path(path), allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"][0]))
            network_config = AlphaZeroNetworkConfig.from_dict(metadata["config"])
            raw = {
                name.removeprefix("raw__"): payload[name].astype(np.float32)
                for name in payload.files
                if name.startswith("raw__")
            }
            ema = {
                name.removeprefix("ema__"): payload[name].astype(np.float32)
                for name in payload.files
                if name.startswith("ema__")
            }
            momentum = {
                name.removeprefix("momentum__"): payload[name].astype(np.float32)
                for name in payload.files
                if name.startswith("momentum__")
            }
        network = AlphaZeroNetwork(network_config, raw, ema_params=ema, metadata=metadata)
        learner = cls(
            network,
            LearnerConfig.from_dict(metadata["learner_config"]),
            game,
            device=device,
        )
        learner.step = int(metadata["step"])
        learner.momentum_buffers = momentum
        learner.rng.bit_generator.state = metadata["rng_state"]
        return learner


def mirror_replay_batch(batch: ReplayBatch, game, flags: np.ndarray) -> ReplayBatch:
    flags = np.asarray(flags, dtype=np.bool_)
    if flags.shape != (batch.observations.shape[0],):
        raise ValueError("mirror flags must have one entry per batch row")
    observations = batch.observations.copy()
    policies = batch.policies.copy()
    action_masks = batch.action_masks.copy()
    opponent_policies = batch.opponent_policies.copy()
    opponent_action_masks = batch.opponent_action_masks.copy()
    for index in np.flatnonzero(flags):
        row = int(index)
        observations[row] = observations[row, :, :, ::-1]
        policies[row] = game.mirror(policies[row])
        action_masks[row] = game.mirror(action_masks[row])
        opponent_policies[row] = game.mirror(opponent_policies[row])
        opponent_action_masks[row] = game.mirror(opponent_action_masks[row])
    return ReplayBatch(
        observations=observations,
        policies=policies,
        values=batch.values.copy(),
        action_masks=action_masks,
        state_keys=batch.state_keys,
        auxiliary_distances=batch.auxiliary_distances.copy(),
        auxiliary_target_mask=batch.auxiliary_target_mask.copy(),
        opponent_policies=opponent_policies,
        opponent_action_masks=opponent_action_masks,
        opponent_policy_target_mask=batch.opponent_policy_target_mask.copy(),
    )


def _masked_policy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    legal_mask: torch.Tensor,
    *,
    row_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    active_rows = (
        torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
        if row_mask is None
        else row_mask.bool()
    )
    if not bool(torch.any(active_rows)):
        zero = logits.sum() * 0.0
        return zero, zero
    selected_logits = logits[active_rows]
    selected_targets = targets[active_rows]
    selected_legal = legal_mask[active_rows]
    masked_logits = torch.where(
        selected_legal, selected_logits, torch.full_like(selected_logits, -1e9)
    )
    log_probabilities = masked_logits - torch.logsumexp(masked_logits, dim=1, keepdim=True)
    loss = -torch.sum(selected_targets * log_probabilities, dim=1).mean()
    probabilities = torch.exp(log_probabilities)
    entropy = -torch.sum(probabilities * log_probabilities, dim=1).mean()
    return loss, entropy


def _masked_mean_square(
    predictions: torch.Tensor, targets: torch.Tensor, row_mask: torch.Tensor
) -> torch.Tensor:
    active_rows = row_mask.bool()
    if not bool(torch.any(active_rows)):
        return predictions.sum() * 0.0
    return torch.mean((predictions[active_rows] - targets[active_rows]) ** 2)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the M2 AlphaZero network from replay.")
    parser.add_argument("--config", type=Path, default=Path("configs/m2_5x5.json"))
    parser.add_argument("--replay", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--git-commit", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    project_config = load_config(args.config)
    game = small_game_from_config(project_config)
    cfg_hash = calculate_config_hash(project_config)
    if args.steps < 0:
        raise ValueError("--steps must be non-negative")
    if args.resume is None:
        learner = AlphaZeroLearner(
            AlphaZeroNetwork.from_config(project_config, seed=args.seed),
            LearnerConfig.from_project_config(project_config),
            game,
            seed=args.seed,
            device=args.device,
        )
    else:
        learner = AlphaZeroLearner.load_checkpoint(args.resume, game, device=args.device)
        checkpoint_hash = learner.network.metadata.get("config_hash")
        if checkpoint_hash not in (None, cfg_hash):
            raise ValueError("resume checkpoint config hash does not match --config")
    if args.steps == 0:
        if args.resume is not None:
            raise ValueError("zero-step initialization cannot be combined with --resume")
        latest = None
    else:
        if args.replay is None:
            raise ValueError("--replay is required when --steps is positive")
        replay = AlphaZeroReplayBuffer.load_npz(args.replay)
        latest = learner.train(replay, steps=args.steps)
    learner.save_checkpoint(
        args.output,
        run_id=args.run_id,
        git_commit=args.git_commit,
        config_hash=cfg_hash,
    )
    if latest is not None:
        replay.save_npz(args.replay)
        payload = latest.to_dict()
    else:
        payload = {
            "step": 0,
            "initialized": True,
            "run_id": args.run_id,
            "config_hash": cfg_hash,
        }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
