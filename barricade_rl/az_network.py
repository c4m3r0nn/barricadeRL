from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class AlphaZeroNetworkConfig:
    board_size: int
    input_planes: int
    action_count: int
    blocks: int
    filters: int
    global_pool_blocks: tuple[int, ...]
    policy_head: str
    auxiliary_targets: tuple[str, ...]
    ema_decay: float
    observation_version: int = 1
    batch_norm_epsilon: float = 1e-5
    batch_norm_momentum: float = 0.1
    policy_conv_filters: int = 4
    value_conv_filters: int = 2
    distance_conv_filters: int = 1
    value_hidden: int = 96

    @classmethod
    def from_project_config(cls, config: Mapping) -> "AlphaZeroNetworkConfig":
        board = config["board"]
        action_space = config["action_space"]
        network = config["network"]
        board_size = int(board["size"])
        filters = int(network["filters"])
        return cls(
            board_size=board_size,
            input_planes=6,
            action_count=int(action_space["action_count"]),
            blocks=int(network["blocks"]),
            filters=filters,
            global_pool_blocks=tuple(int(index) for index in network.get("global_pool_blocks", ())),
            policy_head=str(action_space["policy_head"]),
            auxiliary_targets=tuple(str(target) for target in network.get("auxiliary_targets", ())),
            ema_decay=float(network.get("ema_decay", 0.999)),
            observation_version=int(config.get("observation", {}).get("version", 1)),
            batch_norm_epsilon=float(network.get("batch_norm_epsilon", 1e-5)),
            batch_norm_momentum=float(network.get("batch_norm_momentum", 0.1)),
            value_hidden=int(network.get("value_hidden", filters)),
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["global_pool_blocks"] = list(self.global_pool_blocks)
        payload["auxiliary_targets"] = list(self.auxiliary_targets)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping) -> "AlphaZeroNetworkConfig":
        data = dict(payload)
        data["global_pool_blocks"] = tuple(data["global_pool_blocks"])
        data["auxiliary_targets"] = tuple(data["auxiliary_targets"])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class NetworkOutput:
    policy_logits: np.ndarray
    value: np.ndarray
    auxiliary_distances: np.ndarray
    opponent_policy_logits: np.ndarray


class AlphaZeroNetwork:
    def __init__(
        self,
        config: AlphaZeroNetworkConfig,
        params: Mapping[str, np.ndarray],
        *,
        ema_params: Mapping[str, np.ndarray] | None = None,
        metadata: Mapping | None = None,
    ) -> None:
        self.config = config
        self.params = {name: np.asarray(value, dtype=np.float32).copy() for name, value in params.items()}
        _ensure_batch_norm_params(self.params, config)
        self.ema_params = (
            {name: np.asarray(value, dtype=np.float32).copy() for name, value in ema_params.items()}
            if ema_params is not None
            else {name: value.copy() for name, value in self.params.items()}
        )
        _ensure_batch_norm_params(self.ema_params, config)
        self.metadata = dict(metadata or {})

    @classmethod
    def from_config(cls, config: Mapping, *, seed: int = 0) -> "AlphaZeroNetwork":
        network_config = AlphaZeroNetworkConfig.from_project_config(config)
        rng = np.random.default_rng(seed)
        params = _initial_params(network_config, rng)
        metadata = {
            "schema_version": 1,
            "step": 0,
            "config": network_config.to_dict(),
        }
        return cls(network_config, params, metadata=metadata)

    def forward(self, observations: np.ndarray, *, use_ema: bool = True) -> NetworkOutput:
        observations = np.asarray(observations, dtype=np.float32)
        if observations.ndim == 3:
            observations = observations[None, ...]
        if observations.shape[1:] != (self.config.input_planes, self.config.board_size, self.config.board_size):
            raise ValueError(
                "observations must have shape "
                f"(batch, {self.config.input_planes}, {self.config.board_size}, {self.config.board_size})"
            )

        params = self.ema_params if use_ema else self.params
        trunk = self._trunk(observations, params)
        policy_logits = self._policy_head(trunk, params, prefix="policy")
        opponent_policy_logits = self._policy_head(trunk, params, prefix="opponent_policy")
        value = self._value_head(trunk, params)
        auxiliary_distances = self._distance_head(trunk, params)
        return NetworkOutput(
            policy_logits=policy_logits.astype(np.float32),
            value=value.astype(np.float32),
            auxiliary_distances=auxiliary_distances.astype(np.float32),
            opponent_policy_logits=opponent_policy_logits.astype(np.float32),
        )

    def evaluate(self, game, state) -> tuple[np.ndarray, float]:
        output = self.forward(game.canonical_observation(state)[None, ...])
        return output.policy_logits[0], float(output.value[0])

    def update_ema(self, *, decay: float | None = None) -> None:
        decay = self.config.ema_decay if decay is None else float(decay)
        if not 0.0 <= decay <= 1.0:
            raise ValueError("EMA decay must be between 0 and 1")
        for name, value in self.params.items():
            self.ema_params[name] = (decay * self.ema_params[name] + (1.0 - decay) * value).astype(np.float32)

    def save_checkpoint(self, path: str | Path, *, step: int, config_hash: str | None = None) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(self.metadata)
        metadata["schema_version"] = 1
        metadata["step"] = int(step)
        metadata["config_hash"] = config_hash
        metadata["config"] = self.config.to_dict()
        payload: dict[str, np.ndarray] = {
            "metadata_json": np.asarray([json.dumps(metadata, sort_keys=True)]),
        }
        for name, value in sorted(self.params.items()):
            payload[f"raw__{name}"] = value.astype(np.float32)
        for name, value in sorted(self.ema_params.items()):
            payload[f"ema__{name}"] = value.astype(np.float32)
        np.savez_compressed(output, **payload)

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> "AlphaZeroNetwork":
        with np.load(Path(path), allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"][0]))
            config = AlphaZeroNetworkConfig.from_dict(metadata["config"])
            params = {
                key.removeprefix("raw__"): payload[key].astype(np.float32)
                for key in payload.files
                if key.startswith("raw__")
            }
            ema_params = {
                key.removeprefix("ema__"): payload[key].astype(np.float32)
                for key in payload.files
                if key.startswith("ema__")
            }
        return cls(config, params, ema_params=ema_params, metadata=metadata)

    def _trunk(self, observations: np.ndarray, params: Mapping[str, np.ndarray]) -> np.ndarray:
        x = _relu(
            _batch_norm_inference(
                _conv2d_same(observations, params["stem_w"], params["stem_b"]),
                params,
                "stem",
                self.config.batch_norm_epsilon,
            )
        )
        for block in range(self.config.blocks):
            residual = x
            y = _relu(
                _batch_norm_inference(
                    _conv2d_same(
                        x,
                        params[f"block{block}_conv1_w"],
                        params[f"block{block}_conv1_b"],
                    ),
                    params,
                    f"block{block}_conv1",
                    self.config.batch_norm_epsilon,
                )
            )
            if block in self.config.global_pool_blocks:
                y = y + _global_pool_bias(y, params[f"block{block}_global_w"], params[f"block{block}_global_b"])
            y = _batch_norm_inference(
                _conv2d_same(
                    y,
                    params[f"block{block}_conv2_w"],
                    params[f"block{block}_conv2_b"],
                ),
                params,
                f"block{block}_conv2",
                self.config.batch_norm_epsilon,
            )
            x = _relu(residual + y)
        return x

    def _policy_head(self, trunk: np.ndarray, params: Mapping[str, np.ndarray], *, prefix: str) -> np.ndarray:
        x = _relu(
            _batch_norm_inference(
                _conv2d_1x1(
                    trunk, params[f"{prefix}_conv_w"], params[f"{prefix}_conv_b"]
                ),
                params,
                f"{prefix}_conv",
                self.config.batch_norm_epsilon,
            )
        )
        flat = x.reshape(x.shape[0], -1)
        return flat @ params[f"{prefix}_fc_w"].T + params[f"{prefix}_fc_b"]

    def _value_head(self, trunk: np.ndarray, params: Mapping[str, np.ndarray]) -> np.ndarray:
        x = _relu(
            _batch_norm_inference(
                _conv2d_1x1(trunk, params["value_conv_w"], params["value_conv_b"]),
                params,
                "value_conv",
                self.config.batch_norm_epsilon,
            )
        )
        flat = x.reshape(x.shape[0], -1)
        hidden = _relu(flat @ params["value_fc1_w"].T + params["value_fc1_b"])
        return np.tanh(hidden @ params["value_fc2_w"].T + params["value_fc2_b"]).reshape(-1)

    def _distance_head(self, trunk: np.ndarray, params: Mapping[str, np.ndarray]) -> np.ndarray:
        x = _relu(_conv2d_1x1(trunk, params["distance_conv_w"], params["distance_conv_b"]))
        flat = x.reshape(x.shape[0], -1)
        return flat @ params["distance_fc_w"].T + params["distance_fc_b"]


def _initial_params(config: AlphaZeroNetworkConfig, rng: np.random.Generator) -> dict[str, np.ndarray]:
    params: dict[str, np.ndarray] = {}
    filters = config.filters
    board = config.board_size
    params["stem_w"] = _he(rng, (filters, config.input_planes, 3, 3))
    params["stem_b"] = np.zeros(filters, dtype=np.float32)
    _add_batch_norm_params(params, "stem", filters)
    for block in range(config.blocks):
        params[f"block{block}_conv1_w"] = _he(rng, (filters, filters, 3, 3))
        params[f"block{block}_conv1_b"] = np.zeros(filters, dtype=np.float32)
        _add_batch_norm_params(params, f"block{block}_conv1", filters)
        params[f"block{block}_conv2_w"] = _he(rng, (filters, filters, 3, 3))
        params[f"block{block}_conv2_b"] = np.zeros(filters, dtype=np.float32)
        _add_batch_norm_params(params, f"block{block}_conv2", filters)
        if block in config.global_pool_blocks:
            pooled = max(1, filters // 4)
            params[f"block{block}_global_w"] = _he(rng, (filters, 2 * pooled))
            params[f"block{block}_global_b"] = np.zeros(filters, dtype=np.float32)

    _add_policy_params(params, rng, config, prefix="policy")
    _add_policy_params(params, rng, config, prefix="opponent_policy")
    params["value_conv_w"] = _he(rng, (config.value_conv_filters, filters, 1, 1))
    params["value_conv_b"] = np.zeros(config.value_conv_filters, dtype=np.float32)
    _add_batch_norm_params(params, "value_conv", config.value_conv_filters)
    params["value_fc1_w"] = _he(rng, (config.value_hidden, config.value_conv_filters * board * board))
    params["value_fc1_b"] = np.zeros(config.value_hidden, dtype=np.float32)
    params["value_fc2_w"] = _he(rng, (1, config.value_hidden))
    params["value_fc2_b"] = np.zeros(1, dtype=np.float32)
    params["distance_conv_w"] = _he(rng, (config.distance_conv_filters, filters, 1, 1))
    params["distance_conv_b"] = np.zeros(config.distance_conv_filters, dtype=np.float32)
    params["distance_fc_w"] = _he(rng, (2, config.distance_conv_filters * board * board))
    params["distance_fc_b"] = np.zeros(2, dtype=np.float32)
    return params


def _add_policy_params(
    params: dict[str, np.ndarray],
    rng: np.random.Generator,
    config: AlphaZeroNetworkConfig,
    *,
    prefix: str,
) -> None:
    board = config.board_size
    params[f"{prefix}_conv_w"] = _he(rng, (config.policy_conv_filters, config.filters, 1, 1))
    params[f"{prefix}_conv_b"] = np.zeros(config.policy_conv_filters, dtype=np.float32)
    _add_batch_norm_params(params, f"{prefix}_conv", config.policy_conv_filters)
    params[f"{prefix}_fc_w"] = _he(rng, (config.action_count, config.policy_conv_filters * board * board))
    params[f"{prefix}_fc_b"] = np.zeros(config.action_count, dtype=np.float32)


def _he(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    if len(shape) == 2:
        fan_in = shape[1]
    else:
        fan_in = int(np.prod(shape[1:]))
    scale = np.sqrt(2.0 / max(1, fan_in))
    return rng.normal(0.0, scale, size=shape).astype(np.float32)


def _conv2d_same(x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    kernel_h, kernel_w = int(weights.shape[2]), int(weights.shape[3])
    pad_h, pad_w = kernel_h // 2, kernel_w // 2
    padded = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel_h, kernel_w), axis=(2, 3))
    out = np.tensordot(windows, weights, axes=([1, 4, 5], [1, 2, 3]))
    return np.moveaxis(out, -1, 1).astype(np.float32) + bias[None, :, None, None]


def _conv2d_1x1(x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    out = np.tensordot(x, weights[:, :, 0, 0], axes=([1], [1]))
    return np.moveaxis(out, -1, 1).astype(np.float32) + bias[None, :, None, None]


def _global_pool_bias(x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    pooled_channels = weights.shape[1] // 2
    pooled = x[:, :pooled_channels]
    features = np.concatenate(
        (
            pooled.mean(axis=(2, 3)),
            pooled.max(axis=(2, 3)),
        ),
        axis=1,
    )
    block_bias = features @ weights.T + bias
    return block_bias[:, :, None, None].astype(np.float32)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0).astype(np.float32)


def _add_batch_norm_params(params: dict[str, np.ndarray], prefix: str, channels: int) -> None:
    params[f"{prefix}_bn_scale"] = np.ones(channels, dtype=np.float32)
    params[f"{prefix}_bn_bias"] = np.zeros(channels, dtype=np.float32)
    params[f"{prefix}_bn_mean"] = np.zeros(channels, dtype=np.float32)
    params[f"{prefix}_bn_var"] = np.ones(channels, dtype=np.float32)


def _ensure_batch_norm_params(
    params: dict[str, np.ndarray], config: AlphaZeroNetworkConfig
) -> None:
    prefixes = [("stem", config.filters), ("value_conv", config.value_conv_filters)]
    prefixes.extend(
        (f"block{block}_{convolution}", config.filters)
        for block in range(config.blocks)
        for convolution in ("conv1", "conv2")
    )
    prefixes.extend(
        (f"{head}_conv", config.policy_conv_filters)
        for head in ("policy", "opponent_policy")
    )
    for prefix, channels in prefixes:
        if f"{prefix}_bn_scale" not in params:
            _add_batch_norm_params(params, prefix, channels)


def _batch_norm_inference(
    x: np.ndarray,
    params: Mapping[str, np.ndarray],
    prefix: str,
    epsilon: float,
) -> np.ndarray:
    mean = params[f"{prefix}_bn_mean"][None, :, None, None]
    variance = params[f"{prefix}_bn_var"][None, :, None, None]
    scale = params[f"{prefix}_bn_scale"][None, :, None, None]
    bias = params[f"{prefix}_bn_bias"][None, :, None, None]
    return ((x - mean) / np.sqrt(variance + epsilon) * scale + bias).astype(np.float32)
