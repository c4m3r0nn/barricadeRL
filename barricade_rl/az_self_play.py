from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Protocol

import numpy as np

from .az_replay import AlphaZeroReplayBuffer, ReplaySample, make_replay_sample
from .game import TerminalStatus
from .mcts import Evaluator, MCTS, MCTSConfig, MCTSResult, masked_policy


class RandomSource(Protocol):
    def random(self) -> float: ...

    def choice(self, count: int, p: np.ndarray) -> int: ...


class Search(Protocol):
    def run(self, game, state) -> MCTSResult: ...


SearchFactory = Callable[[MCTSConfig, Evaluator, RandomSource], Search]


@dataclass(frozen=True, slots=True)
class SelfPlayConfig:
    full_simulations: int = 200
    fast_simulations: int = 50
    cpuct: float = 1.6
    temperature_moves: int = 16
    full_search_probability: float = 0.25
    raw_policy_injection_probability: float = 0.04
    diversification_plies: int = 8
    fpu_reduction: float = 0.2
    root_dirichlet_alpha: float = 0.6
    root_noise_fraction: float = 0.25
    forced_playouts: bool = True
    policy_target_pruning: bool = True
    forced_playout_weight: float = 2.0
    observation_version: int = 1
    scoring_scheme: str = "terminal-win-loss-cap-zero"

    def __post_init__(self) -> None:
        if self.full_simulations < 1:
            raise ValueError("full_simulations must be positive")
        if not 1 <= self.fast_simulations <= self.full_simulations:
            raise ValueError("fast_simulations must be between 1 and full_simulations")
        if self.cpuct <= 0:
            raise ValueError("cpuct must be positive")
        if self.temperature_moves < 0:
            raise ValueError("temperature_moves must be non-negative")
        if not 0.0 <= self.full_search_probability <= 1.0:
            raise ValueError("full_search_probability must be in [0, 1]")
        if not 0.0 <= self.raw_policy_injection_probability <= 1.0:
            raise ValueError("raw_policy_injection_probability must be in [0, 1]")
        if self.diversification_plies < 0:
            raise ValueError("diversification_plies must be non-negative")
        if self.fpu_reduction < 0:
            raise ValueError("fpu_reduction must be non-negative")
        if self.root_dirichlet_alpha <= 0:
            raise ValueError("root_dirichlet_alpha must be positive")
        if not 0.0 <= self.root_noise_fraction <= 1.0:
            raise ValueError("root_noise_fraction must be in [0, 1]")
        if self.policy_target_pruning and not self.forced_playouts:
            raise ValueError("policy_target_pruning requires forced_playouts")
        if self.forced_playout_weight <= 0:
            raise ValueError("forced_playout_weight must be positive")
        if self.observation_version < 1:
            raise ValueError("observation_version must be positive")
        if not self.scoring_scheme:
            raise ValueError("scoring_scheme must be non-empty")
        if self.scoring_scheme not in (
            "terminal-win-loss-cap-zero",
            "terminal-win-loss-cap-shortest-path-adjudicated",
        ):
            raise ValueError("unsupported self-play scoring scheme")

    @classmethod
    def from_project_config(cls, config: Mapping) -> "SelfPlayConfig":
        mcts = config["mcts"]
        self_play = config["self_play"]
        diversification = self_play.get("weak_start_state_diversification", {})
        full_simulations = int(mcts["self_play_simulations"])
        fast_fraction = float(self_play.get("fast_search_fraction", 0.25))
        fast_simulations = max(1, int(round(full_simulations * fast_fraction)))
        injection_probability = (
            float(diversification.get("raw_policy_injection_probability", 0.04))
            if diversification.get("enabled", True)
            else 0.0
        )
        return cls(
            full_simulations=full_simulations,
            fast_simulations=fast_simulations,
            cpuct=float(mcts["cpuct_init"]),
            temperature_moves=int(mcts["temperature_moves"]),
            full_search_probability=float(self_play.get("full_search_probability", 0.25)),
            raw_policy_injection_probability=injection_probability,
            diversification_plies=int(diversification.get("diversification_plies", 8)),
            fpu_reduction=float(mcts.get("fpu_reduction", 0.2)),
            root_dirichlet_alpha=float(mcts["root_dirichlet_alpha"]),
            root_noise_fraction=float(mcts["root_noise_fraction"]),
            forced_playouts=bool(mcts.get("forced_playouts", True)),
            policy_target_pruning=bool(mcts.get("policy_target_pruning", True)),
            forced_playout_weight=float(mcts.get("forced_playout_weight", 2.0)),
            observation_version=int(config.get("observation", {}).get("version", 1)),
            scoring_scheme=str(self_play["reward"]),
        )


@dataclass(frozen=True, slots=True)
class SelfPlayGameRecord:
    game_id: str
    samples: tuple[ReplaySample, ...]
    actions: tuple[int, ...]
    terminal_status: TerminalStatus
    winner: int | None
    plies: int
    full_searches: int
    fast_searches: int
    injected_ply: int | None

    @property
    def capped(self) -> bool:
        return self.terminal_status is TerminalStatus.CAPPED


@dataclass(frozen=True, slots=True)
class _PendingSample:
    state: object
    policy: np.ndarray
    root_value: float


def play_self_play_game(
    game,
    evaluator: Evaluator,
    config: SelfPlayConfig,
    *,
    rng: RandomSource | None = None,
    initial_state=None,
    search_factory: SearchFactory | None = None,
    game_id: str = "self-play-00000000",
    run_id: str | None = None,
    config_hash: str | None = None,
    git_commit: str | None = None,
) -> SelfPlayGameRecord:
    """Play one game and return only full-search positions with final mover-frame values."""
    rng = rng or np.random.default_rng()
    search_factory = search_factory or _default_search_factory
    state = game.initial_state() if initial_state is None else initial_state
    starting_ply = int(state.ply)
    pending: list[_PendingSample] = []
    actions: list[int] = []
    full_searches = 0
    fast_searches = 0
    injected_ply: int | None = None
    policies_by_ply: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    while game.is_terminal(state) is TerminalStatus.NOT_TERMINAL:
        relative_ply = int(state.ply) - starting_ply
        can_inject = (
            injected_ply is None
            and relative_ply < config.diversification_plies
            and config.raw_policy_injection_probability > 0.0
            and rng.random() < config.raw_policy_injection_probability
        )

        if can_inject:
            action_mask = game.legal_actions(state)
            logits, _ = evaluator.evaluate(game, state)
            policy = masked_policy(logits, action_mask)
            policies_by_ply[int(state.ply)] = (policy.copy(), action_mask.copy())
            action = _sample_policy(policy, rng)
            injected_ply = int(state.ply)
            # All prior outcomes are now conditional on an off-search intervention.
            pending.clear()
        else:
            full_search = rng.random() < config.full_search_probability
            simulations = config.full_simulations if full_search else config.fast_simulations
            search_config = MCTSConfig(
                simulations=simulations,
                cpuct=config.cpuct,
                temperature=1.0,
                fpu_reduction=config.fpu_reduction,
                root_dirichlet_alpha=config.root_dirichlet_alpha if full_search else None,
                root_noise_fraction=config.root_noise_fraction if full_search else 0.0,
                forced_playouts=config.forced_playouts if full_search else False,
                policy_target_pruning=config.policy_target_pruning if full_search else False,
                forced_playout_weight=config.forced_playout_weight,
            )
            result = search_factory(search_config, evaluator, rng).run(game, state)
            policies_by_ply[int(state.ply)] = (
                np.asarray(result.policy, dtype=np.float32).copy(),
                game.legal_actions(state).copy(),
            )
            if full_search:
                full_searches += 1
                pending.append(
                    _PendingSample(
                        state=state,
                        policy=np.asarray(result.policy, dtype=np.float32).copy(),
                        root_value=float(result.root_value),
                    )
                )
            else:
                fast_searches += 1
            action = (
                _sample_policy(result.policy, rng)
                if relative_ply < config.temperature_moves
                else int(np.argmax(result.visits))
            )

        actions.append(action)
        state = game.next_state(state, action)

    terminal_status = game.is_terminal(state)
    if terminal_status is TerminalStatus.MOVER_LOST:
        winner = 1 - int(state.current_player)
    elif (
        terminal_status is TerminalStatus.CAPPED
        and config.scoring_scheme == "terminal-win-loss-cap-shortest-path-adjudicated"
    ):
        winner = _adjudicated_winner(game, state)
    else:
        winner = None
    source = "self-play"
    samples = tuple(
        make_replay_sample(
            game,
            item.state,
            policy=item.policy,
            value=_value_for_player(int(item.state.current_player), winner),
            source=source,
            config_hash=config_hash,
            root_value=item.root_value,
            observation_version=config.observation_version,
            scoring_scheme=config.scoring_scheme,
            game_id=game_id,
            run_id=run_id,
            git_commit=git_commit,
            opponent_policy=(
                policies_by_ply[int(item.state.ply) + 1][0]
                if int(item.state.ply) + 1 in policies_by_ply
                else None
            ),
            opponent_action_mask=(
                policies_by_ply[int(item.state.ply) + 1][1]
                if int(item.state.ply) + 1 in policies_by_ply
                else None
            ),
        )
        for item in pending
    )
    return SelfPlayGameRecord(
        game_id=game_id,
        samples=samples,
        actions=tuple(actions),
        terminal_status=terminal_status,
        winner=winner,
        plies=int(state.ply) - starting_ply,
        full_searches=full_searches,
        fast_searches=fast_searches,
        injected_ply=injected_ply,
    )


def generate_self_play_games(
    game,
    evaluator: Evaluator,
    config: SelfPlayConfig,
    *,
    games: int,
    replay_buffer: AlphaZeroReplayBuffer,
    run_id: str,
    config_hash: str,
    git_commit: str,
    rng: RandomSource | None = None,
    initial_state_factory: Callable[[int], object] | None = None,
    search_factory: SearchFactory | None = None,
    game_id_prefix: str | None = None,
) -> tuple[SelfPlayGameRecord, ...]:
    if games < 1:
        raise ValueError("games must be positive")
    rng = rng or np.random.default_rng()
    records: list[SelfPlayGameRecord] = []
    prefix = run_id if game_id_prefix is None else game_id_prefix
    for index in range(games):
        initial_state = None if initial_state_factory is None else initial_state_factory(index)
        record = play_self_play_game(
            game,
            evaluator,
            config,
            rng=rng,
            initial_state=initial_state,
            search_factory=search_factory,
            game_id=f"{prefix}-{index:08d}",
            run_id=run_id,
            config_hash=config_hash,
            git_commit=git_commit,
        )
        replay_buffer.extend(record.samples)
        records.append(record)
    return tuple(records)


def _default_search_factory(config: MCTSConfig, evaluator: Evaluator, rng: RandomSource) -> Search:
    return MCTS(config, evaluator, rng=rng)  # type: ignore[arg-type]


def _sample_policy(policy: np.ndarray, rng: RandomSource) -> int:
    probabilities = np.asarray(policy, dtype=np.float64)
    total = float(probabilities.sum())
    if probabilities.ndim != 1 or not np.isfinite(probabilities).all() or total <= 0.0:
        raise ValueError("cannot sample an invalid or empty policy")
    probabilities = probabilities / total
    return int(rng.choice(probabilities.shape[0], p=probabilities))


def _value_for_player(player: int, winner: int | None) -> float:
    if winner is None:
        return 0.0
    return 1.0 if player == winner else -1.0


def _adjudicated_winner(game, state) -> int | None:
    distances = tuple(game.shortest_path_distance(state, player) for player in (0, 1))
    if distances[0] is None or distances[1] is None:
        return None
    if distances[0] < distances[1]:
        return 0
    if distances[1] < distances[0]:
        return 1
    # At equal integer distances, the mover owns the half-tempo tie-break.
    return int(state.current_player)
