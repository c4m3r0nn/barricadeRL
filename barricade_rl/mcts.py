from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .game import TerminalStatus


class Evaluator(Protocol):
    def evaluate(self, game, state) -> tuple[np.ndarray, float]: ...


@dataclass(frozen=True, slots=True)
class MCTSConfig:
    simulations: int = 200
    cpuct: float = 1.6
    temperature: float = 1.0
    fpu_reduction: float = 0.2
    root_dirichlet_alpha: float | None = None
    root_noise_fraction: float = 0.0
    forced_playouts: bool = False
    policy_target_pruning: bool = False
    forced_playout_weight: float = 2.0

    def __post_init__(self) -> None:
        if self.simulations < 1:
            raise ValueError("simulations must be positive")
        if self.cpuct <= 0:
            raise ValueError("cpuct must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.fpu_reduction < 0:
            raise ValueError("fpu_reduction must be non-negative")
        if self.root_dirichlet_alpha is not None and self.root_dirichlet_alpha <= 0:
            raise ValueError("root_dirichlet_alpha must be positive when enabled")
        if not 0.0 <= self.root_noise_fraction <= 1.0:
            raise ValueError("root_noise_fraction must be in [0, 1]")
        if self.root_noise_fraction > 0.0 and self.root_dirichlet_alpha is None:
            raise ValueError("root_dirichlet_alpha is required when root noise is enabled")
        if self.policy_target_pruning and not self.forced_playouts:
            raise ValueError("policy_target_pruning requires forced_playouts")
        if self.forced_playout_weight <= 0:
            raise ValueError("forced_playout_weight must be positive")


@dataclass(frozen=True, slots=True)
class MCTSResult:
    action: int
    policy: np.ndarray
    root_value: float
    visits: np.ndarray
    root_priors: np.ndarray | None = None
    forced_visits: np.ndarray | None = None
    policy_target_visits: np.ndarray | None = None


@dataclass(slots=True)
class _Edge:
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    child: "_Node | None" = None
    forced_visit_count: int = 0

    @property
    def q(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0


@dataclass(slots=True)
class _Node:
    priors: np.ndarray | None = None
    edges: dict[int, _Edge] = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0

    @property
    def expanded(self) -> bool:
        return self.priors is not None

    @property
    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0


class UniformEvaluator:
    def evaluate(self, game, state) -> tuple[np.ndarray, float]:
        return np.zeros(len(game.legal_actions(state)), dtype=np.float32), 0.0


def masked_policy(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Convert logits to a normalized policy while keeping illegal actions at zero."""
    return _masked_softmax(np.asarray(logits, dtype=np.float32), np.asarray(mask, dtype=np.bool_))


class MCTS:
    def __init__(
        self,
        config: MCTSConfig,
        evaluator: Evaluator,
        *,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.config = config
        self.evaluator = evaluator
        self.rng = rng or np.random.default_rng()

    def run(self, game, state) -> MCTSResult:
        action_count = len(game.legal_actions(state))
        immediate_win = _immediate_winning_action(game, state)
        if immediate_win is not None:
            policy = np.zeros(action_count, dtype=np.float32)
            visits = np.zeros(action_count, dtype=np.float32)
            policy[immediate_win] = 1.0
            visits[immediate_win] = float(self.config.simulations)
            return MCTSResult(
                action=immediate_win,
                policy=policy,
                root_value=1.0,
                visits=visits,
                root_priors=policy.copy(),
                forced_visits=np.zeros(action_count, dtype=np.float32),
                policy_target_visits=visits.copy(),
            )

        root = _Node()
        noise_applied = self.config.root_noise_fraction == 0.0
        for _ in range(self.config.simulations):
            self._simulate(game, state, root, is_root=True)
            if root.expanded and not noise_applied:
                self._apply_root_noise(root)
                noise_applied = True

        visits = np.zeros(action_count, dtype=np.float32)
        forced_visits = np.zeros(action_count, dtype=np.float32)
        value_sum = 0.0
        total_visits = 0
        for action, edge in root.edges.items():
            visits[action] = edge.visit_count
            forced_visits[action] = edge.forced_visit_count
            value_sum += edge.value_sum
            total_visits += edge.visit_count

        if total_visits == 0:
            legal = game.legal_actions(state)
            visits = legal.astype(np.float32)
            total_visits = int(visits.sum())
        policy_target_visits = visits.copy()
        if self.config.policy_target_pruning:
            policy_target_visits = np.maximum(0.0, visits - forced_visits)
            if float(policy_target_visits.sum()) <= 0.0:
                policy_target_visits[int(np.argmax(visits))] = 1.0
        policy = _visits_to_policy(policy_target_visits, self.config.temperature)
        action = int(np.argmax(visits))
        root_value = value_sum / total_visits if total_visits else 0.0
        return MCTSResult(
            action=action,
            policy=policy,
            root_value=float(root_value),
            visits=visits,
            root_priors=None if root.priors is None else root.priors.copy(),
            forced_visits=forced_visits,
            policy_target_visits=policy_target_visits,
        )

    def _simulate(self, game, state, node: _Node, *, is_root: bool = False) -> float:
        status = game.is_terminal(state)
        if status is not TerminalStatus.NOT_TERMINAL:
            return float(game.terminal_value(state))

        immediate_win = _immediate_winning_action(game, state)
        if immediate_win is not None:
            priors = np.zeros(len(game.legal_actions(state)), dtype=np.float32)
            priors[immediate_win] = 1.0
            node.priors = priors
            node.edges = {immediate_win: _Edge(prior=1.0)}
            node.visit_count += 1
            node.value_sum += 1.0
            return 1.0

        if not node.expanded:
            logits, value = self.evaluator.evaluate(game, state)
            priors = _masked_softmax(np.asarray(logits, dtype=np.float32), game.legal_actions(state))
            node.priors = priors
            node.edges = {
                int(action): _Edge(prior=float(priors[action]))
                for action in np.flatnonzero(priors > 0)
            }
            node.visit_count += 1
            node.value_sum += float(value)
            return float(value)

        action, edge, forced = self._select_edge(node, is_root=is_root)
        child_state = game.next_state(state, action)
        if edge.child is None:
            edge.child = _Node()
        child_value = self._simulate(game, child_state, edge.child)
        value = -child_value
        edge.visit_count += 1
        if forced:
            edge.forced_visit_count += 1
        edge.value_sum += value
        node.visit_count += 1
        node.value_sum += value
        return value

    def _select_edge(self, node: _Node, *, is_root: bool) -> tuple[int, _Edge, bool]:
        assert node.expanded
        parent_visits = max(1, node.visit_count)
        if is_root and self.config.forced_playouts:
            forced_choice = self._forced_playout_choice(node, parent_visits)
            if forced_choice is not None:
                action, edge = forced_choice
                return action, edge, True
        best_action = -1
        best_edge: _Edge | None = None
        best_score = -float("inf")
        for action, edge in node.edges.items():
            exploration = self.config.cpuct * edge.prior * np.sqrt(parent_visits) / (1 + edge.visit_count)
            q = edge.q if edge.visit_count else node.value - self.config.fpu_reduction
            score = q + exploration
            if score > best_score or (score == best_score and action < best_action):
                best_score = score
                best_action = action
                best_edge = edge
        assert best_edge is not None
        return best_action, best_edge, False

    def _forced_playout_choice(self, node: _Node, parent_visits: int) -> tuple[int, _Edge] | None:
        best: tuple[float, int, _Edge] | None = None
        for action, edge in node.edges.items():
            minimum = np.sqrt(self.config.forced_playout_weight * edge.prior * parent_visits)
            deficit = float(minimum - edge.visit_count)
            if deficit <= 0.0:
                continue
            candidate = (deficit, -action, edge)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        if best is None:
            return None
        return -best[1], best[2]

    def _apply_root_noise(self, root: _Node) -> None:
        assert root.priors is not None
        actions = np.asarray(sorted(root.edges), dtype=np.int64)
        if actions.size == 0:
            return
        alpha = float(self.config.root_dirichlet_alpha)
        noise = self.rng.dirichlet(np.full(actions.size, alpha, dtype=np.float64))
        fraction = self.config.root_noise_fraction
        for index, action in enumerate(actions):
            edge = root.edges[int(action)]
            edge.prior = float((1.0 - fraction) * edge.prior + fraction * noise[index])
            root.priors[int(action)] = edge.prior


def _masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if logits.shape != mask.shape:
        raise ValueError(f"logits shape {logits.shape} does not match mask shape {mask.shape}")
    mask = np.asarray(mask, dtype=np.bool_)
    if not mask.any():
        return np.zeros_like(logits, dtype=np.float32)
    legal_logits = logits[mask]
    shifted = legal_logits - float(np.max(legal_logits))
    exp = np.exp(shifted).astype(np.float32)
    priors = np.zeros_like(logits, dtype=np.float32)
    priors[mask] = exp / float(exp.sum())
    return priors


def _immediate_winning_action(game, state) -> int | None:
    legal = game.legal_actions(state)
    for action in np.flatnonzero(legal[:12]):
        child = game.next_state(state, int(action))
        if game.is_terminal(child) is TerminalStatus.MOVER_LOST:
            return int(action)
    return None


def _visits_to_policy(visits: np.ndarray, temperature: float) -> np.ndarray:
    if visits.sum() <= 0:
        return np.zeros_like(visits, dtype=np.float32)
    if temperature == 0:
        policy = np.zeros_like(visits, dtype=np.float32)
        policy[int(np.argmax(visits))] = 1.0
        return policy
    scaled = visits.astype(np.float64) ** (1.0 / temperature)
    return (scaled / scaled.sum()).astype(np.float32)
