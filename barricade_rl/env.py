from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .game import ACTION_COUNT, BOARD_SIZE, Game, State, TerminalStatus
from .opponents import OpponentPolicy, RandomOpponent


class _LegalDiscrete(spaces.Discrete):
    def __init__(self, n: int, mask_provider) -> None:
        super().__init__(n)
        self._mask_provider = mask_provider

    def sample(self, mask=None, probability=None):
        if mask is None and probability is None:
            mask = self._mask_provider().astype(np.int8)
        return super().sample(mask=mask, probability=probability)


class QuoridorEnv(gym.Env):
    """Single-agent Gymnasium wrapper that executes the opponent reply internally."""

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        opponent_policy: OpponentPolicy | None = None,
        learner_player: int = 0,
        render_mode: str | None = None,
        max_plies: int = 200,
    ) -> None:
        if learner_player not in (0, 1):
            raise ValueError("learner_player must be 0 or 1")
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'")
        self.game = Game(max_plies=max_plies)
        self.opponent_policy = opponent_policy or RandomOpponent()
        self.learner_player = learner_player
        self.render_mode = render_mode
        self.state = self.game.initial_state()
        self.action_space = _LegalDiscrete(ACTION_COUNT, self.action_mask)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(6, BOARD_SIZE, BOARD_SIZE),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options
        self.state = self.game.initial_state()
        extra: dict[str, int] = {}
        if self.learner_player == 1:
            action = self.opponent_policy.select_action(self.game, self.state, self.np_random)
            self._assert_opponent_action(action)
            self.state = self.game.next_state(self.state, action)
            extra["opponent_opening_action"] = action
        return self.game.canonical_observation(self.state), self._info(**extra)

    def step(self, action: int):
        if self.game.is_terminal(self.state) is not TerminalStatus.NOT_TERMINAL:
            raise RuntimeError("cannot step a terminal environment; call reset")
        if self.state.current_player != self.learner_player:
            raise RuntimeError("environment is not at the learner decision point")
        action = int(action)
        if not 0 <= action < ACTION_COUNT or not self.game.legal_actions(self.state)[action]:
            raise ValueError(f"illegal action {action}")

        self.state = self.game.next_state(self.state, action)
        if self.game.is_terminal(self.state) is not TerminalStatus.NOT_TERMINAL:
            return self._transition()

        opponent_action = self.opponent_policy.select_action(self.game, self.state, self.np_random)
        self._assert_opponent_action(opponent_action)
        self.state = self.game.next_state(self.state, opponent_action)
        return self._transition(opponent_action=opponent_action)

    def action_mask(self) -> np.ndarray:
        if self.game.is_terminal(self.state) is not TerminalStatus.NOT_TERMINAL:
            return np.zeros(ACTION_COUNT, dtype=np.bool_)
        return self.game.legal_actions(self.state)

    def action_masks(self) -> np.ndarray:
        return self.action_mask()

    def render(self) -> str:
        return self.game.render(self.state)

    def _assert_opponent_action(self, action: int) -> None:
        if not 0 <= action < ACTION_COUNT or not self.game.legal_actions(self.state)[action]:
            raise RuntimeError(f"opponent policy selected illegal action {action}")

    def _transition(self, **extra):
        status = self.game.is_terminal(self.state)
        terminated = status is TerminalStatus.MOVER_LOST
        truncated = status is TerminalStatus.CAPPED
        winner = 1 - self.state.current_player if terminated else None
        reward = 0.0
        if winner is not None:
            reward = 1.0 if winner == self.learner_player else -1.0
        return (
            self.game.canonical_observation(self.state),
            reward,
            terminated,
            truncated,
            self._info(**extra),
        )

    def _info(self, **extra):
        status = self.game.is_terminal(self.state)
        winner = 1 - self.state.current_player if status is TerminalStatus.MOVER_LOST else None
        info = {
            "action_mask": self.action_mask(),
            "current_player": self.state.current_player,
            "winner": winner,
            "pawns": self.state.pawns,
            "walls_remaining": self.state.walls_remaining,
            "ply": self.state.ply,
        }
        info.update(extra)
        return info
