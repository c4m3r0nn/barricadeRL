from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from barricade_rl.core import ACTION_COUNT, BOARD_SIZE, BarricadeGame
from barricade_rl.opponents import OpponentPolicy, RandomOpponent


class BarricadeSingleAgentEnv(gym.Env):
    """Learner-as-player-0 environment with an automatic opponent turn."""

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        opponent: OpponentPolicy | None = None,
        render_mode: str | None = None,
        max_moves: int = 500,
        invalid_action: str = "raise",
    ):
        self.render_mode = render_mode
        self.invalid_action = invalid_action
        self.opponent = opponent or RandomOpponent()
        self.game = BarricadeGame(max_moves=max_moves)
        self.action_space = spaces.Discrete(ACTION_COUNT)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        self._rng = np.random.default_rng()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self.game.reset()
        return self.game.observation(canonical=True), self._info()

    def step(self, action: int):
        if self.game.state.current_player != 0:
            raise RuntimeError("Single-agent env expected learner turn")

        legal = self.game.apply_action(int(action))
        if not legal:
            if self.invalid_action == "raise":
                raise ValueError(f"Illegal learner action {action}")
            if self.invalid_action == "loss":
                self.game.state.winner = 1
                return self.game.observation(canonical=True), -1.0, True, False, self._info(illegal_action=True)
            return self.game.observation(canonical=True), -0.01, False, self.game.truncated(), self._info(illegal_action=True)

        if self.game.terminated:
            return self.game.observation(canonical=True), 1.0, True, False, self._info()
        if self.game.truncated():
            return self.game.observation(canonical=True), 0.0, False, True, self._info()

        opponent_action = self.opponent.select_action(self.game, self._rng)
        if not self.game.apply_action(opponent_action):
            raise RuntimeError(f"Opponent selected illegal action {opponent_action}")

        terminated = self.game.terminated
        truncated = self.game.truncated()
        reward = -1.0 if terminated and self.game.state.winner == 1 else 0.0
        return self.game.observation(canonical=True), reward, terminated, truncated, self._info(opponent_action=opponent_action)

    def action_mask(self) -> np.ndarray:
        return self.game.legal_actions_mask()

    def action_masks(self) -> np.ndarray:
        return self.action_mask()

    def render(self):
        rows = []
        for row in range(BOARD_SIZE):
            cells = []
            for col in range(BOARD_SIZE):
                pos = (row, col)
                if pos == self.game.state.pawns[0]:
                    cells.append("0")
                elif pos == self.game.state.pawns[1]:
                    cells.append("1")
                else:
                    cells.append(".")
            rows.append(" ".join(cells))
        return "\n".join(rows)

    def _info(self, **extra):
        info = {
            "action_mask": self.action_mask(),
            "current_player": self.game.state.current_player,
            "winner": self.game.state.winner,
            "walls_remaining": tuple(self.game.state.walls_remaining),
            "pawns": tuple(self.game.state.pawns),
        }
        info.update(extra)
        return info
