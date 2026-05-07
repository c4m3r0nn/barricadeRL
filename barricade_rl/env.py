from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from barricade_rl.core import ACTION_COUNT, BOARD_SIZE, BarricadeGame, canonical_action_to_absolute


class BarricadeEnv(gym.Env):
    """Gymnasium environment for compact Barricade training."""

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, render_mode: str | None = None, max_moves: int = 500, invalid_action: str = "raise"):
        self.render_mode = render_mode
        self.invalid_action = invalid_action
        self.game = BarricadeGame(max_moves=max_moves)
        self.action_space = spaces.Discrete(ACTION_COUNT)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.game.reset()
        return self.game.observation(canonical=True), self._info()

    def step(self, action: int):
        actor = self.game.state.current_player
        legal = self.game.apply_action(canonical_action_to_absolute(int(action), actor))
        if not legal:
            if self.invalid_action == "raise":
                raise ValueError(f"Illegal action {action}")
            if self.invalid_action == "loss":
                self.game.state.winner = 1 - actor
                return self.game.observation(canonical=True), -1.0, True, False, self._info(illegal_action=True)
            return self.game.observation(canonical=True), -0.01, False, self.game.truncated(), self._info(illegal_action=True)

        terminated = self.game.terminated
        truncated = self.game.truncated()
        reward = 0.0
        if terminated:
            reward = 1.0 if self.game.state.winner == actor else -1.0
        return self.game.observation(canonical=True), reward, terminated, truncated, self._info()

    def action_mask(self) -> np.ndarray:
        return self.game.legal_actions_mask(canonical=True)

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
