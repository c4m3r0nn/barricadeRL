from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from barricade_rl.core import ACTION_COUNT, BOARD_SIZE, BarricadeGame, canonical_action_to_absolute, opponent_of
from barricade_rl.opponents import OpponentPolicy, RandomOpponent


class BarricadeSingleAgentEnv(gym.Env):
    """Single learner environment with an automatic opponent turn."""

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        opponent: OpponentPolicy | None = None,
        render_mode: str | None = None,
        max_moves: int = 500,
        invalid_action: str = "raise",
        shaped_reward: bool = False,
        path_reward_scale: float = 0.01,
        opponent_path_reward_scale: float = 0.005,
        step_penalty: float = 0.0,
        learner_side: int | None = 0,
    ):
        self.render_mode = render_mode
        self.invalid_action = invalid_action
        self.opponent = opponent or RandomOpponent()
        self.shaped_reward = shaped_reward
        self.path_reward_scale = path_reward_scale
        self.opponent_path_reward_scale = opponent_path_reward_scale
        self.step_penalty = step_penalty
        self.learner_side_setting = learner_side
        self.learner_side = 0 if learner_side is None else learner_side
        self.game = BarricadeGame(max_moves=max_moves)
        self.action_space = spaces.Discrete(ACTION_COUNT)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        self._rng = np.random.default_rng()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self.game.reset()
        if self.learner_side_setting is None:
            self.learner_side = int(self._rng.integers(2))
        else:
            self.learner_side = self.learner_side_setting
        info_extra = {}
        if self.game.state.current_player != self.learner_side:
            opponent_action = self.opponent.select_action(self.game, self._rng)
            if not self.game.apply_action(opponent_action):
                raise RuntimeError(f"Opponent selected illegal opening action {opponent_action}")
            info_extra["opponent_opening_action"] = opponent_action
        return self.game.observation(canonical=True), self._info(**info_extra)

    def step(self, action: int):
        if self.game.state.current_player != self.learner_side:
            raise RuntimeError("Single-agent env expected learner turn")

        opponent_side = opponent_of(self.learner_side)
        prev_learner_path = self.game.shortest_path_length(self.learner_side)
        prev_opponent_path = self.game.shortest_path_length(opponent_side)
        absolute_action = canonical_action_to_absolute(int(action), self.learner_side)
        legal = self.game.apply_action(absolute_action)
        if not legal:
            if self.invalid_action == "raise":
                raise ValueError(f"Illegal learner action {action}")
            if self.invalid_action == "loss":
                self.game.state.winner = opponent_side
                return self.game.observation(canonical=True), -1.0, True, False, self._info(illegal_action=True)
            return self.game.observation(canonical=True), -0.01, False, self.game.truncated(), self._info(illegal_action=True)

        if self.game.terminated:
            reward = 1.0 + self._shaped_reward(prev_learner_path, prev_opponent_path)
            return self.game.observation(canonical=True), reward, True, False, self._info(shaped_reward=reward - 1.0)
        if self.game.truncated():
            return self.game.observation(canonical=True), 0.0, False, True, self._info()

        opponent_action = self.opponent.select_action(self.game, self._rng)
        if not self.game.apply_action(opponent_action):
            raise RuntimeError(f"Opponent selected illegal action {opponent_action}")

        terminated = self.game.terminated
        truncated = self.game.truncated()
        reward = -1.0 if terminated and self.game.state.winner == opponent_side else 0.0
        shaped = self._shaped_reward(prev_learner_path, prev_opponent_path)
        reward += shaped
        return self.game.observation(canonical=True), reward, terminated, truncated, self._info(opponent_action=opponent_action, shaped_reward=shaped)

    def _shaped_reward(self, prev_learner_path: int | None, prev_opponent_path: int | None) -> float:
        if not self.shaped_reward:
            return 0.0
        opponent_side = opponent_of(self.learner_side)
        learner_path = self.game.shortest_path_length(self.learner_side)
        opponent_path = self.game.shortest_path_length(opponent_side)
        shaped = self.step_penalty
        if prev_learner_path is not None and learner_path is not None:
            shaped += self.path_reward_scale * (prev_learner_path - learner_path)
        if prev_opponent_path is not None and opponent_path is not None:
            shaped -= self.opponent_path_reward_scale * (prev_opponent_path - opponent_path)
        return shaped

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
            "learner_side": self.learner_side,
            "winner": self.game.state.winner,
            "walls_remaining": tuple(self.game.state.walls_remaining),
            "pawns": tuple(self.game.state.pawns),
        }
        info.update(extra)
        return info
