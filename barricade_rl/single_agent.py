from __future__ import annotations

import gymnasium as gym
import numpy as np
from dataclasses import dataclass
from gymnasium import spaces

from barricade_rl.core import ACTION_COUNT, BOARD_SIZE, MOVE_ACTIONS, WALLS_PER_PLAYER, BarricadeGame, BarricadeState, canonical_action_to_absolute, opponent_of
from barricade_rl.opponents import OpponentPolicy, RandomOpponent


@dataclass(slots=True)
class EndgameStartSampler:
    min_distance: int = 2
    max_distance: int = 4
    min_walls_remaining: int = 4
    max_walls_remaining: int = WALLS_PER_PLAYER

    def sample(self, rng: np.random.Generator, learner_side: int) -> BarricadeState:
        state = BarricadeState()
        learner_distance = int(rng.integers(self.min_distance, self.max_distance + 1))
        learner_col = int(rng.integers(2, BOARD_SIZE - 2))
        if learner_side == 0:
            learner_pos = (learner_distance, learner_col)
            opponent_pos = (int(rng.integers(5, BOARD_SIZE)), int(rng.integers(2, BOARD_SIZE - 2)))
        else:
            learner_pos = (BOARD_SIZE - 1 - learner_distance, learner_col)
            opponent_pos = (int(rng.integers(0, 4)), int(rng.integers(2, BOARD_SIZE - 2)))
        if opponent_pos == learner_pos:
            opponent_pos = (opponent_pos[0], (opponent_pos[1] + 1) % BOARD_SIZE)
        state.pawns[learner_side] = learner_pos
        state.pawns[opponent_of(learner_side)] = opponent_pos
        state.walls_remaining = [
            int(rng.integers(self.min_walls_remaining, self.max_walls_remaining + 1)),
            int(rng.integers(self.min_walls_remaining, self.max_walls_remaining + 1)),
        ]
        state.current_player = learner_side
        state.move_count = int(rng.integers(12, 31))
        return state


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
        wall_penalty: float = 0.0,
        reverse_move_penalty: float = 0.0,
        progress_reward_scale: float = 0.0,
        survival_reward: float = 0.0,
        opponent_wall_value_penalty_scale: float = 0.0,
        endgame_start_probability: float = 0.0,
        endgame_start_sampler: EndgameStartSampler | None = None,
        learner_side: int | None = 0,
    ):
        self.render_mode = render_mode
        self.invalid_action = invalid_action
        self.opponent = opponent or RandomOpponent()
        self.shaped_reward = shaped_reward
        self.path_reward_scale = path_reward_scale
        self.opponent_path_reward_scale = opponent_path_reward_scale
        self.step_penalty = step_penalty
        self.wall_penalty = wall_penalty
        self.reverse_move_penalty = reverse_move_penalty
        self.progress_reward_scale = progress_reward_scale
        self.survival_reward = survival_reward
        self.opponent_wall_value_penalty_scale = opponent_wall_value_penalty_scale
        self.endgame_start_probability = endgame_start_probability
        self.endgame_start_sampler = endgame_start_sampler or EndgameStartSampler()
        self.learner_side_setting = learner_side
        self.learner_side = 0 if learner_side is None else learner_side
        self.game = BarricadeGame(max_moves=max_moves)
        self.action_space = spaces.Discrete(ACTION_COUNT)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        self._rng = np.random.default_rng()
        self._last_learner_move: int | None = None
        self._episode_endgame_start = False

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self._last_learner_move = None
        self._episode_endgame_start = False
        self.game.reset()
        if self.learner_side_setting is None:
            self.learner_side = int(self._rng.integers(2))
        else:
            self.learner_side = self.learner_side_setting
        info_extra = {"endgame_start": False}
        if self.endgame_start_probability and self._rng.random() < self.endgame_start_probability:
            self.game.set_state(self.endgame_start_sampler.sample(self._rng, self.learner_side))
            info_extra["endgame_start"] = True
            self._episode_endgame_start = True
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
        action = int(action)
        absolute_action = canonical_action_to_absolute(action, self.learner_side)
        legal = self.game.apply_action(absolute_action)
        if not legal:
            if self.invalid_action == "raise":
                raise ValueError(f"Illegal learner action {action}")
            if self.invalid_action == "loss":
                self.game.state.winner = opponent_side
                return self.game.observation(canonical=True), -1.0, True, False, self._info(illegal_action=True)
            return self.game.observation(canonical=True), -0.01, False, self.game.truncated(), self._info(illegal_action=True)

        action_shaping = self._action_shaping_reward(action, prev_learner_path)
        if self.game.terminated:
            shaped = self._shaped_reward(prev_learner_path, prev_opponent_path)
            reward = 1.0 + shaped + action_shaping
            self._remember_learner_move(action)
            return self.game.observation(canonical=True), reward, True, False, self._info(
                shaped_reward=shaped,
                action_shaping_reward=action_shaping,
            )
        if self.game.truncated():
            self._remember_learner_move(action)
            return self.game.observation(canonical=True), action_shaping, False, True, self._info(
                action_shaping_reward=action_shaping,
            )

        learner_path_after_action = self.game.shortest_path_length(self.learner_side)
        opponent_action = self.opponent.select_action(self.game, self._rng)
        if not self.game.apply_action(opponent_action):
            raise RuntimeError(f"Opponent selected illegal action {opponent_action}")

        terminated = self.game.terminated
        truncated = self.game.truncated()
        reward = -1.0 if terminated and self.game.state.winner == opponent_side else 0.0
        shaped = self._shaped_reward(prev_learner_path, prev_opponent_path)
        anti_rush_shaping = self._anti_rush_shaping_reward(opponent_action, learner_path_after_action, terminated)
        reward += shaped + action_shaping + anti_rush_shaping["total"]
        self._remember_learner_move(action)
        return self.game.observation(canonical=True), reward, terminated, truncated, self._info(
            opponent_action=opponent_action,
            shaped_reward=shaped,
            action_shaping_reward=action_shaping,
            survival_reward=anti_rush_shaping["survival"],
            opponent_wall_value_delta=anti_rush_shaping["wall_delta"],
            opponent_wall_value_reward=anti_rush_shaping["wall_reward"],
        )

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

    def _action_shaping_reward(self, action: int, prev_learner_path: int | None) -> float:
        reward = 0.0
        if action >= 4:
            reward -= self.wall_penalty
        elif self._is_reverse_move(action):
            reward -= self.reverse_move_penalty
        if action in MOVE_ACTIONS and prev_learner_path is not None and self.progress_reward_scale:
            learner_path = self.game.shortest_path_length(self.learner_side)
            if learner_path is not None and learner_path < prev_learner_path:
                reward += self.progress_reward_scale * (prev_learner_path - learner_path)
        return reward

    def _anti_rush_shaping_reward(self, opponent_action: int, learner_path_before_opponent: int | None, terminated: bool) -> dict[str, float]:
        survival = self.survival_reward if not terminated else 0.0
        wall_delta = 0.0
        wall_reward = 0.0
        if opponent_action >= 4 and learner_path_before_opponent is not None and self.opponent_wall_value_penalty_scale:
            learner_path_after_opponent = self.game.shortest_path_length(self.learner_side)
            if learner_path_after_opponent is not None:
                wall_delta = max(0.0, float(learner_path_after_opponent - learner_path_before_opponent))
                wall_reward = -self.opponent_wall_value_penalty_scale * wall_delta
        return {
            "survival": survival,
            "wall_delta": wall_delta,
            "wall_reward": wall_reward,
            "total": survival + wall_reward,
        }

    def _is_reverse_move(self, action: int) -> bool:
        if action not in MOVE_ACTIONS or self._last_learner_move is None:
            return False
        return (self._last_learner_move, action) in {(0, 1), (1, 0), (2, 3), (3, 2)}

    def _remember_learner_move(self, action: int) -> None:
        if action in MOVE_ACTIONS:
            self._last_learner_move = action

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
            "survival_reward": 0.0,
            "opponent_wall_value_delta": 0.0,
            "opponent_wall_value_reward": 0.0,
            "endgame_start": self._episode_endgame_start,
        }
        info.update(extra)
        return info
