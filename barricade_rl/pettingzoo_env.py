from __future__ import annotations

import numpy as np
from gymnasium import spaces

from barricade_rl.core import ACTION_COUNT, BOARD_SIZE, BarricadeGame, canonical_action_to_absolute

try:
    from pettingzoo import AECEnv
    from pettingzoo.utils import wrappers
except ImportError:  # Keep base installs usable without PettingZoo.
    AECEnv = object
    wrappers = None


def raw_env(**kwargs):
    return BarricadeAECEnv(**kwargs)


def env(**kwargs):
    environment = raw_env(**kwargs)
    if wrappers is None:
        return environment
    environment = wrappers.AssertOutOfBoundsWrapper(environment)
    environment = wrappers.OrderEnforcingWrapper(environment)
    return environment


class BarricadeAECEnv(AECEnv):
    """Small PettingZoo AEC-style wrapper around BarricadeGame.

    PettingZoo is intentionally optional for now. This class exposes the core
    AEC fields and methods used by PettingZoo tooling while keeping the base
    project install light.
    """

    metadata = {"name": "barricade_v0", "render_modes": ["ansi"]}
    possible_agents = ["player_0", "player_1"]

    def __init__(self, max_moves: int = 500):
        self.game = BarricadeGame(max_moves=max_moves)
        self.agents = []
        self.agent_selection = "player_0"
        self.rewards = {}
        self.terminations = {}
        self.truncations = {}
        self.infos = {}
        self._observation_space = spaces.Dict(
            {
                "observation": spaces.Box(low=0.0, high=1.0, shape=(6, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
                "action_mask": spaces.Box(low=0, high=1, shape=(ACTION_COUNT,), dtype=np.int8),
            }
        )
        self._action_space = spaces.Discrete(ACTION_COUNT)
        self.observation_spaces = {agent: self._observation_space for agent in self.possible_agents}
        self.action_spaces = {agent: self._action_space for agent in self.possible_agents}
        self._cumulative_rewards = {}

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.game.reset()
        self.agents = list(self.possible_agents)
        self.agent_selection = "player_0"
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

    def observe(self, agent: str):
        player = self.possible_agents.index(agent)
        original_player = self.game.state.current_player
        self.game.state.current_player = player
        try:
            obs = self.game.observation(canonical=True)
            mask = self.game.legal_actions_mask(canonical=True).astype(np.int8)
        finally:
            self.game.state.current_player = original_player
        return {"observation": obs, "action_mask": mask}

    def step(self, action: int):
        if not self.agents:
            return
        agent = self.agent_selection
        player = self.possible_agents.index(agent)
        if self.terminations[agent] or self.truncations[agent]:
            if hasattr(self, "_was_dead_step"):
                self._was_dead_step(action)
            else:
                self._advance_agent()
            return
        if self.game.state.current_player != player:
            raise RuntimeError(f"Expected {self.possible_agents[self.game.state.current_player]} to act")

        actor = self.game.state.current_player
        legal = self.game.apply_action(canonical_action_to_absolute(int(action), actor))
        if not legal:
            raise ValueError(f"Illegal action {action}")

        self._cumulative_rewards[agent] = 0.0
        self.rewards = {agent_name: 0.0 for agent_name in self.agents}
        if self.game.terminated:
            winner_agent = self.possible_agents[self.game.state.winner]
            loser_agent = self.possible_agents[1 - self.game.state.winner]
            self.rewards[winner_agent] = 1.0
            self.rewards[loser_agent] = -1.0
            self.terminations = {agent_name: True for agent_name in self.agents}
        elif self.game.truncated():
            self.truncations = {agent_name: True for agent_name in self.agents}
        self.infos[agent]["last_action"] = int(action)
        self.agent_selection = self.possible_agents[self.game.state.current_player] if not self.game.terminated else self.possible_agents[actor]
        if hasattr(self, "_accumulate_rewards"):
            self._accumulate_rewards()

    def action_space(self, agent: str):
        return self._action_space

    def observation_space(self, agent: str):
        return self._observation_space

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

    def close(self):
        return None

    def _advance_agent(self):
        current = self.possible_agents.index(self.agent_selection)
        self.agent_selection = self.possible_agents[1 - current]
