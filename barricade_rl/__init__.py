"""AlphaZero-oriented Quoridor engine."""

from gymnasium.envs.registration import register, registry

from .game import Game, State, TerminalStatus

ENV_ID = "BarricadeRL/Quoridor-v0"

if ENV_ID not in registry:
    register(id=ENV_ID, entry_point="barricade_rl.env:QuoridorEnv")

__all__ = [
    "ENV_ID",
    "Game",
    "QuoridorEnv",
    "SmallBoardSpec",
    "SmallGame",
    "SmallState",
    "SolverOutcome",
    "State",
    "TerminalStatus",
]


def __getattr__(name):
    if name == "QuoridorEnv":
        from .env import QuoridorEnv

        return QuoridorEnv
    if name in {"SmallBoardSpec", "SmallGame", "SmallState", "SolverOutcome"}:
        from . import small_board

        return getattr(small_board, name)
    raise AttributeError(name)
