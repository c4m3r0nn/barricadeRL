"""Barricade RL environment package."""

__all__ = ["BarricadeEnv", "BarricadeSingleAgentEnv"]


def __getattr__(name):
    if name == "BarricadeEnv":
        from barricade_rl.env import BarricadeEnv

        return BarricadeEnv
    if name == "BarricadeSingleAgentEnv":
        from barricade_rl.single_agent import BarricadeSingleAgentEnv

        return BarricadeSingleAgentEnv
    raise AttributeError(name)
