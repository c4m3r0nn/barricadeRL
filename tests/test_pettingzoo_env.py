import numpy as np

from barricade_rl.pettingzoo_env import BarricadeAECEnv


def test_pettingzoo_env_reset_and_step():
    env = BarricadeAECEnv()
    env.reset(seed=0)

    assert env.agents == ["player_0", "player_1"]
    assert env.agent_selection == "player_0"

    obs = env.observe("player_0")
    assert obs["observation"].shape == (6, 9, 9)
    assert obs["action_mask"].shape == (132,)

    legal_action = int(np.flatnonzero(obs["action_mask"])[0])
    env.step(legal_action)
    assert env.agent_selection == "player_1"
