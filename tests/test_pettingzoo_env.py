import numpy as np
import pytest

from barricade_rl.pettingzoo_env import BarricadeAECEnv, env, raw_env


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


def test_pettingzoo_env_last_and_agent_iter():
    env = BarricadeAECEnv()
    env.reset(seed=0)
    agent = next(iter(env.agent_iter(max_iter=1)))
    observation, reward, termination, truncation, info = env.last()

    assert agent == "player_0"
    assert observation["observation"].shape == (6, 9, 9)
    assert observation["action_mask"].dtype == np.int8
    assert reward == 0
    assert not termination
    assert not truncation


def test_pettingzoo_factory_functions():
    assert isinstance(raw_env(), BarricadeAECEnv)
    wrapped = env()
    wrapped.reset(seed=0)
    assert wrapped.agent_selection == "player_0"


def test_pettingzoo_api_contract():
    api_test = pytest.importorskip("pettingzoo.test").api_test
    api_test(BarricadeAECEnv(), num_cycles=20, verbose_progress=False)
