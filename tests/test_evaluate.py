import numpy as np

from barricade_rl.evaluate import select_policy_action
from barricade_rl.single_agent import BarricadeSingleAgentEnv


def test_random_eval_policy_selects_legal_action():
    env = BarricadeSingleAgentEnv()
    env.reset(seed=0)
    action = select_policy_action("random", env, np.random.default_rng(0))
    assert env.action_masks()[action]


def test_greedy_eval_policy_selects_legal_action():
    env = BarricadeSingleAgentEnv()
    env.reset(seed=0)
    action = select_policy_action("greedy", env, np.random.default_rng(0))
    assert env.action_masks()[action]
