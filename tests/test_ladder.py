import numpy as np
import pytest

from barricade_rl.game import Game, State, TerminalStatus
from barricade_rl.opponents import (
    FROZEN_LADDER,
    LADDER_VERSION,
    AlphaBetaOpponent,
    GreedyRacer,
    HeuristicOne,
    RandomOpponent,
)


@pytest.mark.parametrize("policy", [RandomOpponent(), GreedyRacer(), HeuristicOne()])
def test_ladder_policy_always_returns_legal_action(policy):
    game = Game()
    state = game.initial_state()
    rng = np.random.default_rng(4)
    for _ in range(20):
        if game.is_terminal(state) is not TerminalStatus.NOT_TERMINAL:
            break
        action = policy.select_action(game, state, rng)
        assert game.legal_actions(state)[action]
        state = game.next_state(state, action)


def test_greedy_racer_moves_forward_and_never_places_walls():
    action = GreedyRacer().select_action(Game(), Game().initial_state(), np.random.default_rng(0))
    assert action == 0


@pytest.mark.parametrize("depth", [3, 5])
def test_alpha_beta_finds_immediate_win(depth):
    game = Game()
    state = State.from_components(
        pawns=((7, 4), (8, 8)),
        walls_remaining=(0, 0),
        current_player=0,
    )
    policy = AlphaBetaOpponent(depth=depth)
    assert policy.select_action(game, state, np.random.default_rng(0)) == 0


@pytest.mark.parametrize("depth", [3, 5])
def test_alpha_beta_searches_race_position(depth):
    game = Game()
    state = State.from_components(
        pawns=((6, 4), (2, 4)),
        walls_remaining=(0, 0),
        current_player=0,
    )
    assert AlphaBetaOpponent(depth).select_action(game, state, np.random.default_rng(0)) == 0


def test_reference_ladder_is_versioned_and_fixed_in_strength_order():
    assert LADDER_VERSION == 1
    assert tuple(policy.name for policy in FROZEN_LADDER) == (
        "random",
        "greedy-racer",
        "heuristic-1",
        "alpha-beta-d3",
        "alpha-beta-d5",
    )
