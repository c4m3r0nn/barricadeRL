import numpy as np

from barricade_rl.external_oracle import external_legal_actions
from barricade_rl.game import Game, TerminalStatus
from barricade_rl.verify import external_game_verification


def test_external_oracle_matches_initial_position():
    game = Game()
    state = game.initial_state()
    assert np.array_equal(external_legal_actions(state), game.legal_actions(state))


def test_external_oracle_matches_random_reachable_positions():
    game = Game()
    rng = np.random.default_rng(731)
    state = game.initial_state()
    compared = 0
    while compared < 100:
        assert np.array_equal(external_legal_actions(state), game.legal_actions(state))
        compared += 1
        legal = np.flatnonzero(game.legal_actions(state))
        if not legal.size:
            assert game.is_terminal(state) is not TerminalStatus.NOT_TERMINAL
            state = game.initial_state()
        else:
            state = game.next_state(state, int(rng.choice(legal)))


def test_external_oracle_matches_complete_randomized_games():
    metrics = external_game_verification(games=25, seed=991)
    assert metrics["games"] == 25
    assert metrics["states"] > 25
