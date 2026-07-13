import json
from pathlib import Path

import pytest

from barricade_rl.game import Game


def _reference_perft(game, state, depth):
    if depth == 0:
        return 1
    total = 0
    for action, legal in enumerate(game.legal_actions(state)):
        if legal:
            total += _reference_perft(game, game.next_state(state, action), depth - 1)
    return total


def test_native_perft_matches_reference_recursion():
    game = Game()
    state = game.initial_state()
    assert game.perft(state, 0) == 1
    assert game.perft(state, 1) == 131
    assert game.perft(state, 2) == _reference_perft(game, state, 2)


PERFT_CORPUS = json.loads((Path(__file__).parent / "data" / "perft.json").read_text())


@pytest.mark.parametrize("position", PERFT_CORPUS["positions"], ids=lambda position: position["name"])
def test_perft_golden_corpus(position):
    game = Game()
    state = game.initial_state()
    for action in position["actions"]:
        state = game.next_state(state, action)

    assert state.data.hex() == position["state_key"]
    assert [game.perft(state, depth) for depth in range(1, 5)] == position["counts"]
