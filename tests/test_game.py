from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from barricade_rl.game import (
    ACTION_COUNT,
    ACTUAL_TO_CANONICAL,
    CANONICAL_TO_ACTUAL,
    MOVE_ACTIONS,
    TerminalStatus,
    Game,
    State,
    actual_to_canonical_action,
    canonical_to_actual_action,
    mirror_action,
    wall_action,
)


def state_with(
    *,
    pawns: tuple[tuple[int, int], tuple[int, int]] = ((0, 4), (8, 4)),
    horizontal_walls: tuple[tuple[int, int], ...] = (),
    vertical_walls: tuple[tuple[int, int], ...] = (),
    walls_remaining: tuple[int, int] = (10, 10),
    current_player: int = 0,
    ply: int | None = None,
) -> State:
    return State.from_components(
        pawns=pawns,
        horizontal_walls=horizontal_walls,
        vertical_walls=vertical_walls,
        walls_remaining=walls_remaining,
        current_player=current_player,
        ply=current_player if ply is None else ply,
    )


def test_initial_state_and_fixed_action_contract():
    game = Game()
    state = game.initial_state()

    assert ACTION_COUNT == 140
    assert state.pawns == ((0, 4), (8, 4))
    assert state.walls_remaining == (10, 10)
    assert state.current_player == 0
    assert state.ply == 0
    assert len(game.state_key(state)) == 20

    mask = game.legal_actions(state)
    assert mask.shape == (140,)
    assert mask.dtype == np.bool_
    assert mask[:12].tolist() == [True, False, True, True] + [False] * 8
    assert int(mask.sum()) == 131


def test_state_is_immutable_and_successors_do_not_mutate_parent():
    game = Game()
    parent = game.initial_state()
    child = game.next_state(parent, 0)

    assert parent.pawns[0] == (0, 4)
    assert child.pawns[0] == (1, 4)
    assert child.current_player == 1
    with pytest.raises((FrozenInstanceError, AttributeError)):
        child.data = parent.data


@pytest.mark.parametrize(
    ("pawns", "walls", "expected"),
    [
        (((4, 4), (5, 4)), (), {4}),
        (((4, 4), (5, 4)), ((5, 4),), {8, 9}),
        (((7, 4), (8, 4)), (), {8, 9}),
        (((4, 4), (3, 4)), (), {5}),
        (((4, 4), (4, 5)), (), {6}),
        (((4, 4), (4, 3)), (), {7}),
    ],
)
def test_jump_geometry_in_all_directions(pawns, walls, expected):
    game = Game()
    state = state_with(pawns=pawns, horizontal_walls=walls)
    legal_moves = set(np.flatnonzero(game.legal_actions(state)[:12]))

    assert expected <= legal_moves
    if expected & {4, 5, 6, 7}:
        assert not legal_moves & {8, 9, 10, 11}


def test_diagonal_jump_respects_each_side_wall():
    game = Game()
    state = state_with(
        pawns=((4, 4), (5, 4)),
        horizontal_walls=((5, 4),),
        vertical_walls=((5, 3),),
    )
    legal = set(np.flatnonzero(game.legal_actions(state)[:12]))

    assert 8 in legal  # NE: from the opponent to column 5
    assert 9 not in legal  # NW is blocked beside the opponent
    assert 4 not in legal  # N and NN are distinct and neither replaces a diagonal


def test_adjacent_pawns_separated_by_wall_do_not_jump():
    game = Game()
    state = state_with(pawns=((4, 4), (5, 4)), horizontal_walls=((4, 4),))
    legal = set(np.flatnonzero(game.legal_actions(state)[:12]))

    assert not legal & {4, 8, 9}


@pytest.mark.parametrize(
    ("pawns", "horizontal", "vertical", "expected"),
    [
        (((4, 4), (5, 4)), (), (), {1, 2, 3, 4}),
        (((4, 4), (5, 4)), ((5, 4),), (), {1, 2, 3, 8, 9}),
        (((4, 4), (5, 4)), ((5, 4),), ((4, 4),), {1, 3, 9}),
        (((4, 4), (5, 4)), ((5, 4),), ((4, 3),), {1, 2, 8}),
        (((4, 4), (5, 4)), ((5, 4),), ((4, 3), (4, 4)), {1}),
        (((7, 4), (8, 4)), (), (), {1, 2, 3, 8, 9}),
        (((4, 4), (3, 4)), (), (), {0, 2, 3, 5}),
        (((4, 4), (3, 4)), ((2, 4),), (), {0, 2, 3, 10, 11}),
        (((4, 4), (3, 4)), ((2, 4),), ((3, 4),), {0, 3, 11}),
        (((4, 4), (3, 4)), ((2, 4),), ((3, 3),), {0, 2, 10}),
        (((4, 4), (3, 4)), ((2, 4),), ((3, 3), (3, 4)), {0}),
        (((1, 4), (0, 4)), (), (), set()),  # opponent has already won
        (((4, 4), (4, 5)), (), (), {0, 1, 3, 6}),
        (((4, 4), (4, 5)), (), ((4, 5),), {0, 1, 3, 8, 10}),
        (((4, 4), (4, 5)), ((4, 4),), ((4, 5),), {1, 3, 10}),
        (((4, 4), (4, 5)), ((3, 4),), ((4, 5),), {0, 3, 8}),
        (((4, 4), (4, 5)), ((3, 4), (4, 4)), ((4, 5),), {3}),
        (((4, 7), (4, 8)), (), (), {0, 1, 3, 8, 10}),
        (((4, 4), (4, 3)), (), (), {0, 1, 2, 7}),
        (((4, 4), (4, 3)), (), ((4, 2),), {0, 1, 2, 9, 11}),
        (((4, 4), (4, 3)), ((4, 3),), ((4, 2),), {1, 2, 11}),
        (((4, 4), (4, 3)), ((3, 3),), ((4, 2),), {0, 2, 9}),
        (((4, 4), (4, 3)), ((3, 3), (4, 3)), ((4, 2),), {2}),
        (((4, 1), (4, 0)), (), (), {0, 1, 2, 9, 11}),
        (((4, 4), (5, 4)), ((4, 4),), (), {1, 2, 3}),
        (((4, 4), (3, 4)), ((3, 4),), (), {0, 2, 3}),
        (((4, 4), (4, 5)), (), ((4, 4),), {0, 1, 3}),
        (((4, 4), (4, 3)), (), ((4, 3),), {0, 1, 2}),
    ],
)
def test_exact_jump_geometry_battery(pawns, horizontal, vertical, expected):
    state = state_with(pawns=pawns, horizontal_walls=horizontal, vertical_walls=vertical)
    actual = set(np.flatnonzero(Game().legal_actions(state)[:12]))
    assert actual == expected


def test_wall_overlap_crossing_and_last_path_are_illegal():
    game = Game()
    state = state_with(horizontal_walls=((2, 2),))
    mask = game.legal_actions(state)

    assert not mask[wall_action("h", 2, 1)]
    assert not mask[wall_action("h", 2, 2)]
    assert not mask[wall_action("h", 2, 3)]
    assert not mask[wall_action("v", 2, 2)]



LAST_PATH_CASES = (
    (
        ((0, 6), (1, 0), (1, 2), (1, 4), (1, 6), (2, 4), (2, 6), (3, 5), (5, 6), (6, 6)),
        ((0, 0), (2, 2), (3, 1), (3, 7), (4, 6), (5, 2), (5, 3), (5, 4), (7, 5)),
        ("v", 0, 4),
    ),
    (
        ((0, 6), (1, 0), (1, 2), (1, 4), (1, 6), (2, 4), (2, 6), (3, 5), (4, 7), (5, 6), (6, 6)),
        ((0, 0), (2, 2), (3, 1), (3, 7), (4, 6), (5, 2), (5, 3), (5, 4), (7, 5)),
        ("h", 3, 3),
    ),
    (
        ((1, 2), (1, 6), (2, 0), (2, 2), (2, 6), (3, 6), (4, 5), (5, 6)),
        ((0, 7), (1, 1), (2, 4), (3, 0), (5, 1), (5, 3), (5, 4), (5, 7), (7, 5)),
        ("h", 1, 4),
    ),
    (
        ((1, 0), (1, 2), (1, 6), (2, 0), (2, 2), (2, 6), (3, 6), (4, 5), (5, 6)),
        ((0, 7), (1, 1), (2, 4), (3, 0), (5, 1), (5, 3), (5, 4), (5, 7), (7, 5)),
        ("h", 1, 4),
    ),
    (
        ((1, 0), (1, 2), (1, 6), (2, 0), (2, 2), (2, 6), (3, 6), (4, 5), (5, 6), (7, 0)),
        ((0, 7), (1, 1), (2, 4), (3, 0), (5, 1), (5, 3), (5, 4), (5, 7), (7, 5)),
        ("h", 1, 4),
    ),
    (
        ((1, 0), (1, 2), (1, 6), (2, 0), (2, 2), (2, 6), (3, 6), (4, 5), (5, 6), (7, 0)),
        ((0, 0), (0, 7), (1, 1), (2, 4), (3, 0), (5, 1), (5, 3), (5, 4), (5, 7), (7, 5)),
        ("h", 1, 4),
    ),
    (
        ((1, 4), (3, 2), (4, 2), (5, 1), (5, 4), (5, 6), (6, 6)),
        ((0, 0), (0, 1), (0, 2), (1, 5), (2, 2), (4, 3), (4, 4), (5, 0), (5, 5), (5, 7), (6, 1), (7, 4)),
        ("h", 4, 7),
    ),
    (
        ((1, 4), (3, 2), (4, 2), (5, 1), (5, 4), (5, 6), (6, 6)),
        ((0, 0), (0, 1), (0, 2), (0, 3), (1, 5), (2, 2), (4, 3), (4, 4), (5, 0), (5, 5), (5, 7), (6, 1), (7, 4)),
        ("h", 4, 7),
    ),
    (
        ((0, 5), (1, 5), (1, 7), (2, 2), (3, 1), (3, 3), (6, 5), (7, 3)),
        ((0, 3), (0, 6), (1, 1), (2, 4), (5, 7), (6, 1), (6, 6), (7, 5)),
        ("h", 1, 3),
    ),
    (
        ((0, 5), (1, 5), (1, 7), (2, 2), (3, 1), (3, 3), (6, 5), (7, 3)),
        ((0, 3), (0, 6), (1, 1), (2, 4), (2, 5), (5, 7), (6, 1), (6, 6), (7, 5)),
        ("h", 1, 3),
    ),
)


@pytest.mark.parametrize(("horizontal", "vertical", "candidate"), LAST_PATH_CASES)
def test_wall_cannot_seal_the_last_path(horizontal, vertical, candidate):
    state = state_with(horizontal_walls=horizontal, vertical_walls=vertical)
    assert not Game().legal_actions(state)[wall_action(*candidate)]


def test_path_checks_ignore_pawns():
    game = Game()
    state = state_with(pawns=((7, 4), (8, 4)))
    assert game.shortest_path_distance(state, 0) == 1


def test_canonical_action_maps_are_inverse_vertical_flips():
    assert sorted(CANONICAL_TO_ACTUAL[0]) == list(range(ACTION_COUNT))
    assert sorted(CANONICAL_TO_ACTUAL[1]) == list(range(ACTION_COUNT))
    for player in (0, 1):
        for action in range(ACTION_COUNT):
            actual = canonical_to_actual_action(action, player)
            assert actual_to_canonical_action(actual, player) == action
            assert ACTUAL_TO_CANONICAL[player][actual] == action

    assert canonical_to_actual_action(0, 1) == 1
    assert canonical_to_actual_action(2, 1) == 2
    assert canonical_to_actual_action(8, 1) == 10
    assert canonical_to_actual_action(wall_action("h", 0, 1), 1) == wall_action("h", 7, 1)


def test_player_b_wall_action_is_applied_in_canonical_frame():
    game = Game()
    state = state_with(current_player=1, ply=1)
    child = game.next_state(state, wall_action("h", 0, 1))
    assert child.horizontal_walls == ((7, 1),)


def test_player_b_observation_is_only_flipped_vertically():
    game = Game()
    state = state_with(
        pawns=((1, 2), (7, 6)),
        horizontal_walls=((1, 3),),
        vertical_walls=((6, 5),),
        walls_remaining=(8, 3),
        current_player=1,
        ply=1,
    )
    obs = game.canonical_observation(state)

    assert obs.shape == (6, 9, 9)
    assert obs.dtype == np.float32
    assert obs[0, 1, 6] == 1  # mover (7, 6) -> (1, 6)
    assert obs[1, 7, 2] == 1  # opponent (1, 2) -> (7, 2)
    assert obs[2, 6, 3] == 1  # anchor row flips 1 -> 6; column unchanged
    assert obs[3, 1, 5] == 1
    assert np.all(obs[4] == np.float32(0.3))
    assert np.all(obs[5] == np.float32(0.8))


def test_left_right_mirror_is_an_involution_for_state_action_and_policy():
    game = Game()
    state = state_with(
        pawns=((2, 1), (6, 7)),
        horizontal_walls=((1, 2),),
        vertical_walls=((4, 6),),
    )
    mirrored = game.mirror(state)

    assert mirrored.pawns == ((2, 7), (6, 1))
    assert game.mirror(mirrored) == state
    for action in range(ACTION_COUNT):
        assert mirror_action(mirror_action(action)) == action

    policy = np.arange(ACTION_COUNT, dtype=np.float32)
    assert np.array_equal(game.mirror(game.mirror(policy)), policy)
    assert game.transposition_key(state) == game.transposition_key(mirrored)


def test_terminal_value_is_from_the_mover_perspective_and_cap_is_zero():
    game = Game(max_plies=200)
    winning = state_with(pawns=((7, 4), (8, 8)))
    finished = game.next_state(winning, 0)

    assert game.is_terminal(finished) is TerminalStatus.MOVER_LOST
    assert game.terminal_value(finished) == -1.0

    capped = state_with(ply=200, current_player=0)
    assert game.is_terminal(capped) is TerminalStatus.CAPPED
    assert game.terminal_value(capped) == 0.0


def test_ascii_render_draws_walls_in_board_grooves():
    game = Game()
    state = state_with(horizontal_walls=((2, 2),), vertical_walls=((4, 5),))
    rendered = game.render(state)

    assert "A" in rendered and "B" in rendered
    assert "───" in rendered
    assert "│" in rendered
