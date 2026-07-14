import numpy as np

import barricade_rl
from barricade_rl.game import TerminalStatus
from barricade_rl.small_board import (
    SmallBoardSpec,
    SmallGame,
    SmallState,
    SolverOutcome,
    small_wall_action,
    solve_state,
)


def small_state(
    *,
    pawns=((0, 2), (4, 2)),
    horizontal_walls=(),
    vertical_walls=(),
    walls_remaining=(3, 3),
    current_player=0,
    ply=0,
):
    return SmallState.from_components(
        spec=SmallBoardSpec(size=5, walls_per_player=3, max_plies=200),
        pawns=pawns,
        horizontal_walls=horizontal_walls,
        vertical_walls=vertical_walls,
        walls_remaining=walls_remaining,
        current_player=current_player,
        ply=ply,
    )


def test_5x5_initial_contract_and_action_count():
    assert barricade_rl.SmallGame is SmallGame

    game = SmallGame(SmallBoardSpec(size=5, walls_per_player=3, max_plies=80))
    state = game.initial_state()

    assert game.action_count == 44
    assert game.wall_grid_size == 4
    assert state.pawns == ((0, 2), (4, 2))
    assert state.walls_remaining == (3, 3)
    assert state.current_player == 0

    mask = game.legal_actions(state)
    assert mask.shape == (44,)
    assert mask.dtype == np.bool_
    assert mask[:12].tolist() == [True, False, True, True] + [False] * 8
    assert int(mask.sum()) == 35
    assert len(game.state_key(state)) == 12


def test_5x5_opening_perft_is_stable():
    game = SmallGame()
    state = game.initial_state()

    assert [game.perft(state, depth) for depth in range(1, 4)] == [35, 1109, 31540]


def test_5x5_player_b_wall_actions_are_mover_canonical():
    game = SmallGame()
    state = small_state(current_player=1, ply=1)
    child = game.next_state(state, small_wall_action(game.spec, "h", 0, 1))

    assert child.horizontal_walls == ((3, 1),)
    assert child.walls_remaining == (3, 2)
    assert child.current_player == 0


def test_5x5_jump_geometry_uses_same_canonical_move_order():
    game = SmallGame()
    state = small_state(pawns=((2, 2), (3, 2)))
    legal = set(np.flatnonzero(game.legal_actions(state)[:12]))
    assert 4 in legal
    assert not legal & {8, 9}

    blocked = small_state(
        pawns=((2, 2), (3, 2)),
        horizontal_walls=((3, 2),),
    )
    legal = set(np.flatnonzero(game.legal_actions(blocked)[:12]))
    assert {8, 9} <= legal
    assert 4 not in legal


def test_5x5_wall_overlap_crossing_and_path_preservation():
    game = SmallGame()
    state = small_state(horizontal_walls=((1, 1),))
    mask = game.legal_actions(state)

    assert not mask[small_wall_action(game.spec, "h", 1, 0)]
    assert not mask[small_wall_action(game.spec, "h", 1, 1)]
    assert not mask[small_wall_action(game.spec, "h", 1, 2)]
    assert not mask[small_wall_action(game.spec, "v", 1, 1)]

    corridor = small_state(
        pawns=((0, 0), (4, 4)),
        horizontal_walls=((0, 1), (1, 1), (2, 1)),
        vertical_walls=((0, 0), (1, 0), (2, 0)),
        walls_remaining=(3, 3),
    )
    assert game.shortest_path_distance(corridor, 0) == 4
    assert not game.legal_actions(corridor)[small_wall_action(game.spec, "v", 3, 0)]


def test_5x5_observation_mirror_and_state_key_are_stable():
    game = SmallGame()
    state = small_state(
        pawns=((1, 1), (3, 4)),
        horizontal_walls=((1, 2),),
        vertical_walls=((3, 1),),
        walls_remaining=(2, 1),
        current_player=1,
        ply=5,
    )
    observation = game.canonical_observation(state)

    assert observation.shape == (6, 5, 5)
    assert observation.dtype == np.float32
    assert observation[0, 1, 4] == 1
    assert observation[1, 3, 1] == 1
    assert observation[2, 2, 2] == 1
    assert observation[3, 0, 1] == 1
    assert np.all(observation[4] == np.float32(1 / 3))
    assert np.all(observation[5] == np.float32(2 / 3))

    mirrored = game.mirror(state)
    assert mirrored.pawns == ((1, 3), (3, 0))
    assert game.mirror(mirrored) == state
    assert game.state_key(state) == game.state_key(SmallState.from_key(game.spec, game.state_key(state)))


def test_5x5_bounded_solver_handles_terminal_and_forced_race_positions():
    game = SmallGame()
    terminal = small_state(pawns=((4, 2), (0, 0)), current_player=1, ply=7)
    terminal_result = solve_state(game, terminal, max_depth=0)
    assert game.is_terminal(terminal) is TerminalStatus.MOVER_LOST
    assert terminal_result.outcome is SolverOutcome.LOSS
    assert terminal_result.value == -1

    winning = small_state(pawns=((3, 2), (4, 4)), walls_remaining=(0, 0), current_player=0)
    win_result = solve_state(game, winning, max_depth=1)
    assert win_result.outcome is SolverOutcome.WIN
    assert win_result.value == 1
    assert win_result.best_action == 0

    race = small_state(walls_remaining=(0, 0), current_player=0)
    race_result = solve_state(game, race, max_depth=8)
    assert race_result.outcome is SolverOutcome.LOSS
    assert race_result.best_action == 0
