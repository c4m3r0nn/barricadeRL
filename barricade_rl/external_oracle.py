"""Adapter for the independently maintained ``pyquoridor`` rules package.

This module only translates state and action encodings. Rule decisions are made
by pyquoridor 0.0.5, not by either in-repository engine implementation.
"""

from __future__ import annotations

import numpy as np
from pyquoridor.board import Board

from .game import ACTION_COUNT, Game, State, TerminalStatus, wall_action

ORACLE_NAME = "pyquoridor"
ORACLE_VERSION = "0.0.5"
ORACLE_SOURCE = "https://github.com/playquoridor/python-quoridor"

_DELTA_TO_ACTION = {
    (1, 0): 0,
    (-1, 0): 1,
    (0, 1): 2,
    (0, -1): 3,
    (2, 0): 4,
    (-2, 0): 5,
    (0, 2): 6,
    (0, -2): 7,
    (1, 1): 8,
    (1, -1): 9,
    (-1, 1): 10,
    (-1, -1): 11,
}


def board_from_state(state: State) -> Board:
    (white_row, white_col), (black_row, black_col) = state.pawns
    board = Board(
        white_init_row=white_row,
        white_init_col=white_col,
        black_init_row=black_row,
        black_init_col=black_col,
    )
    for row, col in state.horizontal_walls:
        board._place_or_remove_fence(row, col, "h", place=True)
    for row, col in state.vertical_walls:
        board._place_or_remove_fence(row, col, "v", place=True)
    board.turn = state.current_player
    board.fences_left = {
        "white": state.walls_remaining[0],
        "black": state.walls_remaining[1],
    }
    return board


def external_legal_actions(state: State) -> np.ndarray:
    game = Game()
    if game.is_terminal(state) is not TerminalStatus.NOT_TERMINAL:
        return np.zeros(ACTION_COUNT, dtype=np.bool_)

    board = board_from_state(state)
    player = "white" if state.current_player == 0 else "black"
    mask = np.zeros(ACTION_COUNT, dtype=np.bool_)
    pawn_row, pawn_col = state.pawns[state.current_player]

    for square in board.valid_pawn_moves(player, check_winner=False):
        actual_dr = square.row - pawn_row
        canonical_dr = actual_dr if state.current_player == 0 else -actual_dr
        action = _DELTA_TO_ACTION[(canonical_dr, square.col - pawn_col)]
        mask[action] = True

    horizontal, vertical = board.valid_fence_moves(player, check_winner=False)
    for row in range(8):
        for col in range(8):
            canonical_row = 7 - row if state.current_player == 1 else row
            if horizontal[player][(row, col)] >= 0:
                mask[wall_action("h", canonical_row, col)] = True
            if vertical[player][(row, col)] >= 0:
                mask[wall_action("v", canonical_row, col)] = True
    return mask
