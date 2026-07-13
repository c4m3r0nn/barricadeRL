"""Thin ctypes binding for the dependency-free Rust rules engine."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path


def _library_path() -> Path:
    if sys.platform == "darwin":
        filename = "lib_engine.dylib"
    elif sys.platform == "win32":
        filename = "_engine.dll"
    else:
        filename = "lib_engine.so"
    root = Path(__file__).resolve().parent.parent
    for profile in ("release", "debug"):
        candidate = root / "target" / profile / filename
        if candidate.exists():
            return candidate
    raise ImportError("native engine is not built; run `cargo build --release`")


_LIBRARY = ctypes.CDLL(str(_library_path()))
_U8_POINTER = ctypes.POINTER(ctypes.c_uint8)
_LIBRARY.br_legal_actions.argtypes = (_U8_POINTER, ctypes.c_uint8, _U8_POINTER)
_LIBRARY.br_legal_actions.restype = ctypes.c_int32
_LIBRARY.br_next_state.argtypes = (_U8_POINTER, ctypes.c_uint16, ctypes.c_uint8, _U8_POINTER)
_LIBRARY.br_next_state.restype = ctypes.c_int32
_LIBRARY.br_terminal_status.argtypes = (_U8_POINTER, ctypes.c_uint8)
_LIBRARY.br_terminal_status.restype = ctypes.c_int32
_LIBRARY.br_shortest_path_distance.argtypes = (_U8_POINTER, ctypes.c_uint8)
_LIBRARY.br_shortest_path_distance.restype = ctypes.c_int32
_LIBRARY.br_perft.argtypes = (_U8_POINTER, ctypes.c_uint8, ctypes.c_uint8, ctypes.POINTER(ctypes.c_uint64))
_LIBRARY.br_perft.restype = ctypes.c_int32


def _state_buffer(data: bytes):
    if len(data) != 20:
        raise ValueError("state key must contain exactly 20 bytes")
    return (ctypes.c_uint8 * 20).from_buffer_copy(data)


def legal_actions(data: bytes, max_plies: int) -> list[bool]:
    state = _state_buffer(data)
    output = (ctypes.c_uint8 * 140)()
    if _LIBRARY.br_legal_actions(state, max_plies, output):
        raise ValueError("invalid state")
    return [bool(value) for value in output]


def next_state(data: bytes, action: int, max_plies: int) -> bytes:
    state = _state_buffer(data)
    output = (ctypes.c_uint8 * 20)()
    code = _LIBRARY.br_next_state(state, action, max_plies, output)
    if code == 1:
        raise ValueError("invalid state")
    if code == 2:
        raise ValueError("cannot act in a terminal state")
    if code == 3:
        raise ValueError(f"illegal action {action}")
    return bytes(output)


def terminal_status(data: bytes, max_plies: int) -> int:
    result = _LIBRARY.br_terminal_status(_state_buffer(data), max_plies)
    if result < 0:
        raise ValueError("invalid state")
    return result


def shortest_path_distance(data: bytes, player: int) -> int | None:
    result = _LIBRARY.br_shortest_path_distance(_state_buffer(data), player)
    if result == -2:
        raise ValueError("invalid state or player")
    return None if result == -1 else result


def perft(data: bytes, depth: int, max_plies: int) -> int:
    output = ctypes.c_uint64()
    if _LIBRARY.br_perft(_state_buffer(data), depth, max_plies, ctypes.byref(output)):
        raise ValueError("invalid state")
    return output.value
