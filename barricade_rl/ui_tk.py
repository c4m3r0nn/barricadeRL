from __future__ import annotations

import tkinter as tk
from tkinter import messagebox

from barricade_rl.core import (
    BOARD_SIZE,
    MOVE_NAMES,
    WALL_GRID_SIZE,
    BarricadeGame,
    decode_wall_action,
    wall_action,
)

CELL = 56
MARGIN = 32
WALL_THICKNESS = 8
WALL_GAP = 7
SIDE_PANEL_WIDTH = 260
PLAYER_COLORS = ("#1f77b4", "#d1495b")


class BarricadeUI:
    def __init__(self):
        self.game = BarricadeGame()
        self.drag_wall_orientation: str | None = None
        self.drag_xy: tuple[int, int] | None = None
        self.wall_token_bounds: dict[str, tuple[int, int, int, int]] = {}
        self.root = tk.Tk()
        self.root.title("Barricade RL Test UI")
        width = MARGIN * 2 + CELL * BOARD_SIZE + SIDE_PANEL_WIDTH
        height = MARGIN * 2 + CELL * BOARD_SIZE
        self.canvas = tk.Canvas(self.root, width=width, height=height, bg="#f7f4ed", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.root.bind("<Key>", self.on_key)
        self.draw()

    def run(self):
        self.root.mainloop()

    def on_key(self, event):
        keymap = {"Up": 0, "Down": 1, "Left": 2, "Right": 3, "w": 0, "s": 1, "a": 2, "d": 3}
        if event.keysym in keymap:
            self.try_action(keymap[event.keysym])
        elif event.char.lower() == "r":
            self.game.reset()
            self.draw()

    def on_button_press(self, event):
        for orientation, bounds in self.wall_token_bounds.items():
            x0, y0, x1, y1 = bounds
            if x0 <= event.x <= x1 and y0 <= event.y <= y1:
                if self.game.state.walls_remaining[self.game.state.current_player] <= 0:
                    return
                self.drag_wall_orientation = orientation
                self.drag_xy = (event.x, event.y)
                self.draw()
                return
        if self.in_side_panel(event.x):
            return
        row = int((event.y - MARGIN) // CELL)
        col = int((event.x - MARGIN) // CELL)
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            self.try_move_to((row, col))

    def on_drag(self, event):
        if self.drag_wall_orientation is None:
            return
        self.drag_xy = (event.x, event.y)
        self.draw()

    def on_button_release(self, event):
        if self.drag_wall_orientation is None:
            return
        orientation = self.drag_wall_orientation
        row, col = self.nearest_wall_anchor(event.x, event.y)
        self.drag_wall_orientation = None
        self.drag_xy = None
        if row is not None and col is not None:
            self.try_action(wall_action(orientation, row, col))
        else:
            self.draw()

    def try_move_to(self, pos: tuple[int, int]):
        for action in range(4):
            if self.game.move_destination(action) == pos:
                self.try_action(action)
                return

    def try_action(self, action: int):
        if not self.game.apply_action(action):
            messagebox.showinfo("Illegal action", self.describe_action(action))
        self.draw()
        if self.game.state.winner is not None:
            messagebox.showinfo("Game over", f"Player {self.game.state.winner} wins")

    def describe_action(self, action: int) -> str:
        if action < 4:
            return f"Move {MOVE_NAMES[action]} is not legal here."
        orientation, row, col = decode_wall_action(action)
        label = "horizontal" if orientation == "h" else "vertical"
        return f"{label.title()} wall at ({row}, {col}) is not legal here."

    def draw(self):
        self.canvas.delete("all")
        self.draw_board()
        self.draw_walls()
        self.draw_pawns()
        self.draw_side_panel()
        self.draw_drag_preview()

    def xy(self, row: int, col: int) -> tuple[int, int]:
        return MARGIN + col * CELL + CELL // 2, MARGIN + row * CELL + CELL // 2

    def corner_xy(self, row: int, col: int) -> tuple[int, int]:
        return MARGIN + col * CELL, MARGIN + row * CELL

    def side_panel_x(self) -> int:
        return MARGIN * 2 + CELL * BOARD_SIZE + 28

    def in_side_panel(self, x: int) -> bool:
        return x >= self.side_panel_x() - 20

    def nearest_wall_anchor(self, x: int, y: int) -> tuple[int | None, int | None]:
        if self.in_side_panel(x):
            return None, None
        if self.drag_wall_orientation == "h":
            row = round((y - MARGIN) / CELL) - 1
            col = round((x - MARGIN - CELL) / CELL)
        else:
            row = round((y - MARGIN - CELL) / CELL)
            col = round((x - MARGIN) / CELL) - 1
        if 0 <= row < WALL_GRID_SIZE and 0 <= col < WALL_GRID_SIZE:
            return row, col
        return None, None

    def draw_board(self):
        x0, y0 = self.corner_xy(0, 0)
        x1, y1 = self.corner_xy(BOARD_SIZE, BOARD_SIZE)
        self.canvas.create_rectangle(x0, y0, x1, y1, fill="#fcfaf6", outline="#bfb5a7", width=2)
        for idx in range(1, BOARD_SIZE):
            x, _ = self.corner_xy(0, idx)
            self.canvas.create_line(x, y0, x, y1, fill="#d5cbbb", width=1)
            _, y = self.corner_xy(idx, 0)
            self.canvas.create_line(x0, y, x1, y, fill="#d5cbbb", width=1)

    def draw_walls(self):
        for row in range(WALL_GRID_SIZE):
            for col in range(WALL_GRID_SIZE):
                if self.game.state.h_walls[row, col]:
                    self.draw_horizontal_wall(row, col, "#30343f", WALL_THICKNESS)
                if self.game.state.v_walls[row, col]:
                    self.draw_vertical_wall(row, col, "#30343f", WALL_THICKNESS)

    def draw_horizontal_wall(self, row: int, col: int, color: str, width: int):
        x0, y = self.corner_xy(row + 1, col)
        x2, _ = self.corner_xy(row + 1, col + 2)
        self.canvas.create_line(x0 + WALL_GAP, y, x2 - WALL_GAP, y, fill=color, width=width, capstyle="round")

    def draw_vertical_wall(self, row: int, col: int, color: str, width: int):
        x, y0 = self.corner_xy(row, col + 1)
        _, y2 = self.corner_xy(row + 2, col + 1)
        self.canvas.create_line(x, y0 + WALL_GAP, x, y2 - WALL_GAP, fill=color, width=width, capstyle="round")

    def draw_pawns(self):
        active = self.game.state.current_player
        for player, pos in enumerate(self.game.state.pawns):
            x, y = self.xy(*pos)
            if player == active:
                self.canvas.create_oval(x - 24, y - 24, x + 24, y + 24, outline=PLAYER_COLORS[player], width=4)
            self.canvas.create_oval(x - 17, y - 17, x + 17, y + 17, fill=PLAYER_COLORS[player], outline="#ffffff", width=2)
            self.canvas.create_text(x, y, text=str(player), fill="#ffffff", font=("Helvetica", 14, "bold"))

    def draw_side_panel(self):
        x = self.side_panel_x()
        y = MARGIN
        player = self.game.state.current_player
        self.canvas.create_text(x, y, text="Barricade", anchor="nw", fill="#232323", font=("Helvetica", 18, "bold"))
        banner_y = y + 42
        self.canvas.create_rectangle(x, banner_y, x + 190, banner_y + 44, fill=PLAYER_COLORS[player], outline="")
        self.canvas.create_text(x + 14, banner_y + 22, text=f"Player {player} to move", anchor="w", fill="#ffffff", font=("Helvetica", 15, "bold"))

        wall_y = banner_y + 76
        self.canvas.create_text(x, wall_y, text=f"Player {player} walls", anchor="nw", fill="#232323", font=("Helvetica", 12, "bold"))
        self.canvas.create_text(x, wall_y + 24, text=str(self.game.state.walls_remaining[player]), anchor="nw", fill="#232323", font=("Helvetica", 26, "bold"))
        self.draw_wall_tokens(x, wall_y + 74)

        other = 1 - player
        self.canvas.create_text(x, wall_y + 174, text=f"Player {other} walls: {self.game.state.walls_remaining[other]}", anchor="nw", fill="#454545", font=("Helvetica", 12))
        self.canvas.create_text(x, wall_y + 220, text="Arrows/WASD move", anchor="nw", fill="#454545", font=("Helvetica", 12))
        self.canvas.create_text(x, wall_y + 244, text="Drag wall to board", anchor="nw", fill="#454545", font=("Helvetica", 12))
        self.canvas.create_text(x, wall_y + 268, text="R resets", anchor="nw", fill="#454545", font=("Helvetica", 12))

    def draw_wall_tokens(self, x: int, y: int):
        self.wall_token_bounds = {}
        remaining = self.game.state.walls_remaining[self.game.state.current_player]
        fill = "#30343f" if remaining > 0 else "#a9a9a9"
        outline = "#232323" if remaining > 0 else "#8a8a8a"

        h_bounds = (x, y, x + 130, y + 42)
        self.wall_token_bounds["h"] = h_bounds
        self.canvas.create_rectangle(*h_bounds, fill="#ffffff", outline="#c9c1b4", width=1)
        self.canvas.create_line(x + 18, y + 21, x + 112, y + 21, fill=fill, width=WALL_THICKNESS, capstyle="round")

        v_bounds = (x + 148, y, x + 190, y + 130)
        self.wall_token_bounds["v"] = v_bounds
        self.canvas.create_rectangle(*v_bounds, fill="#ffffff", outline="#c9c1b4", width=1)
        self.canvas.create_line(x + 169, y + 18, x + 169, y + 112, fill=fill, width=WALL_THICKNESS, capstyle="round")

        if remaining <= 0:
            self.canvas.create_text(x, y + 148, text="No walls left", anchor="nw", fill="#8a3b3b", font=("Helvetica", 12, "bold"))

    def draw_drag_preview(self):
        if self.drag_wall_orientation is None or self.drag_xy is None:
            return
        x, y = self.drag_xy
        row, col = self.nearest_wall_anchor(x, y)
        legal = row is not None and col is not None and self.game.is_wall_legal(self.drag_wall_orientation, row, col)
        color = "#2f855a" if legal else "#b23b3b"
        if row is not None and col is not None:
            if self.drag_wall_orientation == "h":
                self.draw_horizontal_wall(row, col, color, WALL_THICKNESS + 2)
            else:
                self.draw_vertical_wall(row, col, color, WALL_THICKNESS + 2)
        if self.drag_wall_orientation == "h":
            self.canvas.create_line(x - 48, y, x + 48, y, fill="#30343f", width=WALL_THICKNESS, capstyle="round")
        else:
            self.canvas.create_line(x, y - 48, x, y + 48, fill="#30343f", width=WALL_THICKNESS, capstyle="round")


def main():
    BarricadeUI().run()


if __name__ == "__main__":
    main()
