from __future__ import annotations

import argparse
import os
import platform
import subprocess
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from barricade_rl.experiments import (
    GRAPH_COLORS,
    ExperimentSpec,
    available_replays,
    build_train_command,
    experiment_dir,
    experiment_presets,
    read_jsonl,
    save_graph_svg,
    save_spec,
)


METRIC_OPTIONS = [
    "ep_rew_mean",
    "ep_len_mean",
    "fps",
    "episodes",
    "train_loss",
    "train_value_loss",
    "train_entropy_loss",
    "train_policy_gradient_loss",
]

DEFAULT_CHECKPOINT_GLOB = "runs/maskable_ppo_barricade/best/*.zip"


class ExperimentDashboard:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Barricade RL Experiments")
        self.root.geometry("980x660")
        self.process: subprocess.Popen | None = None
        self.current_dir: Path | None = None
        self.runs_root = Path("runs/ui_experiments")
        self.preset_var = tk.StringVar(value="random")
        self.metric_vars: dict[str, tk.BooleanVar] = {}
        self.name_var = tk.StringVar(value="random")
        self.timesteps_var = tk.StringVar(value="25000")
        self.opponent_var = tk.StringVar(value="random")
        self.seed_var = tk.StringVar(value="0")
        self.replay_freq_var = tk.StringVar(value="5000")
        self.shaped_var = tk.BooleanVar(value=False)
        self.checkpoint_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Idle")
        self._build()
        self.refresh_loop()

    def run(self):
        self.root.mainloop()

    def _build(self):
        left = tk.Frame(self.root, padx=10, pady=10)
        left.pack(side="left", fill="y")
        right = tk.Frame(self.root, padx=10, pady=10)
        right.pack(side="right", fill="both", expand=True)

        tk.Label(left, text="Preset").pack(anchor="w")
        tk.OptionMenu(left, self.preset_var, *experiment_presets().keys()).pack(fill="x")
        tk.Button(left, text="Apply Preset", command=self.apply_preset).pack(fill="x", pady=(0, 10))
        self._field(left, "Name", self.name_var)
        self._field(left, "Timesteps", self.timesteps_var)
        self._field(left, "Seed", self.seed_var)
        self._field(left, "Replay freq", self.replay_freq_var)
        tk.Label(left, text="Opponent").pack(anchor="w")
        tk.OptionMenu(left, self.opponent_var, "random", "greedy", "mixed").pack(fill="x")
        tk.Checkbutton(left, text="Shaped reward", variable=self.shaped_var).pack(anchor="w", pady=(8, 2))
        self._field(left, "Checkpoint glob", self.checkpoint_var)

        tk.Button(left, text="Start", command=self.start_experiment).pack(fill="x", pady=(12, 2))
        tk.Button(left, text="Stop", command=self.stop_experiment).pack(fill="x", pady=2)
        tk.Button(left, text="Open Folder", command=self.open_run_folder).pack(fill="x", pady=(18, 2))
        tk.Button(left, text="Show Replay", command=self.show_replay_selector).pack(fill="x", pady=(18, 2))
        tk.Button(left, text="Record Final Replay", command=self.record_final_replay).pack(fill="x", pady=2)

        tk.Label(left, textvariable=self.status_var, wraplength=220, justify="left").pack(fill="x", pady=(20, 0))

        top = tk.Frame(right)
        top.pack(fill="x")
        tk.Button(top, text="Refresh Runs", command=self.refresh_runs).pack(side="right")
        tk.Button(top, text="Save Graph", command=self.save_current_graph).pack(side="right", padx=(0, 8))

        metrics_frame = tk.LabelFrame(right, text="Metrics")
        metrics_frame.pack(fill="x", pady=(8, 0))
        for index, metric in enumerate(METRIC_OPTIONS):
            variable = tk.BooleanVar(value=metric in {"ep_rew_mean", "ep_len_mean", "train_loss"})
            self.metric_vars[metric] = variable
            tk.Checkbutton(metrics_frame, text=metric, variable=variable, command=self.draw_graph).grid(
                row=index // 4,
                column=index % 4,
                sticky="w",
                padx=(8, 18),
                pady=2,
            )

        self.canvas = tk.Canvas(right, bg="#fbfaf7", height=320, highlightthickness=1, highlightbackground="#c9c1b4")
        self.canvas.pack(fill="both", expand=True, pady=(10, 10))

        tk.Label(right, text="Runs").pack(anchor="w")
        self.runs_list = tk.Listbox(right, height=8)
        self.runs_list.pack(fill="x")
        self.runs_list.bind("<<ListboxSelect>>", self.select_run)
        self.refresh_runs()

    def _field(self, parent, label, variable):
        tk.Label(parent, text=label).pack(anchor="w")
        tk.Entry(parent, textvariable=variable).pack(fill="x", pady=(0, 6))

    def apply_preset(self):
        try:
            timesteps = int(self.timesteps_var.get())
            seed = int(self.seed_var.get())
        except ValueError:
            messagebox.showerror("Invalid preset input", "Timesteps and seed must be integers before applying a preset.")
            return
        checkpoint_glob = self.checkpoint_var.get().strip() or DEFAULT_CHECKPOINT_GLOB
        preset_name = self.preset_var.get()
        spec = experiment_presets(timesteps=timesteps, seed=seed, checkpoint_glob=checkpoint_glob)[preset_name]
        self.name_var.set(spec.name)
        self.timesteps_var.set(str(spec.timesteps))
        self.opponent_var.set(spec.opponent)
        self.seed_var.set(str(spec.seed))
        self.shaped_var.set(spec.shaped_reward)
        self.checkpoint_var.set(" ".join(spec.checkpoint_opponents))

    def spec(self) -> ExperimentSpec:
        checkpoints = [value for value in self.checkpoint_var.get().split() if value]
        return ExperimentSpec(
            name=self.name_var.get().strip() or "experiment",
            timesteps=int(self.timesteps_var.get()),
            opponent=self.opponent_var.get(),
            seed=int(self.seed_var.get()),
            shaped_reward=self.shaped_var.get(),
            checkpoint_opponents=checkpoints,
            replay_freq=int(self.replay_freq_var.get()),
        )

    def start_experiment(self):
        if self.process and self.process.poll() is None:
            messagebox.showinfo("Experiment running", "Stop the current experiment before starting another.")
            return
        spec = self.spec()
        self.current_dir = experiment_dir(self.runs_root, spec)
        save_spec(self.current_dir, spec)
        command = build_train_command(spec, self.runs_root)
        self.process = subprocess.Popen(command)
        self.status_var.set(f"Running {spec.name}")
        self.refresh_runs()

    def stop_experiment(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.status_var.set("Stopping...")

    def refresh_loop(self):
        if self.process:
            code = self.process.poll()
            if code is None:
                self.status_var.set(f"Running: {self.current_dir}")
            else:
                self.status_var.set(f"Finished with code {code}: {self.current_dir}")
        self.draw_graph()
        self.root.after(1000, self.refresh_loop)

    def refresh_runs(self):
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.runs_list.delete(0, tk.END)
        for path in sorted(self.runs_root.iterdir()):
            if path.is_dir():
                self.runs_list.insert(tk.END, str(path))

    def select_run(self, event=None):
        selection = self.runs_list.curselection()
        if selection:
            self.current_dir = Path(self.runs_list.get(selection[0]))
            self.status_var.set(f"Selected {self.current_dir}")
            self.draw_graph()

    def metrics(self):
        if not self.current_dir:
            return []
        return read_jsonl(self.current_dir / "metrics.jsonl")

    def selected_metrics(self) -> list[str]:
        metrics = [metric for metric, variable in self.metric_vars.items() if variable.get()]
        return metrics or ["ep_rew_mean"]

    def draw_graph(self):
        self.canvas.delete("all")
        rows = self.metrics()
        metrics = self.selected_metrics()
        w = max(self.canvas.winfo_width(), 300)
        h = max(self.canvas.winfo_height(), 220)
        title = "Metrics (scaled independently)" if len(metrics) > 1 else metrics[0]
        self.canvas.create_text(12, 12, text=title, anchor="nw", fill="#232323", font=("Helvetica", 13, "bold"))
        all_xs = [row["timesteps"] for row in rows if "timesteps" in row]
        series = []
        for metric in metrics:
            points = [(row["timesteps"], row[metric]) for row in rows if "timesteps" in row and metric in row]
            if len(points) >= 2:
                series.append((metric, points))
        if not all_xs or not series:
            self.canvas.create_text(w / 2, h / 2, text="Waiting for metrics...", fill="#555555")
            return
        min_x, max_x = min(all_xs), max(all_xs)
        pad = 38
        self.canvas.create_line(pad, h - pad, w - pad, h - pad, fill="#c9c1b4")
        self.canvas.create_line(pad, pad, pad, h - pad, fill="#c9c1b4")
        for index, (metric, points) in enumerate(series):
            ys = [p[1] for p in points]
            min_y, max_y = min(ys), max(ys)
            if min_y == max_y:
                min_y -= 1
                max_y += 1
            coords = []
            for x, y in points:
                px = pad + (x - min_x) / max(max_x - min_x, 1) * (w - 2 * pad)
                py = h - pad - (y - min_y) / (max_y - min_y) * (h - 2 * pad)
                coords.extend([px, py])
            color = GRAPH_COLORS[index % len(GRAPH_COLORS)]
            self.canvas.create_line(*coords, fill=color, width=2)
            legend_y = 38 + index * 18
            self.canvas.create_line(14, legend_y, 34, legend_y, fill=color, width=3)
            self.canvas.create_text(
                40,
                legend_y,
                text=f"{metric} ({min_y:.3g} to {max_y:.3g})",
                anchor="w",
                fill="#333333",
            )
        self.canvas.create_text(w - pad, h - 18, text=str(max_x), anchor="e", fill="#555555")
        if self.current_dir:
            save_graph_svg(self.current_dir, rows, metrics)

    def save_current_graph(self):
        if not self.current_dir:
            messagebox.showinfo("No run", "Select or start a run first.")
            return
        path = save_graph_svg(self.current_dir, self.metrics(), self.selected_metrics())
        self.status_var.set(f"Saved graph to {path}")

    def open_run_folder(self):
        if not self.current_dir:
            messagebox.showinfo("No run", "Select or start a run first.")
            return
        self.current_dir.mkdir(parents=True, exist_ok=True)
        system = platform.system()
        if system == "Darwin":
            subprocess.Popen(["open", str(self.current_dir)])
        elif system == "Windows":
            os.startfile(str(self.current_dir))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(self.current_dir)])

    def script_path(self, script_name: str) -> Path:
        scripts_dir = Path(".venv") / ("Scripts" if platform.system() == "Windows" else "bin")
        suffix = ".exe" if platform.system() == "Windows" else ""
        return scripts_dir / f"{script_name}{suffix}"

    def show_replay_selector(self):
        if not self.current_dir:
            messagebox.showinfo("No run", "Select or start a run first.")
            return
        replays = available_replays(self.current_dir)
        if not replays:
            messagebox.showinfo("No replay", "No replay files found for this run.")
            return
        selector = tk.Toplevel(self.root)
        selector.title("Select Replay")
        selector.geometry("420x300")
        selector.transient(self.root)
        tk.Label(selector, text=f"Replays for {self.current_dir.name}").pack(anchor="w", padx=10, pady=(10, 4))
        listbox = tk.Listbox(selector, height=10)
        listbox.pack(fill="both", expand=True, padx=10, pady=4)
        for replay in replays:
            listbox.insert(tk.END, replay.name)
        listbox.selection_set(len(replays) - 1)
        listbox.see(len(replays) - 1)

        def open_selected(event=None):
            selection = listbox.curselection()
            if not selection:
                return
            replay_path = replays[selection[0]]
            subprocess.Popen([str(self.script_path("barricade-play")), "--replay", str(replay_path)])
            self.status_var.set(f"Opened replay {replay_path.name}")
            selector.destroy()

        buttons = tk.Frame(selector)
        buttons.pack(fill="x", padx=10, pady=(4, 10))
        tk.Button(buttons, text="Open Selected", command=open_selected).pack(side="right")
        tk.Button(buttons, text="Cancel", command=selector.destroy).pack(side="right", padx=(0, 8))
        listbox.bind("<Double-Button-1>", open_selected)

    def record_final_replay(self):
        if not self.current_dir:
            return
        model_path = self.current_dir / "final_model.zip"
        if not model_path.exists():
            messagebox.showinfo("No model", "final_model.zip does not exist yet.")
            return
        out = self.current_dir / "manual_replay.json"
        subprocess.Popen([str(self.script_path("barricade-record-model-game")), "--model", str(model_path), "--out", str(out)])
        self.status_var.set(f"Recording replay to {out}")


def main():
    parser = argparse.ArgumentParser(description="Open the Barricade RL experiment dashboard.")
    parser.parse_args()
    ExperimentDashboard().run()


if __name__ == "__main__":
    main()
