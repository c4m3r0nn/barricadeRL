from __future__ import annotations

import argparse
import subprocess
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from barricade_rl.experiments import ExperimentSpec, build_train_command, experiment_dir, read_jsonl, save_spec


class ExperimentDashboard:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Barricade RL Experiments")
        self.root.geometry("980x660")
        self.process: subprocess.Popen | None = None
        self.current_dir: Path | None = None
        self.runs_root = Path("runs/ui_experiments")
        self.metric_name = tk.StringVar(value="ep_rew_mean")
        self.name_var = tk.StringVar(value="experiment")
        self.timesteps_var = tk.StringVar(value="10000")
        self.opponent_var = tk.StringVar(value="random")
        self.seed_var = tk.StringVar(value="0")
        self.replay_freq_var = tk.StringVar(value="1000")
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
        tk.Button(left, text="Open Replay", command=self.open_latest_replay).pack(fill="x", pady=(18, 2))
        tk.Button(left, text="Record Final Replay", command=self.record_final_replay).pack(fill="x", pady=2)

        tk.Label(left, textvariable=self.status_var, wraplength=220, justify="left").pack(fill="x", pady=(20, 0))

        top = tk.Frame(right)
        top.pack(fill="x")
        tk.Label(top, text="Metric").pack(side="left")
        tk.OptionMenu(top, self.metric_name, "ep_rew_mean", "ep_len_mean", "fps", "train_loss", "train_value_loss").pack(side="left")
        tk.Button(top, text="Refresh Runs", command=self.refresh_runs).pack(side="right")

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

    def draw_graph(self):
        self.canvas.delete("all")
        rows = self.metrics()
        metric = self.metric_name.get()
        points = [(row["timesteps"], row[metric]) for row in rows if metric in row]
        w = max(self.canvas.winfo_width(), 300)
        h = max(self.canvas.winfo_height(), 220)
        self.canvas.create_text(12, 12, text=metric, anchor="nw", fill="#232323", font=("Helvetica", 13, "bold"))
        if len(points) < 2:
            self.canvas.create_text(w / 2, h / 2, text="Waiting for metrics...", fill="#555555")
            return
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        if min_y == max_y:
            min_y -= 1
            max_y += 1
        pad = 38
        coords = []
        for x, y in points:
            px = pad + (x - min_x) / max(max_x - min_x, 1) * (w - 2 * pad)
            py = h - pad - (y - min_y) / (max_y - min_y) * (h - 2 * pad)
            coords.extend([px, py])
        self.canvas.create_line(pad, h - pad, w - pad, h - pad, fill="#c9c1b4")
        self.canvas.create_line(pad, pad, pad, h - pad, fill="#c9c1b4")
        self.canvas.create_line(*coords, fill="#1f77b4", width=2)
        self.canvas.create_text(w - pad, h - 18, text=str(max_x), anchor="e", fill="#555555")
        self.canvas.create_text(8, pad, text=f"{max_y:.2f}", anchor="nw", fill="#555555")
        self.canvas.create_text(8, h - pad, text=f"{min_y:.2f}", anchor="sw", fill="#555555")

    def open_latest_replay(self):
        if not self.current_dir:
            return
        replay_dir = self.current_dir / "replays"
        replays = sorted(replay_dir.glob("*.json"))
        if not replays:
            messagebox.showinfo("No replay", "No replay files found for this run.")
            return
        subprocess.Popen([str(Path(".venv/bin/barricade-play")), "--replay", str(replays[-1])])

    def record_final_replay(self):
        if not self.current_dir:
            return
        model_path = self.current_dir / "final_model.zip"
        if not model_path.exists():
            messagebox.showinfo("No model", "final_model.zip does not exist yet.")
            return
        out = self.current_dir / "manual_replay.json"
        subprocess.Popen([str(Path(".venv/bin/barricade-record-model-game")), "--model", str(model_path), "--out", str(out)])
        self.status_var.set(f"Recording replay to {out}")


def main():
    parser = argparse.ArgumentParser(description="Open the Barricade RL experiment dashboard.")
    parser.parse_args()
    ExperimentDashboard().run()


if __name__ == "__main__":
    main()
