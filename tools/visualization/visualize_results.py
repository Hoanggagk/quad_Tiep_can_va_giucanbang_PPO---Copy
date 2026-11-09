"""
Visualization tool for drone PPO training and evaluation results.

Usage examples:
  # Plot training learning curve from a log CSV
  python -m tools.visualization.visualize_results --training-log log_dir/PPO_log_hover_projectairsim.csv

  # Plot trajectories from a folder (test_results/Co_nhieu, control_results, ...)
    python -m tools.visualization.visualize_results --trajectories control_results --goal 0 0 -2

  # Compare two trajectory folders side by side
  python -m tools.visualization.visualize_results --compare test_results/Co_nhieu test_results/Khong_co_nhieu

  # Plot balance / robustness evaluation summary
  python -m tools.visualization.visualize_results --balance balance_eval_results/balance_eval_summary.csv

  # Plot a single balance evaluation trajectory
  python -m tools.visualization.visualize_results --balance-traj balance_eval_results/trajectories/traj_push_north_trial01.csv

  # Combine multiple options in one call
  python -m tools.visualization.visualize_results --training-log log_dir/PPO_log_hover_projectairsim.csv \
      --trajectories control_results --balance balance_eval_results/balance_eval_summary.csv
"""

import os
import argparse
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless / non-GUI backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3D projection)


# ════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════

GOAL_POS = [0.0, 0.0, 3.0]   # Display goal in NED-up frame: Z displayed as -z_ned

COLOR_MAIN   = "#1565C0"
COLOR_LIGHT  = "#90CAF9"
COLOR_ACCENT = "#C62828"
COLOR_GOAL   = "#D32F2F"


def _smooth(data: np.ndarray, window: int = 20) -> np.ndarray:
    """Simple moving average."""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _save(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {path}")


# ════════════════════════════════════════════════════════════════════
# 1. LEARNING CURVE
# ════════════════════════════════════════════════════════════════════

def plot_training_log(csv_path: str, out_dir: str = None, smooth_window: int = 20) -> None:
    """
    Plot the average reward per episode from a PPO training log CSV.

    Expected columns: episode, timestep, avg_reward
    """
    df = pd.read_csv(csv_path)
    if "avg_reward" not in df.columns:
        print(f"[ERROR] Column 'avg_reward' not found in {csv_path}")
        return

    out_dir = out_dir or os.path.dirname(csv_path) or "."
    _ensure_dir(out_dir)

    episodes  = df["episode"].values
    rewards   = df["avg_reward"].values
    timesteps = df["timestep"].values if "timestep" in df.columns else None

    # ── Figure with 1 or 2 subplots ──────────────────────────────
    ncols = 2 if timesteps is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    # Reward vs Episode
    ax = axes[0]
    ax.plot(episodes, rewards, color=COLOR_LIGHT, linewidth=0.8, alpha=0.65, label="Avg reward / log point")
    smoothed = _smooth(rewards, smooth_window)
    offset   = len(rewards) - len(smoothed)
    ax.plot(episodes[offset:], smoothed,
            color=COLOR_MAIN, linewidth=2.0,
            label=f"Moving average ({smooth_window} pts)")
    ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Reward")
    ax.set_title("Learning Curve – Reward vs Episode")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    # Reward vs Timestep
    if timesteps is not None:
        ax2 = axes[1]
        ax2.plot(timesteps, rewards, color=COLOR_LIGHT, linewidth=0.8, alpha=0.65)
        ax2.plot(timesteps[offset:], smoothed, color=COLOR_MAIN, linewidth=2.0,
                 label=f"Moving average ({smooth_window} pts)")
        ax2.axhline(y=0, color="gray", linewidth=0.8, linestyle="--")
        ax2.set_xlabel("Timestep")
        ax2.set_ylabel("Average Reward")
        ax2.set_title("Learning Curve – Reward vs Timestep")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("PPO Training – Learning Curve", fontsize=13, fontweight="bold")
    fig.tight_layout()
    base = os.path.splitext(os.path.basename(csv_path))[0]
    _save(fig, os.path.join(out_dir, f"{base}_learning_curve.png"))


# ════════════════════════════════════════════════════════════════════
# 2. SINGLE TRAJECTORY EPISODE
# ════════════════════════════════════════════════════════════════════

def _load_traj_csv(csv_path: str) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Load a trajectory CSV.
    Supports two layouts:
      A) time_s, x_m, y_m, z_m, roll_deg, pitch_deg, yaw_deg
      B) step, time_s, phase, x_m, y_m, z_m, roll_deg, pitch_deg, yaw_deg  (balance eval)
    Returns (time, data[N,6], phase_label).
    """
    df = pd.read_csv(csv_path)
    phase_label = ""
    if "phase" in df.columns:
        phase_label = df["phase"].iloc[0]

    cols = ["x_m", "y_m", "z_m", "roll_deg", "pitch_deg", "yaw_deg"]
    data = df[cols].values.astype(float)
    time = df["time_s"].values.astype(float)
    return time, data, phase_label


def plot_trajectory(csv_path: str,
                    out_dir:  str  = None,
                    ep_label: str  = None,
                    goal:     list = None,
                    goal_attitude: list = None) -> None:
    """
    Plot one trajectory: 3D path + X/Y/Z and roll/pitch/yaw over time.

    Parameters
    ----------
    csv_path       : path to trajectory CSV
    out_dir        : output folder (defaults to same folder as csv_path)
    ep_label       : title suffix (e.g. "Episode 1")
    goal           : [x, y, z_ned] goal position; z displayed as altitude (-z_ned)
    goal_attitude  : [roll_deg, pitch_deg, yaw_deg] target attitude (default: [0, 0, 0])
    """
    time, data, phase_label = _load_traj_csv(csv_path)

    x_m   = data[:, 0]
    y_m   = data[:, 1]
    z_m   = data[:, 2]    # altitude (positive up) – already stored as -z_NED in CSV
    roll  = data[:, 3]    # degrees
    pitch = data[:, 4]
    yaw   = data[:, 5]

    goal_alt = None if goal is None else -goal[2]   # NED z → altitude
    goal_att = goal_attitude if goal_attitude is not None else [0.0, 0.0, 0.0]

    out_dir = out_dir or os.path.dirname(csv_path) or "."
    _ensure_dir(out_dir)

    title = f"Trajectory – {ep_label}" if ep_label else "Trajectory"
    if phase_label:
        title += f" [{phase_label}]"

    fig = plt.figure(figsize=(14, 13))
    fig.suptitle(title, fontsize=13)

    # ── 3D trajectory ─────────────────────────────────────────────
    ax3 = fig.add_subplot(3, 3, (1, 3), projection="3d")
    ax3.plot(x_m, y_m, z_m, color=COLOR_MAIN, linewidth=1.5)
    ax3.scatter(x_m[0],  y_m[0],  z_m[0],  color="green", marker="o", s=60, zorder=4, label="Start")
    ax3.scatter(x_m[-1], y_m[-1], z_m[-1], color="black", marker="x", s=80, zorder=4, label="End")
    if goal is not None:
        ax3.scatter(goal[0], goal[1], goal_alt, color="red",
                    marker="*", s=300, zorder=5, label="Goal")
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.set_zlabel("Z (m)")
    ax3.set_xlim([-2, 2])
    ax3.set_ylim([-2, 2])
    ax3.set_zlim([0, 2])
    ax3.legend(fontsize=8)
    ax3.set_title("3D")
    ax3.view_init(elev=20, azim=45)

    # ── Position over time ────────────────────────────────────────
    pos_specs = [
        (z_m,  "Z (m)",  goal_alt),
        (x_m,  "X (m)",  goal[0] if goal else None),
        (y_m,  "Y (m)",  goal[1] if goal else None),
    ]
    for i, (series, label, g_val) in enumerate(pos_specs):
        ax = fig.add_subplot(3, 3, 4 + i)
        ax.plot(time, series, color=COLOR_MAIN, linewidth=1.2)
        if g_val is not None:
            ax.axhline(y=g_val, color=COLOR_GOAL, linestyle="--",
                       linewidth=1.0, label="Goal")
            ax.legend(fontsize=7)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_ylim([-0.5, 2.5])
        ax.grid(True, linestyle="--", alpha=0.4)

    # ── Attitude over time ────────────────────────────────────────
    att_specs = [
        (roll,  "Roll (\u00b0)",  goal_att[0]),
        (pitch, "Pitch (\u00b0)", goal_att[1]),
        (yaw,   "Yaw (\u00b0)",   goal_att[2]),
    ]
    for i, (series, label, g_val) in enumerate(att_specs):
        ax = fig.add_subplot(3, 3, 7 + i)
        ax.plot(time, series, color=COLOR_ACCENT, linewidth=1.2)
        if g_val is not None:
            ax.axhline(y=g_val, color=COLOR_GOAL, linestyle="--",
                       linewidth=1.0, label="Goal")
            ax.legend(fontsize=7)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_ylim([-3, 3])
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    base = os.path.splitext(os.path.basename(csv_path))[0]
    _save(fig, os.path.join(out_dir, f"{base}.png"))


# ════════════════════════════════════════════════════════════════════
# 3. MULTI-EPISODE OVERVIEW (folder of trajectory CSVs)
# ════════════════════════════════════════════════════════════════════

def plot_trajectories_folder(folder: str,
                             out_dir: str = None,
                             goal:    list = None,
                             max_eps: int  = None,
                             goal_attitude: list = None) -> None:
    """
    For each trajectory CSV in *folder*, generate an individual plot,
    then produce an overlay summary (altitude + X + Y + roll/pitch/yaw over time).
    """
    pattern = os.path.join(folder, "trajectory_ep*.csv")
    csv_files = sorted(glob.glob(pattern))

    # Fallback: any *.csv in folder
    if not csv_files:
        csv_files = sorted(glob.glob(os.path.join(folder, "*.csv")))

    if not csv_files:
        print(f"[WARN] No trajectory CSV files found in: {folder}")
        return

    if max_eps is not None:
        csv_files = csv_files[:max_eps]

    out_dir = out_dir or folder
    _ensure_dir(out_dir)
    goal_att = goal_attitude if goal_attitude is not None else [0.0, 0.0, 0.0]

    print(f"[INFO] Plotting {len(csv_files)} episodes from: {folder}")

    # ── Individual episode plots ───────────────────────────────────
    all_time, all_alt, all_x, all_y = [], [], [], []
    all_roll, all_pitch, all_yaw = [], [], []
    for idx, csv_path in enumerate(csv_files):
        ep_label = f"Episode {idx + 1}"
        plot_trajectory(csv_path, out_dir=out_dir, ep_label=ep_label, goal=goal, goal_attitude=goal_att)
        time, data, _ = _load_traj_csv(csv_path)
        all_time.append(time)
        all_alt.append(data[:, 2])    # altitude (already positive up in CSV)
        all_x.append(data[:, 0])
        all_y.append(data[:, 1])
        all_roll.append(data[:, 3])
        all_pitch.append(data[:, 4])
        all_yaw.append(data[:, 5])

    # ── Summary overlay: Position ───────────────────────────────────
    cmap   = cm.get_cmap("tab10", len(csv_files))
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=False)
    fig.suptitle(f"Trajectory Overlay (Position) – {os.path.basename(folder)}", fontsize=13, fontweight="bold")

    goal_alt_display = (-goal[2]) if goal is not None else None   # NED z → altitude
    pos_specs = [
        (all_alt, "Z (m)", goal_alt_display),
        (all_x,   "X (m)", goal[0] if goal else None),
        (all_y,   "Y (m)", goal[1] if goal else None),
    ]
    for ax, (series_list, label, g_val) in zip(axes, pos_specs):
        for i, (t, s) in enumerate(zip(all_time, series_list)):
            ax.plot(t, s, color=cmap(i), linewidth=1.2, alpha=0.8, label=f"Ep {i + 1}")
        if g_val is not None:
            ax.axhline(y=g_val, color=COLOR_GOAL, linestyle="--",
                       linewidth=1.2, label="Goal")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=7, ncol=5)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "overlay_summary.png"))

    # ── Summary overlay: Attitude ────────────────────────────────────
    fig2, axes2 = plt.subplots(3, 1, figsize=(12, 11), sharex=False)
    fig2.suptitle(f"Trajectory Overlay (Attitude) – {os.path.basename(folder)}", fontsize=13, fontweight="bold")

    att_specs = [
        (all_roll,  "Roll (\u00b0)",  goal_att[0]),
        (all_pitch, "Pitch (\u00b0)", goal_att[1]),
        (all_yaw,   "Yaw (\u00b0)",   goal_att[2]),
    ]
    for ax, (series_list, label, g_val) in zip(axes2, att_specs):
        for i, (t, s) in enumerate(zip(all_time, series_list)):
            ax.plot(t, s, color=cmap(i), linewidth=1.2, alpha=0.8, label=f"Ep {i + 1}")
        if g_val is not None:
            ax.axhline(y=g_val, color=COLOR_GOAL, linestyle="--",
                       linewidth=1.2, label="Goal")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=7, ncol=5)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig2.tight_layout()
    _save(fig2, os.path.join(out_dir, "overlay_summary_attitude.png"))


# ════════════════════════════════════════════════════════════════════
# 4. COMPARE TWO FOLDERS (e.g. noisy vs no-noise)
# ════════════════════════════════════════════════════════════════════

def plot_compare_folders(folder_a: str, folder_b: str,
                         label_a:  str = None,
                         label_b:  str = None,
                         out_dir:  str = None,
                         goal:     list = None,
                         goal_attitude: list = None) -> None:
    """
    Compare average trajectory statistics between two episode folders.
    Plots mean ± std for altitude, X, Y, and roll/pitch/yaw over time.
    """
    label_a = label_a or os.path.basename(folder_a)
    label_b = label_b or os.path.basename(folder_b)
    out_dir = out_dir or "comparison_plots"
    _ensure_dir(out_dir)
    goal_att = goal_attitude if goal_attitude is not None else [0.0, 0.0, 0.0]

    def _load_folder(folder: str):
        pattern = os.path.join(folder, "trajectory_ep*.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            files = sorted(glob.glob(os.path.join(folder, "*.csv")))
        series_list = {"alt": [], "x": [], "y": [], "roll": [], "pitch": [], "yaw": []}
        max_len = 0
        for f in files:
            _, data, _ = _load_traj_csv(f)
            series_list["alt"].append(data[:, 2])   # altitude already positive up
            series_list["x"].append(data[:, 0])
            series_list["y"].append(data[:, 1])
            series_list["roll"].append(data[:, 3])
            series_list["pitch"].append(data[:, 4])
            series_list["yaw"].append(data[:, 5])
            max_len = max(max_len, len(data))
        # Pad or truncate to common length
        for key in series_list:
            padded = []
            for arr in series_list[key]:
                if len(arr) < max_len:
                    arr = np.pad(arr, (0, max_len - len(arr)), constant_values=arr[-1])
                padded.append(arr[:max_len])
            series_list[key] = np.array(padded)
        time = np.arange(max_len) * 0.1
        return time, series_list

    time_a, data_a = _load_folder(folder_a)
    time_b, data_b = _load_folder(folder_b)

    # ── Position comparison ─────────────────────────────────────────
    titles = ["Z (m)", "X (m)", "Y (m)"]
    keys   = ["alt", "x", "y"]
    goal_vals = [
        ((-goal[2]) if goal else None),   # NED z → altitude
        (goal[0]    if goal else None),
        (goal[1]    if goal else None),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=False)
    fig.suptitle(f"Comparison (Position): {label_a} vs {label_b}", fontsize=13, fontweight="bold")

    for ax, key, title, g_val in zip(axes, keys, titles, goal_vals):
        for time, data, label, color in [
            (time_a, data_a, label_a, COLOR_MAIN),
            (time_b, data_b, label_b, COLOR_ACCENT),
        ]:
            mean = data[key].mean(axis=0)
            std  = data[key].std(axis=0)
            ax.plot(time, mean, color=color, linewidth=2.0, label=f"{label} (mean)")
            ax.fill_between(time, mean - std, mean + std,
                            color=color, alpha=0.2, label=f"{label} (±1 std)")
        if g_val is not None:
            ax.axhline(y=g_val, color=COLOR_GOAL, linestyle="--",
                       linewidth=1.2, label="Goal")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    _save(fig, os.path.join(out_dir, f"compare_{label_a}_vs_{label_b}.png"))

    # ── Attitude comparison ─────────────────────────────────────────
    titles_att = ["Roll (\u00b0)", "Pitch (\u00b0)", "Yaw (\u00b0)"]
    keys_att   = ["roll", "pitch", "yaw"]

    fig2, axes2 = plt.subplots(3, 1, figsize=(12, 11), sharex=False)
    fig2.suptitle(f"Comparison (Attitude): {label_a} vs {label_b}", fontsize=13, fontweight="bold")

    for ax, key, title, g_val in zip(axes2, keys_att, titles_att, goal_att):
        for time, data, label, color in [
            (time_a, data_a, label_a, COLOR_MAIN),
            (time_b, data_b, label_b, COLOR_ACCENT),
        ]:
            mean = data[key].mean(axis=0)
            std  = data[key].std(axis=0)
            ax.plot(time, mean, color=color, linewidth=2.0, label=f"{label} (mean)")
            ax.fill_between(time, mean - std, mean + std,
                            color=color, alpha=0.2, label=f"{label} (±1 std)")
        if g_val is not None:
            ax.axhline(y=g_val, color=COLOR_GOAL, linestyle="--",
                       linewidth=1.2, label="Goal")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig2.tight_layout()
    _save(fig2, os.path.join(out_dir, f"compare_{label_a}_vs_{label_b}_attitude.png"))


# ════════════════════════════════════════════════════════════════════
# 5. BALANCE / ROBUSTNESS EVALUATION SUMMARY
# ════════════════════════════════════════════════════════════════════

def plot_balance_summary(csv_path: str, out_dir: str = None) -> None:
    """
    Visualize the balance evaluation summary CSV with:
      - Survival rate per disturbance configuration
      - Max position error per configuration
      - Max roll / pitch angles
      - Final distance to goal
    """
    df = pd.read_csv(csv_path)
    out_dir = out_dir or os.path.dirname(csv_path) or "."
    _ensure_dir(out_dir)

    configs = df["config"].unique()
    n_cfg   = len(configs)

    # Aggregate per config
    grouped = df.groupby("config").agg(
        survival_rate  = ("survived",    "mean"),
        mean_max_err   = ("max_pos_err_m", "mean"),
        std_max_err    = ("max_pos_err_m", "std"),
        mean_roll      = ("max_roll_deg",   "mean"),
        mean_pitch     = ("max_pitch_deg",  "mean"),
        mean_final_dist= ("final_dist_m",   "mean"),
    ).reindex(configs)  # preserve original order

    x = np.arange(n_cfg)
    bar_w = 0.6

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Balance / Robustness Evaluation Summary", fontsize=14, fontweight="bold")

    # ── Survival rate ─────────────────────────────────────────────
    ax = axes[0, 0]
    colors = [COLOR_MAIN if v > 0 else "#EF9A9A" for v in grouped["survival_rate"]]
    ax.bar(x, grouped["survival_rate"] * 100, color=colors, width=bar_w, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Survival Rate (%)")
    ax.set_title("Survival Rate per Configuration")
    ax.set_ylim([0, 110])
    ax.axhline(y=50, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    # ── Max position error ────────────────────────────────────────
    ax = axes[0, 1]
    ax.bar(x, grouped["mean_max_err"],
           yerr=grouped["std_max_err"].fillna(0),
           color=COLOR_ACCENT, width=bar_w, edgecolor="white",
           capsize=4, error_kw={"linewidth": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Max Position Error (m)")
    ax.set_title("Max Position Error per Configuration")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    # ── Max roll and pitch ────────────────────────────────────────
    ax = axes[1, 0]
    w  = 0.35
    ax.bar(x - w / 2, grouped["mean_roll"],  width=w, label="Max Roll (deg)",  color=COLOR_MAIN,   edgecolor="white")
    ax.bar(x + w / 2, grouped["mean_pitch"], width=w, label="Max Pitch (deg)", color=COLOR_ACCENT, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Max Roll and Pitch per Configuration")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    # ── Final distance to goal ────────────────────────────────────
    ax = axes[1, 1]
    ax.bar(x, grouped["mean_final_dist"], color="#FFA726", width=bar_w, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Final Distance to Goal (m)")
    ax.set_title("Final Distance to Goal per Configuration")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    base = os.path.splitext(os.path.basename(csv_path))[0]
    _save(fig, os.path.join(out_dir, f"{base}_summary_plot.png"))


# ════════════════════════════════════════════════════════════════════
# 6. SINGLE BALANCE-EVAL TRAJECTORY (with phase annotation)
# ════════════════════════════════════════════════════════════════════

def plot_balance_trajectory(csv_path: str, out_dir: str = None) -> None:
    """
    Plot a single balance-evaluation trajectory CSV.
    Columns: step, time_s, phase, x_m, y_m, z_m, roll_deg, pitch_deg, yaw_deg
    Annotates the 'kick' phase with a shaded region.
    """
    df = pd.read_csv(csv_path)
    out_dir = out_dir or os.path.dirname(csv_path) or "."
    _ensure_dir(out_dir)

    time  = df["time_s"].values
    alt   = df["z_m"].values         # altitude (positive up, already -z_NED in CSV)
    x_m   = df["x_m"].values
    y_m   = df["y_m"].values
    roll  = df["roll_deg"].values
    pitch = df["pitch_deg"].values
    yaw   = df["yaw_deg"].values

    # Identify phase boundaries for annotation
    phases = df["phase"].values if "phase" in df.columns else None
    kick_mask = (phases == "kick") if phases is not None else None

    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    fig.suptitle(f"Balance Evaluation – {base_name}", fontsize=13, fontweight="bold")

    def _shade_kick(ax):
        if kick_mask is not None and kick_mask.any():
            t_start = time[kick_mask][0]
            t_end   = time[kick_mask][-1]
            ax.axvspan(t_start, t_end, color="#FFCC80", alpha=0.45, label="Kick phase")

    series_specs = [
        (axes[0, 0], time, alt,   "Altitude (m)",  GOAL_POS[2],  True),
        (axes[0, 1], time, x_m,   "X (m)",         GOAL_POS[0],  False),
        (axes[1, 0], time, y_m,   "Y (m)",         GOAL_POS[1],  False),
        (axes[1, 1], time, roll,  "Roll (deg)",    0,             False),
        (axes[2, 0], time, pitch, "Pitch (deg)",   0,             False),
        (axes[2, 1], time, yaw,   "Yaw (deg)",     0,             False),
    ]

    for ax, t, series, label, g_val, is_alt in series_specs:
        color = COLOR_ACCENT if "Roll" in label or "Pitch" in label or "Yaw" in label else COLOR_MAIN
        ax.plot(t, series, color=color, linewidth=1.3)
        if g_val is not None:
            ax.axhline(y=g_val, color=COLOR_GOAL, linestyle="--",
                       linewidth=1.0, label="Goal / Zero")
        _shade_kick(ax)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    _save(fig, os.path.join(out_dir, f"{base_name}_plot.png"))


# ════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Drone PPO result visualization tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--training-log",
        metavar="CSV",
        help="Path to a PPO training log CSV (columns: episode, timestep, avg_reward)",
    )
    p.add_argument(
        "--trajectories",
        metavar="FOLDER",
        help="Folder containing trajectory_ep*.csv files; generates individual + overlay plots",
    )
    p.add_argument(
        "--compare",
        nargs=2,
        metavar=("FOLDER_A", "FOLDER_B"),
        help="Compare two trajectory folders (mean ± std overlay)",
    )
    p.add_argument(
        "--compare-labels",
        nargs=2,
        metavar=("LABEL_A", "LABEL_B"),
        default=None,
        help="Display labels for --compare folders (default: folder names)",
    )
    p.add_argument(
        "--balance",
        metavar="CSV",
        help="Path to balance_eval_summary.csv; generates bar-chart summary",
    )
    p.add_argument(
        "--balance-traj",
        metavar="CSV",
        help="Path to a single balance-evaluation trajectory CSV (traj_*.csv)",
    )
    p.add_argument(
        "--out-dir",
        metavar="DIR",
        default=None,
        help="Override output directory for all plots (default: next to input files)",
    )
    p.add_argument(
        "--goal",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z_NED"),
        default=None,
        help="Goal position in NED frame, e.g. --goal 0 0 -3  (Z altitude = 3 m)",
    )
    p.add_argument(
        "--goal-attitude",
        nargs=3,
        type=float,
        metavar=("ROLL", "PITCH", "YAW"),
        default=None,
        help="Goal attitude in degrees, e.g. --goal-attitude 0 0 0 (default: 0 0 0)",
    )
    p.add_argument(
        "--smooth",
        type=int,
        default=20,
        metavar="N",
        help="Moving-average window for learning curve (default: 20)",
    )
    p.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        metavar="N",
        help="Limit number of episodes when plotting a folder",
    )
    return p.parse_args()


def main():
    args = parse_args()
    goal = args.goal  # None or [x, y, z_ned]
    goal_attitude = args.goal_attitude  # None or [roll, pitch, yaw] degrees

    did_something = False

    if args.training_log:
        print(f"[PLOT] Training log: {args.training_log}")
        plot_training_log(args.training_log, out_dir=args.out_dir, smooth_window=args.smooth)
        did_something = True

    if args.trajectories:
        print(f"[PLOT] Trajectories folder: {args.trajectories}")
        plot_trajectories_folder(
            args.trajectories,
            out_dir=args.out_dir,
            goal=goal,
            max_eps=args.max_episodes,
            goal_attitude=goal_attitude,
        )
        did_something = True

    if args.compare:
        folder_a, folder_b = args.compare
        label_a = args.compare_labels[0] if args.compare_labels else None
        label_b = args.compare_labels[1] if args.compare_labels else None
        print(f"[PLOT] Comparing: {folder_a}  vs  {folder_b}")
        plot_compare_folders(
            folder_a, folder_b,
            label_a=label_a, label_b=label_b,
            out_dir=args.out_dir,
            goal=goal,
            goal_attitude=goal_attitude,
        )
        did_something = True

    if args.balance:
        print(f"[PLOT] Balance summary: {args.balance}")
        plot_balance_summary(args.balance, out_dir=args.out_dir)
        did_something = True

    if args.balance_traj:
        print(f"[PLOT] Balance trajectory: {args.balance_traj}")
        plot_balance_trajectory(args.balance_traj, out_dir=args.out_dir)
        did_something = True

    if not did_something:
        print("No input specified. Use --help for usage information.")


if __name__ == "__main__":
    main()
