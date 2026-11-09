"""
Đánh giá khả năng giữ tư thế cân bằng của drone sau va chạm.

Phương pháp:
  1. Ổn định hover (STABILIZE_STEPS bước đầu) bằng PPO.
  2. Tiêm nhiễu loạn vận tốc (1 bước) để giả lập va chạm.
  3. Để PPO phục hồi trong RECOVERY_STEPS bước, ghi nhận các chỉ số.
  4. Lặp lại với nhiều hướng / cường độ va chạm khác nhau.

Kết quả:
  - Bảng tóm tắt in ra console
  - CSV chi tiết + biểu đồ PNG lưu vào --save-dir

Sử dụng (từ thư mục gốc):
  python -m drone_control.action_spaces.velocity_ned.evaluate_balance
  python -m drone_control.action_spaces.velocity_ned.evaluate_balance --model <path.pth>
"""

import os
import sys
import asyncio
import argparse
import csv
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drone_control.action_spaces.velocity_ned.env import DroneHoveringEnv
from drone_control.algorithms.ppo import PPO

# ═══════════════════════════════════════════════════════════════════
# CẤU HÌNH
# ═══════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "Ketquahuanluyen", "3_4_2026", "model", "ep2006_step100000.pth",
)
DEFAULT_SAVE_DIR      = "balance_eval_results"
DEFAULT_TRIALS        = 3          # Số lần thử mỗi cấu hình va chạm
STABILIZE_STEPS       = 20         # Bước hover trước khi va chạm (20 × 0.1 s = 2 s)
RECOVERY_STEPS        = 30         # Bước phục hồi sau va chạm (30 × 0.1 s = 3 s)
RECOVERY_THRESHOLD_M  = 0.30       # Ngưỡng khoảng cách đến mục tiêu coi là "đã phục hồi"
ATT_THRESHOLD_DEG     = 10.0       # Ngưỡng góc coi là "tư thế ổn định"

# Siêu tham số PPO (phải khớp với lúc huấn luyện)
STATE_DIM          = 13
ACTION_DIM         = 3
LR_ACTOR           = 3e-4
LR_CRITIC          = 1e-3
GAMMA              = 0.99
K_EPOCHS           = 30
EPS_CLIP           = 0.2
ACTION_STD_EXPLOIT = 0.01   # std nhỏ = khai thác, không khám phá

# ── Cấu hình va chạm (label, v_north, v_east, v_down) ──────────
# Đơn vị: m/s – được clip vào [-1.5, 1.5] bởi env.step()
# Để đạt hiệu ứng va chạm mạnh, dùng giá trị vượt ngưỡng clip:
#   env sẽ clip xuống ±1.5 m/s nhưng tác động vẫn rõ ràng.
DISTURBANCE_CONFIGS = [
    # (nhãn,        v_north, v_east, v_down)
    ("push_north",  1.5,     0.0,    0.0),   # đẩy về phía Bắc
    ("push_south", -1.5,     0.0,    0.0),   # đẩy về phía Nam
    ("push_east",   0.0,     1.5,    0.0),   # đẩy về phía Đông
    ("push_west",   0.0,    -1.5,    0.0),   # đẩy về phía Tây
    ("push_up",     0.0,     0.0,   -1.5),   # đẩy lên trên (v_down âm = lên)
    ("push_down",   0.0,     0.0,    1.5),   # đẩy xuống dưới
    ("push_diag",   1.5,     1.5,    0.0),   # đẩy chéo NE
    ("push_strong", 1.5,     1.5,   -1.5),   # đẩy tổng hợp mạnh
]


# ═══════════════════════════════════════════════════════════════════
# CẤU TRÚC DỮ LIỆU KẾT QUẢ
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TrialResult:
    config_label:     str
    trial_idx:        int
    dist_before_kick: float   # khoảng cách đến goal ngay trước khi va chạm (m)
    survived:         bool    # drone không rơi / không va chạm vật thể
    max_pos_err_m:    float   # sai lệch vị trí tối đa sau va chạm (m)
    max_roll_deg:     float   # roll tối đa sau va chạm (°)
    max_pitch_deg:    float   # pitch tối đa sau va chạm (°)
    recovery_steps:   int     # số bước để đạt RECOVERY_THRESHOLD_M (-1 nếu không phục hồi)
    final_dist_m:     float   # khoảng cách đến goal ở bước cuối (m)
    trajectory:       list    # [[x,y,z,roll,pitch,yaw], ...] toàn bộ episode
    kick_step:        int     # chỉ số bước trong trajectory tại thời điểm va chạm


# ═══════════════════════════════════════════════════════════════════
# CHẠY MỘT TRIAL
# ═══════════════════════════════════════════════════════════════════

async def _run_trial(
    env:          DroneHoveringEnv,
    ppo_agent:    PPO,
    config_label: str,
    kick_vn:      float,
    kick_ve:      float,
    kick_vd:      float,
    trial_idx:    int,
    reload_scene: bool,
) -> TrialResult:
    """Chạy một trial: ổn định → va chạm → phục hồi."""

    obs, _ = env.reset(reload_scene=reload_scene)
    trajectory = [obs[:6].tolist()]
    kick_step = STABILIZE_STEPS   # bước va chạm trong trajectory

    # ── Pha 1: Ổn định hover ──────────────────────────────────────
    for _ in range(STABILIZE_STEPS):
        action = ppo_agent.select_action(obs)
        obs, _, terminated, truncated, _ = await env.step(action)
        trajectory.append(obs[:6].tolist())
        if terminated or truncated:
            break

    dist_before_kick = float(obs[12])   # khoảng cách trước khi bị đẩy

    # ── Pha 2: Tiêm nhiễu loạn (1 bước) ──────────────────────────
    kick_action = np.array([kick_vn, kick_ve, kick_vd], dtype=np.float32)
    obs, _, terminated, truncated, _ = await env.step(kick_action)
    trajectory.append(obs[:6].tolist())

    # ── Pha 3: Phục hồi bằng PPO ──────────────────────────────────
    max_pos_err = float(obs[12])
    max_roll    = abs(float(np.degrees(obs[3])))
    max_pitch   = abs(float(np.degrees(obs[4])))
    recovery_steps = -1   # chưa phục hồi

    survived = not (terminated or truncated)

    if survived:
        for rec_step in range(1, RECOVERY_STEPS + 1):
            action = ppo_agent.select_action(obs)
            obs, _, terminated, truncated, _ = await env.step(action)
            trajectory.append(obs[:6].tolist())

            dist      = float(obs[12])
            roll_deg  = abs(float(np.degrees(obs[3])))
            pitch_deg = abs(float(np.degrees(obs[4])))

            max_pos_err = max(max_pos_err, dist)
            max_roll    = max(max_roll, roll_deg)
            max_pitch   = max(max_pitch, pitch_deg)

            # Đánh dấu phục hồi lần đầu
            if recovery_steps == -1 and dist < RECOVERY_THRESHOLD_M:
                recovery_steps = rec_step

            if terminated or truncated:
                survived = not env.crashed
                break

    final_dist = float(obs[12])

    return TrialResult(
        config_label=config_label,
        trial_idx=trial_idx,
        dist_before_kick=dist_before_kick,
        survived=survived,
        max_pos_err_m=max_pos_err,
        max_roll_deg=max_roll,
        max_pitch_deg=max_pitch,
        recovery_steps=recovery_steps,
        final_dist_m=final_dist,
        trajectory=trajectory,
        kick_step=kick_step,
    )


# ═══════════════════════════════════════════════════════════════════
# LƯU KẾT QUẢ
# ═══════════════════════════════════════════════════════════════════

def _save_results_csv(results: list[TrialResult], save_dir: str) -> str:
    """Lưu tất cả kết quả vào một file CSV."""
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "balance_eval_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "config", "trial",
            "dist_before_kick_m", "survived",
            "max_pos_err_m", "max_roll_deg", "max_pitch_deg",
            "recovery_steps", "final_dist_m",
        ])
        for r in results:
            writer.writerow([
                r.config_label, r.trial_idx + 1,
                f"{r.dist_before_kick:.4f}",
                int(r.survived),
                f"{r.max_pos_err_m:.4f}",
                f"{r.max_roll_deg:.2f}",
                f"{r.max_pitch_deg:.2f}",
                r.recovery_steps,
                f"{r.final_dist_m:.4f}",
            ])
    print(f"[CSV]  Bảng tổng hợp → {csv_path}")
    return csv_path


def _save_trajectory_csv(results: list[TrialResult], save_dir: str) -> None:
    """Lưu quỹ đạo từng trial ra CSV riêng."""
    traj_dir = os.path.join(save_dir, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)
    for r in results:
        fname = f"traj_{r.config_label}_trial{r.trial_idx + 1:02d}.csv"
        path  = os.path.join(traj_dir, fname)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "time_s", "phase",
                             "x_m", "y_m", "z_m",
                             "roll_deg", "pitch_deg", "yaw_deg"])
            for i, pt in enumerate(r.trajectory):
                if i < r.kick_step:
                    phase = "stabilize"
                elif i == r.kick_step:
                    phase = "kick"
                else:
                    phase = "recovery"
                writer.writerow([
                    i, f"{i * 0.1:.1f}", phase,
                    f"{pt[0]:.4f}", f"{pt[1]:.4f}", f"{-pt[2]:.4f}",
                    f"{np.degrees(pt[3]):.3f}",
                    f"{np.degrees(pt[4]):.3f}",
                    f"{np.degrees(pt[5]):.3f}",
                ])


# ═══════════════════════════════════════════════════════════════════
# VẼ BIỂU ĐỒ
# ═══════════════════════════════════════════════════════════════════

def _plot_summary(results: list[TrialResult], save_dir: str) -> None:
    """Vẽ biểu đồ tổng hợp các chỉ số theo từng cấu hình va chạm."""
    os.makedirs(save_dir, exist_ok=True)

    # Nhóm theo config
    configs = list(dict.fromkeys(r.config_label for r in results))
    metrics = {
        "max_pos_err_m":  [],
        "max_roll_deg":   [],
        "max_pitch_deg":  [],
        "recovery_steps": [],
        "survived_rate":  [],
    }
    for cfg in configs:
        grp = [r for r in results if r.config_label == cfg]
        metrics["max_pos_err_m"].append(np.mean([r.max_pos_err_m for r in grp]))
        metrics["max_roll_deg"].append(np.mean([r.max_roll_deg for r in grp]))
        metrics["max_pitch_deg"].append(np.mean([r.max_pitch_deg for r in grp]))
        rec = [r.recovery_steps for r in grp if r.recovery_steps >= 0]
        metrics["recovery_steps"].append(np.mean(rec) if rec else RECOVERY_STEPS + 1)
        metrics["survived_rate"].append(np.mean([float(r.survived) for r in grp]) * 100)

    x = np.arange(len(configs))
    bar_w = 0.6

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("Đánh giá khả năng giữ tư thế sau va chạm", fontsize=14, fontweight="bold")

    def _bar(ax, values, title, ylabel, color, hline=None):
        bars = ax.bar(x, values, width=bar_w, color=color, alpha=0.8, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if hline is not None:
            ax.axhline(y=hline, color="red", linestyle="--", linewidth=1.0, label=f"Ngưỡng = {hline}")
            ax.legend(fontsize=8)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(values + [1]),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7.5)

    _bar(axes[0, 0], metrics["max_pos_err_m"], "Sai lệch vị trí tối đa",
         "Khoảng cách (m)", "#1565C0", hline=RECOVERY_THRESHOLD_M)
    _bar(axes[0, 1], metrics["max_roll_deg"], "Roll tối đa sau va chạm",
         "Roll (°)", "#6A1B9A", hline=ATT_THRESHOLD_DEG)
    _bar(axes[0, 2], metrics["max_pitch_deg"], "Pitch tối đa sau va chạm",
         "Pitch (°)", "#AD1457", hline=ATT_THRESHOLD_DEG)
    _bar(axes[1, 0], metrics["recovery_steps"], "Thời gian phục hồi trung bình",
         "Số bước (× 0.1 s)", "#2E7D32", hline=None)
    _bar(axes[1, 1], metrics["survived_rate"], "Tỷ lệ sống sót",
         "Tỷ lệ (%)", "#E65100", hline=100)

    # Ô cuối: scatter max_roll vs max_pos_err
    ax_sc = axes[1, 2]
    colors_sc = plt.cm.tab10(np.linspace(0, 1, len(configs)))
    for i, cfg in enumerate(configs):
        grp = [r for r in results if r.config_label == cfg]
        ax_sc.scatter(
            [r.max_pos_err_m for r in grp],
            [r.max_roll_deg  for r in grp],
            label=cfg, color=colors_sc[i], s=60, alpha=0.8,
        )
    ax_sc.set_xlabel("Sai lệch vị trí tối đa (m)", fontsize=9)
    ax_sc.set_ylabel("Roll tối đa (°)", fontsize=9)
    ax_sc.set_title("Sai lệch vị trí vs Roll (scatter)", fontsize=10)
    ax_sc.legend(fontsize=6, ncol=2)
    ax_sc.grid(linestyle="--", alpha=0.4)

    fig.tight_layout()
    path = os.path.join(save_dir, "balance_eval_summary.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Biểu đồ tổng hợp → {path}")


def _plot_recovery_curves(results: list[TrialResult], save_dir: str) -> None:
    """Vẽ đường cong phục hồi vị trí và góc theo thời gian (1 hình mỗi config)."""
    configs  = list(dict.fromkeys(r.config_label for r in results))
    curve_dir = os.path.join(save_dir, "recovery_curves")
    os.makedirs(curve_dir, exist_ok=True)

    for cfg in configs:
        grp    = [r for r in results if r.config_label == cfg]
        colors = plt.cm.tab10(np.linspace(0, 1, len(grp)))

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"Phục hồi sau va chạm: {cfg}", fontsize=12)

        for i, r in enumerate(grp):
            if not r.trajectory:
                continue
            t    = np.array(r.trajectory)
            steps = np.arange(len(t))
            time  = steps * 0.1

            # Khoảng cách đến goal (tính lại từ trajectory)
            goal = np.array([0.0, 0.0, -2.0])
            dist = np.linalg.norm(t[:, :3] - goal, axis=1)

            axes[0].plot(time, dist, color=colors[i], linewidth=1.4,
                         label=f"Trial {i+1}", alpha=0.85)
            axes[1].plot(time, np.degrees(t[:, 3]), color=colors[i],
                         linewidth=1.4, label=f"Trial {i+1}", alpha=0.85)
            axes[2].plot(time, np.degrees(t[:, 4]), color=colors[i],
                         linewidth=1.4, label=f"Trial {i+1}", alpha=0.85)

            # Đánh dấu thời điểm va chạm
            kick_t = r.kick_step * 0.1
            for ax in axes:
                ax.axvline(x=kick_t, color="red", linestyle=":", linewidth=1.2,
                           label="Va chạm" if i == 0 else "_nolegend_")

        axes[0].axhline(y=RECOVERY_THRESHOLD_M, color="green", linestyle="--",
                        linewidth=1.0, label=f"Ngưỡng {RECOVERY_THRESHOLD_M} m")
        axes[1].axhline(y=ATT_THRESHOLD_DEG,  color="green", linestyle="--", linewidth=1.0)
        axes[1].axhline(y=-ATT_THRESHOLD_DEG, color="green", linestyle="--", linewidth=1.0)
        axes[2].axhline(y=ATT_THRESHOLD_DEG,  color="green", linestyle="--", linewidth=1.0)
        axes[2].axhline(y=-ATT_THRESHOLD_DEG, color="green", linestyle="--", linewidth=1.0)

        axes[0].set_title("Khoảng cách đến mục tiêu (m)")
        axes[1].set_title("Roll (°)")
        axes[2].set_title("Pitch (°)")
        for ax in axes:
            ax.set_xlabel("Thời gian (s)")
            ax.grid(linestyle="--", alpha=0.4)
            ax.legend(fontsize=7)

        fig.tight_layout()
        path = os.path.join(curve_dir, f"recovery_{cfg}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[PLOT] Đường cong phục hồi [{cfg}] → {path}")


def _print_table(results: list[TrialResult]) -> None:
    """In bảng tóm tắt kết quả ra console."""
    configs = list(dict.fromkeys(r.config_label for r in results))

    header = (
        f"{'Cấu hình va chạm':<18} | "
        f"{'Sống sót':<10} | "
        f"{'Pos err max(m)':<16} | "
        f"{'Roll max(°)':<12} | "
        f"{'Pitch max(°)':<13} | "
        f"{'Phục hồi (bước)':<16} | "
        f"{'Dist cuối (m)'}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    for cfg in configs:
        grp = [r for r in results if r.config_label == cfg]
        n_survived  = sum(1 for r in grp if r.survived)
        avg_pos_err = np.mean([r.max_pos_err_m  for r in grp])
        avg_roll    = np.mean([r.max_roll_deg   for r in grp])
        avg_pitch   = np.mean([r.max_pitch_deg  for r in grp])
        rec_ok      = [r.recovery_steps for r in grp if r.recovery_steps >= 0]
        avg_rec     = f"{np.mean(rec_ok):.1f}" if rec_ok else "KHÔNG PH"
        avg_final   = np.mean([r.final_dist_m   for r in grp])
        survived_str = f"{n_survived}/{len(grp)}"

        print(
            f"{cfg:<18} | "
            f"{survived_str:<10} | "
            f"{avg_pos_err:<16.3f} | "
            f"{avg_roll:<12.2f} | "
            f"{avg_pitch:<13.2f} | "
            f"{avg_rec:<16} | "
            f"{avg_final:.3f}"
        )

    print(sep)
    # Đánh giá tổng thể
    all_survived  = sum(1 for r in results if r.survived)
    all_rec_ok    = [r.recovery_steps for r in results if r.recovery_steps >= 0]
    print(f"\nTổng: {all_survived}/{len(results)} trial sống sót | "
          f"Tỷ lệ phục hồi: {len(all_rec_ok)}/{len(results)} | "
          f"Thời gian phục hồi TB: "
          f"{np.mean(all_rec_ok) * 0.1:.2f} s" if all_rec_ok else "N/A")


# ═══════════════════════════════════════════════════════════════════
# HÀM ĐÁNH GIÁ CHÍNH
# ═══════════════════════════════════════════════════════════════════

async def evaluate(
    model_path:   str  = DEFAULT_MODEL_PATH,
    num_trials:   int  = DEFAULT_TRIALS,
    save_dir:     str  = DEFAULT_SAVE_DIR,
    enable_plot:  bool = True,
) -> None:
    """Chạy toàn bộ đánh giá tư thế cân bằng sau va chạm."""

    if not os.path.isfile(model_path):
        print(f"[LỖI] Không tìm thấy file mô hình: {model_path}")
        sys.exit(1)

    print("=" * 65)
    print("  ĐÁNH GIÁ TƯ THẾ CÂN BẰNG SAU VA CHẠM")
    print("=" * 65)
    print(f"  Model             : {model_path}")
    print(f"  Số trial/config   : {num_trials}")
    print(f"  Bước ổn định      : {STABILIZE_STEPS}  ({STABILIZE_STEPS * 0.1:.1f} s)")
    print(f"  Bước phục hồi     : {RECOVERY_STEPS}  ({RECOVERY_STEPS * 0.1:.1f} s)")
    print(f"  Ngưỡng phục hồi   : {RECOVERY_THRESHOLD_M} m")
    print(f"  Số cấu hình       : {len(DISTURBANCE_CONFIGS)}")
    print(f"  Tổng số trial     : {len(DISTURBANCE_CONFIGS) * num_trials}")
    print(f"  Lưu tại           : {save_dir}")
    print("=" * 65)

    # ── Nạp mô hình ───────────────────────────────────────────────
    ppo_agent = PPO(
        STATE_DIM, ACTION_DIM,
        LR_ACTOR, LR_CRITIC,
        GAMMA, K_EPOCHS, EPS_CLIP,
        action_std_init=ACTION_STD_EXPLOIT,
    )
    ppo_agent.load(model_path)
    print("[MODEL] Đã nạp mô hình.")

    # ── Khởi tạo môi trường ───────────────────────────────────────
    print("[ENV]  Đang kết nối ProjectAirSim...")
    env = DroneHoveringEnv()
    print("[ENV]  Kết nối thành công.")

    all_results: list[TrialResult] = []
    start_time = datetime.now().replace(microsecond=0)
    total_trials = len(DISTURBANCE_CONFIGS) * num_trials
    trial_count  = 0

    try:
        for cfg_label, kick_vn, kick_ve, kick_vd in DISTURBANCE_CONFIGS:
            print(f"\n[CONFIG] {cfg_label}  kick=({kick_vn:+.1f}, {kick_ve:+.1f}, {kick_vd:+.1f}) m/s")

            for t_idx in range(num_trials):
                trial_count += 1
                # Tải lại scene chỉ ở trial đầu tiên của toàn bộ quá trình
                reload = (trial_count == 1)
                result = await _run_trial(
                    env, ppo_agent,
                    config_label=cfg_label,
                    kick_vn=kick_vn, kick_ve=kick_ve, kick_vd=kick_vd,
                    trial_idx=t_idx,
                    reload_scene=reload,
                )
                all_results.append(result)

                rec_str = (f"{result.recovery_steps} bước"
                           if result.recovery_steps >= 0 else "KHÔNG PHỤC HỒI")
                status  = "SỐNG" if result.survived else "CRASH"
                print(
                    f"  Trial {t_idx + 1}/{num_trials} [{trial_count:02d}/{total_trials}] "
                    f"| {status} "
                    f"| pos_err={result.max_pos_err_m:.2f} m "
                    f"| roll={result.max_roll_deg:.1f}° "
                    f"| pitch={result.max_pitch_deg:.1f}° "
                    f"| phục hồi: {rec_str}"
                )

    except KeyboardInterrupt:
        print("\n[STOP] Dừng bởi người dùng (Ctrl+C).")
    finally:
        env.close()

    elapsed = datetime.now().replace(microsecond=0) - start_time

    # ── In bảng tổng hợp ──────────────────────────────────────────
    if all_results:
        _print_table(all_results)

    print(f"\n[TIME] Tổng thời gian đánh giá: {elapsed}")

    # ── Lưu CSV ───────────────────────────────────────────────────
    if all_results:
        _save_results_csv(all_results, save_dir)
        _save_trajectory_csv(all_results, save_dir)

    # ── Vẽ biểu đồ ───────────────────────────────────────────────
    if enable_plot and all_results:
        _plot_summary(all_results, save_dir)
        _plot_recovery_curves(all_results, save_dir)

    print("\n[DONE] Hoàn thành đánh giá.")


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Đánh giá khả năng giữ tư thế cân bằng của drone sau va chạm"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        metavar="PATH",
        help=f"Đường dẫn file .pth (mặc định: model ep2006_step100000)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        metavar="N",
        help=f"Số lần thử mỗi cấu hình va chạm (mặc định: {DEFAULT_TRIALS})",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=DEFAULT_SAVE_DIR,
        metavar="DIR",
        help=f"Thư mục lưu kết quả (mặc định: {DEFAULT_SAVE_DIR})",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Không vẽ biểu đồ",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        evaluate(
            model_path=args.model,
            num_trials=args.trials,
            save_dir=args.save_dir,
            enable_plot=not args.no_plot,
        )
    )
