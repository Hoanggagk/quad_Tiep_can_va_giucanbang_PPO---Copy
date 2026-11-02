"""
Điều khiển drone bằng mô hình PPO đã huấn luyện.

Sử dụng:
  python ControlDrone_disturbance.py                              # Dùng model mặc định, 5 episode
  python ControlDrone_disturbance.py --model <đường_dẫn.pth>     # Chỉ định model khác
  python ControlDrone_disturbance.py --episodes 10               # Chạy 10 episode
  python ControlDrone_disturbance.py --save-dir results/         # Thư mục lưu kết quả
  python ControlDrone_disturbance.py --no-plot                   # Không vẽ đồ thị
"""

import os
import sys
import asyncio
import argparse
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")   # Backend không cần GUI
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs import DroneHoveringEnv
from ppo import PPO

import json
from sympy import true
from websocket import create_connection

# ═══════════════════════════════════════════════════════════════════
# CẤU HÌNH MẶC ĐỊNH
# ═══════════════════════════════════════════════════════════════════

DEFAULT_MODEL_PATH  = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Ketquahuanluyen", "11_6_2026", "ppo_hover_final.pth",
)
DEFAULT_NUM_EPISODES   = 5
DEFAULT_SAVE_DIR       = "control_results"
MAX_STEPS_PER_EPISODE  = 100   # Khớp với cấu hình huấn luyện (100 bước × 0.1 s = 10 s)

# Các siêu tham số phải khớp với lúc huấn luyện
STATE_DIM  = 13
ACTION_DIM = 3
LR_ACTOR   = 3e-4
LR_CRITIC  = 1e-3
GAMMA      = 0.99
K_EPOCHS   = 10
EPS_CLIP   = 0.2
# Dùng action_std nhỏ khi khai thác (không khám phá)
ACTION_STD_EXPLOIT = 0.01


# ═══════════════════════════════════════════════════════════════════
# VẼ BIỂU ĐỒ QUỸ ĐẠO
# ═══════════════════════════════════════════════════════════════════

def _plot_trajectory(
    trajectories: list,
    goal: list = None,
    save_dir: str = DEFAULT_SAVE_DIR,
) -> None:
    """Vẽ quỹ đạo bay 3D và vị trí/góc theo thời gian, lưu thành PNG."""
    os.makedirs(save_dir, exist_ok=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(trajectories), 1)))

    # ── Đồ thị 3D ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")
    for i, traj in enumerate(trajectories):
        if not traj:
            continue
        t = np.array(traj)
        ax.plot(t[:, 0], t[:, 1], -t[:, 2],
                color=colors[i], linewidth=1.5, label=f"Ep {i + 1}", alpha=0.8)
        ax.scatter(t[0, 0],  t[0, 1],  -t[0, 2],  color=colors[i], marker="o", s=50,  zorder=4)
        ax.scatter(t[-1, 0], t[-1, 1], -t[-1, 2], color=colors[i], marker="x", s=80,  zorder=4)
    if goal is not None:
        ax.scatter(goal[0], goal[1], -goal[2],
                   color="red", marker="*", s=300, zorder=5, label="Goal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Độ cao (m)")
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([0, 2])
    ax.set_title("Quỹ đạo bay – 3D")
    ax.view_init(elev=20, azim=45)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path_3d = os.path.join(save_dir, "trajectory_3d.png")
    fig.savefig(path_3d, dpi=150)
    plt.close(fig)
    print(f"[PLOT] 3D Trajectory → {path_3d}")

    # ── Vị trí + Góc theo thời gian ──────────────────────────────
    fig, axes_grid = plt.subplots(3, 2, figsize=(14, 9), sharex=False)
    pos_cfg  = [(2, "Z (m)"), (0, "X (m)"), (1, "Y (m)")]
    ang_cfg  = [(3, "Roll (°)"), (4, "Pitch (°)"), (5, "Yaw (°)")]
    goal_vals = goal if goal is not None else [None, None, None]

    for i, traj in enumerate(trajectories):
        if not traj:
            continue
        t    = np.array(traj)
        time = np.arange(len(t)) * 0.1
        for row, (idx, _) in enumerate(pos_cfg):
            val = -t[:, idx] if idx == 2 else t[:, idx]
            axes_grid[row, 0].plot(time, val, color=colors[i], linewidth=1.2,
                                   alpha=0.75, label=f"Ep {i + 1}")
        for row, (idx, _) in enumerate(ang_cfg):
            axes_grid[row, 1].plot(time, np.degrees(t[:, idx]),
                                   color=colors[i], linewidth=1.2,
                                   alpha=0.75, label=f"Ep {i + 1}")

    for row, (idx, label) in enumerate(pos_cfg):
        ax = axes_grid[row, 0]
        if goal_vals[idx] is not None:
            g_val = -goal_vals[idx] if idx == 2 else goal_vals[idx]
            ax.axhline(y=g_val, color="red", linestyle="--", linewidth=1.0,
                       label="Goal" if row == 0 else "_nolegend_")
        ax.set_ylabel(label)
        ax.set_ylim([-1, 2])
        ax.grid(True, linestyle="--", alpha=0.4)
    for row, (_, label) in enumerate(ang_cfg):
        ax = axes_grid[row, 1]
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylabel(label)
        ax.set_ylim([-90, 90])
        ax.grid(True, linestyle="--", alpha=0.4)

    axes_grid[0, 0].legend(fontsize=7, loc="upper right")
    axes_grid[0, 0].set_title("Vị trí")
    axes_grid[0, 1].set_title("Góc")
    axes_grid[-1, 0].set_xlabel("Thời gian (s)")
    axes_grid[-1, 1].set_xlabel("Thời gian (s)")
    fig.suptitle("Vị trí & Góc theo thời gian")
    fig.tight_layout()
    path_time = os.path.join(save_dir, "trajectory_time.png")
    fig.savefig(path_time, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Trajectory vs Time → {path_time}")


def _save_csv(trajectories: list, save_dir: str) -> None:
    """Xuất quỹ đạo mỗi episode ra file CSV."""
    os.makedirs(save_dir, exist_ok=True)
    for i, traj in enumerate(trajectories):
        if not traj:
            continue
        t    = np.array(traj)
        time = np.arange(len(t)) * 0.1
        csv_path = os.path.join(save_dir, f"trajectory_ep{i + 1:03d}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            f.write("time_s,x_m,y_m,z_m,roll_deg,pitch_deg,yaw_deg\n")
            for k in range(len(t)):
                f.write(
                    f"{time[k]:.1f},"
                    f"{t[k, 0]:.4f},{t[k, 1]:.4f},{-t[k, 2]:.4f},"
                    f"{np.degrees(t[k, 3]):.4f},"
                    f"{np.degrees(t[k, 4]):.4f},"
                    f"{np.degrees(t[k, 5]):.4f}\n"
                )
        print(f"[CSV]  Episode {i + 1} → {csv_path}")


# ═══════════════════════════════════════════════════════════════════
# VÒNG LẶP ĐIỀU KHIỂN
# ═══════════════════════════════════════════════════════════════════

async def run_control(
    model_path:   str  = DEFAULT_MODEL_PATH,
    num_episodes: int  = DEFAULT_NUM_EPISODES,
    save_dir:     str  = DEFAULT_SAVE_DIR,
    enable_plot:  bool = True,
) -> None:
    """
    Nạp mô hình PPO và điều khiển drone trong ProjectAirSim.

    Args:
        model_path:   Đường dẫn file .pth của mô hình đã huấn luyện.
        num_episodes: Số episode chạy kiểm tra.
        save_dir:     Thư mục lưu kết quả (CSV + ảnh).
        enable_plot:  Có vẽ đồ thị hay không.
    """
    # ── Kiểm tra file mô hình ─────────────────────────────────────
    if not os.path.isfile(model_path):
        print(f"[LỖI] Không tìm thấy file mô hình: {model_path}")
        sys.exit(1)

    print("=" * 65)
    print("  ĐIỀU KHIỂN DRONE BẰNG MÔ HÌNH PPO")
    print("=" * 65)
    print(f"  Model     : {model_path}")
    print(f"  Episodes  : {num_episodes}")
    print(f"  Lưu tại   : {save_dir}")
    print("=" * 65)

    # ── Khởi tạo PPO và nạp trọng số ─────────────────────────────
    ppo_agent = PPO(
        STATE_DIM, ACTION_DIM,
        LR_ACTOR, LR_CRITIC,
        GAMMA, K_EPOCHS, EPS_CLIP,
        action_std_init=ACTION_STD_EXPLOIT,
    )
    ppo_agent.load(model_path)
    print(f"[MODEL] Đã nạp mô hình thành công.")

    # ── Khởi tạo môi trường ───────────────────────────────────────
    print("[ENV]  Đang kết nối ProjectAirSim...")
    env = DroneHoveringEnv()
    goal = list(env.goal_pose)
    print(f"[ENV]  Kết nối thành công. Mục tiêu hover: {goal}")

    all_trajectories: list  = []
    all_rewards:      list  = []
    start_time = datetime.now().replace(microsecond=0)

    #Kết nối websocket
    ws = create_connection("ws://127.0.0.1:30020")

    try:
        for ep in range(num_episodes):
            # Reset env; chỉ tải lại scene ở episode đầu tiên
            obs, _ = env.reset(reload_scene=(ep == 0))
            ep_reward = 0.0
            episode_traj: list = [obs[:6].tolist()]  # [x, y, z, roll, pitch, yaw]

            print(f"\n[EP {ep + 1:02d}/{num_episodes}] Bắt đầu episode...")

            for step in range(MAX_STEPS_PER_EPISODE):
                # Chọn hành động (khai thác – không ngẫu nhiên)
                action = ppo_agent.select_action(obs)

                # Thực thi hành động trong sim
                obs, reward, terminated, truncated, info = await env.step(action)
                done = terminated or truncated
                ep_reward += reward

                # Ghi quỹ đạo: [x, y, z, roll, pitch, yaw]
                episode_traj.append(obs[:6].tolist())

                # In trạng thái mỗi 10 bước
                if step % 10 == 0:
                    pos  = obs[:3]
                    rpy  = np.degrees(obs[3:6])
                    vel  = obs[6:9]
                    dist = float(obs[12])
                    print(
                        f"  Bước {step:3d} | "
                        f"pos=({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f}) m | "
                        f"rpy=({rpy[0]:+.1f}°, {rpy[1]:+.1f}°, {rpy[2]:+.1f}°) | "
                        f"vel=({vel[0]:+.2f}, {vel[1]:+.2f}, {vel[2]:+.2f}) m/s | "
                        f"dist={dist:.3f} m"
                    )
                if step == 10 or step == 30:
        
                    message = {
                        "MessageName": "http",
                        "Parameters": {
                            "Url": "/remote/object/property",
                            "Verb": "PUT",
                            "Body": {
                                "ObjectPath": "/Memory/UEDPIE_0_Untitled_MainGrid_L0_X0_Y-1.Untitled:PersistentLevel.Spawn_C_UAID_00155D02090057EF02_1296750191",
                                "propertyName": "flag",
                                "access": "WRITE_ACCESS",
                                "propertyValue": {
                                    "flag": True
                                }
                            }
                        }
                    }
                    ws.send(json.dumps(message))
                    print(ws.recv())
                    
                if 44 <= step < 45 :
                    env.actor.set_external_force([0, 0, -8.86])
                else:
                    env.actor.set_external_force([0, 0, 0])

                if done:
                    status = "[CRASH]" if env.crashed else "[OK]"
                    print(
                        f"  → Kết thúc sớm tại bước {step + 1}  "
                        f"Reward: {ep_reward:.2f}  {status}"
                    )
                    break

            all_trajectories.append(episode_traj)
            all_rewards.append(ep_reward)
            status = "[CRASH]" if env.crashed else "[OK]"
            print(
                f"[EP {ep + 1:02d}/{num_episodes}] Tổng reward: {ep_reward:.2f}  "
                f"Bước hoàn thành: {len(episode_traj) - 1}  {status}"
            )

    except KeyboardInterrupt:
        print("\n[STOP] Dừng bởi người dùng (Ctrl+C).")
    finally:
        env.close()

    # ── Thống kê tổng hợp ─────────────────────────────────────────
    elapsed = datetime.now().replace(microsecond=0) - start_time
    print("\n" + "=" * 65)
    print("  KẾT QUẢ ĐIỀU KHIỂN")
    print("=" * 65)
    if all_rewards:
        print(f"  Số episode hoàn thành : {len(all_rewards)}")
        print(f"  Reward trung bình     : {np.mean(all_rewards):.2f}")
        print(f"  Reward cao nhất       : {np.max(all_rewards):.2f}")
        print(f"  Reward thấp nhất      : {np.min(all_rewards):.2f}")
    print(f"  Thời gian chạy        : {elapsed}")
    print("=" * 65)

    # ── Lưu CSV ───────────────────────────────────────────────────
    if all_trajectories:
        _save_csv(all_trajectories, save_dir)

    # ── Vẽ biểu đồ ───────────────────────────────────────────────
    if enable_plot and all_trajectories:
        _plot_trajectory(all_trajectories, goal=goal, save_dir=save_dir)

    print("\n[DONE] Hoàn thành điều khiển drone.")


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Điều khiển drone bằng mô hình PPO đã huấn luyện (ProjectAirSim)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        metavar="PATH",
        help=f"Đường dẫn file .pth (mặc định: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_NUM_EPISODES,
        metavar="N",
        help=f"Số episode chạy (mặc định: {DEFAULT_NUM_EPISODES})",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=DEFAULT_SAVE_DIR,
        metavar="DIR",
        help=f"Thư mục lưu kết quả CSV và ảnh (mặc định: {DEFAULT_SAVE_DIR})",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Không vẽ đồ thị sau khi kết thúc",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        run_control(
            model_path=args.model,
            num_episodes=args.episodes,
            save_dir=args.save_dir,
            enable_plot=not args.no_plot,
        )
    )
