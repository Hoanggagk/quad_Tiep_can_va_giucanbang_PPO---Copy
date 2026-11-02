"""
Huấn luyện PPO giữ quadcopter ổn định tại điểm mục tiêu (hover)
trong môi trường ProjectAirSim, có xử lý nhiễu gió và va chạm.

Cách dùng:
  python train_hover.py                      # Huấn luyện từ đầu
  python  train_hover.py --resume<ckpt.pth>  # Tiếp tục từ checkpoint
  python train_hover.py --test <ckpt.pth>    # Chỉ kiểm tra
"""

import os
import sys
import asyncio
import argparse
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")          # backend không cần GUI (an toàn khi chạy headless)
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs import DroneHoveringEnv
from ppo import PPO

# ═══════════════════════════════════════════════════════════════════
# THAM SỐ CẤU HÌNH
# ═══════════════════════════════════════════════════════════════════

# Không gian trạng thái và hành động
STATE_DIM  = 13   # [x, y, z, roll, pitch, yaw, vx, vy, vz, roll_rate, pitch_rate, yaw_rate, dist_to_goal]
ACTION_DIM = 3    # [v_north, v_east, v_down] ∈ [-3, 3]

# Cấu hình PPO
# Action space là [-1.5, 1.5]
ACTION_STD_INIT       = 1.5       # Độ lệch chuẩn ban đầu (~33% của range ±1.5)
ACTION_STD_DECAY_RATE = 0.01     # Tốc độ giảm action_std mỗi 100 episode
MIN_ACTION_STD        = 0.01      # Ngưỡng dưới (~1.7% của range, ≈±0.15 m/s khi khai thác)
ACTION_STD_DECAY_FREQ = 50        # Giảm action_std cứ 50 episode (nhanh hơn)

MAX_TRAINING_TIMESTEPS = int(5e5)   # Tổng số timestep tối đa (giảm từ 3e6 → 5e5 = 6x nhanh hơn)
STEPS_PER_EPISODE      = 50         # Số bước tối đa mỗi episode (50 × 0.1s = 5s thực, giảm từ 100 để tăng tốc)
UPDATE_TIMESTEP        = 2000       # Cập nhật PPO sau mỗi N timestep (giữ nguyên)
K_EPOCHS               = 30         # Số epoch per update (giảm từ 40 → 25 để tránh overfit)
EPS_CLIP               = 0.2
GAMMA                  = 0.99
LR_ACTOR               = 3e-4
LR_CRITIC              = 1e-3

# Cấu hình log và lưu model
LOG_DIR        = "log_dir/"
RUN_NAME       = "hover_projectairsim"
PRINT_FREQ     = STEPS_PER_EPISODE * 20  # In log ít hơn (từ ×10 → ×20)
LOG_FREQ       = STEPS_PER_EPISODE * 5   # Ghi CSV ít hơn (từ ×2 → ×5)
SAVE_MODEL_FREQ = int(5e4)                # Lưu checkpoint ít hơn (từ 1e5 → 5e4)

# ── Điều kiện dừng sớm ──────────────────────────────────────────
# Bài toán: DUY TRÌ VÀ CÂN BẰNG tại [0,0,-3] — drone khởi động tại mục tiêu.
# Mỗi episode chạy đầy đủ STEPS_PER_EPISODE bước, không dừng sớm theo reward.
# Dừng chỉ khi đạt MAX_TRAINING_TIMESTEPS hoặc Ctrl+C.


# ═══════════════════════════════════════════════════════════════════
# VẼ BIỂU ĐỒ KẾT QUẢ HUẤN LUYỆN
# ═══════════════════════════════════════════════════════════════════

def _smooth(data: list[float], window: int = 20) -> list[float]:
    """Trung bình động (moving average) để làm mượt đường cong."""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid").tolist()

def _plot_training_results(
    episode_rewards: list[float],
    save_dir:        str = ".",
    smooth_window:   int = 20,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    episodes = list(range(1, len(episode_rewards) + 1))
    ax.plot(episodes, episode_rewards, color="#90CAF9", linewidth=0.8,
            alpha=0.6, label="Reward mỗi episode")
    smoothed = _smooth(episode_rewards, smooth_window)
    offset   = len(episode_rewards) - len(smoothed)
    ax.plot(
        list(range(offset + 1, len(episode_rewards) + 1)),
        smoothed,
        color="#1565C0", linewidth=2.0,
        label=f"Trung bình động ({smooth_window} ep)",
    )
    ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Tổng Reward")
    ax.set_title("Learning Curve")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    lc_path = os.path.join(save_dir, "learning_curve.png")
    fig.savefig(lc_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Learning Curve → {lc_path}")


# ═══════════════════════════════════════════════════════════════════
# VÒNG LẶP HUẤN LUYỆN (ASYNC – vì env.step() là coroutine)
# ═══════════════════════════════════════════════════════════════════

async def train(resume_path: str = None):
    """
    Vòng lặp huấn luyện PPO với DroneHoveringEnv (ProjectAirSim).

    Args:
        resume_path: Đường dẫn checkpoint .pth để tiếp tục huấn luyện.
                     None = huấn luyện từ đầu.

    Returns:
        Đường dẫn checkpoint cuối cùng đã lưu.
    """

    # ── Chuẩn bị thư mục log ──────────────────────────────────────
    run_dir = os.path.join(LOG_DIR, RUN_NAME)
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(LOG_DIR, f"PPO_log_{RUN_NAME}.csv")
    log_f = open(log_path, "w+")
    log_f.write("episode,timestep,avg_reward\n")

    # ── Khởi tạo môi trường ───────────────────────────────────────
    print("[ENV] Đang kết nối ProjectAirSim...")
    env = DroneHoveringEnv()
    print("[ENV] Kết nối thành công.")

    # ── Khởi tạo PPO ──────────────────────────────────────────────
    ppo_agent = PPO(
        STATE_DIM, ACTION_DIM,
        LR_ACTOR, LR_CRITIC,
        GAMMA, K_EPOCHS, EPS_CLIP,
        ACTION_STD_INIT,
    )

    if resume_path is not None:
        print(f"[PPO] Nạp checkpoint: {resume_path}")
        ppo_agent.load(resume_path)
        print("[PPO] Đã nạp xong.")

    # ── Biến theo dõi tiến trình ───────────────────────────────────
    start_time = datetime.now().replace(microsecond=0)
    print(f"[TRAIN] Bắt đầu lúc: {start_time}")
    print(f"[TRAIN] Mục tiêu hover: {env.goal_pose if hasattr(env, 'goal_pose') else '[0, 0, -5]'}")

    time_step              = 0
    i_episode              = 0
    print_running_reward   = 0
    print_running_episodes = 0
    log_running_reward     = 0
    log_running_episodes   = 0
    final_path = os.path.join(run_dir, "ppo_hover_final.pth")

    # ── Lịch sử để vẽ biểu đồ ────────────────────────────────────
    history_episode_rewards: list[float] = []

    stop_reason = "max_timesteps"

    try:
        # ── Vòng lặp huấn luyện ────────────────────────────────────
        while time_step <= MAX_TRAINING_TIMESTEPS:
            obs, _ = env.reset()
            current_ep_reward = 0

            for step in range(STEPS_PER_EPISODE):
                # Chọn hành động từ policy
                action = ppo_agent.select_action(obs)

                # Thực thi bước (async)
                obs, reward, terminated, truncated, info = await env.step(action)
                done = terminated or truncated

                # Ghi vào buffer
                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(done)

                time_step         += 1
                current_ep_reward += reward

                # ── Cập nhật PPO ──────────────────────────────────────
                if time_step % UPDATE_TIMESTEP == 0:
                    ppo_agent.update()

                # ── Ghi log CSV ───────────────────────────────────────
                if time_step % LOG_FREQ == 0 and log_running_episodes > 0:
                    avg = round(log_running_reward / log_running_episodes, 4)
                    log_f.write(f"{i_episode},{time_step},{avg}\n")
                    log_f.flush()
                    log_running_reward     = 0
                    log_running_episodes   = 0

                # ── In console ────────────────────────────────────────
                if time_step % PRINT_FREQ == 0 and print_running_episodes > 0:
                    avg = round(print_running_reward / print_running_episodes, 2)
                    elapsed = datetime.now().replace(microsecond=0) - start_time
                    print(
                        f"  Episode: {i_episode:<6}  "
                        f"Timestep: {time_step:<9}  "
                        f"Avg Reward: {avg:<10}  "
                        f"Elapsed: {elapsed}"
                    )
                    print_running_reward   = 0
                    print_running_episodes = 0

                # ── Lưu checkpoint ────────────────────────────────────
                if time_step % SAVE_MODEL_FREQ == 0:
                    ckpt = os.path.join(run_dir, f"ep{i_episode}_step{time_step}.pth")
                    ppo_agent.save(ckpt)
                    elapsed = datetime.now().replace(microsecond=0) - start_time
                    print(f"  [SAVE] {ckpt}  |  elapsed: {elapsed}")

                if done:
                    print(
                        f"  [Episode {i_episode}] "
                        f"Step {step + 1}/{STEPS_PER_EPISODE}  "
                        f"Reward: {current_ep_reward:.2f}  "
                        f"{'[CRASH]' if env.crashed else '[OK]'}"
                    )
                    break

            history_episode_rewards.append(current_ep_reward)
            print_running_reward   += current_ep_reward
            print_running_episodes += 1
            log_running_reward     += current_ep_reward
            log_running_episodes   += 1
            i_episode              += 1

            # ── Giảm action_std (episode-based) ──────────────────────────────
            if i_episode % ACTION_STD_DECAY_FREQ == 0 and i_episode > 0:
                old_std = ppo_agent.action_std
                ppo_agent.decay_action_std(ACTION_STD_DECAY_RATE, MIN_ACTION_STD)
                new_std = ppo_agent.action_std
                print(
                    f"  [STD] Episode {i_episode}: action_std "
                    f"{old_std:.4f} → {new_std:.4f}"
                    + (" (đã đạt MIN)" if new_std <= MIN_ACTION_STD else "")
                )



    except KeyboardInterrupt:
        stop_reason = "user interrupt (Ctrl+C)"
        print(f"\n[TRAIN] Dừng bởi người dùng (Ctrl+C).")
    finally:
        log_f.close()
        env.close()

    end_time = datetime.now().replace(microsecond=0)
    print("=" * 65)
    print(f"Bắt đầu : {start_time}")
    print(f"Kết thúc : {end_time}")
    print(f"Tổng thời gian: {end_time - start_time}")
    print(f"Lý do dừng   : {stop_reason}")
    print("=" * 65)

    # Lưu model cuối
    ppo_agent.save(final_path)
    print(f"[SAVE] Model cuối: {final_path}")

    # ── Vẽ và lưu biểu đồ ────────────────────────────────────────
    _plot_training_results(
        history_episode_rewards,
        save_dir=run_dir,
    )

    return final_path


# ═══════════════════════════════════════════════════════════════════
# VẼ BIỂU ĐỒ QUỸ ĐẠO BAY
# ═══════════════════════════════════════════════════════════════════

def _plot_trajectory(
    trajectories: list,
    goal: list = None,
    save_dir: str = "test_results",
) -> None:
    """Vẽ quỹ đạo bay 3D và vị trí theo thời gian cho tất cả episode kiểm tra."""
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
        ax.scatter(t[0, 0],  t[0, 1],  -t[0, 2],  color=colors[i], marker="o", s=50,  zorder=4)  # điểm đầu
        ax.scatter(t[-1, 0], t[-1, 1], -t[-1, 2], color=colors[i], marker="x", s=80,  zorder=4)  # điểm cuối

    if goal is not None:
        ax.scatter(goal[0], goal[1], -goal[2], color="red", marker="*", s=300, zorder=5, label="Goal")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([0, 2])
    ax.set_title("Quỹ đạo bay – 3D")
    ax.view_init(elev=20, azim=45)  # Góc nhìn: X (trái-phải), Y (chiều sâu), Z (lên)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path_3d = os.path.join(save_dir, "trajectory_3d.png")
    fig.savefig(path_3d, dpi=150)
    plt.close(fig)
    print(f"[PLOT] 3D Trajectory → {path_3d}")

    # ── Vị trí + Góc theo thời gian ──────────────────────────────
    fig, axes_grid = plt.subplots(3, 2, figsize=(14, 9), sharex=False)
    pos_cfg   = [(2, "Z (m)"), (0, "X (m)"), (1, "Y (m)")]
    ang_cfg   = [(3, "Roll (°)"), (4, "Pitch (°)"), (5, "Yaw (°)")]
    goal_vals = goal if goal is not None else [None, None, None]

    for i, traj in enumerate(trajectories):
        if not traj:
            continue
        t    = np.array(traj)
        time = np.arange(len(t)) * 0.1   # 0.1 s / step
        for row, (data_idx, _) in enumerate(pos_cfg):
            val = -t[:, data_idx] if data_idx == 2 else t[:, data_idx]
            axes_grid[row, 0].plot(time, val, color=colors[i], linewidth=1.2,
                                   alpha=0.75, label=f"Ep {i + 1}")
        for row, (data_idx, _) in enumerate(ang_cfg):
            axes_grid[row, 1].plot(time, np.degrees(t[:, data_idx]),
                                   color=colors[i], linewidth=1.2,
                                   alpha=0.75, label=f"Ep {i + 1}")

    for row, (data_idx, label) in enumerate(pos_cfg):
        ax = axes_grid[row, 0]
        if goal_vals[data_idx] is not None:
            g_val = -goal_vals[data_idx] if data_idx == 2 else goal_vals[data_idx]
            ax.axhline(y=g_val, color="red", linestyle="--", linewidth=1.0,
                       label="Goal" if row == 0 else "_nolegend_")
        ax.set_ylabel(label)
        ax.set_ylim([-0.1, 2])
        ax.grid(True, linestyle="--", alpha=0.4)

    for row, (_, label) in enumerate(ang_cfg):
        ax = axes_grid[row, 1]
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylabel(label)
        ax.set_ylim([-2, 2])
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

    # ── Xuất CSV từng episode ──────────────────────────────────────
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
        print(f"[CSV] Episode {i + 1} → {csv_path}")


# ═══════════════════════════════════════════════════════════════════
# KIỂM TRA MÔ HÌNH ĐÃ HUẤN LUYỆN
# ═══════════════════════════════════════════════════════════════════

async def test(checkpoint_path: str, num_episodes: int = 10):
    """
    Kiểm tra mô hình đã huấn luyện trong ProjectAirSim.

    Args:
        checkpoint_path: Đường dẫn file .pth đã lưu.
        num_episodes: Số episode kiểm tra.
    """
    print(f"[TEST] Nạp checkpoint: {checkpoint_path}")

    ppo_agent = PPO(
        STATE_DIM, ACTION_DIM,
        LR_ACTOR, LR_CRITIC,
        GAMMA, K_EPOCHS, EPS_CLIP,
        action_std_init=MIN_ACTION_STD,  # dùng std nhỏ khi test (khai thác)
    )
    ppo_agent.load(checkpoint_path)
    print("[TEST] Đã nạp model.")

    env = DroneHoveringEnv()
    goal = list(env.goal_pose) if hasattr(env, "goal_pose") else [0.0, 0.0, -5.0]

    all_trajectories: list = []

    for ep in range(num_episodes):
        obs, _ = env.reset(reload_scene=(ep == 0))
        ep_reward    = 0
        step         = 0
        episode_traj: list = [obs[:6].tolist()]  # [x, y, z, roll, pitch, yaw]

        print(f"\n[TEST] Episode {ep + 1}/{num_episodes}")

        for step in range(STEPS_PER_EPISODE):
            action = ppo_agent.select_action(obs)
            obs, reward, terminated, truncated, info = await env.step(action)
            done = terminated or truncated
            ep_reward += reward

            # Ghi nhận vị trí và góc vào quỹ đạo
            episode_traj.append(obs[:6].tolist())

            # In trạng thái mỗi 10 bước
            if step % 10 == 0:
                pos  = obs[:3]
                rpy  = np.degrees(obs[3:6])
                vel  = obs[6:9]
                dist = float(obs[12])   # dist_to_goal đã có sẵn trong observation
                print(
                    f"  Step {step:4d} | "
                    f"pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) | "
                    f"rpy=({rpy[0]:.1f}°, {rpy[1]:.1f}°, {rpy[2]:.1f}°) | "
                    f"vel=({vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}) | "
                    f"dist={dist:.2f}m"
                )

            if done:
                print(
                    f"  [Episode {ep + 1}] Tổng reward: {ep_reward:.2f}  "
                    f"Bước: {step + 1}  "
                    f"{'[CRASH]' if env.crashed else '[OK]'}"
                )
                break

        all_trajectories.append(episode_traj)
        print(
            f"  Episode {ep + 1}/{num_episodes} | Reward: {ep_reward:.2f}  "
            f"Bước: {step + 1}  "
            f"{'[CRASH]' if env.crashed else '[OK]'}"
        )

    env.close()

    # ── Vẽ và lưu quỹ đạo ────────────────────────────────────────
    _plot_trajectory(all_trajectories, goal=goal, save_dir="test_results")

    print("\n[TEST] Hoàn thành.")


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Huấn luyện / kiểm tra PPO hover quadcopter (ProjectAirSim)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CKPT",
        help="Tiếp tục huấn luyện từ checkpoint .pth",
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        metavar="CKPT",
        help="Chỉ kiểm tra mô hình từ checkpoint .pth",
    )
    parser.add_argument(
        "--num-test-episodes",
        type=int,
        default=5,
        help="Số episode khi kiểm tra (mặc định: 5)",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    if args.test is not None:
        # Chế độ chỉ kiểm tra
        await test(args.test, num_episodes=args.num_test_episodes)
    else:
        # Chế độ huấn luyện (có thể resume)
        final_ckpt = await train(resume_path=args.resume)
        print(f"\n[DONE] Checkpoint cuối: {final_ckpt}")


if __name__ == "__main__":
    asyncio.run(main())
