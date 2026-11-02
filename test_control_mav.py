import asyncio
import os
import sys
import numpy as np
import math
import time
from mavsdk import System
from mavsdk.offboard import (
    OffboardError,
    VelocityNedYaw
)
from mavsdk.action import ActionError

# Import PPO agent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ppo import PPO

# ═══════════════════════════════════════════════════════════════════
# CẤU HÌNH PPO VÀ DRONE
# ═══════════════════════════════════════════════════════════════════

STATE_DIM  = 13   # [x, y, z, roll, pitch, yaw, vx, vy, vz, roll_rate, pitch_rate, yaw_rate, dist_to_goal]
ACTION_DIM = 3    # [v_north, v_east, v_down]

# Đường dẫn checkpoint PPO (thay đổi theo checkpoint bạn muốn dùng)
CHECKPOINT_PATH = "Ketquahuanluyen\\3_4_2026\\model\\ep2006_step100000.pth"

# Điểm mục tiêu hover (NED frame)
GOAL_POSITION = [0.0, 0.0, -2.0]  # x=0, y=0, z=-2m (2m trên mặt đất)

# Thời gian điều khiển
CONTROL_DURATION = 10  # giây
CONTROL_RATE = 10      # Hz (0.1s mỗi bước)

# Hyperparameters PPO (phải khớp với train)
LR_ACTOR  = 3e-4
LR_CRITIC = 1e-3
GAMMA     = 0.99
K_EPOCHS  = 10
EPS_CLIP  = 0.2
MIN_ACTION_STD = 0.01  # Dùng std nhỏ khi test (khai thác)


async def run():
    """
    Chạy điều khiển drone bằng PPO qua MAVSDK.
    """
    
    # ── Khởi tạo PPO Agent ────────────────────────────────────────
    # State history để tính vi phân hữu hạn (giống envs.py trong training)
    print(f"[PPO] Đang nạp model từ: {CHECKPOINT_PATH}")
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[ERROR] Không tìm thấy checkpoint: {CHECKPOINT_PATH}")
        print("[INFO] Hãy chạy train_hover.py trước để tạo model.")
        return
    
    ppo_agent = PPO(
        STATE_DIM, ACTION_DIM,
        LR_ACTOR, LR_CRITIC,
        GAMMA, K_EPOCHS, EPS_CLIP,
        action_std_init=MIN_ACTION_STD,
    )
    ppo_agent.load(CHECKPOINT_PATH)
    print("[PPO] Đã nạp model thành công.")

    # ── Kết nối drone ─────────────────────────────────────────────
    drone = System()

    print("[MAVSDK] Đang kết nối PX4...")
    await drone.connect(system_address="udpin://127.0.0.1:14550")

    print("[MAVSDK] Chờ kết nối...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("[MAVSDK] Đã kết nối với PX4")
            break

    print("[MAVSDK] Chờ GPS...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("[MAVSDK] GPS OK")
            break

    print("[MAVSDK] Đang arm...")
    # Thử arm với retry và in trạng thái health khi bị từ chối
    async def try_arm(drone: System, max_attempts: int = 6, delay: float = 2.0) -> bool:
        for attempt in range(1, max_attempts + 1):
            try:
                print(f"[MAVSDK] Gửi lệnh arm (thử {attempt}/{max_attempts})")
                await drone.action.arm()
                # Kiểm tra trạng thái armed một lần
                try:
                    async for armed in drone.telemetry.armed():
                        if hasattr(armed, 'is_armed') and armed.is_armed:
                            print("[MAVSDK] Drone đã armed.")
                            return True
                        break
                except Exception:
                    pass
                # Nếu không có thông tin armed nhưng không ném lỗi, coi là success
                return True

            except ActionError as e:
                print(f"[WARNING] Arm thất bại: {e}")
                # In một mẫu health để debug nguyên nhân
                try:
                    async for health in drone.telemetry.health():
                        print(f"[MAVSDK] Health sample: {health}")
                        break
                except Exception:
                    pass
                try:
                    async for batt in drone.telemetry.battery():
                        print(f"[MAVSDK] Battery sample: {batt}")
                        break
                except Exception:
                    pass
                await asyncio.sleep(delay)
        return False

    armed_ok = await try_arm(drone)
    if not armed_ok:
        print("[ERROR] Không thể arm drone sau nhiều lần thử. Kiểm tra PX4, safety switch, và pre-arm checks.")
        return
    
    # Takeoff lên độ cao nhỏ trước khi chuyển sang Offboard
    TAKEOFF_ALT = 1.0  # mét
    print(f"[MAVSDK] Thiết lập takeoff altitude {TAKEOFF_ALT}m và takeoff...")
    try:
        # một số firmware có hỗ trợ set_takeoff_altitude
        await drone.action.set_takeoff_altitude(TAKEOFF_ALT)
    except Exception:
        # Không bắt buộc, nên chỉ cảnh báo nếu không hỗ trợ
        print("[WARNING] Không thể set_takeoff_altitude (firmware có thể không hỗ trợ)")

    try:
        await drone.action.takeoff()
    except Exception as e:
        print(f"[ERROR] Lỗi khi takeoff: {e}")
        await drone.action.disarm()
        return

    # Chờ đến khi đạt độ cao takeoff (down_m nhỏ hơn hoặc bằng -TAKEOFF_ALT + tol)
    tol = 0.15
    max_wait = 20.0
    start_t = time.time()
    print("[MAVSDK] Đang chờ đạt độ cao takeoff...")
    try:
        async for pos_vel in drone.telemetry.position_velocity_ned():
            z_down = pos_vel.position.down_m
            # Giá trị down_m âm khi lên cao theo quy ước trong code này
            if z_down <= -TAKEOFF_ALT + tol:
                print(f"[MAVSDK] Đã đạt takeoff độ cao (down_m={z_down:.2f})")
                break
            if time.time() - start_t > max_wait:
                print("[WARNING] Timeout khi chờ takeoff — tiếp tục")
                break
            await asyncio.sleep(0.1)
    except Exception as e:
        print(f"[WARNING] Lỗi khi đọc telemetry trong khi chờ takeoff: {e}")

    # Gửi setpoint ban đầu trước khi start Offboard
    await drone.offboard.set_velocity_ned(
        VelocityNedYaw(
            north_m_s=0.0,
            east_m_s=0.0,
            down_m_s=0.0,
            yaw_deg=0.0
        )
    )

    print("[MAVSDK] Khởi động Offboard mode...")
    try:
        await drone.offboard.start()
    except OffboardError as error:
        print(f"[ERROR] Lỗi Offboard: {error}")
        await drone.action.disarm()
        return

    print(f"[CONTROL] Bắt đầu điều khiển PPO (hover tại {GOAL_POSITION})")
    print(f"[CONTROL] Thời gian: {CONTROL_DURATION}s, tần số: {CONTROL_RATE}Hz")
    
    try:
        # ── Vòng lặp điều khiển PPO ──────────────────────────────
        control_steps = CONTROL_DURATION * CONTROL_RATE
        dt = 1.0 / CONTROL_RATE
        
        # Lưu trữ trajectory để phân tích
        trajectory = []
        
        for step in range(control_steps):
            # ── Thu thập telemetry để tạo observation ────────────
            obs = await get_observation(drone, GOAL_POSITION)
            
            if obs is None:
                print("[WARNING] Không lấy được telemetry, giữ nguyên lệnh")
                await asyncio.sleep(dt)
                continue
            
            # Lưu trajectory
            trajectory.append(obs[:6].tolist())  # [x, y, z, roll, pitch, yaw]
            
            # ── Chọn action từ PPO ────────────────────────────────
            action = ppo_agent.select_action(obs)  # [v_north, v_east, v_down]
            
            v_north = float(action[0])
            v_east  = float(action[1])
            v_down  = float(action[2])
            
            # ── Áp dụng action qua MAVSDK ─────────────────────────
            await drone.offboard.set_velocity_ned(
                VelocityNedYaw(
                    north_m_s=v_north,
                    east_m_s=v_east,
                    down_m_s=v_down,
                    yaw_deg=0.0  # Giữ yaw = 0
                )
            )
            
            # ── In trạng thái mỗi 1s (10 bước) ────────────────────
            if step % 10 == 0:
                x, y, z = obs[0], obs[1], obs[2]
                roll, pitch, yaw = np.degrees(obs[3:6])
                dist = obs[12]
                print(
                    f"  [{step * dt:5.1f}s] "
                    f"Pos=({x:6.2f}, {y:6.2f}, {z:6.2f}) | "
                    f"RPY=({roll:5.1f}°, {pitch:5.1f}°, {yaw:5.1f}°) | "
                    f"Dist={dist:5.2f}m | "
                    f"Cmd=({v_north:5.2f}, {v_east:5.2f}, {v_down:5.2f}) m/s"
                )
            
            await asyncio.sleep(dt)
        
        print(f"[CONTROL] Hoàn thành {CONTROL_DURATION}s điều khiển.")
        
        # ── Lưu trajectory ─────────────────────────────────────────
        save_trajectory(trajectory, "ppo_mavsdk_trajectory.csv")

    finally:
        print("[MAVSDK] Dừng Offboard...")
        await drone.offboard.stop()

        print("[MAVSDK] Hạ cánh...")
        await drone.action.land()

        await asyncio.sleep(10)


async def get_observation(drone: System, goal: list) -> np.ndarray:
    """
    Thu thập telemetry từ MAVSDK và tạo observation vector 13 chiều.
    
    Format: [x, y, z, roll, pitch, yaw, vx, vy, vz, roll_rate, pitch_rate, yaw_rate, dist_to_goal]
    
    Args:
        drone: MAVSDK System object
        goal: Mục tiêu [x, y, z] trong NED frame
    
    Returns:
        Observation vector 13 chiều hoặc None nếu không lấy được telemetry
    """
    try:
        # Lấy vị trí và vận tốc NED
        async for pos_vel in drone.telemetry.position_velocity_ned():
            position_ned = pos_vel.position
            velocity_ned = pos_vel.velocity
            break
        
        # Lấy góc Euler (roll, pitch, yaw)
        async for attitude in drone.telemetry.attitude_euler():
            roll_deg  = attitude.roll_deg
            pitch_deg = attitude.pitch_deg
            yaw_deg   = attitude.yaw_deg
            break
        
        # Lấy vận tốc góc (roll_rate, pitch_rate, yaw_rate)
        async for angular_vel in drone.telemetry.attitude_angular_velocity_body():
            roll_rate_rad  = angular_vel.roll_rad_s
            pitch_rate_rad = angular_vel.pitch_rad_s
            yaw_rate_rad   = angular_vel.yaw_rad_s
            break
        
        # Chuyển đổi sang đúng định dạng
        x = position_ned.north_m
        y = position_ned.east_m
        z = position_ned.down_m
        
        vx = velocity_ned.north_m_s
        vy = velocity_ned.east_m_s
        vz = velocity_ned.down_m_s
        
        # Chuyển góc từ độ sang radian
        roll  = math.radians(roll_deg)
        pitch = math.radians(pitch_deg)
        yaw   = math.radians(yaw_deg)
        
        # Tính khoảng cách đến mục tiêu
        dist_to_goal = np.linalg.norm(np.array([x, y, z]) - np.array(goal))
        
        # Tạo observation vector
        obs = np.array(
            [x, y, z, roll, pitch, yaw, vx, vy, vz, 
             roll_rate_rad, pitch_rate_rad, yaw_rate_rad, dist_to_goal],
            dtype=np.float32
        )
        
        return obs
        
    except Exception as e:
        print(f"[ERROR] Không lấy được telemetry: {e}")
        return None


def save_trajectory(trajectory: list, filename: str = "trajectory.csv"):
    """
    Lưu trajectory vào file CSV.
    
    Args:
        trajectory: List of [x, y, z, roll, pitch, yaw]
        filename: Tên file output
    """
    import csv
    
    if not trajectory:
        print("[WARNING] Trajectory rỗng, không lưu file.")
        return
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['time_s', 'x_m', 'y_m', 'z_m', 'roll_deg', 'pitch_deg', 'yaw_deg'])
            
            for i, state in enumerate(trajectory):
                time = i * 0.1  # 10 Hz = 0.1s per step
                x, y, z, roll, pitch, yaw = state
                writer.writerow([
                    f"{time:.1f}",
                    f"{x:.4f}",
                    f"{y:.4f}",
                    f"{z:.4f}",
                    f"{math.degrees(roll):.4f}",
                    f"{math.degrees(pitch):.4f}",
                    f"{math.degrees(yaw):.4f}"
                ])
        
        print(f"[SAVE] Trajectory đã lưu: {filename}")
    
    except Exception as e:
        print(f"[ERROR] Không lưu được trajectory: {e}")


if __name__ == "__main__":
    asyncio.run(run())