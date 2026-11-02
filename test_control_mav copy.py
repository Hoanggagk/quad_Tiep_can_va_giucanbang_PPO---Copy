import asyncio
import math
import numpy as np
from mavsdk import System
from mavsdk.action import ActionError


async def run():

    drone = System()
    monitor_task = None
    stop_event = asyncio.Event()
    # Mục tiêu theo NED (mặc định về home)
    goal = [0.0, 0.0, 0.0]

    print("Đang kết nối PX4...")
    await drone.connect(system_address="udpin://127.0.0.1:14550")

    print("Chờ kết nối...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Đã kết nối với PX4")
            break

    print("Chờ GPS...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("GPS OK")
            break
    
    # Bắt đầu nhiệm vụ giám sát telemetry
    monitor_task = asyncio.create_task(monitor(drone, goal, stop_event, interval=1.0))

    # Pre-arm: kiểm tra health trước khi gọi arm()
    try:
        async for health in drone.telemetry.health():
            print(f"[PRE-ARM] health: global_ok={health.is_global_position_ok} home_ok={health.is_home_position_ok}")
            # Nếu GPS hoặc home chưa OK, thông báo và chờ thêm
            if not (health.is_global_position_ok and health.is_home_position_ok):
                print("[PRE-ARM] GPS/home chưa sẵn sàng. Chờ thêm 2s...")
                await asyncio.sleep(2.0)
                continue
            break
    except Exception as e:
        print(f"[PRE-ARM] Không thể đọc health: {e}")

    print("Arm...")
    try:
        await drone.action.arm()
    except ActionError as e:
        print(f"[ERROR] Arm thất bại: {e}")
        await print_diagnostics(drone)
        # Không raise nữa — ta sẽ hạ cánh/dọn dẹp trong finally
    except Exception as e:
        print(f"[ERROR] Arm gây exception: {e}")
        await print_diagnostics(drone)

    TAKEOFF_ALTITUDE = 3.0
    try:
        # Nếu API hỗ trợ, cố gắng đặt độ cao takeoff
        try:
            await drone.action.set_takeoff_altitude(TAKEOFF_ALTITUDE)
        except Exception:
            pass

        print(f"Takeoff → {TAKEOFF_ALTITUDE} m...")
        await drone.action.takeoff()

        # Chờ vài giây để đạt độ cao và hover
        await asyncio.sleep(8)

        print("Hover...")
        await asyncio.sleep(5)

    finally:
        print("Hạ cánh...")
        await drone.action.land()
        await asyncio.sleep(8)

        print("Disarm...")
        try:
            await drone.action.disarm()
        except Exception:
            pass
        # Dừng nhiệm vụ giám sát và chờ nó kết thúc
        stop_event.set()
        if monitor_task is not None:
            try:
                await monitor_task
            except Exception:
                pass

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


async def print_diagnostics(drone: System):
    """
    In ra một số thông tin chẩn đoán khi lệnh arm() bị từ chối.
    """
    try:
        async for health in drone.telemetry.health():
            try:
                print(
                    f"[DIAG] Health: gyro_ok={health.is_gyrometer_calibration_ok} "
                    f"accel_ok={health.is_accelerometer_calibration_ok} "
                    f"mag_ok={health.is_magnetometer_calibration_ok} "
                    f"local_ok={health.is_local_position_ok} "
                    f"global_ok={health.is_global_position_ok} "
                    f"home_ok={health.is_home_position_ok}"
                )
            except Exception:
                # Some MAVSDK versions expose slightly different fields
                print(f"[DIAG] Health object: {health}")
            break
    except Exception as e:
        print(f"[DIAG] Không đọc được health: {e}")

    try:
        async for is_armed in drone.telemetry.armed():
            print(f"[DIAG] Armed: {is_armed}")
            break
    except Exception as e:
        print(f"[DIAG] Không đọc được trạng thái 'armed': {e}")

    try:
        async for conn in drone.core.connection_state():
            print(f"[DIAG] Connection: is_connected={conn.is_connected}")
            break
    except Exception as e:
        print(f"[DIAG] Không đọc được connection_state: {e}")

    # Thử in một vài status text (nếu có)
    try:
        count = 0
        async for st in drone.telemetry.status_text():
            print(f"[DIAG] Status: severity={st.severity} text={st.text}")
            count += 1
            if count >= 3:
                break
    except Exception:
        # Tiện lợi: không phải mọi hệ thống đều hỗ trợ status_text()
        pass


async def monitor(drone: System, goal: list, stop_event: asyncio.Event, interval: float = 1.0):
    """
    Giám sát telemetry theo chu kỳ và in các thông số chính.
    """
    while not stop_event.is_set():
        obs = await get_observation(drone, goal)
        if obs is None:
            print("[MONITOR] Không có telemetry.")
        else:
            x, y, z, roll, pitch, yaw, vx, vy, vz, roll_rate, pitch_rate, yaw_rate, dist = obs
            print(
                f"[MONITOR] Pos N:{x:.2f} E:{y:.2f} D:{z:.2f} | "
                f"Vel N:{vx:.2f} E:{vy:.2f} D:{vz:.2f} | "
                f"Att r:{math.degrees(roll):.1f}° p:{math.degrees(pitch):.1f}° y:{math.degrees(yaw):.1f}° | "
                f"Rates r:{roll_rate:.2f} p:{pitch_rate:.2f} y:{yaw_rate:.2f} | dist:{dist:.2f}"
            )
        await asyncio.sleep(interval)

if __name__ == "__main__":
    asyncio.run(run())