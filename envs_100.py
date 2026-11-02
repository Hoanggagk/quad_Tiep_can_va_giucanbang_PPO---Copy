import asyncio
import math
import threading

import numpy as np
from gymnasium.spaces import Box
from projectairsim.gym_envs.gym_env import ProjectAirSimEnv
from projectairsim import Drone
from projectairsim.utils import quaternion_to_rpy, load_scene_config_as_dict

class DroneHoveringEnv(ProjectAirSimEnv):
    MAX_STEPS_PER_EPISODE = 100   # Số bước tối đa: 50 × 0.1s = 5s thực

    def __init__(self):
        self.sim_config_fname = "scene_basic_drone.jsonc"
        # Thời gian giữa các lần gửi lệnh hành động (> 20ms của flight controller 50Hz)
        self.dt = 0.1
        super().__init__(self.sim_config_fname)
        self.observation_shape = (13,)
        self.action_shape = (3,)

        self.observation_space = Box(-math.inf, math.inf, shape=self.observation_shape, dtype=np.float32)
        self.action_space = Box(-1.5, 1.5, shape=self.action_shape, dtype=np.float32)

        # Khởi tạo giá trị mặc định để tránh AttributeError trước reset()
        self.goal_pose = [0, 0, -2]
        self._hover_steps = 0
        self._pose_lock = threading.Lock()
        self._subscribed_topics = []
        self._init_episode_state()

    def _init_episode_state(self):
        """Khởi tạo/reset toàn bộ biến trạng thái của một episode."""
        self.done = False
        self.crashed = False
        self.ground_touch = False
        self.actor_pose = None
        self.observation = None
        self.collision_info = None
        self._prev_pos = None
        self._prev_rpy = None
        self._prev_t = None        # nanosec sim timestamp (thay wall-clock)
        self._target_simtime = 0   # nanosec: mốc thời gian sim để đồng bộ lock-step
        self._pose_sync = False
        self.linear_velocity = (0.0, 0.0, 0.0)
        self.angular_velocity = (0.0, 0.0, 0.0)
        self.prev_action = np.zeros(3)
        self.prev_state = None
        self._hover_steps = 0
        self._step_count = 0

    def set_target_sim_time(self, t_tgt):
        """Đặt mốc sim time cần đợi, reset cờ sync cho lock-step."""
        self._target_simtime = int(t_tgt)  # nanosec
        self._pose_sync = False

    def is_pose_synced(self) -> bool:
        return self._pose_sync

    def reset(self, seed=None, options=None):
        # Hủy subscription cũ để tránh leak callback qua các episode
        if self._subscribed_topics:
            self.client.unsubscribe(self._subscribed_topics)
            self._subscribed_topics = []

        self._init_episode_state()

        # Load scene configuration from file
        config_dict, _ = load_scene_config_as_dict(self.sim_config_fname)
        self.world.load_scene(config_dict, delay_after_load_sec=0)
        self.actor = Drone(self.client, self.world, "Drone1")

        self.goal_pose = [0, 0, -2]  # Mục tiêu: bay lên cao 2m so với vị trí xuất phát

        pose_topic = self.actor.robot_info["actual_pose"]
        self.client.subscribe(pose_topic, lambda _, pose: self.pose_callback(pose))
        self._subscribed_topics = [pose_topic]

        self.actor.enable_api_control()
        self.actor.arm()
        self.world.pause()  # Đảm bảo sim đang dừng cho manual stepping

        obs = self.get_observation()
        return obs, {}
    async def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.tolist()
        action = np.clip(np.array(action, dtype=np.float32), -1.5, 1.5)
        v_north = float(action[0])
        v_east = float(action[1])
        v_down = float(action[2])

        time_float_precision_offset = 1e-3
        await self.actor.move_by_velocity_async(
            v_north, v_east, v_down,
            duration=(self.dt - time_float_precision_offset),
            yaw_is_rate=True,
            yaw=0.0,
        )

        # Lock-step: đặt mốc target timestamp, tiến sim không blocking
        t_next = self.world.get_sim_time() + int(self.dt * 1e9)
        self.set_target_sim_time(t_next)
        self.world.continue_for_sim_time(self.dt * 1e9, wait_until_complete=False)

        # Spin đến khi nhận được pose với đúng sim timestamp
        while not self.is_pose_synced():
            await asyncio.sleep(0.001)

        next_obs = self.get_observation()
        self._check_termination()
        reward = self.get_reward(action)
        terminated = self.done
        self._step_count += 1
        truncated = self._step_count >= self.MAX_STEPS_PER_EPISODE
        info = {"status": "Running"}

        self.prev_state = next_obs.copy() if next_obs is not None else None
        self.prev_action = action.copy()

        return (next_obs, reward, terminated, truncated, info)

    def render(self):
        if self.observation is not None:
            print(f"Obs:{self.observation}")
    def get_observation(self):
        # Nếu chưa nhận được pose từ callback, lấy trực tiếp từ sim
        if self.actor_pose is None:
            raw_pose = self.world.get_object_pose(self.actor.name)
            self.pose_callback(raw_pose)

        with self._pose_lock:
            x, y, z, roll, pitch, yaw = self.actor_pose
            vx, vy, vz = self.linear_velocity
            roll_rate, pitch_rate, yaw_rate = self.angular_velocity

        # Vector trạng thái 13 chiều:
        # [x, y, z, roll, pitch, yaw, vx, vy, vz, roll_rate, pitch_rate, yaw_rate, dist_to_goal]
        dist_to_goal = float(np.linalg.norm(np.array([x, y, z]) - np.array(self.goal_pose)))
        self.observation = np.array(
            [x, y, z, roll, pitch, yaw, vx, vy, vz, roll_rate, pitch_rate, yaw_rate, dist_to_goal],
            dtype=np.float32,
        )
        return self.observation

    def pose_callback(self, pose):
        """Callback cập nhật vị trí, tư thế và ước lượng vận tốc bằng đạo hàm hữu hạn."""
        sim_t_ns = pose.get("time_stamp")  # nanosec từ sim clock

        # Hỗ trợ cả hai định dạng key: translation/position, rotation/orientation
        p = pose.get("translation") or pose.get("position", {})
        r = pose.get("rotation")    or pose.get("orientation", {})

        x = float(p.get("x", 0.0))
        y = float(p.get("y", 0.0))
        z = float(p.get("z", 0.0))

        roll, pitch, yaw = quaternion_to_rpy(
            float(r.get("w", 1.0)),
            float(r.get("x", 0.0)),
            float(r.get("y", 0.0)),
            float(r.get("z", 0.0)),
        )

        pos = np.array([x, y, z])
        rpy = np.array([roll, pitch, yaw])

        with self._pose_lock:
            # Ước lượng vận tốc bằng vi phân hữu hạn trên sim timestamp
            # Dùng sim time thay wall-clock để chính xác khi sim chạy nhanh/chậm hơn real-time
            if self._prev_t is not None and sim_t_ns is not None:
                dt = (sim_t_ns - self._prev_t) / 1e9  # nanosec → giây
                if dt > 1e-9:
                    self.linear_velocity = tuple((pos - self._prev_pos) / dt)
                    # Xử lý wraparound góc yaw (±π) để tránh spike vận tốc góc
                    ang_diff = rpy - self._prev_rpy
                    ang_diff[2] = math.atan2(math.sin(ang_diff[2]), math.cos(ang_diff[2]))
                    self.angular_velocity = tuple(ang_diff / dt)

            self.actor_pose = [x, y, z, roll, pitch, yaw]
            self._prev_pos = pos
            self._prev_rpy = rpy
            self._prev_t = sim_t_ns  # lưu sim timestamp (nanosec)

        # Đánh dấu pose đã sync khi timestamp đạt target (lock-step pattern)
        if sim_t_ns is not None and sim_t_ns >= self._target_simtime:
            self._pose_sync = True

    def collision_callback(self, collision_info):
        self.collision_info = collision_info
        self.crashed = True
        self.done = True
        print(f"[TERM] Va chạm phát hiện: {collision_info}")

    def _check_termination(self):
        """Kiểm tra các điều kiện kết thúc sớm để tránh lãng phí bước huấn luyện."""
        if self.actor_pose is None:
            return

        with self._pose_lock:
            x, y, z = self.actor_pose[:3]
            roll, pitch = self.actor_pose[3], self.actor_pose[4]
            vx, vy, vz = self.linear_velocity

        # 1. Drone đang lật (roll hoặc pitch vượt quá 90°)
        if abs(roll) > math.pi / 2 or abs(pitch) > math.pi / 2:
            print("[TERM] Drone đang lật (attitude out of bounds).")
            self.crashed = True
            self.done = True
            return

        # 2. Drone rơi gần mặt đất (NED: z=0 là mặt đất, z âm là lên cao)
        if z > -0.2:
            print(f"[TERM] Drone gần mặt đất (z={z:.2f}m).")
            self.ground_touch = True
            self.crashed = True
            self.done = True
            return

        # 3. Quá xa mục tiêu
        target = np.array(self.goal_pose)
        pos_err = np.linalg.norm(np.array([x, y, z]) - target)
        if pos_err > 5.0:
            print(f"[TERM] Quá xa mục tiêu (pos_err={pos_err:.1f}m).")
            self.crashed = True
            self.done = True
            return

        # 4. Vận tốc nguy hiểm
        vel_norm = math.sqrt(vx**2 + vy**2 + vz**2)
        if vel_norm > 5.0:
            print(f"[TERM] Vận tốc quá cao (vel={vel_norm:.1f} m/s).")
            self.crashed = True
            self.done = True

    def get_reward(self, action):
        if self.actor_pose is None: return 0.0
        
        # 1. Tính toán các thông số cơ bản
        curr_pos = self.actor_pose[:3]
        target = np.array(self.goal_pose)
        pos_err = np.linalg.norm(curr_pos - target)
        vel_norm = np.linalg.norm(self.linear_velocity)
        
        # 2. Phạt kết thúc (Termination)
        if self.ground_touch: return -2.0
        if pos_err > 5.0: return -2.0

        # 3. Thành phần Reward chính (đã đơn giản hóa)
        # Vị trí: 1.0 khi ở tâm, giảm dần ra xa
        r_pos = max(0, 1.0 - pos_err / 3.0) 
        
        # Tiếp cận: Thưởng khi thu hẹp khoảng cách
        r_approach = 0.0
        if self.prev_state is not None:
            prev_err = np.linalg.norm(self.prev_state[:3] - target)
            r_approach = (prev_err - pos_err) * 2.0 # Hệ số 2.0 để tín hiệu rõ hơn

        # Vận tốc: Không áp dụng phạt
        r_vel = 0.0

        # Tư thế: Phạt nếu nghiêng quá nhiều
        r_att = -0.1 * (abs(self.actor_pose[3]) + abs(self.actor_pose[4]))

        # 4. Tổng hợp (Weighting)
        # Tổng này thường nằm trong khoảng [0, 1] cho mỗi bước
        reward = r_pos + r_approach + r_vel + r_att + 0.01 # survival

        # 5. Bonus dừng hẳn (Hover success)
        if pos_err < 0.2 and vel_norm < 0.2:
            reward += 0.5
            self._hover_steps += 1
            if self._hover_steps > 20:
                reward += 5.0 # Thắng lợi hoàn toàn
                self.done = True
        else:
            self._hover_steps = 0

        return float(reward)

