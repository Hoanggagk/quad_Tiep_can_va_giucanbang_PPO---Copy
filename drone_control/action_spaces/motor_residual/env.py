"""Gymnasium environment điều khiển trực tiếp bốn rotor của quadrotor.

Action được chuẩn hóa trong ``[-1, 1]^4`` và ánh xạ thành residual RPM quanh
operating point đã hiệu chuẩn trong ``config/motor_control.json``.
"""

from __future__ import annotations

import asyncio
from collections import deque
import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from projectairsim import Drone, ProjectAirSimClient, World
from projectairsim.utils import load_scene_config_as_dict, quaternion_to_rpy

from drone_control.action_spaces.motor_residual.curriculum import (
    EpisodeConditions,
    MotorCurriculum,
)

try:
    from gymnasium import Env
    from gymnasium.spaces import Box
except ModuleNotFoundError:
    # Interpreter ProjectAirSim hiện tại có thể chưa cài gymnasium. Fallback này
    # giữ API tối thiểu cần cho environment; khi gymnasium được cài, class chuẩn
    # phía trên sẽ tự động được sử dụng.
    class Env:  # type: ignore[no-redef]
        metadata: dict[str, Any] = {}

        def __init__(self) -> None:
            self.np_random = np.random.default_rng()

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ) -> None:
            if seed is not None:
                self.np_random = np.random.default_rng(seed)

    class Box:  # type: ignore[no-redef]
        def __init__(
            self,
            low: float,
            high: float,
            *,
            shape: tuple[int, ...],
            dtype: type[np.floating[Any]],
        ) -> None:
            self.shape = shape
            self.dtype = np.dtype(dtype)
            self.low = np.full(shape, low, dtype=self.dtype)
            self.high = np.full(shape, high, dtype=self.dtype)

        def contains(self, value: Any) -> bool:
            array = np.asarray(value)
            return (
                array.shape == self.shape
                and np.all(array >= self.low)
                and np.all(array <= self.high)
            )


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCENE_CONFIG = "scene_basic_drone_motor.jsonc"
MOTOR_CONFIG_PATH = PROJECT_ROOT / "config" / "motor_control.json"
DRONE_NAME = "Drone1"
ACTUATOR_ORDER = (
    "Prop_FL_actuator",
    "Prop_FR_actuator",
    "Prop_RL_actuator",
    "Prop_RR_actuator",
)
NANOSECONDS_PER_SECOND = 1_000_000_000


@dataclass(frozen=True)
class MotorCommand:
    normalized_action: np.ndarray
    rpm: np.ndarray
    control_signal: np.ndarray
    actuator_map: dict[str, float]


@dataclass(frozen=True)
class RewardConfig:
    """Trọng số reward tại tần số tham chiếu 50 Hz."""

    position_weight: float = 1.0
    position_scale: float = 0.5
    velocity_weight: float = 0.10
    attitude_weight: float = 0.50
    yaw_weight: float = 0.05
    angular_rate_weight: float = 0.05
    motor_effort_weight: float = 0.02
    motor_smoothness_weight: float = 0.10
    saturation_weight: float = 0.20
    alive_bonus: float = 0.01
    hover_bonus: float = 0.50
    crash_penalty: float = 10.0
    hover_position_tolerance_m: float = 0.20
    hover_velocity_tolerance_mps: float = 0.20
    hover_attitude_tolerance_rad: float = math.radians(10.0)
    saturation_margin: float = 0.02
    reference_dt: float = 0.02

    def validate(self) -> None:
        numeric_values = (
            self.position_weight,
            self.position_scale,
            self.velocity_weight,
            self.attitude_weight,
            self.yaw_weight,
            self.angular_rate_weight,
            self.motor_effort_weight,
            self.motor_smoothness_weight,
            self.saturation_weight,
            self.alive_bonus,
            self.hover_bonus,
            self.crash_penalty,
            self.hover_position_tolerance_m,
            self.hover_velocity_tolerance_mps,
            self.hover_attitude_tolerance_rad,
            self.saturation_margin,
            self.reference_dt,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in numeric_values):
            raise ValueError("RewardConfig chỉ chấp nhận trọng số hữu hạn, không âm")
        if self.position_scale <= 0.0 or self.reference_dt <= 0.0:
            raise ValueError("position_scale và reference_dt phải lớn hơn 0")
        if not 0.0 <= self.saturation_margin < 0.5:
            raise ValueError("saturation_margin phải thuộc [0, 0.5)")


def load_motor_calibration(path: Path = MOTOR_CONFIG_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy calibration motor: {path}. "
            "Hãy chạy module motor_residual.calibrate trước."
        )
    with path.open("r", encoding="utf-8") as config_file:
        calibration = json.load(config_file)

    required = {"max_rpm", "hover_rpm", "hover_control_signal", "actuator_order"}
    missing = required.difference(calibration)
    if missing:
        raise ValueError(f"Calibration thiếu trường: {sorted(missing)}")
    if tuple(calibration["actuator_order"]) != ACTUATOR_ORDER:
        raise ValueError("Thứ tự actuator trong calibration không khớp environment")
    if not calibration.get("verification", {}).get("passed", False):
        raise ValueError("Calibration chưa vượt qua bước verification")
    return calibration


class DroneMotorHoverEnv(Env):
    """Environment hover với action là tốc độ của bốn rotor độc lập."""

    metadata = {"render_modes": ["human"]}
    MAX_STEPS_PER_EPISODE = 500

    def __init__(
        self,
        *,
        sim_config_fname: str = SCENE_CONFIG,
        motor_config_path: Path = MOTOR_CONFIG_PATH,
        dt: float = 0.02,
        goal_pose: tuple[float, float, float] = (0.0, 0.0, -2.0),
        min_rpm: float = 0.0,
        rpm_delta: float | None = None,
        pose_timeout_seconds: float = 5.0,
        reward_config: RewardConfig | None = None,
        curriculum: MotorCurriculum | None = None,
        connect: bool = True,
    ) -> None:
        super().__init__()
        if dt <= 0.0:
            raise ValueError("dt phải lớn hơn 0")
        if pose_timeout_seconds <= 0.0:
            raise ValueError("pose_timeout_seconds phải lớn hơn 0")

        calibration = load_motor_calibration(Path(motor_config_path))
        self.sim_config_fname = sim_config_fname
        self.dt = float(dt)
        self.goal_pose = np.asarray(goal_pose, dtype=np.float64)
        self.min_rpm = float(min_rpm)
        self.max_rpm = float(calibration["max_rpm"])
        self.hover_rpm = float(calibration["hover_rpm"])
        self.hover_control_signal = float(calibration["hover_control_signal"])
        self.rpm_delta = float(
            rpm_delta
            if rpm_delta is not None
            else min(self.hover_rpm - self.min_rpm, self.max_rpm - self.hover_rpm)
        )
        self.pose_timeout_seconds = float(pose_timeout_seconds)
        self.reward_config = reward_config or RewardConfig()
        self.reward_config.validate()
        self.curriculum = curriculum
        self.episode_conditions: EpisodeConditions | None = None
        self.sensor_noise_std = 0.0
        self.hover_signal_bias = 0.0
        self.action_delay_steps = 0
        self._action_delay_queue: deque[np.ndarray] = deque()
        self.max_tilt_radians = math.radians(70.0)

        if not 0.0 <= self.min_rpm < self.hover_rpm < self.max_rpm:
            raise ValueError("Cần min_rpm < hover_rpm < max_rpm")
        if self.rpm_delta <= 0.0:
            raise ValueError("rpm_delta phải lớn hơn 0")
        if self.hover_rpm - self.rpm_delta < self.min_rpm:
            raise ValueError("rpm_delta làm RPM thấp hơn min_rpm")
        if self.hover_rpm + self.rpm_delta > self.max_rpm:
            raise ValueError("rpm_delta làm RPM cao hơn max_rpm")

        self.observation_shape = (17,)
        self.action_shape = (4,)
        self.observation_space = Box(
            -math.inf, math.inf, shape=self.observation_shape, dtype=np.float32
        )
        self.action_space = Box(
            -1.0, 1.0, shape=self.action_shape, dtype=np.float32
        )

        self.client: ProjectAirSimClient | None = None
        self.world: World | None = None
        self.actor: Drone | None = None
        self._connected = False
        self._pose_lock = threading.Lock()
        self._pose_event = threading.Event()
        self._subscribed_topics: list[str] = []
        self._closed = False
        self._init_episode_state()

        if connect:
            self.connect()

    def connect(self) -> None:
        if self._connected:
            return
        self.client = ProjectAirSimClient()
        self.client.connect()
        self.world = World(self.client, self.sim_config_fname, delay_after_load_sec=0)
        self._connected = True
        self._closed = False

    def _init_episode_state(self) -> None:
        self.done = False
        self.crashed = False
        self.ground_touch = False
        self.collision_info: dict[str, Any] | None = None
        self.actor_pose: list[float] | None = None
        self.observation: np.ndarray | None = None
        self.linear_velocity = np.zeros(3, dtype=np.float64)
        self.angular_velocity = np.zeros(3, dtype=np.float64)
        self.prev_motor_signal = np.zeros(4, dtype=np.float64)
        self.prev_motor_rpm = np.zeros(4, dtype=np.float64)
        self.prev_action = np.zeros(4, dtype=np.float64)
        self.prev_state: np.ndarray | None = None
        self._prev_pos: np.ndarray | None = None
        self._prev_rpy: np.ndarray | None = None
        self._prev_t_ns: int | None = None
        self._target_simtime_ns = 0
        self._step_count = 0
        self._hover_steps = 0
        self.termination_reason: str | None = None
        self.last_reward_terms: dict[str, float] = {}
        self._episode_position_errors_m: list[float] = []
        self._episode_attitude_errors_rad: list[float] = []
        self._curriculum_episode_recorded = False
        self._action_delay_queue.clear()
        self._pose_event.clear()

    def action_to_motor_command(self, action: np.ndarray | list[float]) -> MotorCommand:
        action_array = np.asarray(action, dtype=np.float64)
        if action_array.shape != self.action_shape:
            raise ValueError(
                f"Action phải có shape {self.action_shape}, nhận được {action_array.shape}"
            )
        if not np.all(np.isfinite(action_array)):
            raise ValueError("Action chứa NaN hoặc infinity")

        normalized_action = np.clip(action_array, -1.0, 1.0)
        rpm = np.clip(
            self.hover_rpm + normalized_action * self.rpm_delta,
            self.min_rpm,
            self.max_rpm,
        )
        control_signal = np.clip(
            np.square(rpm / self.max_rpm) + self.hover_signal_bias,
            0.0,
            1.0,
        )
        actuator_map = {
            actuator_id: float(control_signal[index])
            for index, actuator_id in enumerate(ACTUATOR_ORDER)
        }
        return MotorCommand(
            normalized_action=normalized_action.astype(np.float32),
            rpm=rpm.astype(np.float32),
            control_signal=control_signal.astype(np.float32),
            actuator_map=actuator_map,
        )

    def _require_runtime(self) -> tuple[ProjectAirSimClient, World, Drone]:
        if not self._connected or self.client is None or self.world is None:
            raise RuntimeError("Environment chưa kết nối ProjectAirSim")
        if self.actor is None:
            raise RuntimeError("Environment chưa reset")
        return self.client, self.world, self.actor

    def _stop_all_rotors(self, *, raise_on_failure: bool = True) -> bool:
        if self.actor is None:
            return True
        result = self.actor.set_control_signals(
            {actuator_id: 0.0 for actuator_id in ACTUATOR_ORDER}
        )
        if result is not True and raise_on_failure:
            raise RuntimeError("ProjectAirSim từ chối lệnh dừng rotor")
        return result is True

    def _unsubscribe_all(self) -> None:
        if self.client is not None and self._subscribed_topics:
            self.client.unsubscribe(self._subscribed_topics)
        self._subscribed_topics = []

    def _set_target_sim_time(self, target_time_ns: int) -> None:
        with self._pose_lock:
            self._target_simtime_ns = int(target_time_ns)
            self._pose_event.clear()

    def _wait_for_pose_sync_blocking(self) -> None:
        if not self._pose_event.wait(self.pose_timeout_seconds):
            raise TimeoutError(
                f"Không nhận được actual_pose trong {self.pose_timeout_seconds:.1f}s"
            )

    async def _wait_for_pose_sync_async(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.pose_timeout_seconds
        while not self._pose_event.is_set():
            if loop.time() >= deadline:
                raise TimeoutError(
                    f"Không nhận được actual_pose trong {self.pose_timeout_seconds:.1f}s"
                )
            await asyncio.sleep(0.001)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        reload_scene: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if not self._connected:
            self.connect()
        assert self.client is not None and self.world is not None

        self._stop_all_rotors(raise_on_failure=False)
        self._unsubscribe_all()
        self._init_episode_state()

        if self.curriculum is not None:
            self.episode_conditions = self.curriculum.sample_episode()
            conditions = self.episode_conditions
            self.goal_pose = conditions.goal_position_m.copy()
            self.linear_velocity = conditions.initial_velocity_mps.copy()
            self.sensor_noise_std = conditions.sensor_noise_std
            self.hover_signal_bias = conditions.hover_signal_bias
            self.action_delay_steps = conditions.action_delay_steps
            self._action_delay_queue.extend(
                np.zeros(4, dtype=np.float64)
                for _ in range(self.action_delay_steps)
            )
            # Mỗi episode curriculum cần scene mới để initial pose và các lực
            # còn lưu từ episode trước không bị rò rỉ sang episode hiện tại.
            reload_scene = True
        else:
            self.episode_conditions = None
            self.sensor_noise_std = 0.0
            self.hover_signal_bias = 0.0
            self.action_delay_steps = 0

        if reload_scene or self.actor is None:
            config_dict, _ = load_scene_config_as_dict(self.sim_config_fname)
            if self.episode_conditions is not None:
                self._apply_initial_pose_to_scene(config_dict, self.episode_conditions)
                self._apply_domain_randomization_to_scene(
                    config_dict, self.episode_conditions
                )
            self.world.load_scene(config_dict, delay_after_load_sec=0)

        self.actor = Drone(self.client, self.world, DRONE_NAME)
        if self.episode_conditions is not None:
            conditions = self.episode_conditions
            # Một impulse dài đúng một control period tạo initial velocity vật lý.
            nominal_mass_kg = 1.22
            impulse_force = (
                nominal_mass_kg
                * conditions.mass_scale
                * conditions.initial_velocity_mps
                / self.dt
            )
            initial_force = impulse_force + conditions.disturbance_force_n
            if np.any(initial_force):
                self.actor.set_external_force(initial_force.tolist())
        self.world.pause()
        pose_topic = self.actor.robot_info["actual_pose"]
        collision_topic = self.actor.robot_info["collision_info"]
        self.client.subscribe(pose_topic, self.pose_callback)
        self.client.subscribe(collision_topic, self.collision_callback)
        self._subscribed_topics = [pose_topic, collision_topic]

        # Reset kết thúc với rotor bằng 0; action đầu tiên của policy mới đưa
        # rotor tới operating point hover (action=zeros(4)).
        if not self._stop_all_rotors():
            raise RuntimeError("Không dừng được rotor khi reset")
        self.prev_motor_signal = np.zeros(4, dtype=np.float64)
        self.prev_motor_rpm = np.zeros(4, dtype=np.float64)

        initial_step_ns = max(3_000_000, int(round(self.dt * NANOSECONDS_PER_SECOND)))
        target_time = self.world.get_sim_time() + initial_step_ns
        self._set_target_sim_time(target_time)
        self.world.continue_for_sim_time(initial_step_ns, wait_until_complete=False)
        self._wait_for_pose_sync_blocking()
        if self.episode_conditions is not None:
            self.actor.set_external_force(
                self.episode_conditions.disturbance_force_n.tolist()
            )

        observation = self.get_observation()
        info = {
            "motor_rpm": self.prev_motor_rpm.copy(),
            "motor_signal": self.prev_motor_signal.copy(),
        }
        if self.episode_conditions is not None:
            info["curriculum"] = self._curriculum_info()
        return observation, info

    @staticmethod
    def _apply_initial_pose_to_scene(
        config_dict: dict[str, Any], conditions: EpisodeConditions
    ) -> None:
        actor = next(
            (
                item
                for item in config_dict.get("actors", [])
                if item.get("name") == DRONE_NAME
            ),
            None,
        )
        if actor is None:
            raise ValueError(f"Scene không có actor {DRONE_NAME}")
        position = conditions.initial_position_m
        rpy_deg = np.degrees(conditions.initial_rpy_rad)
        actor["origin"] = {
            "xyz": " ".join(f"{value:.9g}" for value in position),
            "rpy-deg": " ".join(f"{value:.9g}" for value in rpy_deg),
        }

    @staticmethod
    def _apply_domain_randomization_to_scene(
        config_dict: dict[str, Any], conditions: EpisodeConditions
    ) -> None:
        actor = next(
            (
                item
                for item in config_dict.get("actors", [])
                if item.get("name") == DRONE_NAME
            ),
            None,
        )
        if actor is None or not isinstance(actor.get("robot-config"), dict):
            raise ValueError(f"Scene không chứa robot-config đã nạp cho {DRONE_NAME}")
        robot_config = actor["robot-config"]
        for link in robot_config.get("links", []):
            inertial = link.get("inertial", {})
            if "mass" in inertial:
                inertial["mass"] = float(inertial["mass"]) * conditions.mass_scale
        for actuator in robot_config.get("actuators", []):
            if actuator.get("type") != "rotor":
                continue
            settings = actuator.get("rotor-settings", {})
            settings["coeff-of-thrust"] = (
                float(settings["coeff-of-thrust"]) * conditions.thrust_scale
            )
            settings["smoothing-tc"] = (
                float(settings["smoothing-tc"])
                * conditions.motor_time_constant_scale
            )

    def _curriculum_info(self) -> dict[str, Any]:
        assert self.episode_conditions is not None
        conditions = self.episode_conditions
        return {
            "stage_index": conditions.stage_index,
            "stage_name": conditions.stage_name,
            "goal_position_m": conditions.goal_position_m.copy(),
            "initial_position_m": conditions.initial_position_m.copy(),
            "initial_rpy_rad": conditions.initial_rpy_rad.copy(),
            "initial_velocity_mps": conditions.initial_velocity_mps.copy(),
            "disturbance_force_n": conditions.disturbance_force_n.copy(),
            "mass_scale": conditions.mass_scale,
            "thrust_scale": conditions.thrust_scale,
            "motor_time_constant_scale": conditions.motor_time_constant_scale,
            "sensor_noise_std": conditions.sensor_noise_std,
            "action_delay_steps": conditions.action_delay_steps,
            "hover_signal_bias": conditions.hover_signal_bias,
        }

    async def step(
        self, action: np.ndarray | list[float]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        _, world, actor = self._require_runtime()
        if self.done:
            raise RuntimeError("Episode đã kết thúc; hãy gọi reset()")

        requested_command = self.action_to_motor_command(action)
        requested_action = requested_command.normalized_action.astype(np.float64)
        if self.action_delay_steps:
            self._action_delay_queue.append(requested_action.copy())
            applied_action = self._action_delay_queue.popleft()
        else:
            applied_action = requested_action
        command = self.action_to_motor_command(applied_action)
        previous_motor_signal = self.prev_motor_signal.copy()
        try:
            if actor.set_control_signals(command.actuator_map) is not True:
                raise RuntimeError("ProjectAirSim từ chối motor command")

            step_ns = int(round(self.dt * NANOSECONDS_PER_SECOND))
            target_time = world.get_sim_time() + step_ns
            self._set_target_sim_time(target_time)
            world.continue_for_sim_time(step_ns, wait_until_complete=False)
            await self._wait_for_pose_sync_async()

            self.prev_action = command.normalized_action.astype(np.float64)
            self.prev_motor_rpm = command.rpm.astype(np.float64)
            self.prev_motor_signal = command.control_signal.astype(np.float64)
            observation = self.get_observation()
            self._check_termination()
            reward, reward_terms = self.calculate_reward(
                command.control_signal,
                previous_motor_signal,
            )
            self.last_reward_terms = reward_terms

            self._step_count += 1
            terminated = bool(self.done)
            truncated = self._step_count >= self.MAX_STEPS_PER_EPISODE
            pose = np.asarray(self.actor_pose, dtype=np.float64)
            self._episode_position_errors_m.append(
                float(np.linalg.norm(pose[:3] - self.goal_pose))
            )
            self._episode_attitude_errors_rad.append(
                float(np.linalg.norm(pose[3:6]))
            )
            if terminated or truncated:
                self._stop_all_rotors(raise_on_failure=False)

            self.prev_state = observation.copy()
            info = {
                "status": "Terminated" if terminated else "Running",
                "motor_rpm": command.rpm.copy(),
                "motor_signal": command.control_signal.copy(),
                "normalized_action": command.normalized_action.copy(),
                "requested_action": requested_action.astype(np.float32),
                "reward_terms": reward_terms.copy(),
                "termination_reason": self.termination_reason,
            }
            if self.episode_conditions is not None:
                info["curriculum"] = self._curriculum_info()
                if terminated or truncated:
                    info["curriculum_result"] = self._record_curriculum_episode(
                        survived=truncated and not self.crashed
                    )
            return observation, reward, terminated, truncated, info
        except Exception:
            self._stop_all_rotors(raise_on_failure=False)
            raise

    def _record_curriculum_episode(self, *, survived: bool) -> dict[str, Any]:
        if self.curriculum is None or self._curriculum_episode_recorded:
            return {"recorded": False, "stage_advanced": False}
        metrics = self.curriculum.record_episode(
            survived=survived,
            position_errors_m=np.asarray(self._episode_position_errors_m),
            attitude_errors_rad=np.asarray(self._episode_attitude_errors_rad),
        )
        stage_advanced = self.curriculum.advance_if_ready()
        self._curriculum_episode_recorded = True
        return {
            "recorded": True,
            "survived": survived,
            "episode_count": metrics.episode_count,
            "survival_rate": metrics.survival_rate,
            "position_rmse_m": metrics.position_rmse_m,
            "attitude_rmse_rad": metrics.attitude_rmse_rad,
            "stage_advanced": stage_advanced,
            "next_stage_index": self.curriculum.stage_index,
            "next_stage_name": self.curriculum.current_stage.name,
        }

    def pose_callback(self, _topic: str, pose: dict[str, Any]) -> None:
        sim_time_ns = pose.get("time_stamp")
        position = pose.get("translation") or pose.get("position", {})
        rotation = pose.get("rotation") or pose.get("orientation", {})
        pos = np.array(
            [
                float(position.get("x", 0.0)),
                float(position.get("y", 0.0)),
                float(position.get("z", 0.0)),
            ],
            dtype=np.float64,
        )
        rpy = np.array(
            quaternion_to_rpy(
                float(rotation.get("w", 1.0)),
                float(rotation.get("x", 0.0)),
                float(rotation.get("y", 0.0)),
                float(rotation.get("z", 0.0)),
            ),
            dtype=np.float64,
        )

        with self._pose_lock:
            if self._prev_t_ns is not None and sim_time_ns is not None:
                delta_time = (int(sim_time_ns) - self._prev_t_ns) / NANOSECONDS_PER_SECOND
                if delta_time > 1e-9 and self._prev_pos is not None and self._prev_rpy is not None:
                    self.linear_velocity = (pos - self._prev_pos) / delta_time
                    angular_delta = rpy - self._prev_rpy
                    angular_delta = np.arctan2(np.sin(angular_delta), np.cos(angular_delta))
                    self.angular_velocity = angular_delta / delta_time

            self.actor_pose = [*pos.tolist(), *rpy.tolist()]
            self._prev_pos = pos
            self._prev_rpy = rpy
            if sim_time_ns is not None:
                self._prev_t_ns = int(sim_time_ns)
                if self._prev_t_ns >= self._target_simtime_ns:
                    self._pose_event.set()

    def collision_callback(self, _topic: str, collision_info: dict[str, Any]) -> None:
        self.collision_info = collision_info
        self.crashed = True
        self.done = True
        self.termination_reason = "collision"

    def get_observation(self) -> np.ndarray:
        with self._pose_lock:
            if self.actor_pose is None:
                raise RuntimeError("Chưa nhận được actual_pose")
            pose = np.asarray(self.actor_pose, dtype=np.float64)
            linear_velocity = self.linear_velocity.copy()
            angular_velocity = self.angular_velocity.copy()

        distance_to_goal = float(np.linalg.norm(pose[:3] - self.goal_pose))
        observation = np.concatenate(
            (
                pose,
                linear_velocity,
                angular_velocity,
                np.array([distance_to_goal], dtype=np.float64),
                self.prev_motor_signal,
            )
        ).astype(np.float32)
        if self.sensor_noise_std > 0.0:
            observation[:13] += self.np_random.normal(
                0.0, self.sensor_noise_std, size=13
            ).astype(np.float32)
        if observation.shape != self.observation_shape:
            raise RuntimeError(f"Observation sai shape: {observation.shape}")
        self.observation = observation
        return observation.copy()

    def _check_termination(self) -> None:
        if self.actor_pose is None:
            return
        if self.done and self.termination_reason == "collision":
            return
        pose = np.asarray(self.actor_pose, dtype=np.float64)
        position = pose[:3]
        roll, pitch = pose[3], pose[4]
        velocity_norm = float(np.linalg.norm(self.linear_velocity))
        position_error = float(np.linalg.norm(position - self.goal_pose))

        if abs(roll) > self.max_tilt_radians or abs(pitch) > self.max_tilt_radians:
            self.crashed = True
            self.done = True
            self.termination_reason = "attitude_limit"
        elif position[2] > -0.2:
            self.ground_touch = True
            self.crashed = True
            self.done = True
            self.termination_reason = "ground_touch"
        elif position_error > 5.0:
            self.crashed = True
            self.done = True
            self.termination_reason = "position_limit"
        elif velocity_norm > 5.0:
            self.crashed = True
            self.done = True
            self.termination_reason = "velocity_limit"

    def calculate_reward(
        self,
        current_motor_signal: np.ndarray,
        previous_motor_signal: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        """Tính reward và trả về từng thành phần đã áp dụng trọng số."""
        if self.actor_pose is None:
            return 0.0, {"total": 0.0}
        current_signal = np.asarray(current_motor_signal, dtype=np.float64)
        previous_signal = np.asarray(previous_motor_signal, dtype=np.float64)
        if current_signal.shape != (4,) or previous_signal.shape != (4,):
            raise ValueError("Motor signal dùng tính reward phải có shape (4,)")
        if not np.all(np.isfinite(current_signal)) or not np.all(
            np.isfinite(previous_signal)
        ):
            raise ValueError("Motor signal dùng tính reward phải hữu hạn")

        config = self.reward_config
        time_scale = self.dt / config.reference_dt
        pose = np.asarray(self.actor_pose, dtype=np.float64)
        position_error = float(np.linalg.norm(pose[:3] - self.goal_pose))
        velocity_norm = float(np.linalg.norm(self.linear_velocity))
        angular_rate_norm = float(np.linalg.norm(self.angular_velocity))
        roll, pitch, yaw = float(pose[3]), float(pose[4]), float(pose[5])
        yaw_error = math.atan2(math.sin(yaw), math.cos(yaw))
        hover_signal = np.full(4, self.hover_control_signal, dtype=np.float64)
        saturation_fraction = float(
            np.mean(
                (current_signal <= config.saturation_margin)
                | (current_signal >= 1.0 - config.saturation_margin)
            )
        )

        terms = {
            "position": time_scale
            * config.position_weight
            * math.exp(-config.position_scale * position_error**2),
            "alive": time_scale * config.alive_bonus,
            "velocity": -time_scale * config.velocity_weight * velocity_norm**2,
            "attitude": -time_scale
            * config.attitude_weight
            * (roll**2 + pitch**2),
            "yaw": -time_scale * config.yaw_weight * yaw_error**2,
            "angular_rate": -time_scale
            * config.angular_rate_weight
            * angular_rate_norm**2,
            "motor_effort": -time_scale
            * config.motor_effort_weight
            * float(np.mean(np.square(current_signal - hover_signal))),
            "motor_smoothness": -time_scale
            * config.motor_smoothness_weight
            * float(np.mean(np.square(current_signal - previous_signal))),
            "saturation": -time_scale
            * config.saturation_weight
            * saturation_fraction,
            "hover_bonus": 0.0,
            "crash": -config.crash_penalty if self.crashed else 0.0,
        }

        attitude_norm = math.sqrt(roll**2 + pitch**2)
        if (
            position_error < config.hover_position_tolerance_m
            and velocity_norm < config.hover_velocity_tolerance_mps
            and attitude_norm < config.hover_attitude_tolerance_rad
        ):
            terms["hover_bonus"] = time_scale * config.hover_bonus
            self._hover_steps += 1
        else:
            self._hover_steps = 0

        total = float(sum(terms.values()))
        terms["total"] = total
        terms["position_error_m"] = position_error
        terms["velocity_norm_mps"] = velocity_norm
        terms["angular_rate_norm_radps"] = angular_rate_norm
        terms["saturation_fraction"] = saturation_fraction
        return total, {name: float(value) for name, value in terms.items()}

    def get_reward(self) -> float:
        """API tương thích: reward theo motor command hiện tại, không có delta mới."""
        reward, terms = self.calculate_reward(
            self.prev_motor_signal,
            self.prev_motor_signal,
        )
        self.last_reward_terms = terms
        return reward

    def render(self) -> None:
        if self.observation is not None:
            print(f"Obs: {self.observation}")

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._stop_all_rotors(raise_on_failure=False)
            self._unsubscribe_all()
        finally:
            if self.client is not None and self._connected:
                self.client.disconnect()
            self._connected = False
            self._closed = True
            self.actor = None
            self.world = None
            self.client = None

