"""Versioned task definition and success tracking shared by both action spaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


@dataclass(frozen=True)
class TaskContract:
    version: int = 1
    coordinate_frame: str = "NED"
    goal_position_m: tuple[float, float, float] = (0.0, 0.0, -2.0)
    yaw_target_rad: float = 0.0
    require_yaw: bool = True
    reach_position_tolerance_m: float = 0.20
    hover_velocity_tolerance_mps: float = 0.20
    hover_attitude_tolerance_rad: float = math.radians(10.0)
    hover_yaw_tolerance_rad: float = math.radians(15.0)
    training_sustained_seconds: float = 2.0
    evaluation_sustained_seconds: float = 5.0
    stress_sustained_seconds: float = 10.0
    max_position_error_m: float = 5.0
    max_velocity_mps: float = 5.0
    velocity_episode_seconds: float = 15.0
    motor_episode_seconds: float = 10.0

    def validate(self) -> None:
        if self.version < 1:
            raise ValueError("Task contract version phải lớn hơn hoặc bằng 1")
        if self.coordinate_frame != "NED":
            raise ValueError("Checkout hiện tại chỉ hỗ trợ coordinate_frame='NED'")
        if len(self.goal_position_m) != 3:
            raise ValueError("goal_position_m phải có ba phần tử")
        numeric = (
            *self.goal_position_m,
            self.yaw_target_rad,
            self.reach_position_tolerance_m,
            self.hover_velocity_tolerance_mps,
            self.hover_attitude_tolerance_rad,
            self.hover_yaw_tolerance_rad,
            self.training_sustained_seconds,
            self.evaluation_sustained_seconds,
            self.stress_sustained_seconds,
            self.max_position_error_m,
            self.max_velocity_mps,
            self.velocity_episode_seconds,
            self.motor_episode_seconds,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("Task contract chỉ chấp nhận giá trị hữu hạn")
        positive = numeric[4:]
        if not all(value > 0.0 for value in positive):
            raise ValueError("Các tolerance/duration/limit phải lớn hơn 0")
        if self.training_sustained_seconds > self.motor_episode_seconds:
            raise ValueError("Training sustain không được dài hơn motor episode")
        if self.evaluation_sustained_seconds > self.velocity_episode_seconds:
            raise ValueError("Evaluation sustain không được dài hơn velocity episode")

    def to_metadata(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


DEFAULT_TASK_CONTRACT = TaskContract()
DEFAULT_TASK_CONTRACT.validate()


@dataclass(frozen=True)
class SuccessStatus:
    reached: bool
    reached_once: bool
    sustaining: bool
    success: bool
    success_event: bool
    time_to_target_seconds: float | None
    sustained_seconds: float

    def to_info(self) -> dict[str, Any]:
        return asdict(self)


class SuccessTracker:
    """Tracks reaching and continuous hover independently from reward."""

    def __init__(
        self,
        contract: TaskContract,
        *,
        dt: float,
        required_sustained_seconds: float | None = None,
    ) -> None:
        contract.validate()
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt phải lớn hơn 0")
        required = (
            contract.training_sustained_seconds
            if required_sustained_seconds is None
            else required_sustained_seconds
        )
        if not math.isfinite(required) or required <= 0.0:
            raise ValueError("required_sustained_seconds phải lớn hơn 0")
        self.contract = contract
        self.dt = float(dt)
        self.required_steps = max(1, math.ceil(required / self.dt))
        self.reset()

    def reset(self) -> None:
        self.elapsed_steps = 0
        self.sustained_steps = 0
        self.reached_once = False
        self.time_to_target_seconds: float | None = None
        self.success = False

    def update(
        self,
        *,
        position_error_m: float,
        velocity_norm_mps: float,
        attitude_norm_rad: float,
        yaw_error_rad: float,
    ) -> SuccessStatus:
        values = (
            position_error_m,
            velocity_norm_mps,
            attitude_norm_rad,
            yaw_error_rad,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("Success metrics phải hữu hạn và không âm")

        self.elapsed_steps += 1
        reached = position_error_m < self.contract.reach_position_tolerance_m
        if reached and not self.reached_once:
            self.reached_once = True
            self.time_to_target_seconds = self.elapsed_steps * self.dt

        yaw_ok = (
            not self.contract.require_yaw
            or yaw_error_rad < self.contract.hover_yaw_tolerance_rad
        )
        sustaining = (
            reached
            and velocity_norm_mps < self.contract.hover_velocity_tolerance_mps
            and attitude_norm_rad < self.contract.hover_attitude_tolerance_rad
            and yaw_ok
        )
        self.sustained_steps = self.sustained_steps + 1 if sustaining else 0
        success_event = not self.success and self.sustained_steps >= self.required_steps
        if success_event:
            self.success = True

        return SuccessStatus(
            reached=reached,
            reached_once=self.reached_once,
            sustaining=sustaining,
            success=self.success,
            success_event=success_event,
            time_to_target_seconds=self.time_to_target_seconds,
            sustained_seconds=self.sustained_steps * self.dt,
        )
