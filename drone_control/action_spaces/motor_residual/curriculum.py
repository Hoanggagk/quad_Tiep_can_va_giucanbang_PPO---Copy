"""Curriculum cho bài toán hover điều khiển trực tiếp bốn motor.

Module này không chạy simulator hay training. Nó chỉ sinh điều kiện episode và
quyết định chuyển stage từ các metric độc lập với tổng reward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    position_range_m: float = 0.0
    tilt_range_rad: float = 0.0
    velocity_range_mps: float = 0.0
    target_z_range_m: tuple[float, float] = (-2.0, -2.0)
    initial_z_range_m: tuple[float, float] = (-2.0, -2.0)
    disturbance_force_range_n: float = 0.0
    mass_scale_range: tuple[float, float] = (1.0, 1.0)
    thrust_scale_range: tuple[float, float] = (1.0, 1.0)
    motor_time_constant_scale_range: tuple[float, float] = (1.0, 1.0)
    sensor_noise_std: float = 0.0
    action_delay_steps: int = 0
    hover_signal_bias_range: float = 0.0
    min_survival_rate: float = 0.90
    max_position_rmse_m: float = 0.30
    max_attitude_rmse_rad: float = math.radians(12.0)
    min_evaluation_episodes: int = 20


@dataclass(frozen=True)
class EpisodeConditions:
    stage_index: int
    stage_name: str
    initial_position_m: np.ndarray
    initial_rpy_rad: np.ndarray
    initial_velocity_mps: np.ndarray
    goal_position_m: np.ndarray
    disturbance_force_n: np.ndarray
    mass_scale: float
    thrust_scale: float
    motor_time_constant_scale: float
    sensor_noise_std: float
    action_delay_steps: int
    hover_signal_bias: float


@dataclass(frozen=True)
class CurriculumMetrics:
    episode_count: int
    survival_rate: float
    position_rmse_m: float
    attitude_rmse_rad: float


@dataclass
class CurriculumWindow:
    survived: list[bool] = field(default_factory=list)
    squared_position_errors: list[float] = field(default_factory=list)
    squared_attitude_errors: list[float] = field(default_factory=list)

    def add_episode(
        self,
        *,
        survived: bool,
        position_errors_m: np.ndarray,
        attitude_errors_rad: np.ndarray,
    ) -> None:
        position = np.asarray(position_errors_m, dtype=np.float64)
        attitude = np.asarray(attitude_errors_rad, dtype=np.float64)
        if position.size == 0 or attitude.size == 0:
            raise ValueError("Metric episode không được rỗng")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(attitude)):
            raise ValueError("Metric episode phải hữu hạn")
        self.survived.append(bool(survived))
        self.squared_position_errors.append(float(np.mean(np.square(position))))
        self.squared_attitude_errors.append(float(np.mean(np.square(attitude))))

    def summarize(self) -> CurriculumMetrics:
        if not self.survived:
            return CurriculumMetrics(0, 0.0, math.inf, math.inf)
        return CurriculumMetrics(
            episode_count=len(self.survived),
            survival_rate=float(np.mean(self.survived)),
            position_rmse_m=math.sqrt(float(np.mean(self.squared_position_errors))),
            attitude_rmse_rad=math.sqrt(float(np.mean(self.squared_attitude_errors))),
        )

    def clear(self) -> None:
        self.survived.clear()
        self.squared_position_errors.clear()
        self.squared_attitude_errors.clear()


def default_motor_curriculum() -> tuple[CurriculumStage, ...]:
    """Bảy stage theo đúng thứ tự tăng độ khó của kế hoạch triển khai."""
    return (
        CurriculumStage(name="attitude_hover"),
        CurriculumStage(name="random_attitude", tilt_range_rad=math.radians(8.0)),
        CurriculumStage(
            name="random_position_velocity",
            position_range_m=0.25,
            tilt_range_rad=math.radians(8.0),
            velocity_range_mps=0.25,
        ),
        CurriculumStage(
            name="altitude_change",
            position_range_m=0.30,
            tilt_range_rad=math.radians(10.0),
            velocity_range_mps=0.30,
            target_z_range_m=(-3.0, -1.0),
        ),
        CurriculumStage(
            name="takeoff",
            position_range_m=0.20,
            tilt_range_rad=math.radians(5.0),
            initial_z_range_m=(-0.70, -0.35),
            target_z_range_m=(-2.5, -1.5),
        ),
        CurriculumStage(
            name="disturbance",
            position_range_m=0.35,
            tilt_range_rad=math.radians(12.0),
            velocity_range_mps=0.40,
            target_z_range_m=(-3.0, -1.0),
            disturbance_force_range_n=2.0,
        ),
        CurriculumStage(
            name="domain_randomization",
            position_range_m=0.40,
            tilt_range_rad=math.radians(15.0),
            velocity_range_mps=0.50,
            target_z_range_m=(-3.0, -1.0),
            disturbance_force_range_n=2.5,
            mass_scale_range=(0.90, 1.10),
            thrust_scale_range=(0.90, 1.10),
            motor_time_constant_scale_range=(0.90, 1.10),
            sensor_noise_std=0.01,
            action_delay_steps=2,
            hover_signal_bias_range=0.03,
        ),
    )


class MotorCurriculum:
    def __init__(
        self,
        stages: tuple[CurriculumStage, ...] | None = None,
        *,
        initial_stage: int = 0,
        seed: int | None = None,
    ) -> None:
        self.stages = stages or default_motor_curriculum()
        if not self.stages:
            raise ValueError("Curriculum phải có ít nhất một stage")
        if not 0 <= initial_stage < len(self.stages):
            raise ValueError("initial_stage nằm ngoài curriculum")
        self.stage_index = initial_stage
        self.rng = np.random.default_rng(seed)
        self.window = CurriculumWindow()

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self.stage_index]

    def sample_episode(self) -> EpisodeConditions:
        stage = self.current_stage
        xy = self.rng.uniform(-stage.position_range_m, stage.position_range_m, 2)
        initial_z = self.rng.uniform(*stage.initial_z_range_m)
        roll_pitch = self.rng.uniform(-stage.tilt_range_rad, stage.tilt_range_rad, 2)
        velocity = self.rng.uniform(-stage.velocity_range_mps, stage.velocity_range_mps, 3)
        target_z = self.rng.uniform(*stage.target_z_range_m)
        force = self.rng.uniform(
            -stage.disturbance_force_range_n, stage.disturbance_force_range_n, 3
        )
        return EpisodeConditions(
            stage_index=self.stage_index,
            stage_name=stage.name,
            initial_position_m=np.array([xy[0], xy[1], initial_z]),
            initial_rpy_rad=np.array([roll_pitch[0], roll_pitch[1], 0.0]),
            initial_velocity_mps=velocity,
            goal_position_m=np.array([0.0, 0.0, target_z]),
            disturbance_force_n=force,
            mass_scale=float(self.rng.uniform(*stage.mass_scale_range)),
            thrust_scale=float(self.rng.uniform(*stage.thrust_scale_range)),
            motor_time_constant_scale=float(
                self.rng.uniform(*stage.motor_time_constant_scale_range)
            ),
            sensor_noise_std=stage.sensor_noise_std,
            action_delay_steps=stage.action_delay_steps,
            hover_signal_bias=float(
                self.rng.uniform(
                    -stage.hover_signal_bias_range, stage.hover_signal_bias_range
                )
            ),
        )

    def record_episode(self, **kwargs: object) -> CurriculumMetrics:
        self.window.add_episode(**kwargs)  # type: ignore[arg-type]
        return self.window.summarize()

    def can_advance(self) -> bool:
        metrics = self.window.summarize()
        stage = self.current_stage
        return (
            self.stage_index < len(self.stages) - 1
            and metrics.episode_count >= stage.min_evaluation_episodes
            and metrics.survival_rate >= stage.min_survival_rate
            and metrics.position_rmse_m <= stage.max_position_rmse_m
            and metrics.attitude_rmse_rad <= stage.max_attitude_rmse_rad
        )

    def advance_if_ready(self) -> bool:
        if not self.can_advance():
            return False
        self.stage_index += 1
        self.window.clear()
        return True

