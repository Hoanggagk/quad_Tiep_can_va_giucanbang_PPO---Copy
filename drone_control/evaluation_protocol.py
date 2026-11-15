"""Frozen evaluation protocol and simulator-independent metric aggregation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from drone_control.task_contract import DEFAULT_TASK_CONTRACT, TaskContract


EVALUATION_PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class EvaluationScenario:
    scenario_id: str
    description: str
    sustained_seconds: float
    disturbance_group: str | None = None


FROZEN_SCENARIOS = (
    EvaluationScenario("A_hover", "Hover gần cân bằng", 5.0),
    EvaluationScenario("B_reaching", "Tiếp cận từ reset ngoài goal", 0.0),
    EvaluationScenario("C_reach_hover", "Tiếp cận rồi hover liên tục", 5.0),
    EvaluationScenario("C_reach_hover_stress", "Stress hover sau tiếp cận", 10.0),
    EvaluationScenario("D_mass", "Sai lệch mass", 5.0, "mass"),
    EvaluationScenario("D_thrust", "Sai lệch thrust", 5.0, "thrust"),
    EvaluationScenario("D_motor_lag", "Sai lệch motor lag", 5.0, "motor_lag"),
    EvaluationScenario("D_sensor", "Sensor noise", 5.0, "sensor_noise"),
    EvaluationScenario("D_delay", "Action delay", 5.0, "action_delay"),
    EvaluationScenario("D_wind", "Wind/gust", 5.0, "wind"),
)


@dataclass(frozen=True)
class EpisodeMetrics:
    scenario_id: str
    seed: int
    success: bool
    reached_once: bool
    crashed: bool
    timed_out: bool
    time_to_target_seconds: float | None
    overshoot_m: float
    path_efficiency: float
    position_rmse_m: float
    velocity_rmse_mps: float
    attitude_rmse_rad: float
    angular_rate_rmse_radps: float
    control_effort: float
    smoothness: float
    saturation_fraction: float
    action_near_bound_fraction: float
    recovery_time_seconds: float | None = None
    peak_deviation_m: float | None = None

    def validate(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id không được rỗng")
        nonnegative = (
            self.overshoot_m,
            self.path_efficiency,
            self.position_rmse_m,
            self.velocity_rmse_mps,
            self.attitude_rmse_rad,
            self.angular_rate_rmse_radps,
            self.control_effort,
            self.smoothness,
            self.saturation_fraction,
            self.action_near_bound_fraction,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in nonnegative):
            raise ValueError("Evaluation metrics phải hữu hạn và không âm")
        for fraction in (self.saturation_fraction, self.action_near_bound_fraction):
            if fraction > 1.0:
                raise ValueError("Metric dạng fraction phải thuộc [0, 1]")


def aggregate_metrics(records: Iterable[EpisodeMetrics]) -> dict[str, Any]:
    rows = list(records)
    if not rows:
        raise ValueError("Cần ít nhất một episode để aggregate")
    for row in rows:
        row.validate()

    def mean(field: str) -> float:
        return float(sum(float(getattr(row, field)) for row in rows) / len(rows))

    finite_times = [
        row.time_to_target_seconds
        for row in rows
        if row.time_to_target_seconds is not None
    ]
    return {
        "episode_count": len(rows),
        "seed_count": len({row.seed for row in rows}),
        "success_rate": sum(row.success for row in rows) / len(rows),
        "reach_rate": sum(row.reached_once for row in rows) / len(rows),
        "crash_rate": sum(row.crashed for row in rows) / len(rows),
        "timeout_rate": sum(row.timed_out for row in rows) / len(rows),
        "mean_time_to_target_seconds": (
            float(sum(finite_times) / len(finite_times)) if finite_times else None
        ),
        "mean_overshoot_m": mean("overshoot_m"),
        "mean_path_efficiency": mean("path_efficiency"),
        "position_rmse_m": mean("position_rmse_m"),
        "velocity_rmse_mps": mean("velocity_rmse_mps"),
        "attitude_rmse_rad": mean("attitude_rmse_rad"),
        "angular_rate_rmse_radps": mean("angular_rate_rmse_radps"),
        "mean_control_effort": mean("control_effort"),
        "mean_smoothness": mean("smoothness"),
        "saturation_fraction": mean("saturation_fraction"),
        "action_near_bound_fraction": mean("action_near_bound_fraction"),
    }


def build_evaluation_manifest(
    *,
    action_space: str,
    checkpoint: str,
    seeds: Iterable[int],
    task_contract: TaskContract = DEFAULT_TASK_CONTRACT,
) -> dict[str, Any]:
    task_contract.validate()
    seed_list = [int(seed) for seed in seeds]
    if action_space not in {"velocity_ned", "motor_residual"}:
        raise ValueError("action_space không được hỗ trợ")
    if not seed_list or len(seed_list) != len(set(seed_list)):
        raise ValueError("seeds phải khác nhau và không được rỗng")
    scenario_data = [asdict(scenario) for scenario in FROZEN_SCENARIOS]
    fingerprint_payload = json.dumps(
        {
            "version": EVALUATION_PROTOCOL_VERSION,
            "task_contract": task_contract.to_metadata(),
            "scenarios": scenario_data,
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return {
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "protocol_sha256": hashlib.sha256(fingerprint_payload).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "action_space": action_space,
        "checkpoint": str(Path(checkpoint)),
        "seeds": seed_list,
        "task_contract": task_contract.to_metadata(),
        "scenarios": scenario_data,
        "status": "planned",
        "results": [],
    }
