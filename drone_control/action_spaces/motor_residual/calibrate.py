"""Hiệu chuẩn control signal hover cho quadrotor manual-controller.

Script dùng binary search trên cùng một control signal cho bốn rotor. Mỗi phép
đo reload scene ở độ cao an toàn, chạy theo sim time và tính vận tốc NED trục z.
Mặc định chỉ dry-run; truyền ``--execute`` mới kết nối ProjectAirSim.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from projectairsim import Drone, ProjectAirSimClient, World
from projectairsim.utils import load_scene_config_as_dict

from drone_control.action_spaces.motor_residual.check_actuators import (
    ACTUATOR_ORDER,
    DRONE_NAME,
    SCENE_CONFIG,
    all_stop_command,
    require_success,
    stop_all_rotors,
    validate_motor_scene,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "config" / "motor_control.json"
NANOSECONDS_PER_SECOND = 1_000_000_000


@dataclass(frozen=True)
class CalibrationSettings:
    lower_signal: float = 0.40
    upper_signal: float = 0.95
    iterations: int = 9
    warmup_seconds: float = 0.10
    measure_seconds: float = 0.25
    verification_trials: int = 3
    velocity_tolerance_mps: float = 0.08
    start_altitude_m: float = 2.0

    def validate(self) -> None:
        if not 0.0 <= self.lower_signal < self.upper_signal <= 1.0:
            raise ValueError("Khoảng signal phải thỏa 0 <= lower < upper <= 1")
        if self.iterations < 1:
            raise ValueError("iterations phải lớn hơn hoặc bằng 1")
        if self.warmup_seconds < 0.0:
            raise ValueError("warmup_seconds không được âm")
        if self.measure_seconds <= 0.0:
            raise ValueError("measure_seconds phải lớn hơn 0")
        if self.verification_trials < 1:
            raise ValueError("verification_trials phải lớn hơn hoặc bằng 1")
        if self.velocity_tolerance_mps <= 0.0:
            raise ValueError("velocity_tolerance_mps phải lớn hơn 0")
        if self.start_altitude_m < 1.0:
            raise ValueError("start_altitude_m phải ít nhất 1 m")


@dataclass(frozen=True)
class TrialResult:
    control_signal: float
    start_z_m: float
    end_z_m: float
    vertical_velocity_mps: float


class PoseTracker:
    """Theo dõi actual_pose và đợi pose đạt một sim timestamp cụ thể."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._target_time_ns = 0
        self._latest_pose: dict[str, Any] | None = None

    def callback(self, _topic: str, pose: dict[str, Any]) -> None:
        with self._lock:
            self._latest_pose = pose
            timestamp = pose.get("time_stamp")
            if timestamp is not None and int(timestamp) >= self._target_time_ns:
                self._event.set()

    def wait_for_time(self, target_time_ns: int, timeout_seconds: float = 5.0) -> dict[str, Any]:
        with self._lock:
            self._target_time_ns = int(target_time_ns)
            self._event.clear()
            latest = self._latest_pose
            if latest is not None:
                timestamp = latest.get("time_stamp")
                if timestamp is not None and int(timestamp) >= self._target_time_ns:
                    self._event.set()

        if not self._event.wait(timeout_seconds):
            raise TimeoutError(
                f"Không nhận được actual_pose tại sim time {target_time_ns}"
            )
        with self._lock:
            if self._latest_pose is None:
                raise RuntimeError("Pose event được set nhưng không có pose")
            return copy.deepcopy(self._latest_pose)


def equal_motor_command(control_signal: float) -> dict[str, float]:
    if not 0.0 <= control_signal <= 1.0:
        raise ValueError("control_signal phải thuộc [0, 1]")
    return {actuator_id: float(control_signal) for actuator_id in ACTUATOR_ORDER}


def extract_z(pose: dict[str, Any]) -> float:
    position = pose.get("translation") or pose.get("position")
    if position is None or "z" not in position:
        raise KeyError("Pose không chứa translation.z hoặc position.z")
    return float(position["z"])


def seconds_to_nanos(seconds: float) -> int:
    return int(round(seconds * NANOSECONDS_PER_SECOND))


def get_robot_metadata(scene_config: str = SCENE_CONFIG) -> dict[str, Any]:
    config, _ = load_scene_config_as_dict(scene_config)
    robot = config["actors"][0]["robot-config"]
    rotors = [
        actuator for actuator in robot["actuators"] if actuator["type"] == "rotor"
    ]
    max_rpms = {
        float(actuator["rotor-settings"]["max-rpm"]) for actuator in rotors
    }
    if len(rotors) != 4 or len(max_rpms) != 1:
        raise ValueError("Cần đúng bốn rotor có cùng max-rpm để hiệu chuẩn")
    return {
        "max_rpm": max_rpms.pop(),
        "actuator_order": list(ACTUATOR_ORDER),
    }


def make_trial_scene(
    base_scene: dict[str, Any], start_altitude_m: float
) -> dict[str, Any]:
    scene = copy.deepcopy(base_scene)
    actor = next(actor for actor in scene["actors"] if actor["name"] == DRONE_NAME)
    actor["origin"]["xyz"] = f"0 0 {-start_altitude_m}"
    return scene


def run_trial(
    client: ProjectAirSimClient,
    world: World,
    trial_scene: dict[str, Any],
    control_signal: float,
    settings: CalibrationSettings,
) -> TrialResult:
    """Đo vận tốc z của một signal; luôn dừng rotor trước khi trả về."""
    world.load_scene(copy.deepcopy(trial_scene), delay_after_load_sec=0)
    drone = Drone(client, world, DRONE_NAME)
    world.pause()
    pose_tracker = PoseTracker()
    pose_topic = drone.robot_info["actual_pose"]
    client.subscribe(pose_topic, pose_tracker.callback)

    stop_all_rotors(drone)
    try:
        require_success(
            drone.set_control_signals(equal_motor_command(control_signal)),
            f"set all rotors to {control_signal:.6f}",
        )
        warmup_ns = seconds_to_nanos(max(settings.warmup_seconds, 0.003))
        warmup_target = world.get_sim_time() + warmup_ns
        world.continue_for_sim_time(warmup_ns, wait_until_complete=False)
        start_pose = pose_tracker.wait_for_time(warmup_target)
        start_z = extract_z(start_pose)

        measure_ns = seconds_to_nanos(settings.measure_seconds)
        measure_target = world.get_sim_time() + measure_ns
        world.continue_for_sim_time(
            measure_ns,
            wait_until_complete=False,
        )
        end_pose = pose_tracker.wait_for_time(measure_target)
        end_z = extract_z(end_pose)
        velocity = (end_z - start_z) / settings.measure_seconds
        return TrialResult(control_signal, start_z, end_z, velocity)
    finally:
        stop_all_rotors(drone)
        client.unsubscribe([pose_topic])


def choose_best_trial(trials: list[TrialResult]) -> TrialResult:
    if not trials:
        raise ValueError("Không có trial để lựa chọn")
    return min(trials, key=lambda trial: abs(trial.vertical_velocity_mps))


def binary_search_hover(
    client: ProjectAirSimClient,
    world: World,
    trial_scene: dict[str, Any],
    settings: CalibrationSettings,
) -> tuple[TrialResult, list[TrialResult]]:
    lower = settings.lower_signal
    upper = settings.upper_signal
    trials: list[TrialResult] = []

    lower_trial = run_trial(client, world, trial_scene, lower, settings)
    upper_trial = run_trial(client, world, trial_scene, upper, settings)
    trials.extend((lower_trial, upper_trial))
    print(
        f"[BRACKET LOW]  u={lower:.6f}, "
        f"v_down={lower_trial.vertical_velocity_mps:+.5f} m/s"
    )
    print(
        f"[BRACKET HIGH] u={upper:.6f}, "
        f"v_down={upper_trial.vertical_velocity_mps:+.5f} m/s"
    )
    if lower_trial.vertical_velocity_mps <= 0.0:
        raise RuntimeError("Lower signal không làm drone đi xuống; bracket không hợp lệ")
    if upper_trial.vertical_velocity_mps >= 0.0:
        raise RuntimeError("Upper signal không làm drone đi lên; bracket không hợp lệ")

    for iteration in range(1, settings.iterations + 1):
        signal = (lower + upper) / 2.0
        trial = run_trial(client, world, trial_scene, signal, settings)
        trials.append(trial)
        print(
            f"[SEARCH {iteration:02d}/{settings.iterations}] "
            f"u={signal:.6f}, v_down={trial.vertical_velocity_mps:+.5f} m/s"
        )

        # NED: v_down > 0 là đang rơi, do đó cần tăng lực đẩy.
        if trial.vertical_velocity_mps > 0.0:
            lower = signal
        else:
            upper = signal

    return choose_best_trial(trials), trials


def verify_hover_signal(
    client: ProjectAirSimClient,
    world: World,
    trial_scene: dict[str, Any],
    control_signal: float,
    settings: CalibrationSettings,
) -> list[TrialResult]:
    results = []
    for index in range(1, settings.verification_trials + 1):
        trial = run_trial(client, world, trial_scene, control_signal, settings)
        results.append(trial)
        print(
            f"[VERIFY {index:02d}/{settings.verification_trials}] "
            f"u={control_signal:.6f}, "
            f"v_down={trial.vertical_velocity_mps:+.5f} m/s"
        )
    return results


def build_output(
    metadata: dict[str, Any],
    settings: CalibrationSettings,
    best_trial: TrialResult,
    search_trials: list[TrialResult],
    verification_trials: list[TrialResult],
) -> dict[str, Any]:
    velocities = [trial.vertical_velocity_mps for trial in verification_trials]
    mean_velocity = fmean(velocities)
    std_velocity = pstdev(velocities)
    hover_signal = best_trial.control_signal
    max_rpm = float(metadata["max_rpm"])
    hover_rpm = max_rpm * math.sqrt(hover_signal)
    velocity_span = max(
        trial.vertical_velocity_mps for trial in search_trials
    ) - min(trial.vertical_velocity_mps for trial in search_trials)
    passed = (
        abs(mean_velocity) <= settings.velocity_tolerance_mps
        and velocity_span > settings.velocity_tolerance_mps
    )

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scene_config": SCENE_CONFIG,
        "actuator_order": metadata["actuator_order"],
        "max_rpm": max_rpm,
        "hover_control_signal": hover_signal,
        "hover_rpm": hover_rpm,
        "rpm_conversion": "control_signal = (rpm / max_rpm)^2",
        "verification": {
            "mean_vertical_velocity_mps": mean_velocity,
            "std_vertical_velocity_mps": std_velocity,
            "tolerance_mps": settings.velocity_tolerance_mps,
            "search_velocity_span_mps": velocity_span,
            "passed": passed,
        },
        "settings": asdict(settings),
        "search_trials": [asdict(trial) for trial in search_trials],
        "verification_trials": [asdict(trial) for trial in verification_trials],
    }


def save_output(data: dict[str, Any], output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Không ghi đè {output_path}; dùng --overwrite nếu thực sự cần"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    os.replace(temporary_path, output_path)


def execute_calibration(
    settings: CalibrationSettings, output_path: Path, overwrite: bool
) -> dict[str, Any]:
    metadata = get_robot_metadata()
    base_scene, _ = load_scene_config_as_dict(SCENE_CONFIG)
    trial_scene = make_trial_scene(base_scene, settings.start_altitude_m)
    client = ProjectAirSimClient()
    world: World | None = None
    last_drone: Drone | None = None

    try:
        print("[CONNECT] Đang kết nối ProjectAirSim...")
        client.connect()
        world = World(client, SCENE_CONFIG, delay_after_load_sec=0)
        print("[CALIBRATE] Bắt đầu binary search.")
        best_trial, search_trials = binary_search_hover(
            client, world, trial_scene, settings
        )
        print(
            f"[BEST] u={best_trial.control_signal:.6f}, "
            f"v_down={best_trial.vertical_velocity_mps:+.5f} m/s"
        )
        verification_trials = verify_hover_signal(
            client,
            world,
            trial_scene,
            best_trial.control_signal,
            settings,
        )
        output = build_output(
            metadata,
            settings,
            best_trial,
            search_trials,
            verification_trials,
        )
        if not output["verification"]["passed"]:
            raise RuntimeError(
                "Kết quả chưa đạt tolerance; không lưu motor_control.json"
            )
        save_output(output, output_path, overwrite)
        print(f"[SAVED] Đã lưu kết quả: {output_path}")
        return output
    finally:
        if world is not None:
            try:
                last_drone = Drone(client, world, DRONE_NAME)
                require_success(
                    last_drone.set_control_signals(all_stop_command()),
                    "final stop all rotors",
                )
                print("[SAFE] Đã gửi lệnh dừng cả bốn rotor.")
            except Exception as stop_error:
                print(f"[SAFETY ERROR] Không xác nhận được lệnh dừng: {stop_error}")
        client.disconnect()
        print("[DISCONNECT] Đã ngắt kết nối ProjectAirSim.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hiệu chuẩn hover control signal cho bốn rotor"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Cho phép kết nối simulator và chạy hiệu chuẩn",
    )
    parser.add_argument("--lower", type=float, default=0.40)
    parser.add_argument("--upper", type=float, default=0.95)
    parser.add_argument("--iterations", type=int, default=9)
    parser.add_argument("--warmup-seconds", type=float, default=0.10)
    parser.add_argument("--measure-seconds", type=float, default=0.25)
    parser.add_argument("--verification-trials", type=int, default=3)
    parser.add_argument("--velocity-tolerance", type=float, default=0.08)
    parser.add_argument("--start-altitude", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Cho phép ghi đè file output đã tồn tại",
    )
    return parser.parse_args()


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    settings = CalibrationSettings(
        lower_signal=args.lower,
        upper_signal=args.upper,
        iterations=args.iterations,
        warmup_seconds=args.warmup_seconds,
        measure_seconds=args.measure_seconds,
        verification_trials=args.verification_trials,
        velocity_tolerance_mps=args.velocity_tolerance,
        start_altitude_m=args.start_altitude,
    )
    settings.validate()
    validate_motor_scene()
    metadata = get_robot_metadata()

    if not args.execute:
        estimated_signal = 0.72
        estimated_rpm = metadata["max_rpm"] * math.sqrt(estimated_signal)
        print("[DRY RUN] Không kết nối simulator.")
        print(f"[CONFIG] max_rpm={metadata['max_rpm']:.3f}")
        print(
            f"[ESTIMATE] u_hover≈{estimated_signal:.3f}, "
            f"rpm_hover≈{estimated_rpm:.1f} RPM"
        )
        print(
            f"[PLAN] Binary search [{settings.lower_signal:.3f}, "
            f"{settings.upper_signal:.3f}] trong {settings.iterations} vòng"
        )
        print("[INFO] Dùng --execute để chạy hiệu chuẩn trong ProjectAirSim.")
        return

    output = execute_calibration(settings, args.output, args.overwrite)
    print(
        f"[RESULT] hover_control_signal={output['hover_control_signal']:.6f}, "
        f"hover_rpm={output['hover_rpm']:.2f} RPM, "
        f"mean_v_down="
        f"{output['verification']['mean_vertical_velocity_mps']:+.5f} m/s"
    )


if __name__ == "__main__":
    main()
