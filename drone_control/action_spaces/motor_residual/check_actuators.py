"""Smoke test an toàn cho bốn rotor của cấu hình manual-controller.

Mặc định script chỉ kiểm tra config và in chuỗi lệnh dự kiến. Phải truyền
``--execute`` mới kết nối ProjectAirSim và gửi tín hiệu tới actuator.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Mapping, Protocol

from projectairsim import Drone, ProjectAirSimClient, World
from projectairsim.utils import load_scene_config_as_dict


SCENE_CONFIG = "scene_basic_drone_motor.jsonc"
DRONE_NAME = "Drone1"
ACTUATOR_ORDER = (
    "Prop_FL_actuator",
    "Prop_FR_actuator",
    "Prop_RL_actuator",
    "Prop_RR_actuator",
)
INVALID_ACTUATOR_ID = "Invalid_actuator"
DEFAULT_SIGNAL = 0.05
MAX_SMOKE_TEST_SIGNAL = 0.20


class ControlSignalTarget(Protocol):
    def set_control_signals(self, control_signal_map: Mapping[str, float]) -> bool:
        """Gửi control signal tới một hoặc nhiều actuator."""


@dataclass(frozen=True)
class SmokeTestSettings:
    signal: float = DEFAULT_SIGNAL
    pulse_seconds: float = 0.5
    settle_seconds: float = 0.25

    def validate(self) -> None:
        if not 0.0 < self.signal <= MAX_SMOKE_TEST_SIGNAL:
            raise ValueError(
                f"signal phải thuộc (0, {MAX_SMOKE_TEST_SIGNAL}], "
                f"nhận được {self.signal}"
            )
        if self.pulse_seconds <= 0.0:
            raise ValueError("pulse_seconds phải lớn hơn 0")
        if self.settle_seconds < 0.0:
            raise ValueError("settle_seconds không được âm")


def all_stop_command() -> dict[str, float]:
    """Tạo lệnh dừng đồng thời cả bốn rotor."""
    return {actuator_id: 0.0 for actuator_id in ACTUATOR_ORDER}


def single_rotor_command(actuator_id: str, signal: float) -> dict[str, float]:
    """Tạo lệnh chỉ kích hoạt một rotor, ba rotor còn lại bằng 0."""
    if actuator_id not in ACTUATOR_ORDER:
        raise ValueError(f"Actuator không hợp lệ: {actuator_id}")
    command = all_stop_command()
    command[actuator_id] = float(signal)
    return command


def require_success(result: bool, operation: str) -> None:
    if result is not True:
        raise RuntimeError(f"ProjectAirSim từ chối thao tác: {operation}")


def stop_all_rotors(drone: ControlSignalTarget) -> None:
    """Dừng cả bốn rotor và báo lỗi nếu simulator từ chối lệnh."""
    require_success(drone.set_control_signals(all_stop_command()), "stop all rotors")


def validate_motor_scene(scene_config: str = SCENE_CONFIG) -> None:
    """Xác thực schema và mapping actuator của scene motor-level."""
    config, _ = load_scene_config_as_dict(scene_config)
    robot = config["actors"][0]["robot-config"]
    controller = robot["controller"]
    configured_ids = {
        item["id"]
        for item in controller["manual-controller-api-settings"]["actuator-order"]
    }
    actuator_ids = {item["name"] for item in robot["actuators"]}
    expected_ids = set(ACTUATOR_ORDER)

    if controller["type"] != "manual-controller-api":
        raise ValueError("Scene không sử dụng manual-controller-api")
    if configured_ids != expected_ids:
        raise ValueError(
            f"Actuator order không khớp: {configured_ids} != {expected_ids}"
        )
    if actuator_ids != expected_ids:
        raise ValueError(f"Danh sách actuator không khớp: {actuator_ids}")


def run_invalid_actuator_check(
    drone: ControlSignalTarget, signal: float = DEFAULT_SIGNAL
) -> None:
    """Xác nhận controller từ chối một actuator ID không tồn tại."""
    result = drone.set_control_signals({INVALID_ACTUATOR_ID: float(signal)})
    if result is not False:
        raise RuntimeError(
            "Controller không từ chối actuator ID không hợp lệ như mong đợi"
        )


def run_rotor_sequence(
    drone: ControlSignalTarget,
    settings: SmokeTestSettings,
    sleep_fn=time.sleep,
) -> None:
    """Kích hoạt lần lượt FL, FR, RL, RR và luôn dừng rotor khi thoát."""
    settings.validate()
    stop_all_rotors(drone)

    try:
        for index, actuator_id in enumerate(ACTUATOR_ORDER, start=1):
            print(
                f"[ROTOR {index}/4] {actuator_id}: "
                f"signal={settings.signal:.3f}"
            )
            require_success(
                drone.set_control_signals(
                    single_rotor_command(actuator_id, settings.signal)
                ),
                f"pulse {actuator_id}",
            )
            sleep_fn(settings.pulse_seconds)
            stop_all_rotors(drone)
            sleep_fn(settings.settle_seconds)

        print(f"[NEGATIVE TEST] Kiểm tra ID không hợp lệ: {INVALID_ACTUATOR_ID}")
        run_invalid_actuator_check(drone, settings.signal)
    finally:
        # Safety invariant: mọi đường thoát đều cố gắng dừng đồng thời 4 rotor.
        stop_all_rotors(drone)


def print_dry_run(settings: SmokeTestSettings) -> None:
    print("[DRY RUN] Không kết nối simulator và không gửi lệnh actuator.")
    print(f"[CONFIG] Scene hợp lệ: {SCENE_CONFIG}")
    for index, actuator_id in enumerate(ACTUATOR_ORDER, start=1):
        print(
            f"  {index}. {actuator_id}: signal={settings.signal:.3f}, "
            f"duration={settings.pulse_seconds:.2f}s"
        )
    print(f"  5. Negative test: {INVALID_ACTUATOR_ID} phải trả về False")
    print("  6. Final safety command: cả bốn rotor = 0.0")
    print("[INFO] Dùng --execute để thực thi smoke test trong ProjectAirSim.")


def execute_smoke_test(settings: SmokeTestSettings) -> None:
    client = ProjectAirSimClient()
    drone: Drone | None = None

    try:
        print("[CONNECT] Đang kết nối ProjectAirSim...")
        client.connect()
        world = World(client, SCENE_CONFIG, delay_after_load_sec=0)
        drone = Drone(client, world, DRONE_NAME)
        print("[CONNECT] Đã load scene motor-level.")
        run_rotor_sequence(drone, settings)
        print("[PASS] Hoàn thành chuỗi kiểm tra actuator.")
    finally:
        if drone is not None:
            try:
                stop_all_rotors(drone)
                print("[SAFE] Đã gửi lệnh dừng cả bốn rotor.")
            except Exception as stop_error:
                print(f"[SAFETY ERROR] Không xác nhận được lệnh dừng rotor: {stop_error}")
        client.disconnect()
        print("[DISCONNECT] Đã ngắt kết nối ProjectAirSim.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test bốn rotor của manual-controller ProjectAirSim"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Cho phép kết nối simulator và gửi lệnh rotor; mặc định là dry-run",
    )
    parser.add_argument(
        "--signal",
        type=float,
        default=DEFAULT_SIGNAL,
        help=f"Control signal mỗi rotor, tối đa {MAX_SMOKE_TEST_SIGNAL}",
    )
    parser.add_argument(
        "--pulse-seconds",
        type=float,
        default=0.5,
        help="Thời gian giữ tín hiệu cho mỗi rotor",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=0.25,
        help="Thời gian chờ sau khi dừng một rotor",
    )
    return parser.parse_args()


def main() -> None:
    # Windows có thể mặc định dùng cp1252, không biểu diễn đầy đủ tiếng Việt.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    settings = SmokeTestSettings(
        signal=args.signal,
        pulse_seconds=args.pulse_seconds,
        settle_seconds=args.settle_seconds,
    )
    settings.validate()
    validate_motor_scene()

    if args.execute:
        execute_smoke_test(settings)
    else:
        print_dry_run(settings)


if __name__ == "__main__":
    main()
