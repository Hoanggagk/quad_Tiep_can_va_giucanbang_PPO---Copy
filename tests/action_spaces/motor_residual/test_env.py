import asyncio
import math
import unittest
from unittest.mock import AsyncMock, Mock

import numpy as np

from drone_control.action_spaces.motor_residual.env import (
    ACTUATOR_ORDER,
    DroneMotorHoverEnv,
)


def make_pose(
    timestamp_ns: int,
    *,
    x: float = 0.0,
    y: float = 0.0,
    z: float = -2.0,
) -> dict:
    return {
        "time_stamp": timestamp_ns,
        "translation": {"x": x, "y": y, "z": z},
        "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }


class TestMotorActionMapping(unittest.TestCase):
    def setUp(self) -> None:
        self.env = DroneMotorHoverEnv(connect=False)

    def tearDown(self) -> None:
        self.env.close()

    def test_spaces_have_expected_shapes_and_bounds(self) -> None:
        self.assertEqual(self.env.action_space.shape, (4,))
        self.assertEqual(self.env.observation_space.shape, (17,))
        np.testing.assert_allclose(self.env.action_space.low, -1.0)
        np.testing.assert_allclose(self.env.action_space.high, 1.0)

    def test_action_is_clipped_to_normalized_bounds(self) -> None:
        command = self.env.action_to_motor_command([2.0, -2.0, 1.5, -1.5])
        np.testing.assert_array_equal(
            command.normalized_action,
            np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32),
        )

    def test_zero_action_maps_to_hover_operating_point(self) -> None:
        command = self.env.action_to_motor_command(np.zeros(4))
        np.testing.assert_allclose(command.rpm, self.env.hover_rpm, rtol=1e-6)
        np.testing.assert_allclose(
            command.control_signal,
            self.env.hover_control_signal,
            rtol=1e-6,
        )

    def test_rpm_is_always_within_configured_bounds(self) -> None:
        for action in (
            np.full(4, -100.0),
            np.zeros(4),
            np.full(4, 100.0),
        ):
            rpm = self.env.action_to_motor_command(action).rpm
            self.assertTrue(np.all(rpm >= self.env.min_rpm))
            self.assertTrue(np.all(rpm <= self.env.max_rpm))

    def test_rpm_to_control_signal_uses_square_relationship(self) -> None:
        command = self.env.action_to_motor_command([-1.0, -0.25, 0.25, 1.0])
        expected = np.square(command.rpm.astype(np.float64) / self.env.max_rpm)
        np.testing.assert_allclose(command.control_signal, expected, rtol=1e-6)

    def test_actuator_map_uses_fl_fr_rl_rr_order(self) -> None:
        command = self.env.action_to_motor_command([-1.0, -0.5, 0.5, 1.0])
        self.assertEqual(tuple(command.actuator_map), ACTUATOR_ORDER)
        for index, actuator_id in enumerate(ACTUATOR_ORDER):
            self.assertAlmostEqual(
                command.actuator_map[actuator_id],
                float(command.control_signal[index]),
                places=7,
            )

    def test_prev_action_and_motor_state_have_four_elements(self) -> None:
        self.assertEqual(self.env.prev_action.shape, (4,))
        self.assertEqual(self.env.prev_motor_rpm.shape, (4,))
        self.assertEqual(self.env.prev_motor_signal.shape, (4,))

    def test_invalid_action_shape_and_nonfinite_value_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.env.action_to_motor_command([0.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            self.env.action_to_motor_command([0.0, math.nan, 0.0, 0.0])


class TestMotorObservation(unittest.TestCase):
    def setUp(self) -> None:
        self.env = DroneMotorHoverEnv(connect=False)

    def tearDown(self) -> None:
        self.env.close()

    def test_observation_has_17_float32_elements(self) -> None:
        self.env.prev_motor_signal = np.array([0.1, 0.2, 0.3, 0.4])
        self.env.pose_callback("pose", make_pose(1_000_000_000))
        observation = self.env.get_observation()
        self.assertEqual(observation.shape, (17,))
        self.assertEqual(observation.dtype, np.float32)
        np.testing.assert_allclose(observation[13:17], [0.1, 0.2, 0.3, 0.4])

    def test_pose_callback_estimates_linear_velocity_from_sim_time(self) -> None:
        self.env.pose_callback("pose", make_pose(1_000_000_000, x=0.0))
        self.env.pose_callback("pose", make_pose(1_020_000_000, x=0.02))
        observation = self.env.get_observation()
        self.assertAlmostEqual(float(observation[6]), 1.0, places=6)

    def test_observation_requires_actual_pose(self) -> None:
        with self.assertRaises(RuntimeError):
            self.env.get_observation()


class TestMotorRuntimeSafety(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.env = DroneMotorHoverEnv(connect=False, pose_timeout_seconds=0.001)
        self.world = Mock()
        self.world.get_sim_time.return_value = 1_000_000_000
        self.client = Mock()
        self.actor = Mock()
        self.env.client = self.client
        self.env.world = self.world
        self.env.actor = self.actor
        self.env._connected = True
        self.env.pose_callback("pose", make_pose(1_000_000_000))

    def tearDown(self) -> None:
        self.env.close()

    async def test_rejected_motor_command_raises_and_stops_all_rotors(self) -> None:
        # Lần 1: command bị từ chối; lần 2: emergency stop; lần 3: teardown stop.
        self.actor.set_control_signals.side_effect = [False, True, True]

        with self.assertRaisesRegex(RuntimeError, "từ chối motor command"):
            await self.env.step(np.zeros(4))

        final_command = self.actor.set_control_signals.call_args_list[-1].args[0]
        self.assertEqual(final_command, {key: 0.0 for key in ACTUATOR_ORDER})

    async def test_pose_timeout_stops_all_rotors(self) -> None:
        self.actor.set_control_signals.return_value = True

        with self.assertRaises(TimeoutError):
            await self.env.step(np.zeros(4))

        final_command = self.actor.set_control_signals.call_args_list[-1].args[0]
        self.assertEqual(final_command, {key: 0.0 for key in ACTUATOR_ORDER})

    async def test_simulation_exception_stops_all_rotors(self) -> None:
        self.actor.set_control_signals.return_value = True
        self.world.continue_for_sim_time.side_effect = RuntimeError("sim failure")

        with self.assertRaisesRegex(RuntimeError, "sim failure"):
            await self.env.step(np.zeros(4))

        final_command = self.actor.set_control_signals.call_args_list[-1].args[0]
        self.assertEqual(final_command, {key: 0.0 for key in ACTUATOR_ORDER})

    async def test_successful_step_updates_previous_motor_state(self) -> None:
        self.actor.set_control_signals.return_value = True
        self.env._wait_for_pose_sync_async = AsyncMock(return_value=None)

        observation, _, terminated, truncated, info = await self.env.step(
            np.zeros(4)
        )

        self.assertEqual(observation.shape, (17,))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIn("reward_terms", info)
        self.assertIn("total", info["reward_terms"])
        np.testing.assert_allclose(
            self.env.prev_motor_signal,
            self.env.hover_control_signal,
            rtol=1e-6,
        )
        np.testing.assert_allclose(info["motor_rpm"], self.env.hover_rpm, rtol=1e-6)

    async def test_close_stops_all_rotors(self) -> None:
        self.actor.set_control_signals.return_value = True
        self.env.close()
        final_command = self.actor.set_control_signals.call_args_list[-1].args[0]
        self.assertEqual(final_command, {key: 0.0 for key in ACTUATOR_ORDER})


if __name__ == "__main__":
    unittest.main()
