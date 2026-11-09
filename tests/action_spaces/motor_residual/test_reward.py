import math
import unittest

import numpy as np

from drone_control.action_spaces.motor_residual.env import (
    DroneMotorHoverEnv,
    RewardConfig,
)


class TestMotorReward(unittest.TestCase):
    def setUp(self) -> None:
        self.env = DroneMotorHoverEnv(connect=False)
        self.hover_signal = np.full(4, self.env.hover_control_signal)
        self.set_state()

    def tearDown(self) -> None:
        self.env.close()

    def set_state(
        self,
        *,
        x: float = 0.0,
        y: float = 0.0,
        z: float = -2.0,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        velocity=(0.0, 0.0, 0.0),
        angular_rate=(0.0, 0.0, 0.0),
    ) -> None:
        self.env.actor_pose = [x, y, z, roll, pitch, yaw]
        self.env.linear_velocity = np.asarray(velocity, dtype=np.float64)
        self.env.angular_velocity = np.asarray(angular_rate, dtype=np.float64)
        self.env.done = False
        self.env.crashed = False
        self.env.ground_touch = False
        self.env.termination_reason = None

    def reward(self, current=None, previous=None):
        current = self.hover_signal if current is None else np.asarray(current)
        previous = self.hover_signal if previous is None else np.asarray(previous)
        return self.env.calculate_reward(current, previous)

    def test_stable_hover_has_positive_reward_and_no_penalties(self) -> None:
        total, terms = self.reward()
        self.assertGreater(total, 1.0)
        self.assertGreater(terms["position"], 0.0)
        self.assertGreater(terms["hover_bonus"], 0.0)
        for name in (
            "velocity",
            "attitude",
            "yaw",
            "angular_rate",
            "motor_effort",
            "motor_smoothness",
            "saturation",
            "crash",
        ):
            self.assertAlmostEqual(terms[name], 0.0)

    def test_tilted_state_has_attitude_penalty_and_lower_reward(self) -> None:
        hover_total, _ = self.reward()
        self.set_state(roll=math.radians(30.0), pitch=math.radians(-20.0))
        tilted_total, terms = self.reward()
        self.assertLess(terms["attitude"], 0.0)
        self.assertLess(tilted_total, hover_total)

    def test_velocity_and_angular_rate_are_penalized_in_correct_units(self) -> None:
        self.set_state(velocity=(1.0, 0.0, 0.0), angular_rate=(0.0, 1.0, 0.0))
        _, terms = self.reward()
        self.assertAlmostEqual(terms["velocity"], -0.10, places=7)
        self.assertAlmostEqual(terms["angular_rate"], -0.05, places=7)

    def test_yaw_error_is_wrapped_and_penalized(self) -> None:
        self.set_state(yaw=2.0 * math.pi - 0.1)
        _, terms = self.reward()
        self.assertAlmostEqual(terms["yaw"], -0.05 * 0.1**2, places=7)

    def test_ground_touch_applies_strong_crash_penalty(self) -> None:
        self.set_state(z=-0.1)
        self.env._check_termination()
        total, terms = self.reward()
        self.assertTrue(self.env.crashed)
        self.assertEqual(self.env.termination_reason, "ground_touch")
        self.assertEqual(terms["crash"], -self.env.reward_config.crash_penalty)
        self.assertLess(total, 0.0)

    def test_motor_saturation_is_penalized(self) -> None:
        hover_total, _ = self.reward()
        saturated = np.ones(4)
        saturated_total, terms = self.reward(saturated, saturated)
        self.assertAlmostEqual(terms["saturation_fraction"], 1.0)
        self.assertLess(terms["saturation"], 0.0)
        self.assertLess(saturated_total, hover_total)

    def test_motor_command_delta_is_penalized(self) -> None:
        current = np.ones(4)
        _, terms = self.reward(current, self.hover_signal)
        self.assertLess(terms["motor_smoothness"], 0.0)

    def test_reward_terms_contain_all_logged_components(self) -> None:
        _, terms = self.reward()
        expected = {
            "position",
            "alive",
            "velocity",
            "attitude",
            "yaw",
            "angular_rate",
            "motor_effort",
            "motor_smoothness",
            "saturation",
            "hover_bonus",
            "crash",
            "total",
            "position_error_m",
            "velocity_norm_mps",
            "angular_rate_norm_radps",
            "saturation_fraction",
        }
        self.assertEqual(set(terms), expected)

    def test_tilt_limit_is_70_degrees(self) -> None:
        self.set_state(roll=math.radians(69.0))
        self.env._check_termination()
        self.assertFalse(self.env.done)
        self.set_state(roll=math.radians(71.0))
        self.env._check_termination()
        self.assertTrue(self.env.done)
        self.assertEqual(self.env.termination_reason, "attitude_limit")

    def test_reward_scales_with_control_period(self) -> None:
        total_50hz, _ = self.reward()
        env_100hz = DroneMotorHoverEnv(connect=False, dt=0.01)
        try:
            env_100hz.actor_pose = list(self.env.actor_pose)
            env_100hz.linear_velocity = self.env.linear_velocity.copy()
            env_100hz.angular_velocity = self.env.angular_velocity.copy()
            signal = np.full(4, env_100hz.hover_control_signal)
            total_100hz, _ = env_100hz.calculate_reward(signal, signal)
            self.assertAlmostEqual(total_100hz, total_50hz / 2.0, places=6)
        finally:
            env_100hz.close()

    def test_invalid_reward_config_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DroneMotorHoverEnv(
                connect=False,
                reward_config=RewardConfig(velocity_weight=-1.0),
            )


if __name__ == "__main__":
    unittest.main()
