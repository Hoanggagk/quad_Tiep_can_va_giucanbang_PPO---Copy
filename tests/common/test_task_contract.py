import math
import unittest

from drone_control.task_contract import SuccessTracker, TaskContract


class TestTaskContract(unittest.TestCase):
    def test_default_contract_is_valid_and_serializable(self) -> None:
        contract = TaskContract()
        contract.validate()
        metadata = contract.to_metadata()
        self.assertEqual(metadata["coordinate_frame"], "NED")
        self.assertEqual(metadata["goal_position_m"], (0.0, 0.0, -2.0))

    def test_invalid_duration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TaskContract(training_sustained_seconds=0.0).validate()


class TestSuccessTracker(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = TaskContract(
            training_sustained_seconds=0.2,
            motor_episode_seconds=1.0,
        )
        self.tracker = SuccessTracker(self.contract, dt=0.1)

    def update(self, **overrides):
        values = {
            "position_error_m": 0.1,
            "velocity_norm_mps": 0.1,
            "attitude_norm_rad": math.radians(5.0),
            "yaw_error_rad": math.radians(5.0),
        }
        values.update(overrides)
        return self.tracker.update(**values)

    def test_success_requires_continuous_qualified_steps_and_emits_once(self) -> None:
        first = self.update()
        self.assertFalse(first.success)
        second = self.update()
        self.assertTrue(second.success)
        self.assertTrue(second.success_event)
        third = self.update()
        self.assertTrue(third.success)
        self.assertFalse(third.success_event)

    def test_violation_resets_sustain_but_not_reached_history(self) -> None:
        self.update()
        failed = self.update(velocity_norm_mps=1.0)
        self.assertTrue(failed.reached_once)
        self.assertEqual(failed.sustained_seconds, 0.0)
        self.assertFalse(failed.success)

    def test_yaw_can_be_excluded_from_contract(self) -> None:
        tracker = SuccessTracker(
            TaskContract(
                require_yaw=False,
                training_sustained_seconds=0.1,
                motor_episode_seconds=1.0,
            ),
            dt=0.1,
        )
        status = tracker.update(
            position_error_m=0.1,
            velocity_norm_mps=0.1,
            attitude_norm_rad=0.1,
            yaw_error_rad=math.pi,
        )
        self.assertTrue(status.success)


if __name__ == "__main__":
    unittest.main()
