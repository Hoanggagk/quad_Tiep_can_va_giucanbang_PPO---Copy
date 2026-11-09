import math
import unittest

import numpy as np

from drone_control.action_spaces.motor_residual.curriculum import (
    CurriculumStage,
    MotorCurriculum,
    default_motor_curriculum,
)
from drone_control.action_spaces.motor_residual.env import DroneMotorHoverEnv


class TestMotorCurriculum(unittest.TestCase):
    def test_default_curriculum_has_required_order(self) -> None:
        self.assertEqual(
            [stage.name for stage in default_motor_curriculum()],
            [
                "attitude_hover",
                "random_attitude",
                "random_position_velocity",
                "altitude_change",
                "takeoff",
                "disturbance",
                "domain_randomization",
            ],
        )

    def test_sampling_is_reproducible_and_bounded(self) -> None:
        left = MotorCurriculum(initial_stage=2, seed=123).sample_episode()
        right = MotorCurriculum(initial_stage=2, seed=123).sample_episode()
        np.testing.assert_allclose(left.initial_position_m, right.initial_position_m)
        np.testing.assert_allclose(left.initial_velocity_mps, right.initial_velocity_mps)
        self.assertTrue(np.all(np.abs(left.initial_position_m[:2]) <= 0.25))
        self.assertTrue(np.all(np.abs(left.initial_velocity_mps) <= 0.25))

    def test_takeoff_starts_near_ground_and_targets_hover_height(self) -> None:
        conditions = MotorCurriculum(initial_stage=4, seed=7).sample_episode()
        self.assertGreaterEqual(conditions.initial_position_m[2], -0.70)
        self.assertLessEqual(conditions.initial_position_m[2], -0.35)
        self.assertGreaterEqual(conditions.goal_position_m[2], -2.5)
        self.assertLessEqual(conditions.goal_position_m[2], -1.5)

    def test_domain_randomization_ranges(self) -> None:
        conditions = MotorCurriculum(initial_stage=6, seed=8).sample_episode()
        for scale in (
            conditions.mass_scale,
            conditions.thrust_scale,
            conditions.motor_time_constant_scale,
        ):
            self.assertGreaterEqual(scale, 0.9)
            self.assertLessEqual(scale, 1.1)
        self.assertEqual(conditions.action_delay_steps, 2)
        self.assertEqual(conditions.sensor_noise_std, 0.01)
        self.assertLessEqual(abs(conditions.hover_signal_bias), 0.03)

    def test_physical_domain_randomization_is_applied_to_robot_config(self) -> None:
        conditions = MotorCurriculum(initial_stage=6, seed=8).sample_episode()
        scene = {
            "actors": [
                {
                    "name": "Drone1",
                    "robot-config": {
                        "links": [{"inertial": {"mass": 2.0}}],
                        "actuators": [
                            {
                                "type": "rotor",
                                "rotor-settings": {
                                    "coeff-of-thrust": 0.1,
                                    "smoothing-tc": 0.01,
                                },
                            }
                        ],
                    },
                }
            ]
        }
        DroneMotorHoverEnv._apply_domain_randomization_to_scene(scene, conditions)
        robot = scene["actors"][0]["robot-config"]
        settings = robot["actuators"][0]["rotor-settings"]
        self.assertAlmostEqual(
            robot["links"][0]["inertial"]["mass"], 2.0 * conditions.mass_scale
        )
        self.assertAlmostEqual(
            settings["coeff-of-thrust"], 0.1 * conditions.thrust_scale
        )
        self.assertAlmostEqual(
            settings["smoothing-tc"],
            0.01 * conditions.motor_time_constant_scale,
        )

    def test_advance_requires_all_metrics_not_reward(self) -> None:
        stages = (
            CurriculumStage(
                name="first",
                min_evaluation_episodes=2,
                min_survival_rate=1.0,
                max_position_rmse_m=0.2,
                max_attitude_rmse_rad=0.1,
            ),
            CurriculumStage(name="second"),
        )
        curriculum = MotorCurriculum(stages=stages)
        for _ in range(2):
            curriculum.record_episode(
                survived=True,
                position_errors_m=np.array([0.1, 0.1]),
                attitude_errors_rad=np.array([0.05, 0.05]),
            )
        self.assertTrue(curriculum.can_advance())
        self.assertTrue(curriculum.advance_if_ready())
        self.assertEqual(curriculum.stage_index, 1)
        self.assertEqual(curriculum.window.summarize().episode_count, 0)

    def test_failed_survival_or_rmse_blocks_advance(self) -> None:
        stages = (
            CurriculumStage(name="first", min_evaluation_episodes=1),
            CurriculumStage(name="second"),
        )
        curriculum = MotorCurriculum(stages=stages)
        curriculum.record_episode(
            survived=False,
            position_errors_m=np.array([0.01]),
            attitude_errors_rad=np.array([0.01]),
        )
        self.assertFalse(curriculum.advance_if_ready())

    def test_environment_records_metrics_and_advances_stage_once(self) -> None:
        stages = (
            CurriculumStage(
                name="first",
                min_evaluation_episodes=1,
                min_survival_rate=1.0,
                max_position_rmse_m=0.2,
                max_attitude_rmse_rad=0.1,
            ),
            CurriculumStage(name="second"),
        )
        curriculum = MotorCurriculum(stages=stages)
        env = DroneMotorHoverEnv(connect=False, curriculum=curriculum)
        try:
            env._episode_position_errors_m = [0.1, 0.1]
            env._episode_attitude_errors_rad = [0.05, 0.05]
            result = env._record_curriculum_episode(survived=True)
            self.assertTrue(result["recorded"])
            self.assertTrue(result["stage_advanced"])
            self.assertEqual(result["next_stage_name"], "second")
            self.assertFalse(
                env._record_curriculum_episode(survived=True)["recorded"]
            )
        finally:
            env.close()

    def test_scene_initial_pose_uses_meters_and_degrees(self) -> None:
        conditions = MotorCurriculum(initial_stage=1, seed=9).sample_episode()
        scene = {"actors": [{"name": "Drone1", "origin": {}}]}
        DroneMotorHoverEnv._apply_initial_pose_to_scene(scene, conditions)
        xyz = np.fromstring(scene["actors"][0]["origin"]["xyz"], sep=" ")
        rpy_deg = np.fromstring(scene["actors"][0]["origin"]["rpy-deg"], sep=" ")
        np.testing.assert_allclose(xyz, conditions.initial_position_m)
        np.testing.assert_allclose(
            rpy_deg, np.degrees(conditions.initial_rpy_rad), rtol=1e-7, atol=1e-7
        )

    def test_stage_one_is_near_exact_hover(self) -> None:
        conditions = MotorCurriculum(seed=1).sample_episode()
        np.testing.assert_allclose(conditions.initial_position_m, [0.0, 0.0, -2.0])
        np.testing.assert_allclose(conditions.initial_rpy_rad, 0.0)
        np.testing.assert_allclose(conditions.initial_velocity_mps, 0.0)
        np.testing.assert_allclose(conditions.goal_position_m, [0.0, 0.0, -2.0])


if __name__ == "__main__":
    unittest.main()
