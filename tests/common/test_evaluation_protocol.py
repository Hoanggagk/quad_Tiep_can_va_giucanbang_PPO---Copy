import unittest

from drone_control.evaluation_protocol import (
    EpisodeMetrics,
    FROZEN_SCENARIOS,
    aggregate_metrics,
    build_evaluation_manifest,
)


class TestEvaluationProtocol(unittest.TestCase):
    def make_record(self, *, seed: int, success: bool) -> EpisodeMetrics:
        return EpisodeMetrics(
            scenario_id="C_reach_hover",
            seed=seed,
            success=success,
            reached_once=True,
            crashed=not success,
            timed_out=False,
            time_to_target_seconds=2.0,
            overshoot_m=0.1,
            path_efficiency=1.1,
            position_rmse_m=0.2,
            velocity_rmse_mps=0.1,
            attitude_rmse_rad=0.05,
            angular_rate_rmse_radps=0.04,
            control_effort=0.3,
            smoothness=0.02,
            saturation_fraction=0.01,
            action_near_bound_fraction=0.02,
        )

    def test_manifest_freezes_tests_a_to_d_and_multiple_seeds(self) -> None:
        manifest = build_evaluation_manifest(
            action_space="motor_residual",
            checkpoint="checkpoint.pth",
            seeds=[1, 2, 3],
        )
        self.assertEqual(manifest["seeds"], [1, 2, 3])
        self.assertTrue(manifest["protocol_sha256"])
        identifiers = {item["scenario_id"] for item in manifest["scenarios"]}
        self.assertTrue({"A_hover", "B_reaching", "C_reach_hover"}.issubset(identifiers))
        self.assertTrue(any(item.scenario_id.startswith("D_") for item in FROZEN_SCENARIOS))

    def test_aggregate_reports_completion_safety_and_control(self) -> None:
        aggregate = aggregate_metrics(
            [self.make_record(seed=1, success=True), self.make_record(seed=2, success=False)]
        )
        self.assertEqual(aggregate["seed_count"], 2)
        self.assertEqual(aggregate["success_rate"], 0.5)
        self.assertEqual(aggregate["crash_rate"], 0.5)
        self.assertAlmostEqual(aggregate["position_rmse_m"], 0.2)

    def test_duplicate_seeds_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_evaluation_manifest(
                action_space="velocity_ned",
                checkpoint="checkpoint.pth",
                seeds=[1, 1],
            )


if __name__ == "__main__":
    unittest.main()
