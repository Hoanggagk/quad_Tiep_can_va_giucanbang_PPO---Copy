import json
import tempfile
import unittest
from pathlib import Path

from drone_control.action_spaces.motor_residual.monitoring import TrainingMonitor


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


class TestTrainingMonitor(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.status_path = Path(self.temporary_directory.name) / "training_status.json"
        self.clock = FakeClock()
        self.messages: list[str] = []
        self.monitor = TrainingMonitor(
            run_id="motor_test",
            max_timesteps=1_000,
            status_path=self.status_path,
            report_every_episodes=2,
            rolling_window=2,
            clock=self.clock,
            printer=self.messages.append,
        )

    def record(self, episode: int, timestep: int, reward: float, survived: bool):
        return self.monitor.record_episode(
            episode=episode,
            timestep=timestep,
            reward=reward,
            steps=100,
            survived=survived,
            stage_index=0,
            stage_name="stabilize",
            survival_rate=0.5,
            position_rmse_m=0.2,
            attitude_rmse_rad=0.1,
            action_std=0.35,
        )

    def test_snapshot_contains_progress_rolling_metrics_and_eta(self) -> None:
        self.clock.value = 10.0
        self.record(0, 100, 2.0, True)
        self.clock.value = 20.0
        csv_metrics = self.record(1, 200, 4.0, False)

        status = json.loads(self.status_path.read_text(encoding="utf-8"))
        self.assertEqual(status["state"], "running")
        self.assertEqual(status["progress_percent"], 20.0)
        self.assertEqual(status["rolling"]["reward_mean"], 3.0)
        self.assertEqual(status["rolling"]["survival_fraction"], 0.5)
        self.assertEqual(status["timesteps_per_second"], 10.0)
        self.assertEqual(status["eta_seconds"], 80.0)
        self.assertEqual(csv_metrics["reward_mean_window"], "3")

    def test_console_reports_first_episode_and_configured_interval(self) -> None:
        self.clock.value = 1.0
        self.record(0, 100, 1.0, True)
        message_count = len(self.messages)
        self.clock.value = 2.0
        self.record(1, 200, 1.0, True)

        self.assertGreater(len(self.messages), message_count)
        self.assertIn("ep=2", self.messages[-1])
        self.assertIn("20.0%", self.messages[-1])

    def test_checkpoint_and_completion_are_persisted(self) -> None:
        self.clock.value = 5.0
        self.record(0, 100, 1.0, True)
        checkpoint = Path("checkpoints_motor/test/final.pth")
        self.monitor.record_checkpoint(checkpoint)
        self.monitor.finish("completed", final_checkpoint=checkpoint)

        status = json.loads(self.status_path.read_text(encoding="utf-8"))
        self.assertEqual(status["state"], "completed")
        self.assertEqual(status["last_checkpoint"], str(checkpoint))
        self.assertEqual(status["final_checkpoint"], str(checkpoint))

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "report_every_episodes"):
            TrainingMonitor(
                run_id="bad",
                max_timesteps=1,
                status_path=self.status_path,
                report_every_episodes=0,
            )


if __name__ == "__main__":
    unittest.main()
