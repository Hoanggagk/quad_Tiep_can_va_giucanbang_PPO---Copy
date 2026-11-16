import csv
import tempfile
import unittest
from pathlib import Path

from tools.analyze_motor_training import build_report, load_training_csv


class TestAnalyzeMotorTraining(unittest.TestCase):
    def test_report_identifies_short_failed_stage_locked_training(self) -> None:
        columns = [
            "episode", "timestep", "reward", "steps", "survived", "success",
            "stage_name", "position_rmse_m", "attitude_rmse_rad", "action_std",
            "ppo_approx_kl", "ppo_clip_fraction", "ppo_explained_variance",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                for episode in range(10):
                    writer.writerow(
                        {
                            "episode": episode,
                            "timestep": (episode + 1) * 20,
                            "reward": 10 + episode,
                            "steps": 20 + episode,
                            "survived": False,
                            "success": False,
                            "stage_name": "attitude_hover",
                            "position_rmse_m": 0.5,
                            "attitude_rmse_rad": 0.7,
                            "action_std": 0.5,
                            "ppo_approx_kl": 0.01,
                            "ppo_clip_fraction": 0.1,
                            "ppo_explained_variance": 0.9,
                        }
                    )

            rows = load_training_csv(path)
            report = build_report(path, rows)

        self.assertIn("Episode kết thúc quá sớm", report)
        self.assertIn("Mất ổn định attitude", report)
        self.assertIn("Curriculum bị khóa", report)
        self.assertIn("termination_reason", report)

    def test_missing_required_column_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.csv"
            path.write_text("episode,reward\n0,1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "CSV thiếu cột"):
                load_training_csv(path)


if __name__ == "__main__":
    unittest.main()
