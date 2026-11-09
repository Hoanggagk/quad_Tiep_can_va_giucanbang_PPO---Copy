import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from drone_control.action_spaces.motor_residual.curriculum import MotorCurriculum
from drone_control.action_spaces.motor_residual.env import (
    ACTUATOR_ORDER,
    DroneMotorHoverEnv,
)
from drone_control.action_spaces.motor_residual.train import (
    ACTION_DIM,
    CHECKPOINT_ROOT,
    LOG_ROOT,
    OBSERVATION_LAYOUT,
    STATE_DIM,
    build_checkpoint_metadata,
    make_run_id,
    metadata_path_for,
    save_checkpoint_with_metadata,
    validate_resume_checkpoint,
)


class FakeAgent:
    def save(self, path: str) -> None:
        Path(path).write_bytes(b"fake motor checkpoint")


class TestMotorTrainingEntryPoint(unittest.TestCase):
    def setUp(self) -> None:
        self.curriculum = MotorCurriculum(seed=1)
        self.env = DroneMotorHoverEnv(connect=False, curriculum=self.curriculum)

    def tearDown(self) -> None:
        self.env.close()

    def test_motor_dimensions_and_observation_layout(self) -> None:
        self.assertEqual(STATE_DIM, 17)
        self.assertEqual(ACTION_DIM, 4)
        self.assertEqual(len(OBSERVATION_LAYOUT), STATE_DIM)

    def test_output_namespaces_are_separate_from_velocity_training(self) -> None:
        self.assertEqual(LOG_ROOT.as_posix(), "log_dir/hover_motor")
        self.assertEqual(CHECKPOINT_ROOT.as_posix(), "checkpoints_motor")

    def test_run_id_is_deterministic_for_timestamp_and_motor_prefixed(self) -> None:
        run_id = make_run_id(datetime(2026, 1, 2, 3, 4, 5, 6000))
        self.assertEqual(run_id, "motor_20260102_030405_006000")

    def test_metadata_contains_all_required_fields(self) -> None:
        metadata = build_checkpoint_metadata(
            self.env,
            self.curriculum,
            run_id="motor_test",
            episode=3,
            timestep=100,
        )
        required = {
            "state_dim",
            "action_dim",
            "actuator_order",
            "max_rpm",
            "hover_rpm",
            "rpm_delta",
            "observation_layout",
            "reward_weights",
            "control_frequency_hz",
            "source_id",
        }
        self.assertTrue(required.issubset(metadata))
        self.assertEqual(metadata["actuator_order"], list(ACTUATOR_ORDER))
        self.assertEqual(metadata["state_dim"], 17)
        self.assertEqual(metadata["action_dim"], 4)
        self.assertTrue(metadata["source_id"].startswith("sha256:"))

    def test_checkpoint_and_sidecar_are_saved_without_overwrite(self) -> None:
        metadata = build_checkpoint_metadata(
            self.env, self.curriculum, run_id="motor_test", episode=0, timestep=0
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "checkpoint.pth"
            save_checkpoint_with_metadata(FakeAgent(), checkpoint, metadata)
            self.assertTrue(checkpoint.is_file())
            sidecar = metadata_path_for(checkpoint)
            self.assertTrue(sidecar.is_file())
            self.assertEqual(
                json.loads(sidecar.read_text(encoding="utf-8"))["policy_type"],
                "motor_level_ppo",
            )
            with self.assertRaises(FileExistsError):
                save_checkpoint_with_metadata(FakeAgent(), checkpoint, metadata)

    def test_resume_rejects_checkpoint_without_motor_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "velocity.pth"
            checkpoint.write_bytes(b"velocity checkpoint")
            with self.assertRaisesRegex(ValueError, "thiếu metadata"):
                validate_resume_checkpoint(checkpoint)

    def test_resume_rejects_wrong_dimensions(self) -> None:
        metadata = build_checkpoint_metadata(
            self.env, self.curriculum, run_id="motor_test", episode=0, timestep=0
        )
        metadata["action_dim"] = 3
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "wrong.pth"
            checkpoint.write_bytes(b"wrong checkpoint")
            metadata_path_for(checkpoint).write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "action_dim"):
                validate_resume_checkpoint(checkpoint)


if __name__ == "__main__":
    unittest.main()
