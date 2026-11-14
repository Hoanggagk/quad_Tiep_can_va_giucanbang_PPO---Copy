"""Entry point huấn luyện PPO cho action là tốc độ bốn motor.

Lệnh này chỉ bắt đầu kết nối simulator khi gọi ``train``/chạy script. Import
module để kiểm tra cấu hình hoặc metadata không tạo kết nối và không ghi file.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from drone_control.action_spaces.motor_residual.curriculum import MotorCurriculum
from drone_control.action_spaces.motor_residual.env import (
    ACTUATOR_ORDER,
    DroneMotorHoverEnv,
)
from drone_control.action_spaces.motor_residual.monitoring import TrainingMonitor


STATE_DIM = 17
ACTION_DIM = 4
OBSERVATION_LAYOUT = (
    "x_m", "y_m", "z_m", "roll_rad", "pitch_rad", "yaw_rad",
    "vx_mps", "vy_mps", "vz_mps", "roll_rate_radps",
    "pitch_rate_radps", "yaw_rate_radps", "distance_to_goal_m",
    "motor_fl_signal", "motor_fr_signal", "motor_rl_signal", "motor_rr_signal",
)
CONTROL_FREQUENCY_HZ = 50.0
CONTROL_DT = 1.0 / CONTROL_FREQUENCY_HZ

LOG_ROOT = Path("log_dir") / "hover_motor"
CHECKPOINT_ROOT = Path("checkpoints_motor")

ACTION_STD_INIT = 0.35
ACTION_STD_DECAY_RATE = 0.01
ACTION_STD_DECAY_FREQ_EPISODES = 50
MIN_ACTION_STD = 0.05
MAX_TRAINING_TIMESTEPS = 500_000
UPDATE_TIMESTEP = 2_000
SAVE_MODEL_FREQ = 50_000
STEPS_PER_EPISODE = 500
MONITOR_EVERY_EPISODES = 10
MONITOR_ROLLING_WINDOW = 20

K_EPOCHS = 30
EPS_CLIP = 0.2
GAMMA = 0.99
LR_ACTOR = 3e-4
LR_CRITIC = 1e-3


def make_run_id(now: datetime | None = None) -> str:
    instant = now or datetime.now()
    return instant.strftime("motor_%Y%m%d_%H%M%S_%f")


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    project_root = Path(__file__).resolve().parents[3]
    source_paths = (
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("env.py"),
        Path(__file__).resolve().with_name("curriculum.py"),
        project_root / "drone_control" / "algorithms" / "ppo.py",
    )
    for path in source_paths:
        digest.update(path.relative_to(project_root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return f"sha256:{digest.hexdigest()}"


def build_checkpoint_metadata(
    env: DroneMotorHoverEnv,
    curriculum: MotorCurriculum,
    *,
    run_id: str,
    episode: int,
    timestep: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "policy_type": "motor_level_ppo",
        "run_id": run_id,
        "episode": episode,
        "timestep": timestep,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "action_bounds": {"low": -1.0, "high": 1.0},
        "actuator_order": list(ACTUATOR_ORDER),
        "max_rpm": env.max_rpm,
        "hover_rpm": env.hover_rpm,
        "rpm_delta": env.rpm_delta,
        "observation_layout": list(OBSERVATION_LAYOUT),
        "reward_weights": asdict(env.reward_config),
        "control_frequency_hz": 1.0 / env.dt,
        "control_dt_seconds": env.dt,
        "curriculum": {
            "stage_index": curriculum.stage_index,
            "stage_name": curriculum.current_stage.name,
            "stages": [asdict(stage) for stage in curriculum.stages],
        },
        "source_id": source_fingerprint(),
        "created_at": datetime.now().astimezone().isoformat(),
    }


def metadata_path_for(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(checkpoint_path.suffix + ".json")


def save_checkpoint_with_metadata(
    agent: Any,
    checkpoint_path: Path,
    metadata: dict[str, Any],
) -> None:
    checkpoint_path = Path(checkpoint_path)
    metadata_path = metadata_path_for(checkpoint_path)
    if checkpoint_path.exists() or metadata_path.exists():
        raise FileExistsError(f"Không overwrite checkpoint đã tồn tại: {checkpoint_path}")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(checkpoint_path))
    try:
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        # Checkpoint không có metadata là artifact không hợp lệ; xóa file vừa tạo.
        checkpoint_path.unlink(missing_ok=True)
        raise


def validate_resume_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")
    sidecar = metadata_path_for(checkpoint_path)
    if not sidecar.is_file():
        raise ValueError("Checkpoint resume thiếu metadata motor-level")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    expected = {
        "policy_type": "motor_level_ppo",
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "actuator_order": list(ACTUATOR_ORDER),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Checkpoint không tương thích: {key}={metadata.get(key)!r}, "
                f"cần {value!r}"
            )
    return metadata


async def train(
    *,
    resume_path: Path | None = None,
    max_timesteps: int = MAX_TRAINING_TIMESTEPS,
    seed: int | None = None,
    monitor_every_episodes: int = MONITOR_EVERY_EPISODES,
    monitor_rolling_window: int = MONITOR_ROLLING_WINDOW,
    monitor_enabled: bool = True,
) -> Path:
    # Import trễ để các công cụ kiểm tra metadata không bắt buộc cài PyTorch.
    from drone_control.algorithms.ppo import PPO

    if max_timesteps <= 0:
        raise ValueError("max_timesteps phải lớn hơn 0")
    if monitor_every_episodes <= 0:
        raise ValueError("monitor_every_episodes phải lớn hơn 0")
    if monitor_rolling_window <= 0:
        raise ValueError("monitor_rolling_window phải lớn hơn 0")
    resume_metadata = (
        validate_resume_checkpoint(resume_path) if resume_path is not None else None
    )
    run_id = make_run_id()
    log_dir = LOG_ROOT / run_id
    checkpoint_dir = CHECKPOINT_ROOT / run_id
    log_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)

    curriculum = MotorCurriculum(
        initial_stage=int(resume_metadata["curriculum"]["stage_index"])
        if resume_metadata is not None
        else 0,
        seed=seed,
    )
    env = DroneMotorHoverEnv(dt=CONTROL_DT, curriculum=curriculum)
    agent = PPO(
        STATE_DIM,
        ACTION_DIM,
        LR_ACTOR,
        LR_CRITIC,
        GAMMA,
        K_EPOCHS,
        EPS_CLIP,
        ACTION_STD_INIT,
        action_low=-1.0,
        action_high=1.0,
    )
    if resume_path is not None:
        agent.load(str(resume_path))

    log_path = log_dir / "training.csv"
    monitor = TrainingMonitor(
        run_id=run_id,
        max_timesteps=max_timesteps,
        status_path=log_dir / "training_status.json",
        report_every_episodes=monitor_every_episodes,
        rolling_window=monitor_rolling_window,
        enabled=monitor_enabled,
    )
    timestep = 0
    episode = 0
    last_checkpoint: Path | None = None
    try:
        with log_path.open("x", newline="", encoding="utf-8") as log_file:
            writer = csv.DictWriter(
                log_file,
                fieldnames=(
                    "episode", "timestep", "reward", "steps", "survived",
                    "stage_index", "stage_name", "survival_rate",
                    "position_rmse_m", "attitude_rmse_rad",
                    "progress_percent", "elapsed_seconds", "timesteps_per_second",
                    "eta_seconds", "reward_mean_window",
                    "survival_fraction_window", "position_rmse_mean_window_m",
                    "attitude_rmse_mean_window_rad", "action_std",
                ),
            )
            writer.writeheader()
            while timestep < max_timesteps:
                observation, reset_info = env.reset(seed=None if seed is None else seed + episode)
                episode_reward = 0.0
                final_info: dict[str, Any] = {}
                step_count = 0
                for step_count in range(1, STEPS_PER_EPISODE + 1):
                    action = agent.select_action(observation)
                    observation, reward, terminated, truncated, final_info = await env.step(action)
                    done = terminated or truncated
                    agent.buffer.rewards.append(reward)
                    agent.buffer.is_terminals.append(done)
                    episode_reward += reward
                    timestep += 1
                    if timestep % UPDATE_TIMESTEP == 0:
                        agent.update()
                    if timestep % SAVE_MODEL_FREQ == 0:
                        path = checkpoint_dir / f"episode_{episode:06d}_step_{timestep:09d}.pth"
                        save_checkpoint_with_metadata(
                            agent,
                            path,
                            build_checkpoint_metadata(
                                env, curriculum, run_id=run_id,
                                episode=episode, timestep=timestep,
                            ),
                        )
                        last_checkpoint = path
                        monitor.record_checkpoint(path)
                    if done or timestep >= max_timesteps:
                        break

                curriculum_result = final_info.get("curriculum_result", {})
                curriculum_info = final_info.get(
                    "curriculum", reset_info.get("curriculum", {})
                )
                episode_data = {
                    "episode": episode,
                    "timestep": timestep,
                    "reward": f"{episode_reward:.9g}",
                    "steps": step_count,
                    "survived": curriculum_result.get("survived", False),
                    "stage_index": curriculum_info.get("stage_index"),
                    "stage_name": curriculum_info.get("stage_name"),
                    "survival_rate": curriculum_result.get("survival_rate"),
                    "position_rmse_m": curriculum_result.get("position_rmse_m"),
                    "attitude_rmse_rad": curriculum_result.get("attitude_rmse_rad"),
                }
                episode_data.update(
                    monitor.record_episode(
                        episode=episode,
                        timestep=timestep,
                        reward=episode_reward,
                        steps=step_count,
                        survived=bool(episode_data["survived"]),
                        stage_index=episode_data["stage_index"],
                        stage_name=episode_data["stage_name"],
                        survival_rate=episode_data["survival_rate"],
                        position_rmse_m=episode_data["position_rmse_m"],
                        attitude_rmse_rad=episode_data["attitude_rmse_rad"],
                        action_std=agent.action_std,
                    )
                )
                writer.writerow(episode_data)
                log_file.flush()
                episode += 1
                if episode % ACTION_STD_DECAY_FREQ_EPISODES == 0:
                    agent.decay_action_std(ACTION_STD_DECAY_RATE, MIN_ACTION_STD)
    except BaseException as error:
        monitor.finish(
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        env.close()

    final_path = checkpoint_dir / f"final_episode_{episode:06d}_step_{timestep:09d}.pth"
    try:
        save_checkpoint_with_metadata(
            agent,
            final_path,
            build_checkpoint_metadata(
                env, curriculum, run_id=run_id, episode=episode, timestep=timestep
            ),
        )
        monitor.record_checkpoint(final_path)
        monitor.finish("completed", final_checkpoint=final_path)
    except BaseException as error:
        monitor.finish("failed", error=f"{type(error).__name__}: {error}")
        raise
    return final_path


def parse_args() -> argparse.Namespace:
    # Giữ help CLI ở ASCII để hoạt động cả trên Windows console dùng cp1252.
    parser = argparse.ArgumentParser(
        description="Train the motor-level PPO policy for the quadrotor"
    )
    parser.add_argument("--resume", type=Path, default=None, metavar="CHECKPOINT")
    parser.add_argument("--max-timesteps", type=int, default=MAX_TRAINING_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--monitor-every",
        type=int,
        default=MONITOR_EVERY_EPISODES,
        metavar="EPISODES",
        help="Print a monitoring summary every N episodes (default: %(default)s)",
    )
    parser.add_argument(
        "--monitor-window",
        type=int,
        default=MONITOR_ROLLING_WINDOW,
        metavar="EPISODES",
        help="Rolling metric window in episodes (default: %(default)s)",
    )
    parser.add_argument(
        "--quiet-monitor",
        action="store_true",
        help="Disable console reports; still update CSV and training_status.json",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    final_path = await train(
        resume_path=args.resume,
        max_timesteps=args.max_timesteps,
        seed=args.seed,
        monitor_every_episodes=args.monitor_every,
        monitor_rolling_window=args.monitor_window,
        monitor_enabled=not args.quiet_monitor,
    )
    print(f"[DONE] Checkpoint motor-level cuối: {final_path}")


if __name__ == "__main__":
    asyncio.run(main())
