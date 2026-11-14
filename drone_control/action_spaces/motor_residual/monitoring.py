"""Giám sát nhẹ cho quá trình huấn luyện motor-level PPO.

Module này không phụ thuộc simulator hay thư viện hiển thị. Mỗi episode tạo một
snapshot máy đọc được và, theo chu kỳ, in tóm tắt tiến độ cho người vận hành.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(values: deque[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--"
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class TrainingMonitor:
    """Theo dõi rolling metrics, tiến độ, throughput và ETA của một run."""

    def __init__(
        self,
        *,
        run_id: str,
        max_timesteps: int,
        status_path: Path,
        report_every_episodes: int = 10,
        rolling_window: int = 20,
        enabled: bool = True,
        clock: Callable[[], float] = time.perf_counter,
        printer: Callable[[str], None] = print,
    ) -> None:
        if max_timesteps <= 0:
            raise ValueError("max_timesteps phải lớn hơn 0")
        if report_every_episodes <= 0:
            raise ValueError("report_every_episodes phải lớn hơn 0")
        if rolling_window <= 0:
            raise ValueError("rolling_window phải lớn hơn 0")

        self.run_id = run_id
        self.max_timesteps = max_timesteps
        self.status_path = Path(status_path)
        self.report_every_episodes = report_every_episodes
        self.rolling_window = rolling_window
        self.enabled = enabled
        self._clock = clock
        self._printer = printer
        self._started_at = clock()
        self._reward_window: deque[float] = deque(maxlen=rolling_window)
        self._survival_window: deque[float] = deque(maxlen=rolling_window)
        self._position_rmse_window: deque[float] = deque(maxlen=rolling_window)
        self._attitude_rmse_window: deque[float] = deque(maxlen=rolling_window)
        self._last_stage_index: int | None = None
        self._last_snapshot: dict[str, Any] = {}
        self._last_checkpoint: str | None = None

        self._last_snapshot = {
            "schema_version": 1,
            "run_id": run_id,
            "state": "starting",
            "max_timesteps": max_timesteps,
            "rolling_window": rolling_window,
            "updated_at": datetime.now().astimezone().isoformat(),
        }
        self._write_status(self._last_snapshot)
        if enabled:
            printer(
                f"[MONITOR] Run {run_id} | target={max_timesteps:,} timesteps | "
                f"window={rolling_window} episodes"
            )

    def record_episode(
        self,
        *,
        episode: int,
        timestep: int,
        reward: float,
        steps: int,
        survived: bool,
        stage_index: int | None,
        stage_name: str | None,
        survival_rate: Any,
        position_rmse_m: Any,
        attitude_rmse_rad: Any,
        action_std: Any,
    ) -> dict[str, Any]:
        """Ghi nhận một episode và trả về các trường monitoring để ghi CSV."""
        reward = float(reward)
        self._reward_window.append(reward)
        self._survival_window.append(float(bool(survived)))
        position_rmse = _optional_float(position_rmse_m)
        attitude_rmse = _optional_float(attitude_rmse_rad)
        if position_rmse is not None:
            self._position_rmse_window.append(position_rmse)
        if attitude_rmse is not None:
            self._attitude_rmse_window.append(attitude_rmse)

        elapsed = max(0.0, self._clock() - self._started_at)
        throughput = timestep / elapsed if elapsed > 0 else 0.0
        remaining = max(0, self.max_timesteps - timestep)
        eta = remaining / throughput if throughput > 0 else None
        progress = min(100.0, max(0.0, timestep / self.max_timesteps * 100.0))
        rolling_reward = _mean(self._reward_window)
        rolling_survival = _mean(self._survival_window)
        rolling_position_rmse = _mean(self._position_rmse_window)
        rolling_attitude_rmse = _mean(self._attitude_rmse_window)

        snapshot = {
            "schema_version": 1,
            "run_id": self.run_id,
            "state": "running",
            "episode": episode,
            "timestep": timestep,
            "max_timesteps": self.max_timesteps,
            "progress_percent": progress,
            "elapsed_seconds": elapsed,
            "timesteps_per_second": throughput,
            "eta_seconds": eta,
            "reward": reward,
            "steps": steps,
            "survived": bool(survived),
            "stage_index": stage_index,
            "stage_name": stage_name,
            "survival_rate": _optional_float(survival_rate),
            "position_rmse_m": position_rmse,
            "attitude_rmse_rad": attitude_rmse,
            "action_std": _optional_float(action_std),
            "rolling": {
                "window_episodes": len(self._reward_window),
                "reward_mean": rolling_reward,
                "survival_fraction": rolling_survival,
                "position_rmse_mean_m": rolling_position_rmse,
                "attitude_rmse_mean_rad": rolling_attitude_rmse,
            },
            "last_checkpoint": self._last_checkpoint,
            "updated_at": datetime.now().astimezone().isoformat(),
        }
        self._last_snapshot = snapshot
        self._write_status(snapshot)

        stage_changed = (
            self._last_stage_index is not None
            and stage_index is not None
            and stage_index != self._last_stage_index
        )
        self._last_stage_index = stage_index
        should_report = (
            episode == 0
            or (episode + 1) % self.report_every_episodes == 0
            or timestep >= self.max_timesteps
            or stage_changed
        )
        if self.enabled and should_report:
            position_text = (
                "--"
                if rolling_position_rmse is None
                else f"{rolling_position_rmse:.3f}m"
            )
            attitude_text = (
                "--"
                if rolling_attitude_rmse is None
                else f"{rolling_attitude_rmse:.3f}rad"
            )
            reward_text = "--" if rolling_reward is None else f"{rolling_reward:.2f}"
            survival_text = (
                "--"
                if rolling_survival is None
                else f"{rolling_survival * 100:.0f}%"
            )
            self._printer(
                f"[MONITOR] ep={episode + 1} step={timestep:,}/{self.max_timesteps:,} "
                f"({progress:5.1f}%) | reward={reward:.2f} avg={reward_text} | "
                f"survival={survival_text} pos_rmse={position_text} "
                f"att_rmse={attitude_text} | "
                f"stage={stage_index}:{stage_name or '--'} | "
                f"speed={throughput:.1f} step/s ETA={_format_duration(eta)}"
            )

        return {
            "progress_percent": f"{progress:.6g}",
            "elapsed_seconds": f"{elapsed:.6g}",
            "timesteps_per_second": f"{throughput:.6g}",
            "eta_seconds": "" if eta is None else f"{eta:.6g}",
            "reward_mean_window": (
                "" if rolling_reward is None else f"{rolling_reward:.9g}"
            ),
            "survival_fraction_window": (
                "" if rolling_survival is None else f"{rolling_survival:.9g}"
            ),
            "position_rmse_mean_window_m": (
                "" if rolling_position_rmse is None else f"{rolling_position_rmse:.9g}"
            ),
            "attitude_rmse_mean_window_rad": (
                "" if rolling_attitude_rmse is None else f"{rolling_attitude_rmse:.9g}"
            ),
            "action_std": (
                ""
                if snapshot["action_std"] is None
                else f"{snapshot['action_std']:.9g}"
            ),
        }

    def record_checkpoint(self, checkpoint_path: Path) -> None:
        self._last_checkpoint = str(checkpoint_path)
        if self._last_snapshot:
            self._last_snapshot["last_checkpoint"] = self._last_checkpoint
            self._last_snapshot["updated_at"] = datetime.now().astimezone().isoformat()
            self._write_status(self._last_snapshot)
        if self.enabled:
            self._printer(f"[MONITOR] Đã lưu checkpoint: {checkpoint_path}")

    def finish(
        self,
        state: str,
        *,
        final_checkpoint: Path | None = None,
        error: str | None = None,
    ) -> None:
        snapshot = dict(self._last_snapshot)
        snapshot.update(
            {
                "schema_version": 1,
                "run_id": self.run_id,
                "state": state,
                "final_checkpoint": (
                    str(final_checkpoint) if final_checkpoint is not None else None
                ),
                "error": error,
                "updated_at": datetime.now().astimezone().isoformat(),
            }
        )
        self._write_status(snapshot)
        if self.enabled:
            elapsed = _optional_float(snapshot.get("elapsed_seconds"))
            self._printer(
                f"[MONITOR] Run {state} | elapsed={_format_duration(elapsed)} | "
                f"checkpoint={final_checkpoint or '--'}"
            )

    def _write_status(self, payload: dict[str, Any]) -> None:
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.status_path.with_suffix(self.status_path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_path.replace(self.status_path)
