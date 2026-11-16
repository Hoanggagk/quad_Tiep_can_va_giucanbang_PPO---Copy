"""Analyze a motor-level PPO training CSV and write a Markdown report."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


REQUIRED_COLUMNS = {
    "episode",
    "timestep",
    "reward",
    "steps",
    "survived",
    "success",
    "stage_name",
    "position_rmse_m",
    "attitude_rmse_rad",
    "action_std",
}
FLOAT_COLUMNS = REQUIRED_COLUMNS - {"survived", "success", "stage_name"}
OPTIONAL_FLOAT_COLUMNS = {
    "ppo_approx_kl",
    "ppo_clip_fraction",
    "ppo_explained_variance",
    "ppo_actor_grad_norm",
    "ppo_critic_grad_norm",
    "ppo_epochs_completed",
    "action_near_bound_fraction",
}
MAX_EPISODE_STEPS = 500
TARGET_KL = 0.02
CURRICULUM_POSITION_RMSE_M = 0.30
CURRICULUM_ATTITUDE_RMSE_RAD = math.radians(12.0)


def _float(value: str, column: str, row_number: int) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Dòng {row_number}: {column} không phải số") from error
    if not math.isfinite(result):
        raise ValueError(f"Dòng {row_number}: {column} phải hữu hạn")
    return result


def _optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def load_training_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"CSV thiếu cột: {', '.join(sorted(missing))}")
        rows: list[dict[str, object]] = []
        for row_number, source in enumerate(reader, start=2):
            row: dict[str, object] = {
                column: _float(source[column], column, row_number)
                for column in FLOAT_COLUMNS
            }
            row["survived"] = source["survived"].strip().lower() == "true"
            row["success"] = source["success"].strip().lower() == "true"
            row["stage_name"] = source["stage_name"].strip() or "(trống)"
            for column in OPTIONAL_FLOAT_COLUMNS:
                row[column] = _optional_float(source.get(column))
            rows.append(row)
    if not rows:
        raise ValueError("CSV không có episode")
    return rows


def _values(rows: Iterable[dict[str, object]], column: str) -> list[float]:
    return [float(row[column]) for row in rows if row.get(column) is not None]


def _mean(rows: Iterable[dict[str, object]], column: str) -> float:
    return statistics.fmean(_values(rows, column))


def _rate(rows: list[dict[str, object]], column: str) -> float:
    return statistics.fmean(float(bool(row[column])) for row in rows)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def _correlation(rows: list[dict[str, object]], left: str, right: str) -> float | None:
    pairs = [
        (float(row[left]), float(row[right]))
        for row in rows
        if row.get(left) is not None and row.get(right) is not None
    ]
    if len(pairs) < 2:
        return None
    xs, ys = zip(*pairs)
    x_mean, y_mean = statistics.fmean(xs), statistics.fmean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in xs) * sum((y - y_mean) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else None


def _fmt(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _unique_ppo_snapshots(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    snapshots: list[dict[str, object]] = []
    previous: tuple[object, ...] | None = None
    columns = tuple(sorted(OPTIONAL_FLOAT_COLUMNS - {"action_near_bound_fraction"}))
    for row in rows:
        current = tuple(row.get(column) for column in columns)
        if current != previous and any(value is not None for value in current):
            snapshots.append(row)
            previous = current
    return snapshots


def build_report(path: Path, rows: list[dict[str, object]]) -> str:
    count = len(rows)
    window_size = min(500, max(1, count // 10))
    first, last = rows[:window_size], rows[-window_size:]
    reward_step_correlation = _correlation(rows, "reward", "steps")
    first_reward_per_step = statistics.fmean(
        float(row["reward"]) / max(1.0, float(row["steps"])) for row in first
    )
    last_reward_per_step = statistics.fmean(
        float(row["reward"]) / max(1.0, float(row["steps"])) for row in last
    )
    stage_counts = Counter(str(row["stage_name"]) for row in rows)
    survival_rate, success_rate = _rate(rows, "survived"), _rate(rows, "success")
    median_steps = statistics.median(_values(rows, "steps"))
    last_attitude_rmse = _mean(last, "attitude_rmse_rad")
    last_position_rmse = _mean(last, "position_rmse_m")
    ppo = _unique_ppo_snapshots(rows)
    kl = _values(ppo, "ppo_approx_kl")
    clip = _values(ppo, "ppo_clip_fraction")
    explained = _values(ppo, "ppo_explained_variance")
    kl_over_target = (
        sum(value > TARGET_KL for value in kl) / len(kl) if kl else None
    )

    diagnoses: list[str] = []
    if survival_rate == 0.0 and success_rate == 0.0:
        diagnoses.append(
            "**Không có tín hiệu thành công:** toàn bộ episode có `survived=0` và "
            "`success=0`, nên curriculum không có điều kiện chuyển stage."
        )
    if median_steps < MAX_EPISODE_STEPS * 0.25:
        diagnoses.append(
            f"**Episode kết thúc quá sớm:** median chỉ {_fmt(median_steps, 1)}/"
            f"{MAX_EPISODE_STEPS} bước. Policy gần như không thu được trajectory hover dài."
        )
    if last_attitude_rmse > CURRICULUM_ATTITUDE_RMSE_RAD:
        diagnoses.append(
            "**Mất ổn định attitude là nguyên nhân có khả năng cao nhất:** attitude "
            f"RMSE cuối {_fmt(math.degrees(last_attitude_rmse), 1)}°, vượt gate 12°. "
            "CSV không có `termination_reason`, nên chưa thể khẳng định crash do "
            "`attitude_limit` hay một điều kiện kết thúc khác."
        )
    if last_position_rmse > CURRICULUM_POSITION_RMSE_M:
        diagnoses.append(
            f"**Position gate cũng không đạt:** RMSE cuối {_fmt(last_position_rmse)} m "
            f"> {CURRICULUM_POSITION_RMSE_M:.2f} m."
        )
    if reward_step_correlation is not None and reward_step_correlation > 0.8:
        diagnoses.append(
            "**Reward tổng chủ yếu đo độ dài episode:** tương quan reward–steps là "
            f"{reward_step_correlation:.3f}. Vì vậy reward không tăng rõ không đồng nghĩa "
            "PPO loss không học; drone đang không sống đủ lâu để tích lũy reward."
        )
    if len(stage_counts) == 1:
        stage = next(iter(stage_counts))
        diagnoses.append(
            f"**Curriculum bị khóa:** {count:,} episode đều ở `{stage}`."
        )
    if kl_over_target is not None and kl_over_target > 0.10:
        diagnoses.append(
            "**Một phần PPO update quá mạnh:** "
            f"{kl_over_target:.1%} snapshot vượt target KL {TARGET_KL:.2f}. "
            "Đây là yếu tố phụ; nó không giải thích riêng việc survival luôn bằng 0."
        )

    by_std: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        action_std = float(row["action_std"])
        lower = min(0.4, max(0.1, math.floor((action_std + 1e-9) * 10.0) / 10.0))
        upper = min(0.5, lower + 0.1)
        by_std[f"{lower:.1f}–{upper:.1f}"].append(row)

    lines = [
        "# Báo cáo phân tích huấn luyện PPO 4 động cơ",
        "",
        f"- Nguồn: `{path.as_posix()}`",
        f"- Số episode: **{count:,}**",
        f"- Timestep cuối: **{int(float(rows[-1]['timestep'])):,}**",
        f"- Cửa sổ so sánh: **{window_size} episode** đầu và cuối",
        "",
        "## Kết luận chính",
        "",
    ]
    lines.extend(f"{index}. {item}" for index, item in enumerate(diagnoses, start=1))
    lines.extend(
        [
            "",
            "## So sánh đầu và cuối",
            "",
            "| Chỉ số | Đầu | Cuối |",
            "|---|---:|---:|",
            f"| Reward/episode trung bình | {_fmt(_mean(first, 'reward'))} | {_fmt(_mean(last, 'reward'))} |",
            f"| Reward/bước trung bình | {_fmt(first_reward_per_step, 4)} | {_fmt(last_reward_per_step, 4)} |",
            f"| Số bước/episode trung bình | {_fmt(_mean(first, 'steps'), 1)} | {_fmt(_mean(last, 'steps'), 1)} |",
            f"| Survival rate | {_rate(first, 'survived'):.1%} | {_rate(last, 'survived'):.1%} |",
            f"| Success rate | {_rate(first, 'success'):.1%} | {_rate(last, 'success'):.1%} |",
            f"| Position RMSE (m) | {_fmt(_mean(first, 'position_rmse_m'))} | {_fmt(last_position_rmse)} |",
            f"| Attitude RMSE (độ) | {_fmt(math.degrees(_mean(first, 'attitude_rmse_rad')), 1)} | {_fmt(math.degrees(last_attitude_rmse), 1)} |",
            f"| Action std | {_fmt(_mean(first, 'action_std'))} | {_fmt(_mean(last, 'action_std'))} |",
            "",
            f"Tương quan Pearson giữa reward và số bước: **{_fmt(reward_step_correlation)}**.",
            "",
            "## Curriculum",
            "",
            "| Stage | Episode | Tỷ lệ |",
            "|---|---:|---:|",
        ]
    )
    for stage, stage_count in stage_counts.most_common():
        lines.append(f"| `{stage}` | {stage_count:,} | {stage_count / count:.1%} |")

    lines.extend(
        [
            "",
            "## Kết quả theo action standard deviation",
            "",
            "| Action std | Episode | Reward TB | Steps TB | Attitude RMSE (độ) | Survival |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for action_std in sorted(by_std, reverse=True):
        group = by_std[action_std]
        lines.append(
            f"| {action_std} | {len(group):,} | {_mean(group, 'reward'):.3f} | "
            f"{_mean(group, 'steps'):.1f} | "
            f"{math.degrees(_mean(group, 'attitude_rmse_rad')):.1f} | "
            f"{_rate(group, 'survived'):.1%} |"
        )

    lines.extend(
        [
            "",
            "## PPO diagnostics",
            "",
            f"Phân tích **{len(ppo)}** snapshot PPO khác nhau, loại các giá trị lặp giữa các episode.",
            "",
            "| Chỉ số | Median | P95 | Max/cuối |",
            "|---|---:|---:|---:|",
            f"| Approx KL | {_fmt(_percentile(kl, 0.5), 4)} | {_fmt(_percentile(kl, 0.95), 4)} | {_fmt(max(kl) if kl else None, 4)} |",
            f"| Clip fraction | {_fmt(_percentile(clip, 0.5), 4)} | {_fmt(_percentile(clip, 0.95), 4)} | {_fmt(max(clip) if clip else None, 4)} |",
            f"| Explained variance | {_fmt(_percentile(explained, 0.5), 4)} | {_fmt(_percentile(explained, 0.95), 4)} | {_fmt(explained[-1] if explained else None, 4)} |",
            "",
            f"Tỷ lệ update có `approx_kl > {TARGET_KL}`: "
            + ("**N/A**." if kl_over_target is None else f"**{kl_over_target:.1%}**."),
            "",
            "## Nguyên nhân và hành động đề xuất",
            "",
            "1. **Ưu tiên xác nhận termination:** thêm `termination_reason` vào `training.csv`; dữ liệu hiện tại chỉ cho phép suy luận attitude instability từ episode ngắn và attitude RMSE cao.",
            "2. **Không tiếp tục tăng exploration:** action std ban đầu 0.50 trên bốn motor độc lập tạo differential thrust lớn. Hãy thử lại từ checkpoint sạch với std thấp hơn sau khi xác nhận neutral action giữ hover ổn định.",
            "3. **Kiểm tra motor mapping trước PPO:** chạy bài kiểm tra bốn motor cùng action 0 và perturbation đối xứng nhỏ; xác nhận thứ tự FL/FR/RL/RR và chiều moment roll/pitch đúng.",
            "4. **Đánh giá bằng survival và RMSE, không chỉ reward:** reward episode bị chi phối mạnh bởi số bước sống. Curriculum chỉ nên tiến khi survival, success và hai RMSE đạt gate.",
            "5. **Không resume checkpoint hiện tại để kiểm tra cấu hình exploration mới:** policy đã trải qua hàng nghìn episode crash ở stage đầu; dùng run mới giúp tách lỗi policy cũ khỏi lỗi cấu hình/mapping.",
            "",
            "## Giới hạn của báo cáo",
            "",
            "CSV không lưu `termination_reason`, reward components hoặc action theo từng rotor. Vì vậy báo cáo xác định được triệu chứng và nguyên nhân có khả năng cao, nhưng chưa chứng minh được rotor/moment nào gây mất ổn định.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.csv_path.with_name("TRAINING_ANALYSIS_REPORT.md")
    if output.exists():
        raise FileExistsError(f"Không overwrite báo cáo đã tồn tại: {output}")
    report = build_report(args.csv_path, load_training_csv(args.csv_path))
    output.write_text(report, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
