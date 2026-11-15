"""Create a frozen evaluation manifest without starting ProjectAirSim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drone_control.evaluation_protocol import build_evaluation_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action-space",
        required=True,
        choices=("velocity_ned", "motor_residual"),
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Không overwrite manifest đã tồn tại: {args.output}")
    manifest = build_evaluation_manifest(
        action_space=args.action_space,
        checkpoint=args.checkpoint,
        seeds=args.seeds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
