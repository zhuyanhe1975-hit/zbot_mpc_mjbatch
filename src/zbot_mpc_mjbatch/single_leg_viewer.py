"""Plan a single-leg stand and replay the measured process in MuJoCo Viewer."""

from __future__ import annotations

import argparse
import json
import math

from .single_leg import simulate_single_leg_stand
from .viewer import play_viewer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--support-foot",
        type=int,
        choices=(0, 1),
        default=0,
        help="foot that remains planted (blue in the viewer)",
    )
    parser.add_argument("--hold-seconds", type=float, default=5.0)
    parser.add_argument("--unfold-seconds", type=float, default=6.0)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--once", action="store_true", help="close after one replay")
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be finite and positive")
    try:
        run = simulate_single_leg_stand(
            support_foot=args.support_foot,
            hold_seconds=args.hold_seconds,
            num_candidates=args.candidates,
            iterations=args.iterations,
            num_threads=args.threads,
            seed=args.seed,
            unfold_seconds=args.unfold_seconds,
        )
    except ValueError as error:
        parser.error(str(error))

    print(json.dumps(run.summary(), indent=2), flush=True)
    if run.stop_reason is not None:
        raise SystemExit(run.stop_reason)
    print(
        "MuJoCo Viewer：蓝色为支撑脚，橙色为摆动脚；"
        "空格暂停/继续，N 单帧前进，R 从头回放，关闭窗口退出。",
        flush=True,
    )
    try:
        play_viewer(run, speed=args.speed, once=args.once)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
