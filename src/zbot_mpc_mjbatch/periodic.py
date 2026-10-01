"""Run a validated periodic stepping segment from 0.5 through 2 Hz."""

from __future__ import annotations

import argparse
import json

from .stepping import simulate_periodic_steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", type=float, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=4)
    args = parser.parse_args()
    try:
        run = simulate_periodic_steps(
            args.frequency,
            steps=args.steps,
            num_threads=args.threads,
            seed=args.seed,
        )
    except ValueError as error:
        parser.error(str(error))
    result = {**run.summary(), "landing_times_s": run.landing_times_s.round(3).tolist()}
    print(json.dumps(result, indent=2))
    if run.stop_reason is not None:
        raise SystemExit(run.stop_reason)


if __name__ == "__main__":
    main()
