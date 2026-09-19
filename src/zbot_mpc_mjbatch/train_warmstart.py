"""Bootstrap a safe MPC warm-start policy from MPC expert plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .stepping import simulate_steps
from .warmstart import LinearWarmStartPolicy, WarmStartDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dist/zbot_warmstart.npz"))
    parser.add_argument("--dataset", type=Path, default=Path("dist/zbot_mpc_expert.npz"))
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=4.2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--hold-seconds", type=float, default=0.1)
    parser.add_argument("--frequency-min", type=float, default=0.5)
    parser.add_argument("--frequency-max", type=float, default=2.0)
    parser.add_argument("--candidates", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=1e-3)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if not 0.4 <= args.frequency_min <= args.frequency_max <= 2.0:
        parser.error("frequencies must satisfy 0.4 <= min <= max <= 2.0")

    datasets: list[WarmStartDataset] = []
    failures: list[str] = []
    policy: LinearWarmStartPolicy | None = None
    frequencies = np.linspace(args.frequency_min, args.frequency_max, args.episodes)
    for episode, frequency in enumerate(frequencies):
        try:
            run = simulate_steps(
                frequency_hz=float(frequency),
                seconds=args.seconds,
                num_candidates=args.candidates,
                iterations=args.iterations,
                num_threads=args.threads,
                seed=args.seed + episode,
                warm_start_policy=policy,
                max_steps=args.steps,
                minimum_hold_s=args.hold_seconds,
            )
        except RuntimeError as error:
            failures.append(f"episode {episode}: {error}")
            continue
        if run.plans:
            datasets.append(WarmStartDataset.from_plans(run.plans))
            aggregate = WarmStartDataset.concatenate(datasets)
            if len(aggregate.observations) >= 2:
                policy = LinearWarmStartPolicy.fit(
                    aggregate.observations, aggregate.targets, ridge=args.ridge
                )
        if run.stop_reason is not None:
            failures.append(f"episode {episode}: {run.stop_reason}")

    if not datasets:
        raise SystemExit("MPC produced no safe expert plans; warm-start policy was not written")
    dataset = WarmStartDataset.concatenate(datasets)
    if len(dataset.observations) < 2:
        raise SystemExit("at least two safe expert plans are required")
    assert policy is not None
    predictions = np.stack([policy.predict(observation) for observation in dataset.observations])
    error = predictions - dataset.targets

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.dataset.parent.mkdir(parents=True, exist_ok=True)
    policy.save(args.output)
    dataset.save(args.dataset)
    print(
        json.dumps(
            {
                "safe_expert_plans": len(dataset.observations),
                "episodes_with_plans": len(datasets),
                "collection_warnings": failures,
                "training_rmse": round(float(np.sqrt(np.mean(error**2))), 6),
                "policy": str(args.output.resolve()),
                "dataset": str(args.dataset.resolve()),
                "note": "policy only seeds MPC; rollout safety checks remain authoritative",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
