"""Verify that the copied model and mjbatch hold a standing ZBot stable."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .model import load_model, standing_joint_targets
from .rollout import BatchRollout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-sims", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    if args.num_sims <= 0 or args.seconds <= 0:
        parser.error("--num-sims and --seconds must be positive")
    model = load_model()
    simulator = BatchRollout(model, args.num_sims, num_threads=args.threads)
    simulator.reset_to_stand()
    stand = standing_joint_targets(model)
    horizon = int(np.ceil(args.seconds / simulator.control_dt))
    controls = np.broadcast_to(stand, (args.num_sims, horizon, model.nu)).copy()
    start = time.perf_counter()
    trace = simulator.rollout(controls)
    wall_s = time.perf_counter() - start
    initial_base = trace.base_pos_w[0]
    final_base = trace.base_pos_w[-1]
    drift = np.linalg.norm(final_base[:, :2] - initial_base[:, :2], axis=1)
    result = {
        "num_sims": args.num_sims,
        "simulated_seconds": horizon * simulator.control_dt,
        "wall_seconds": round(wall_s, 4),
        "minimum_base_height_m": round(float(trace.base_pos_w[:, :, 2].min()), 4),
        "maximum_xy_drift_m": round(float(drift.max()), 4),
        "both_feet_contact_at_end": bool(trace.foot_contacts[-1].all()),
    }
    print(json.dumps(result, indent=2))
    if (
        not np.isfinite(trace.base_pos_w).all()
        or result["minimum_base_height_m"] < 0.2
        or result["maximum_xy_drift_m"] > 0.05
        or not result["both_feet_contact_at_end"]
    ):
        raise SystemExit("standing batch rollout failed")


if __name__ == "__main__":
    main()
