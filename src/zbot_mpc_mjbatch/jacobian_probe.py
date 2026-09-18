"""Measure how well a standing COM Jacobian predicts contact-rich step rollouts."""

from __future__ import annotations

import argparse
import json

import mujoco
import numpy as np

from .model import load_model
from .mpc import StepMPC


def _correlation(predicted: np.ndarray, measured: np.ndarray) -> float | None:
    if np.std(predicted) < 1e-12 or np.std(measured) < 1e-12:
        return None
    return round(float(np.corrcoef(predicted, measured)[0, 1]), 3)


def probe(num_candidates: int = 128, num_threads: int = 8, seed: int = 4) -> dict:
    """Compare local COM derivatives with the first MPC proposal population."""
    model = load_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
    mujoco.mj_forward(model, data)
    root = model.body("foot_0").id
    jac_com = np.zeros((3, model.nv))
    jac_support = np.zeros((3, model.nv))
    mujoco.mj_jacSubtreeCom(model, data, jac_com, root)
    mujoco.mj_jacBody(model, data, jac_support, None, root)
    relative_com_jacobian = (jac_com - jac_support)[:2, 6:]
    initial_com_relative = data.subtree_com[root, :2] - data.body("foot_0").xpos[:2]

    mpc = StepMPC(model, num_candidates, iterations=1, num_threads=num_threads, seed=seed)
    warm = np.r_[mpc.warm_offsets[1], np.zeros(12)]
    bounds = np.r_[np.full(6, 0.6), np.full(12, 0.4)]
    offsets = np.clip(
        mpc.rng.normal(warm, 0.15, (num_candidates, 18)), -bounds, bounds
    )
    times = np.arange(100) * mpc.rollout.control_dt
    swing = mpc.amplitude(times, 2.0)
    transfer = mpc.transfer_amplitude(times, 2.0)
    hold = mpc.hold_amplitude(times, 2.0)
    targets = (
        mpc.stand[None, None, :]
        + offsets[:, None, :6] * swing[None, :, None]
        + offsets[:, None, 6:12] * transfer[None, :, None]
        + offsets[:, None, 12:] * hold[None, :, None]
    )
    mpc.rollout.reset_to_stand()
    trace = mpc.rollout.rollout(targets, max_target_rate_rad_s=4.0)
    samples = {}
    for index in (9, 69):  # 0.20 s transfer and 1.40 s single support
        joint_delta = (
            offsets[:, :6] * swing[index]
            + offsets[:, 6:12] * transfer[index]
            + offsets[:, 12:] * hold[index]
        )
        predicted = joint_delta @ relative_com_jacobian.T
        measured = (
            trace.com_pos_w[index, :, :2]
            - trace.foot_pos_w[index, :, 0, :2]
            - initial_com_relative
        )
        samples[f"{times[index] + mpc.rollout.control_dt:.2f}s"] = {
            "com_x_correlation": _correlation(predicted[:, 0], measured[:, 0]),
            "com_y_correlation": _correlation(predicted[:, 1], measured[:, 1]),
            "both_feet_in_contact": int(trace.foot_contacts[index].all(axis=1).sum()),
        }
    return {
        "candidate_count": num_candidates,
        "samples": samples,
        "upright_with_support_contact_full_step": int(
            ((trace.base_pos_w[:, :, 2].min(axis=0) > 0.22)
             & trace.foot_contacts[:, :, 0].all(axis=0)).sum()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=4)
    args = parser.parse_args()
    if args.candidates < 8 or args.threads < 0:
        parser.error("at least eight candidates and a nonnegative thread count required")
    print(json.dumps(probe(args.candidates, args.threads, args.seed), indent=2))


if __name__ == "__main__":
    main()
