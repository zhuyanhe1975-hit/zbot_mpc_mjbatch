"""Contact-aware MPC from double support to a raised right foot."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np

from .cadence import CadenceStateMachine
from .model import load_model
from .mpc import StepMPC
from .rollout import BatchRollout
from .support import FootSupportGeometry


@dataclass(frozen=True)
class SingleSupportRun:
    times_s: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    foot_contacts: np.ndarray
    foot_clearance_m: np.ndarray
    foot_normal_force_n: np.ndarray
    foot_foot_distance_m: np.ndarray
    base_pos_w: np.ndarray
    support_zmp_margin_m: np.ndarray
    raised_foot_up_axis_z: np.ndarray
    raised_foot_height_m: np.ndarray
    stop_reason: str | None

    def summary(self) -> dict[str, object]:
        hold_frames = max(1, int(round(0.1 / self.times_s[0])))
        held = slice(-hold_frames, None)
        single = self.foot_contacts[:, 0] & ~self.foot_contacts[:, 1]
        lift_single = single & (self.times_s > 1.3)
        straight = bool(
            single[-1]
            and self.raised_foot_up_axis_z[-1] >= 0.98
            and self.raised_foot_height_m[-1] >= 0.55
            and np.linalg.norm(self.qpos[-1, 7:]) <= 0.15
        )
        return {
            "simulated_seconds": round(float(self.times_s[-1]), 3),
            "single_support_at_end": bool(self.foot_contacts[-1, 0] and not self.foot_contacts[-1, 1]),
            "minimum_final_100ms_clearance_m": round(float(self.foot_clearance_m[held, 1].min()), 4),
            "right_foot_off_ground_final_100ms": bool((~self.foot_contacts[held, 1]).all()),
            "minimum_single_support_force_n": (
                round(float(self.foot_normal_force_n[single, 0].min()), 2)
                if single.any() else None
            ),
            "minimum_lift_support_force_n": (
                round(float(self.foot_normal_force_n[lift_single, 0].min()), 2)
                if lift_single.any() else None
            ),
            "right_only_transfer_frames": int((
                ~self.foot_contacts[:, 0]
                & self.foot_contacts[:, 1]
                & (self.times_s <= 1.3)
            ).sum()),
            "minimum_single_support_zmp_margin_m": (
                round(float(self.support_zmp_margin_m[single].min()), 6)
                if single.any() else None
            ),
            "minimum_active_support_zmp_margin_m": round(
                float(self.support_zmp_margin_m.min()), 6
            ),
            "raised_foot_height_m": round(float(self.raised_foot_height_m[-1]), 4),
            "raised_foot_up_axis_z": round(float(self.raised_foot_up_axis_z[-1]), 4),
            "joint_distance_to_straight_rad": round(float(np.linalg.norm(self.qpos[-1, 7:])), 4),
            "straight_goal_reached": straight,
            "minimum_base_height_m": round(float(self.base_pos_w[:, 2].min()), 4),
            "minimum_foot_foot_distance_m": round(float(self.foot_foot_distance_m.min()), 4),
            "final_root_speed": round(float(np.linalg.norm(self.qvel[-1, :6])), 4),
            "final_joint_speed": round(float(np.linalg.norm(self.qvel[-1, 6:])), 4),
            "stop_reason": self.stop_reason,
        }


class LiftHoldMPC:
    """Optimize a short raised-foot target from the measured transfer state."""

    def __init__(self, plant: BatchRollout, candidates: int, iterations: int,
                 threads: int, seed: int) -> None:
        if candidates < 8 or iterations < 1:
            raise ValueError("at least eight candidates and one iteration required")
        self.rollout = BatchRollout(
            plant.model, candidates, substeps_per_control=1, num_threads=threads
        )
        self.candidates = candidates
        self.iterations = iterations
        self.rng = np.random.default_rng(seed)
        self.support_geometry = FootSupportGeometry(plant.model)

    def plan(self, qpos: np.ndarray, qvel: np.ndarray, ctrl: np.ndarray,
             time_s: float, horizon_steps: int = 60) -> np.ndarray:
        # A warm start found from the same transfer state; CEM still searches
        # and checks complete MuJoCo trajectories at every candidate.
        warm = np.array([0.02, -0.26, 0.15, -0.14, -0.06, 0.07])
        mean = warm.copy()
        spread = np.full(6, 0.1)
        best = None
        best_score = -np.inf
        for _ in range(self.iterations):
            offsets = np.clip(self.rng.normal(mean, spread, (self.candidates, 6)), -0.4, 0.4)
            offsets[0] = 0
            if best is not None:
                offsets[1] = best
            targets = np.broadcast_to(
                (ctrl + offsets)[:, None, :], (self.candidates, horizon_steps, 6)
            ).copy()
            self.rollout.set_state(qpos, qvel, ctrl, time_s)
            trace = self.rollout.rollout(targets, max_target_rate_rad_s=4.0)
            physical = (
                trace.foot_contacts[:, :, 0].all(axis=0)
                & np.all(
                    trace.foot_contacts[:, :, 1]
                    | (trace.foot_normal_force_n[:, :, 0] >= 15), axis=0
                )
                & ~trace.foot_foot_contact.any(axis=0)
                & (trace.foot_foot_distance_m.min(axis=0) >= 0.002)
                & (trace.base_pos_w[:, :, 2].min(axis=0) >= 0.22)
            )
            single = trace.foot_contacts[:, :, 0] & ~trace.foot_contacts[:, :, 1]
            zmp_margin = self.support_geometry.margin(
                np.nan_to_num(trace.foot_cop_w[:, :, 0]),
                trace.foot_geom_pos_w[:, :, 0],
                trace.foot_geom_mat_w[:, :, 0], 0,
            )
            minimum_zmp_margin = np.min(np.where(single, zmp_margin, 1.0), axis=0)
            safe = physical & (minimum_zmp_margin >= 0.0)
            clearance = trace.foot_clearance_m[-20:, :, 1].min(axis=0)
            foot_up = trace.foot_geom_mat_w[-1, :, 1].reshape(-1, 3, 3)[:, 2, 2]
            foot_height = trace.foot_pos_w[-1, :, 1, 2]
            score = (
                clearance + 0.01 * foot_up + 0.01 * foot_height
                - 0.001 * np.linalg.norm(trace.qvel[-1], axis=1)
                - 50.0 * np.maximum(0.0, -minimum_zmp_margin)
            )
            valid = np.flatnonzero(safe)
            if len(valid):
                candidate = valid[np.argmax(score[valid])]
                if score[candidate] > best_score:
                    best = offsets[candidate].copy()
                    best_score = float(score[candidate])
            physical_candidates = np.flatnonzero(physical)
            if len(physical_candidates):
                elite = physical_candidates[
                    np.argsort(score[physical_candidates])[-max(4, self.candidates // 16):]
                ]
            else:
                elite = np.argsort(score)[-4:]
            mean = offsets[elite].mean(axis=0)
            spread = np.maximum(0.02, offsets[elite].std(axis=0))
        if best is None:
            raise RuntimeError("no safe raised-foot candidate")
        return ctrl + best


def simulate_raise(transfer_s: float = 1.3, lift_s: float = 0.3,
                   candidates: int = 256, iterations: int = 40,
                   threads: int = 0, seed: int = 4) -> SingleSupportRun:
    """Plan weight transfer, measure state, then replan the final foot lift."""
    model = load_model()
    plant = BatchRollout(model, 1, substeps_per_control=1)
    plant.reset_to_stand()
    dt = plant.control_dt
    if abs(transfer_s - 1.3) > 1e-9 or abs(lift_s - 0.3) > 1e-9:
        raise ValueError("this verified transition uses 1.3 s transfer and 0.3 s lift")
    transfer_mpc = StepMPC(model, num_candidates=128, iterations=8,
                           num_threads=threads, seed=seed)
    state = CadenceStateMachine(0.5, first_swing_foot=1).state()
    transfer_plan = transfer_mpc.plan(plant.qpos[0].copy(), plant.qvel[0].copy(),
                                      plant.ctrl[0].copy(), state)
    count = int(round((transfer_s + lift_s) / dt))
    transfer_count = int(round(transfer_s / dt))
    times = (np.arange(count) + 1) * dt
    qpos = np.empty((count, model.nq))
    qvel = np.empty((count, model.nv))
    ctrl = np.empty((count, model.nu))
    contacts = np.empty((count, 2), dtype=bool)
    clearance = np.empty((count, 2))
    force = np.empty((count, 2))
    distance = np.empty(count)
    base = np.empty((count, 3))
    zmp_margin = np.empty(count)
    foot_up = np.empty(count)
    foot_height = np.empty(count)
    support_geometry = FootSupportGeometry(model)
    lift_mpc = LiftHoldMPC(plant, candidates, iterations, threads, seed)
    raised_target = None
    applied_command = plant.ctrl[0].copy()
    reason = None
    completed = 0
    for i in range(count):
        if i == transfer_count:
            try:
                raised_target = lift_mpc.plan(plant.qpos[0].copy(), plant.qvel[0].copy(),
                                               plant.ctrl[0].copy(), i * dt)
            except RuntimeError as error:
                reason = str(error)
                break
        if raised_target is None:
            if i % 4 == 0:
                command = transfer_mpc.target(transfer_plan, i * dt)
                applied_command = np.clip(command, plant.ctrl[0] - 0.08, plant.ctrl[0] + 0.08)
        else:
            applied_command = np.clip(
                raised_target, plant.ctrl[0] - 4.0 * dt, plant.ctrl[0] + 4.0 * dt
            )
        trace = plant.rollout(applied_command[None, None, :])
        qpos[i], qvel[i], ctrl[i] = plant.qpos[0], plant.qvel[0], plant.ctrl[0]
        contacts[i] = trace.foot_contacts[0, 0]
        clearance[i] = trace.foot_clearance_m[0, 0]
        force[i] = trace.foot_normal_force_n[0, 0]
        distance[i] = trace.foot_foot_distance_m[0, 0]
        base[i] = trace.base_pos_w[0, 0]
        foot_up[i] = trace.foot_geom_mat_w[0, 0, 1].reshape(3, 3)[2, 2]
        foot_height[i] = trace.foot_pos_w[0, 0, 1, 2]
        loaded = contacts[i] & np.isfinite(trace.foot_cop_w[0, 0]).all(axis=1) & (force[i] > 1)
        if loaded.all():
            total_cop = np.average(trace.foot_cop_w[0, 0], axis=0, weights=force[i])
            zmp_margin[i] = support_geometry.double_support_margin(
                total_cop, trace.foot_geom_pos_w[0, 0], trace.foot_geom_mat_w[0, 0]
            )
        elif loaded.any():
            foot = int(np.flatnonzero(loaded)[0])
            zmp_margin[i] = support_geometry.margin(
                trace.foot_cop_w[0, 0, foot],
                trace.foot_geom_pos_w[0, 0, foot],
                trace.foot_geom_mat_w[0, 0, foot], foot,
            )
        else:
            zmp_margin[i] = np.nan
        completed = i + 1
        if (not contacts[i].any()
                or (i >= transfer_count and not contacts[i, 0])
                or (i % 4 == 3 and not contacts[i, 1] and force[i, 0] < 15)):
            reason = f"support contact or load lost at {times[i]:.3f} s"
            break
        if trace.foot_foot_contact[0, 0] or distance[i] < 0.002:
            reason = f"feet too close at {times[i]:.2f} s"
            break
        if base[i, 2] < 0.22:
            reason = f"base fell at {times[i]:.2f} s"
            break
        if not np.isfinite(zmp_margin[i]) or zmp_margin[i] < 0.0:
            reason = f"ZMP left the active support polygon at {times[i]:.3f} s"
            break
    if completed == 0:
        raise RuntimeError(reason or "no simulation frames")
    return SingleSupportRun(times[:completed], qpos[:completed], qvel[:completed],
                            ctrl[:completed], contacts[:completed], clearance[:completed],
                            force[:completed], distance[:completed], base[:completed],
                            zmp_margin[:completed], foot_up[:completed], foot_height[:completed], reason)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--trajectory", type=str, help="save measured states to an NPZ file")
    parser.add_argument("--require-straight", action="store_true",
                        help="fail unless the raised leg is straight and points upward")
    args = parser.parse_args()
    run = simulate_raise(candidates=args.candidates, iterations=args.iterations,
                         threads=args.threads, seed=args.seed)
    quality = run.summary()
    if args.trajectory:
        from pathlib import Path

        output = Path(args.trajectory)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output, times_s=run.times_s, qpos=run.qpos, qvel=run.qvel,
            ctrl=run.ctrl, foot_contacts=run.foot_contacts,
            foot_clearance_m=run.foot_clearance_m,
            foot_normal_force_n=run.foot_normal_force_n,
            support_zmp_margin_m=run.support_zmp_margin_m,
            raised_foot_up_axis_z=run.raised_foot_up_axis_z,
            raised_foot_height_m=run.raised_foot_height_m,
        )
        quality["trajectory"] = str(output.resolve())
    print(json.dumps(quality, indent=2))
    if (run.stop_reason or not quality["single_support_at_end"]
            or not quality["right_foot_off_ground_final_100ms"]
            or quality["minimum_final_100ms_clearance_m"] < 0.02):
        raise SystemExit(run.stop_reason or "raised single-support goal was not reached")
    if args.require_straight and not quality["straight_goal_reached"]:
        raise SystemExit("straight upward single-support goal was not reached")


if __name__ == "__main__":
    main()
