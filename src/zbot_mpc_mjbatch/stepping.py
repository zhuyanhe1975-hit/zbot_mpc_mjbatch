"""Closed-loop stepping simulation driven by the cadence guard and step MPC."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import numpy as np

from .cadence import CadenceStateMachine, Phase
from .model import load_model
from .mpc import StepMPC, StepPlan
from .rollout import BatchRollout


@dataclass(frozen=True)
class StepRun:
    times_s: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    base_pos_w: np.ndarray
    com_pos_w: np.ndarray
    foot_pos_w: np.ndarray
    foot_contacts: np.ndarray
    foot_cop_w: np.ndarray
    foot_normal_force_n: np.ndarray
    foot_clearance_m: np.ndarray
    foot_foot_distance_m: np.ndarray
    foot_foot_contact: np.ndarray
    support_com_margin_m: np.ndarray
    support_cop_margin_m: np.ndarray
    swing_feet: np.ndarray
    phases: tuple[Phase, ...]
    landing_times_s: np.ndarray
    plans: tuple[StepPlan, ...]
    wall_seconds: float

    def summary(self) -> dict[str, float | int | bool | None]:
        initial_xy = self.base_pos_w[0, :2]
        drift = np.linalg.norm(self.base_pos_w[:, :2] - initial_xy, axis=1)
        support = 1 - self.swing_feet
        single = self.foot_contacts[np.arange(len(support)), support] & ~self.foot_contacts[
            np.arange(len(support)), self.swing_feet
        ]
        com_margins = self.support_com_margin_m[single]
        cop_margins = self.support_cop_margin_m[single]
        cop_margins = cop_margins[np.isfinite(cop_margins)]
        dt = float(self.times_s[0])
        boundaries = np.searchsorted(self.times_s, self.landing_times_s, side="right")
        peak_clearances: list[float] = []
        clear_runs: list[float] = []
        support_contacts: list[float] = []
        support_motions: list[float] = []
        support_forces: list[float] = []
        for start, end in zip(np.r_[0, boundaries[:-1]], boundaries, strict=True):
            held = np.array([phase == Phase.HOLD for phase in self.phases[start:end]])
            if not held.any():
                continue
            swing = int(self.swing_feet[start])
            support_foot = 1 - swing
            clearance = self.foot_clearance_m[start:end, swing][held]
            above = np.r_[False, clearance >= 0.003, False].astype(np.int8)
            changes = np.diff(above)
            run_lengths = np.where(changes == -1)[0] - np.where(changes == 1)[0]
            clear_runs.append(float(np.max(run_lengths, initial=0) * dt))
            peak_clearances.append(float(clearance.max()))
            support_contacts.append(float(self.foot_contacts[start:end, support_foot][held].mean()))
            positions = self.foot_pos_w[start:end, support_foot, :2][held]
            support_motions.append(float(np.linalg.norm(positions - positions[0], axis=1).max()))
            support_forces.append(
                float(np.percentile(self.foot_normal_force_n[start:end, support_foot][held], 10))
            )
        return {
            "simulated_seconds": round(float(self.times_s[-1]), 3),
            "completed_steps": len(self.landing_times_s),
            "minimum_base_height_m": round(float(self.base_pos_w[:, 2].min()), 4),
            "maximum_xy_drift_m": round(float(drift.max()), 4),
            "wall_seconds": round(self.wall_seconds, 3),
            "both_feet_contact_at_end": bool(self.foot_contacts[-1].all()),
            "minimum_single_support_com_margin_m": (
                round(float(com_margins.min()), 4) if len(com_margins) else None
            ),
            "minimum_single_support_cop_margin_m": (
                round(float(cop_margins.min()), 4) if len(cop_margins) else None
            ),
            "minimum_step_peak_clearance_m": (
                round(min(peak_clearances), 4) if peak_clearances else None
            ),
            "minimum_step_3mm_clearance_run_s": (
                round(min(clear_runs), 3) if clear_runs else None
            ),
            "minimum_step_support_contact_fraction": (
                round(min(support_contacts), 3) if support_contacts else None
            ),
            "maximum_step_support_xy_motion_m": (
                round(max(support_motions), 4) if support_motions else None
            ),
            "minimum_step_support_force_p10_n": (
                round(min(support_forces), 2) if support_forces else None
            ),
            "minimum_foot_foot_distance_m": round(float(self.foot_foot_distance_m.min()), 4),
            "foot_foot_contact_frames": int(self.foot_foot_contact.sum()),
        }


def simulate_steps(
    frequency_hz: float = 1.0,
    seconds: float = 5.0,
    num_candidates: int = 256,
    iterations: int = 16,
    num_threads: int = 0,
    seed: int = 4,
) -> StepRun:
    """Optimize each step from the latest measured state and simulate its execution."""
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be finite and positive")
    model = load_model()
    plant = BatchRollout(model, 1)
    plant.reset_to_stand()
    cadence = CadenceStateMachine(frequency_hz, first_swing_foot=1)
    mpc = StepMPC(
        model,
        num_candidates=num_candidates,
        iterations=iterations,
        num_threads=num_threads,
        seed=seed,
    )
    count = int(np.ceil(seconds / plant.control_dt))
    times = np.arange(1, count + 1) * plant.control_dt
    qpos = np.empty((count, model.nq))
    qvel = np.empty((count, model.nv))
    ctrl = np.empty((count, model.nu))
    base = np.empty((count, 3))
    com = np.empty((count, 3))
    feet = np.empty((count, 2, 3))
    contacts = np.empty((count, 2), dtype=np.bool_)
    cop = np.empty((count, 2, 2))
    normal_force = np.empty((count, 2))
    clearance = np.empty((count, 2))
    foot_foot_distance = np.empty(count)
    foot_foot_contact = np.empty(count, dtype=np.bool_)
    com_margin = np.empty(count)
    cop_margin = np.empty(count)
    swings = np.empty(count, dtype=np.int8)
    phases: list[Phase] = []
    landings: list[float] = []
    plans: list[StepPlan] = []
    active_plan: StepPlan | None = None
    start = time.perf_counter()
    for i in range(count):
        state = cadence.state()
        if active_plan is None or active_plan.swing_foot != state.swing_foot:
            active_plan = mpc.plan(
                plant.qpos[0].copy(),
                plant.qvel[0].copy(),
                plant.ctrl[0].copy(),
                state,
                time_s=float(plant.time[0]),
            )
            plans.append(active_plan)
        command = mpc.target(active_plan, state.elapsed_s)
        trace = plant.rollout(command[None, None, :], max_target_rate_rad_s=4.0)
        qpos[i] = plant.qpos[0]
        qvel[i] = plant.qvel[0]
        ctrl[i] = plant.ctrl[0]
        base[i] = trace.base_pos_w[0, 0]
        com[i] = trace.com_pos_w[0, 0]
        feet[i] = trace.foot_pos_w[0, 0]
        contacts[i] = trace.foot_contacts[0, 0]
        cop[i] = trace.foot_cop_w[0, 0]
        normal_force[i] = trace.foot_normal_force_n[0, 0]
        clearance[i] = trace.foot_clearance_m[0, 0]
        foot_foot_distance[i] = trace.foot_foot_distance_m[0, 0]
        foot_foot_contact[i] = trace.foot_foot_contact[0, 0]
        if foot_foot_contact[i] or foot_foot_distance[i] < 0:
            raise RuntimeError(f"feet collided at simulation time {times[i]:.3f} s")
        support = state.support_foot
        geom_pos = trace.foot_geom_pos_w[0, 0, support]
        geom_mat = trace.foot_geom_mat_w[0, 0, support]
        com_margin[i] = mpc.support_geometry.margin(com[i, :2], geom_pos, geom_mat, support)
        cop_margin[i] = (
            mpc.support_geometry.margin(cop[i, support], geom_pos, geom_mat, support)
            if np.isfinite(cop[i, support]).all()
            else np.nan
        )
        swings[i] = state.swing_foot
        phases.append(state.phase)
        next_state = cadence.advance(
            plant.control_dt, (bool(contacts[i, 0]), bool(contacts[i, 1]))
        )
        if next_state.swing_foot != state.swing_foot:
            landings.append(float(times[i]))
    return StepRun(
        times,
        qpos,
        qvel,
        ctrl,
        base,
        com,
        feet,
        contacts,
        cop,
        normal_force,
        clearance,
        foot_foot_distance,
        foot_foot_contact,
        com_margin,
        cop_margin,
        swings,
        tuple(phases),
        np.asarray(landings),
        tuple(plans),
        time.perf_counter() - start,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", type=float, default=1.0, help="alternating landings per second")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--candidates", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=4)
    args = parser.parse_args()
    try:
        run = simulate_steps(
            args.frequency,
            args.seconds,
            args.candidates,
            args.iterations,
            args.threads,
            args.seed,
        )
    except ValueError as error:
        parser.error(str(error))
    result = {**run.summary(), "landing_times_s": run.landing_times_s.round(3).tolist()}
    print(json.dumps(result, indent=2))
    if result["minimum_base_height_m"] < 0.2:
        raise SystemExit("stepping simulation fell")


if __name__ == "__main__":
    main()
