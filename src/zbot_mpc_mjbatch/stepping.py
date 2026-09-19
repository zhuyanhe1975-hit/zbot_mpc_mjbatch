"""Closed-loop stepping simulation driven by the cadence guard and step MPC."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, replace

import numpy as np

from .cadence import CadenceStateMachine, Phase
from .model import load_model
from .mpc import MIN_SINGLE_SUPPORT_FORCE_N, StepMPC, StepPlan
from .rollout import BatchRollout
from .warmstart import LinearWarmStartPolicy, WarmStartPolicy


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
    stop_reason: str | None = None

    def summary(self) -> dict[str, float | int | bool | str | None]:
        initial_xy = self.base_pos_w[0, :2]
        drift = np.linalg.norm(self.base_pos_w[:, :2] - initial_xy, axis=1)
        support = 1 - self.swing_feet
        single = self.foot_contacts[np.arange(len(support)), support] & ~self.foot_contacts[
            np.arange(len(support)), self.swing_feet
        ]
        com_margins = self.support_com_margin_m[single]
        cop_margins = self.support_cop_margin_m[single]
        cop_margins = cop_margins[np.isfinite(cop_margins)]
        support_contacts_all = self.foot_contacts[np.arange(len(support)), support]
        single_support_forces = self.foot_normal_force_n[np.arange(len(support)), support][single]
        dt = float(self.times_s[0])
        boundaries = np.searchsorted(self.times_s, self.landing_times_s, side="right")
        maximum_support_motion = 0.0
        for step_start, step_end in zip(
            np.r_[0, boundaries], np.r_[boundaries, len(self.times_s)], strict=True
        ):
            if step_start == step_end:
                continue
            stance = 1 - int(self.swing_feet[step_start])
            foot_xy = self.foot_pos_w[step_start:step_end, stance, :2]
            maximum_support_motion = max(
                maximum_support_motion,
                float(np.linalg.norm(foot_xy - foot_xy[0], axis=1).max()),
            )
        peak_clearances: list[float] = []
        clear_runs: list[float] = []
        support_contacts: list[float] = []
        all_phase_support_contacts: list[float] = []
        support_motions: list[float] = []
        support_forces: list[float] = []
        for index, end in enumerate(boundaries):
            start = 0 if index == 0 else boundaries[index - 1]
            held = np.array([phase == Phase.HOLD for phase in self.phases[start:end]])
            if not held.any():
                continue
            swing = int(self.swing_feet[start])
            support_foot = 1 - swing
            clearance = self.foot_clearance_m[start:end, swing]
            above = np.r_[False, clearance >= 0.003, False].astype(np.int8)
            changes = np.diff(above)
            run_lengths = np.where(changes == -1)[0] - np.where(changes == 1)[0]
            clear_runs.append(float(np.max(run_lengths, initial=0) * dt))
            peak_clearances.append(float(clearance.max()))
            support_contacts.append(float(self.foot_contacts[start:end, support_foot][held].mean()))
            all_phase_support_contacts.append(
                float(self.foot_contacts[start:end, support_foot].mean())
            )
            positions = self.foot_pos_w[start:end, support_foot, :2]
            support_motions.append(float(np.linalg.norm(positions - positions[0], axis=1).max()))
            support_forces.append(
                float(np.percentile(self.foot_normal_force_n[start:end, support_foot][held], 10))
            )
        return {
            "simulated_seconds": round(float(self.times_s[-1]), 3),
            "terminated_early": self.stop_reason is not None,
            "stop_reason": self.stop_reason,
            "support_contact_loss_frames": int((~support_contacts_all).sum()),
            "minimum_single_support_force_n": (
                round(float(single_support_forces.min()), 2)
                if len(single_support_forces) else None
            ),
            "weak_single_support_frames": int(
                (single_support_forces < MIN_SINGLE_SUPPORT_FORCE_N).sum()
            ),
            "maximum_support_xy_motion_all_frames_m": round(maximum_support_motion, 4),
            "completed_steps": len(self.landing_times_s),
            "maximum_planned_step_duration_s": round(
                max((plan.period_s for plan in self.plans), default=0.0), 3
            ),
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
            "minimum_step_hold_s": round(min(clear_runs), 3) if clear_runs else None,
            "minimum_step_support_contact_fraction": (
                round(min(support_contacts), 3) if support_contacts else None
            ),
            "minimum_step_support_contact_fraction_all_phases": (
                round(min(all_phase_support_contacts), 3) if all_phase_support_contacts else None
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
    frequency_hz: float = 0.5,
    seconds: float = 4.2,
    num_candidates: int = 256,
    iterations: int = 16,
    num_threads: int = 0,
    seed: int = 4,
    warm_start_policy: WarmStartPolicy | None = None,
    max_steps: int | None = None,
    minimum_hold_s: float = 0.1,
) -> StepRun:
    """Optimize each step from the latest measured state and simulate its execution."""
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be finite and positive")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive when provided")
    model = load_model()
    plant = BatchRollout(model, 1)
    plant.reset_to_stand()
    cadence = CadenceStateMachine(
        frequency_hz, first_swing_foot=1, minimum_clearance_s=minimum_hold_s
    )
    mpc = StepMPC(
        model,
        num_candidates=num_candidates,
        iterations=iterations,
        num_threads=num_threads,
        seed=seed,
        warm_start_policy=warm_start_policy,
        minimum_clearance_s=minimum_hold_s,
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
    settle_frames = 0
    stop_reason: str | None = None
    completed_frames = 0
    start = time.perf_counter()
    for i in range(count):
        state = cadence.state()
        settling = settle_frames > 0
        if not settling and (active_plan is None or active_plan.swing_foot != state.swing_foot):
            active_plan = None
            error: RuntimeError | None = None
            hold_extension_s = minimum_hold_s if minimum_hold_s >= 1.0 else 0.0
            maximum_period_s = max(2.0 + hold_extension_s, state.target_period_s)
            periods = sorted(
                {
                    min(maximum_period_s, state.target_period_s * value)
                    for value in (1.0, 1.2, 1.5, 2.0, 3.0, 4.0)
                }
            )
            for period_s in periods:
                try:
                    active_plan = mpc.plan(
                        plant.qpos[0].copy(),
                        plant.qvel[0].copy(),
                        plant.ctrl[0].copy(),
                        replace(state, target_period_s=period_s),
                        time_s=float(plant.time[0]),
                    )
                    break
                except RuntimeError as candidate_error:
                    error = candidate_error
            if active_plan is None:
                assert error is not None
                if i == 0:
                    raise error
                stop_reason = str(error)
                break
            cadence.frequency_hz = 1.0 / active_plan.period_s
            state = cadence.state()
            plans.append(active_plan)
        command = mpc.stand if settling else mpc.target(active_plan, state.elapsed_s)
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
        completed_frames = i + 1
        if settling and (
            not contacts[i].any()
            or normal_force[i, contacts[i]].sum() < MIN_SINGLE_SUPPORT_FORCE_N
        ):
            stop_reason = (
                f"settling lost loaded ground support at {times[i]:.2f} s; "
                f"contacts={contacts[i].tolist()}, forces={normal_force[i].round(1).tolist()}"
            )
            break
        if not settling and not contacts[i, support]:
            stop_reason = (
                f"support foot {support} lost contact at {times[i]:.2f} s; "
                f"other foot contact={bool(contacts[i, state.swing_foot])}, "
                f"force={normal_force[i, state.swing_foot]:.1f} N"
            )
            break
        if not settling and (
            not contacts[i, state.swing_foot]
            and normal_force[i, support] < MIN_SINGLE_SUPPORT_FORCE_N
        ):
            stop_reason = (
                f"support foot {support} carried only {normal_force[i, support]:.1f} N "
                f"during single support at {times[i]:.2f} s"
            )
            break
        if settling:
            settle_frames -= 1
            if settle_frames == 0:
                active_plan = None
            continue
        next_state = cadence.advance(
            plant.control_dt,
            (bool(contacts[i, 0]), bool(contacts[i, 1])),
            float(clearance[i, state.swing_foot]),
        )
        if next_state.swing_foot != state.swing_foot:
            landings.append(float(times[i]))
            if max_steps is not None and len(landings) >= max_steps:
                break
            if minimum_hold_s >= 1.0:
                settle_frames = int(np.ceil(1.0 / plant.control_dt))
    return StepRun(
        times[:completed_frames],
        qpos[:completed_frames],
        qvel[:completed_frames],
        ctrl[:completed_frames],
        base[:completed_frames],
        com[:completed_frames],
        feet[:completed_frames],
        contacts[:completed_frames],
        cop[:completed_frames],
        normal_force[:completed_frames],
        clearance[:completed_frames],
        foot_foot_distance[:completed_frames],
        foot_foot_contact[:completed_frames],
        com_margin[:completed_frames],
        cop_margin[:completed_frames],
        swings[:completed_frames],
        tuple(phases),
        np.asarray(landings),
        tuple(plans),
        time.perf_counter() - start,
        stop_reason,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", type=float, default=0.5, help="alternating landings per second")
    parser.add_argument("--seconds", type=float, default=4.2)
    parser.add_argument("--candidates", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--steps", type=int, help="stop successfully after this many landings")
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=0.1,
        help="required continuous swing-foot clearance per step",
    )
    parser.add_argument(
        "--warm-start",
        type=str,
        help="NPZ policy whose proposal seeds MPC; all safety checks remain active",
    )
    args = parser.parse_args()
    try:
        policy = LinearWarmStartPolicy.load(args.warm_start) if args.warm_start else None
        run = simulate_steps(
            args.frequency,
            args.seconds,
            args.candidates,
            args.iterations,
            args.threads,
            args.seed,
            policy,
            args.steps,
            args.hold_seconds,
        )
    except ValueError as error:
        parser.error(str(error))
    result = {**run.summary(), "landing_times_s": run.landing_times_s.round(3).tolist()}
    print(json.dumps(result, indent=2))
    if run.stop_reason is not None:
        raise SystemExit(run.stop_reason)
    if result["minimum_base_height_m"] < 0.2:
        raise SystemExit("stepping simulation fell")


if __name__ == "__main__":
    main()
