"""Transition the ZBot from two-foot support to a sustained single-leg stand."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import mujoco
import numpy as np

from .cadence import CadenceStateMachine, phase_durations
from .model import load_model
from .mpc import MIN_SINGLE_SUPPORT_FORCE_N, StepMPC, StepPlan
from .rollout import BatchRollout


@dataclass(frozen=True)
class SingleLegRun:
    """Recorded transition and hold for one stationary support foot."""

    times_s: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    base_pos_w: np.ndarray
    com_pos_w: np.ndarray
    foot_pos_w: np.ndarray
    foot_contacts: np.ndarray
    foot_normal_force_n: np.ndarray
    foot_clearance_m: np.ndarray
    foot_foot_distance_m: np.ndarray
    foot_foot_contact: np.ndarray
    support_foot: int
    hold_start_s: float
    requested_hold_s: float
    unfold_s: float
    plan: StepPlan
    wall_seconds: float
    stop_reason: str | None = None
    physics_timestep_s: float = 0.001

    @property
    def swing_foot(self) -> int:
        return 1 - self.support_foot

    def summary(self) -> dict[str, float | int | bool | str | None]:
        hold = self.times_s >= self.hold_start_s
        support_contact = self.foot_contacts[:, self.support_foot]
        swing_contact = self.foot_contacts[:, self.swing_foot]
        single_support = support_contact & ~swing_contact
        valid_hold = (
            hold
            & single_support
            & (self.foot_clearance_m[:, self.swing_foot] >= 0.003)
            & (
                self.foot_normal_force_n[:, self.support_foot]
                >= MIN_SINGLE_SUPPORT_FORCE_N
            )
        )
        trailing_frames = 0
        for valid in valid_hold[::-1]:
            if not valid:
                break
            trailing_frames += 1
        hold_clearance = self.foot_clearance_m[hold, self.swing_foot]
        hold_force = self.foot_normal_force_n[hold, self.support_foot]
        support_xy = self.foot_pos_w[:, self.support_foot, :2]
        settled = hold & (self.times_s >= self.times_s[-1] - 1.0)
        return {
            "support_foot": self.support_foot,
            "swing_foot": self.swing_foot,
            "simulated_seconds": round(float(self.times_s[-1]), 3),
            "transition_seconds": round(self.hold_start_s, 3),
            "unfold_seconds": round(self.unfold_s, 3),
            "requested_hold_seconds": round(self.requested_hold_s, 3),
            "achieved_continuous_hold_seconds": round(
                trailing_frames * float(self.times_s[0]), 3
            ),
            "terminated_early": self.stop_reason is not None,
            "stop_reason": self.stop_reason,
            "final_single_support": bool(single_support[-1]),
            "minimum_hold_clearance_m": (
                round(float(hold_clearance.min()), 4) if len(hold_clearance) else None
            ),
            "minimum_hold_support_force_n": (
                round(float(hold_force.min()), 2) if len(hold_force) else None
            ),
            "minimum_base_height_m": round(float(self.base_pos_w[:, 2].min()), 4),
            "final_foot_height_difference_m": round(
                float(abs(self.foot_pos_w[-1, 1, 2] - self.foot_pos_w[-1, 0, 2])), 4
            ),
            "final_max_abs_joint_rad": round(float(np.abs(self.qpos[-1, 7:]).max()), 5),
            "last_second_hold_qvel_rms": (
                round(float(np.sqrt(np.mean(self.qvel[settled] ** 2))), 6)
                if settled.any() else None
            ),
            "last_second_hold_base_motion_m": (
                round(float(np.linalg.norm(np.ptp(self.base_pos_w[settled], axis=0))), 6)
                if settled.any() else None
            ),
            "maximum_support_xy_motion_m": round(
                float(np.linalg.norm(support_xy - support_xy[0], axis=1).max()), 4
            ),
            "minimum_foot_foot_distance_m": round(
                float(self.foot_foot_distance_m.min()), 4
            ),
            "foot_foot_contact_frames": int(self.foot_foot_contact.sum()),
            "wall_seconds": round(self.wall_seconds, 3),
        }


def _support_relative_com_xy(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_target: np.ndarray,
    support_foot: int,
) -> np.ndarray:
    """Return the kinematic COM projection in the support foot frame."""
    data.qpos[:] = model.key("stand").qpos
    data.qpos[7:] = joint_target
    mujoco.mj_forward(model, data)
    support = data.body(f"foot_{support_foot}")
    rotation = support.xmat.reshape(3, 3)
    com = data.subtree_com[model.body("foot_0").id]
    return (rotation.T @ (com - support.xpos))[:2]


def _balanced_unfold_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    nominal: np.ndarray,
    desired_com_xy: np.ndarray,
    support_foot: int,
) -> np.ndarray:
    """Correct a zeroing target so its COM stays above the planted foot."""
    target = nominal.copy()
    regularization = 0.3
    epsilon = 1e-5
    for _ in range(8):
        com_xy = _support_relative_com_xy(model, data, target, support_foot)
        jacobian = np.empty((2, model.nu))
        for joint in range(model.nu):
            perturbed = target.copy()
            perturbed[joint] += epsilon
            jacobian[:, joint] = (
                _support_relative_com_xy(model, data, perturbed, support_foot) - com_xy
            ) / epsilon
        system = np.vstack((regularization * np.eye(model.nu), 20.0 * jacobian))
        residual = np.r_[
            regularization * (nominal - target),
            20.0 * (desired_com_xy - com_xy),
        ]
        delta = np.linalg.solve(
            system.T @ system + 1e-5 * np.eye(model.nu), system.T @ residual
        )
        target += np.clip(delta, -0.15, 0.15)
    return target


def simulate_single_leg_stand(
    support_foot: int = 0,
    hold_seconds: float = 3.0,
    num_candidates: int = 64,
    iterations: int = 8,
    num_threads: int = 0,
    seed: int = 4,
    unfold_seconds: float = 6.0,
) -> SingleLegRun:
    """Shift the COM, lift the other foot, and unfold into a zero-joint column.

    The MPC first validates a complete three-second lift-and-land trajectory.  We
    execute it through the middle of single support, then zero all six joints while
    a COM Jacobian correction keeps the projection centered over the support foot.
    """
    if support_foot not in (0, 1):
        raise ValueError("support_foot must be 0 or 1")
    if not np.isfinite(hold_seconds) or hold_seconds <= 0:
        raise ValueError("hold_seconds must be finite and positive")
    if not np.isfinite(unfold_seconds) or unfold_seconds <= 0:
        raise ValueError("unfold_seconds must be finite and positive")

    model = load_model()
    # Resolve contact rocking of the tall column without changing the 20 ms
    # control period. Planning and execution use the same refined physics.
    model.opt.timestep = 0.001
    plant = BatchRollout(model, 1, substeps_per_control=20)
    plant.reset_to_stand()
    period_s = 3.0
    swing_foot = 1 - support_foot
    cadence = CadenceStateMachine(
        1.0 / period_s,
        first_swing_foot=swing_foot,
        minimum_clearance_s=1.0,
    )
    mpc = StepMPC(
        model,
        num_candidates=num_candidates,
        iterations=iterations,
        num_threads=num_threads,
        seed=seed,
        minimum_clearance_s=1.0,
        substeps_per_control=20,
    )
    plan = mpc.plan(
        plant.qpos[0].copy(),
        plant.qvel[0].copy(),
        plant.ctrl[0].copy(),
        cadence.state(),
    )
    transfer_s, lift_s, lower_s = phase_durations(period_s)
    hold_start = transfer_s + lift_s
    hold_end = period_s - lower_s
    freeze_s = 0.5 * (hold_start + hold_end)
    column_start_s = freeze_s + unfold_seconds
    count = int(np.ceil((column_start_s + hold_seconds) / plant.control_dt))
    times = np.arange(1, count + 1) * plant.control_dt
    qpos = np.empty((count, model.nq))
    qvel = np.empty((count, model.nv))
    ctrl = np.empty((count, model.nu))
    base = np.empty((count, 3))
    com = np.empty((count, 3))
    feet = np.empty((count, 2, 3))
    contacts = np.empty((count, 2), dtype=np.bool_)
    normal_force = np.empty((count, 2))
    clearance = np.empty((count, 2))
    foot_distance = np.empty(count)
    foot_contact = np.empty(count, dtype=np.bool_)
    completed_frames = 0
    stop_reason: str | None = None
    kinematics = mujoco.MjData(model)
    lifted_target = mpc.target(plan, freeze_s)
    lifted_com_xy = _support_relative_com_xy(
        model, kinematics, lifted_target, support_foot
    )
    column_com_xy = _support_relative_com_xy(
        model, kinematics, np.zeros(model.nu), support_foot
    )
    start = time.perf_counter()

    for index in range(count):
        command_time = index * plant.control_dt
        if command_time < freeze_s:
            command = mpc.target(plan, command_time)
            target_rate = 4.0
        elif command_time < column_start_s:
            progress = np.clip((command_time - freeze_s) / unfold_seconds, 0.0, 1.0)
            # Zero velocity and acceleration at both ends of the unfold.
            progress = progress**3 * (10.0 - 15.0 * progress + 6.0 * progress**2)
            nominal = lifted_target * (1.0 - progress)
            desired_com_xy = (
                lifted_com_xy * (1.0 - progress) + column_com_xy * progress
            )
            command = _balanced_unfold_target(
                model, kinematics, nominal, desired_com_xy, support_foot
            )
            target_rate = 0.8
        else:
            command = np.zeros(model.nu)
            target_rate = 0.8
        trace = plant.rollout(
            command[None, None, :],
            max_target_rate_rad_s=target_rate,
        )
        qpos[index] = plant.qpos[0]
        qvel[index] = plant.qvel[0]
        ctrl[index] = plant.ctrl[0]
        base[index] = trace.base_pos_w[0, 0]
        com[index] = trace.com_pos_w[0, 0]
        feet[index] = trace.foot_pos_w[0, 0]
        contacts[index] = trace.foot_contacts[0, 0]
        normal_force[index] = trace.foot_normal_force_n[0, 0]
        clearance[index] = trace.foot_clearance_m[0, 0]
        foot_distance[index] = trace.foot_foot_distance_m[0, 0]
        foot_contact[index] = trace.foot_foot_contact[0, 0]
        completed_frames = index + 1

        if foot_contact[index] or foot_distance[index] < 0.002:
            stop_reason = f"feet approached or collided at {times[index]:.2f} s"
        elif base[index, 2] < 0.22:
            stop_reason = f"base fell below 0.22 m at {times[index]:.2f} s"
        elif not contacts[index, support_foot]:
            stop_reason = f"support foot {support_foot} lost contact at {times[index]:.2f} s"
        elif (
            not contacts[index, swing_foot]
            and normal_force[index, support_foot] < MIN_SINGLE_SUPPORT_FORCE_N
        ):
            stop_reason = (
                f"support foot {support_foot} carried only "
                f"{normal_force[index, support_foot]:.1f} N at {times[index]:.2f} s"
            )
        elif index * plant.control_dt >= freeze_s and (
            contacts[index, swing_foot] or clearance[index, swing_foot] < 0.003
        ):
            stop_reason = f"swing foot {swing_foot} did not remain lifted at {times[index]:.2f} s"
        if stop_reason is not None:
            break

    return SingleLegRun(
        times[:completed_frames],
        qpos[:completed_frames],
        qvel[:completed_frames],
        ctrl[:completed_frames],
        base[:completed_frames],
        com[:completed_frames],
        feet[:completed_frames],
        contacts[:completed_frames],
        normal_force[:completed_frames],
        clearance[:completed_frames],
        foot_distance[:completed_frames],
        foot_contact[:completed_frames],
        support_foot,
        column_start_s,
        hold_seconds,
        unfold_seconds,
        plan,
        time.perf_counter() - start,
        stop_reason,
        physics_timestep_s=float(model.opt.timestep),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--support-foot",
        type=int,
        choices=(0, 1),
        default=0,
        help="foot that remains planted",
    )
    parser.add_argument("--hold-seconds", type=float, default=3.0)
    parser.add_argument("--unfold-seconds", type=float, default=6.0)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=4)
    args = parser.parse_args()
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
    print(json.dumps(run.summary(), indent=2))
    if run.stop_reason is not None:
        raise SystemExit(run.stop_reason)


if __name__ == "__main__":
    main()
