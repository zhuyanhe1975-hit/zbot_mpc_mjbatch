"""Sampling MPC to return a measured single-leg column to two-foot standing."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import mujoco
import numpy as np

from .model import load_model, standing_joint_targets
from .mpc import MIN_SINGLE_SUPPORT_FORCE_N
from .rollout import BatchRollout, RolloutTrace
from .single_leg import SingleLegRun, simulate_single_leg_stand


@dataclass(frozen=True)
class ReturnPlan:
    targets: np.ndarray
    cost: float
    safe_candidates: int


def _safety(trace: RolloutTrace, support_foot: int) -> np.ndarray:
    """Apply the same safety conditions during search and measured execution."""
    alone = ~trace.foot_contacts[:, :, 1 - support_foot]
    return (
        (trace.base_pos_w[:, :, 2] >= 0.22)
        & (trace.foot_foot_distance_m >= 0.002)
        & ~trace.foot_foot_contact
        & trace.foot_contacts[:, :, support_foot]
        & (~alone | (trace.foot_normal_force_n[:, :, support_foot] >= MIN_SINGLE_SUPPORT_FORCE_N))
    )


class ReturnMPC:
    """Optimize smooth corrections to a reverse reference from measured states.

    A reference is only a warm start: every candidate is simulated forwards in
    time, including a terminal settling interval, before it can be executed.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        num_candidates: int = 32,
        iterations: int = 4,
        num_threads: int = 0,
        seed: int = 4,
    ) -> None:
        if num_candidates < 8 or iterations < 1:
            raise ValueError("at least eight candidates and one iteration required")
        self.rollout = BatchRollout(
            model, num_candidates, substeps_per_control=20, num_threads=num_threads
        )
        self.iterations = iterations
        self.rng = np.random.default_rng(seed)
        self.stand = standing_joint_targets(model)

    def plan(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        ctrl: np.ndarray,
        reference: np.ndarray,
        support_foot: int,
        *,
        single_support_steps: int = 0,
    ) -> ReturnPlan:
        if support_foot not in (0, 1):
            raise ValueError("support_foot must be 0 or 1")
        reference = np.asarray(reference, dtype=float)
        if (
            reference.ndim != 2
            or reference.shape[1] != len(self.stand)
            or len(reference) < 2
            or not np.isfinite(reference).all()
        ):
            raise ValueError("reference must contain at least two finite six-joint targets")
        if not 0 <= single_support_steps <= len(reference):
            raise ValueError("single support interval must fit inside the reference")
        n = self.rollout.num_candidates
        # Three local correction pulses have zero endpoint velocity/acceleration.
        phase = np.linspace(0.0, 1.0, len(reference))
        centers = np.array([0.25, 0.5, 0.75])
        distance = np.abs(phase[:, None] - centers) / 0.25
        weight = np.clip(1.0 - distance, 0.0, 1.0)
        basis = weight**3 * (10.0 - 15.0 * weight + 6.0 * weight**2)
        settle_steps = round(1.0 / self.rollout.control_dt)
        mean = np.zeros((3, len(self.stand)))
        deviation = np.full_like(mean, 0.025)
        best = None
        best_offsets = mean.copy()
        safe_count = 0
        for _ in range(self.iterations):
            offsets = np.clip(self.rng.normal(mean, deviation, (n, *mean.shape)), -0.12, 0.12)
            offsets[0] = 0.0  # Always retain the unmodified reference.
            offsets[1] = best_offsets
            targets = reference[None] + np.einsum("tk,nkj->ntj", basis, offsets)
            terminal = np.broadcast_to(reference[-1], (n, settle_steps, len(self.stand)))
            predicted = np.concatenate((targets, terminal), axis=1)
            self.rollout.set_state(qpos, qvel, ctrl)
            trace = self.rollout.rollout(predicted, max_target_rate_rad_s=0.8)
            safe = _safety(trace, support_foot).all(axis=0)
            terminal_contacts = trace.foot_contacts[-settle_steps:]
            safe &= terminal_contacts.all(axis=(0, 2))
            safe &= (trace.foot_normal_force_n[-settle_steps:] >= 1.0).all(axis=(0, 2))
            safe &= np.abs(trace.joint_pos[-1] - self.stand).max(axis=1) < 0.025
            safe &= np.sqrt(np.mean(trace.qvel[-settle_steps:] ** 2, axis=(0, 2))) < 0.02
            if single_support_steps:
                fold = slice(0, single_support_steps)
                safe &= ~trace.foot_contacts[fold, :, 1 - support_foot].any(axis=0)
                safe &= (trace.foot_clearance_m[fold, :, 1 - support_foot] >= 0.003).all(axis=0)
            safe_count += int(safe.sum())
            speed = np.mean(trace.qvel**2, axis=(0, 2))
            terminal_speed = np.mean(trace.qvel[-settle_steps:] ** 2, axis=(0, 2))
            acceleration = np.diff(trace.applied_targets, n=2, axis=0)
            smoothness = np.mean(acceleration**2, axis=(0, 2)) / self.rollout.control_dt**4
            support_xy = trace.foot_pos_w[:, :, support_foot, :2]
            motion = np.mean(np.sum((support_xy - support_xy[:1]) ** 2, axis=2), axis=0)
            cost = speed + 20.0 * terminal_speed + 0.05 * smoothness + 1000.0 * motion
            cost += 0.1 * np.mean(offsets**2, axis=(1, 2))
            cost += 1000.0 * np.mean(~_safety(trace, support_foot), axis=0)
            cost += 100.0 * np.mean(~terminal_contacts, axis=(0, 2))
            cost += 50.0 * np.mean((trace.joint_pos[-1] - self.stand) ** 2, axis=1)
            ranked = np.argsort(np.where(safe, cost, cost + 1e6))
            if safe[ranked[0]] and (best is None or cost[ranked[0]] < best.cost):
                winner = ranked[0]
                best_offsets = offsets[winner].copy()
                best = ReturnPlan(targets[winner].copy(), float(cost[winner]), safe_count)
            elite = offsets[ranked[: max(4, n // 8)]]
            mean = elite.mean(axis=0)
            deviation = np.maximum(0.005, elite.std(axis=0))
        if best is None:
            raise RuntimeError("no safe reverse MPC candidate for two-foot landing")
        return ReturnPlan(best.targets, best.cost, safe_count)


@dataclass(frozen=True)
class DoubleLegRun:
    times_s: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    base_pos_w: np.ndarray
    foot_contacts: np.ndarray
    foot_normal_force_n: np.ndarray
    foot_foot_distance_m: np.ndarray
    foot_foot_contact: np.ndarray
    support_foot: int
    hold_start_s: float
    requested_hold_s: float
    plans: tuple[ReturnPlan, ...]
    wall_seconds: float
    stop_reason: str | None
    physics_timestep_s: float = 0.001

    @property
    def swing_foot(self) -> int:
        return 1 - self.support_foot

    def summary(self) -> dict[str, float | int | bool | str | None]:
        hold = self.times_s >= self.hold_start_s
        valid = hold & self.foot_contacts.all(axis=1)
        valid &= (self.foot_normal_force_n >= 1.0).all(axis=1)
        trailing = 0
        for frame in valid[::-1]:
            if not frame:
                break
            trailing += 1
        settled = hold & (self.times_s >= self.times_s[-1] - 1.0)
        return {
            "initial_support_foot": self.support_foot,
            "transition_seconds": round(self.hold_start_s, 3),
            "simulated_seconds": round(float(self.times_s[-1]), 3),
            "requested_hold_seconds": self.requested_hold_s,
            "achieved_continuous_double_support_seconds": round(
                trailing * float(self.times_s[0]), 3
            ),
            "final_double_support": bool(self.foot_contacts[-1].all()),
            "minimum_base_height_m": round(float(self.base_pos_w[:, 2].min()), 4),
            "minimum_foot_foot_distance_m": round(float(self.foot_foot_distance_m.min()), 4),
            "foot_foot_contact_frames": int(self.foot_foot_contact.sum()),
            "last_second_hold_qvel_rms": (
                round(float(np.sqrt(np.mean(self.qvel[settled] ** 2))), 6)
                if settled.any()
                else None
            ),
            "mpc_replans": len(self.plans),
            "safe_candidate_evaluations": sum(p.safe_candidates for p in self.plans),
            "terminated_early": self.stop_reason is not None,
            "stop_reason": self.stop_reason,
            "wall_seconds": round(self.wall_seconds, 3),
        }


def simulate_double_leg_stand(
    support_foot: int = 0,
    hold_seconds: float = 3.0,
    num_candidates: int = 32,
    iterations: int = 4,
    num_threads: int = 0,
    seed: int = 4,
    source_run: SingleLegRun | None = None,
) -> DoubleLegRun:
    """Prepare a measured column, then replan at folding and landing boundaries.

    Only the reverse motion is recorded. An existing successful SingleLegRun can
    be passed to continue from its final state without regenerating the column.
    """
    if support_foot not in (0, 1):
        raise ValueError("support_foot must be 0 or 1")
    if not np.isfinite(hold_seconds) or hold_seconds <= 0:
        raise ValueError("hold_seconds must be finite and positive")
    if num_candidates < 8 or iterations < 1:
        raise ValueError("at least eight candidates and one iteration required")
    if source_run is None:
        source_run = simulate_single_leg_stand(
            support_foot=support_foot, hold_seconds=1.0, num_threads=num_threads, seed=seed
        )
    if (
        source_run.support_foot != support_foot
        or source_run.stop_reason is not None
        or not source_run.foot_contacts[-1, support_foot]
        or source_run.foot_contacts[-1, 1 - support_foot]
        or np.abs(source_run.qpos[-1, 7:]).max() > 0.01
    ):
        raise ValueError("source_run must end in a successful column on the requested foot")
    model = load_model()
    model.opt.timestep = 0.001
    plant = BatchRollout(model, 1, substeps_per_control=20)
    plant.set_state(source_run.qpos[-1], source_run.qvel[-1], source_run.ctrl[-1])
    optimizer = ReturnMPC(model, num_candidates, iterations, num_threads, seed)
    stand = standing_joint_targets(model)
    reverse = source_run.ctrl[source_run.times_s < source_run.hold_start_s][::-1]
    # Landing has a transient even with a zero endpoint target velocity.
    # Allow one second to settle before the requested stable hold begins.
    settle_steps = round(1.0 / plant.control_dt)
    reference = np.vstack((source_run.ctrl[-1], reverse, np.tile(stand, (settle_steps + 1, 1))))
    fold_steps = round(source_run.unfold_s / plant.control_dt)
    hold_steps = int(np.ceil(hold_seconds / plant.control_dt))
    hold_start = len(reference) * plant.control_dt
    plans = []
    frames = []
    stop_reason = None
    start = time.perf_counter()
    for stage in range(2):
        # Include landing in the first horizon: a locally stable folded pose
        # alone does not guarantee that the remaining landing is feasible.
        warm = reference if stage == 0 else plans[0].targets[fold_steps:]
        try:
            plan = optimizer.plan(
                plant.qpos[0].copy(),
                plant.qvel[0].copy(),
                plant.ctrl[0].copy(),
                warm,
                support_foot,
                single_support_steps=fold_steps if stage == 0 else 0,
            )
        except RuntimeError as error:
            if not frames:
                raise
            stop_reason = f"reverse MPC stopped at {len(frames) * plant.control_dt:.2f} s: {error}"
            break
        plans.append(plan)
        targets = plan.targets[:fold_steps] if stage == 0 else plan.targets
        if stage == 1:
            targets = np.vstack((targets, np.tile(stand, (hold_steps, 1))))
        for command in targets:
            trace = plant.rollout(command[None, None], max_target_rate_rad_s=0.8)
            frames.append(
                (
                    plant.qpos[0].copy(),
                    plant.qvel[0].copy(),
                    plant.ctrl[0].copy(),
                    trace.base_pos_w[0, 0].copy(),
                    trace.foot_contacts[0, 0].copy(),
                    trace.foot_normal_force_n[0, 0].copy(),
                    trace.foot_foot_distance_m[0, 0],
                    trace.foot_foot_contact[0, 0],
                )
            )
            now = len(frames) * plant.control_dt
            if not _safety(trace, support_foot)[0, 0]:
                stop_reason = (
                    f"reverse execution violated support, height, or foot clearance at {now:.2f} s"
                )
            elif stage == 0 and (
                trace.foot_contacts[0, 0, 1 - support_foot]
                or trace.foot_clearance_m[0, 0, 1 - support_foot] < 0.003
            ):
                stop_reason = f"swing foot landed before folding completed at {now:.2f} s"
            elif now > hold_start and not (
                trace.foot_contacts[0, 0].all() and (trace.foot_normal_force_n[0, 0] >= 1.0).all()
            ):
                stop_reason = f"double support did not remain loaded at {now:.2f} s"
            if stop_reason:
                break
        if stop_reason:
            break
    arrays = [np.asarray(values) for values in zip(*frames, strict=True)]
    times = np.arange(1, len(frames) + 1) * plant.control_dt
    if stop_reason is None:
        settled = (times > hold_start) & (times >= times[-1] - 1.0)
        if np.abs(arrays[0][-1, 7:] - stand).max() >= 0.025:
            stop_reason = "measured terminal standing joint error exceeded 0.025 rad"
        elif np.sqrt(np.mean(arrays[1][settled] ** 2)) >= 0.02:
            stop_reason = "measured terminal velocity RMS exceeded 0.02"
    return DoubleLegRun(
        times,
        *arrays,
        support_foot,
        hold_start,
        hold_seconds,
        tuple(plans),
        time.perf_counter() - start,
        stop_reason,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-foot", type=int, choices=(0, 1), default=0)
    parser.add_argument("--hold-seconds", type=float, default=3.0)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error("speed must be finite and positive")
    try:
        run = simulate_double_leg_stand(
            args.support_foot,
            args.hold_seconds,
            args.candidates,
            args.iterations,
            args.threads,
            args.seed,
        )
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(run.summary(), indent=2), flush=True)
    if run.stop_reason:
        raise SystemExit(run.stop_reason)
    if args.viewer:
        from .viewer import play_viewer

        play_viewer(run, speed=args.speed, once=args.once)


if __name__ == "__main__":
    main()
