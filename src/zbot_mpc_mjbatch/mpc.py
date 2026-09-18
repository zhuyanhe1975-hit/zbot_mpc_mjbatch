"""Step-level sampling MPC using parallel MuJoCo rollouts."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .cadence import StepState
from .model import standing_joint_targets
from .rollout import BatchRollout, RolloutTrace
from .support import FootSupportGeometry


@dataclass(frozen=True)
class StepPlan:
    """One optimized joint-target pulse, followed by a standing target."""

    swing_foot: int
    period_s: float
    joint_offset: np.ndarray
    transfer_offset: np.ndarray
    cost: float


class StepMPC:
    """Replan at each contact-validated step boundary.

    Each candidate uses six peak joint-position offsets. Fast steps also use six
    brief transfer offsets to prepare single support before the swing pulse.
    Candidates are evaluated for a complete step plus a settling interval; only
    the current step is executed.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        num_candidates: int = 256,
        iterations: int = 16,
        num_threads: int = 0,
        seed: int = 0,
        settle_s: float = 0.5,
    ) -> None:
        if num_candidates < 8 or iterations < 1 or settle_s <= 0:
            raise ValueError("at least eight candidates, one iteration, and positive settling time required")
        self.rollout = BatchRollout(model, num_candidates, num_threads=num_threads)
        self.num_candidates = num_candidates
        self.iterations = iterations
        self.settle_s = settle_s
        self.stand = standing_joint_targets(model)
        self.rng = np.random.default_rng(seed)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
        mujoco.mj_forward(model, data)
        self.home_base_quat = data.xquat[model.body("base").id].copy()
        self.home_com = data.subtree_com[model.body("foot_0").id].copy()
        self.support_geometry = FootSupportGeometry(model)
        # Feasible warm starts; the optimizer is free to move away from them.
        self.warm_offsets = np.array(
            [
                [-0.11, 0.20, -0.07, 0.12, 0.13, -0.05],
                [0.15, -0.10, -0.17, -0.15, 0.23, 0.05],
            ]
        )
        self.previous_offsets: list[np.ndarray | None] = [None, None]

    @staticmethod
    def amplitude(elapsed_s: np.ndarray | float, period_s: float) -> np.ndarray:
        """Rise during lift, hold on one foot, and lower before touchdown."""
        t = np.asarray(elapsed_s)
        transfer_s = min(0.08, 0.08 * period_s)
        rise_s = min(0.12, 0.2 * period_s)
        return np.clip(
            np.minimum((t - transfer_s) / rise_s, (period_s - t) / 0.12), 0.0, 1.0
        )

    @staticmethod
    def transfer_amplitude(elapsed_s: np.ndarray | float, period_s: float) -> np.ndarray:
        """Short early pulse that shifts weight before a fast swing."""
        t = np.asarray(elapsed_s)
        if period_s >= 0.8:
            return np.zeros_like(t, dtype=np.float64)
        peak_s = 0.08 * period_s
        end_s = peak_s + 0.14
        return np.clip(np.minimum(t / peak_s, (end_s - t) / (end_s - peak_s)), 0.0, 1.0)

    def target(self, plan: StepPlan, elapsed_s: float) -> np.ndarray:
        return (
            self.stand
            + plan.joint_offset * self.amplitude(elapsed_s, plan.period_s)
            + plan.transfer_offset * self.transfer_amplitude(elapsed_s, plan.period_s)
        )

    def _cost(self, trace: RolloutTrace, swing: int, period_steps: int) -> np.ndarray:
        base = trace.base_pos_w
        contacts = trace.foot_contacts
        support = 1 - swing
        lift = slice(max(0, int(0.25 * period_steps)), max(1, int(0.75 * period_steps)))
        late = slice(period_steps, None)
        airborne_fraction = np.mean(~contacts[lift, :, swing], axis=0)
        swing_clearance = trace.foot_clearance_m[lift, :, swing]
        clearance_error = np.maximum(0.0, 0.008 - swing_clearance)
        clear_fraction = np.mean(swing_clearance >= 0.004, axis=0)
        support_force = trace.foot_normal_force_n[lift, :, support]
        support_xy = trace.foot_geom_pos_w[lift, :, support, :2]
        support_motion = support_xy - support_xy[0:1]
        xy_error = trace.com_pos_w[:, :, :2] - self.home_com[:2]
        final_xy_error = trace.com_pos_w[-1, :, :2] - self.home_com[:2]
        min_height = base[:, :, 2].min(axis=0)
        final_height = base[-1, :, 2]
        qvel = self.rollout.qvel
        geom_pos = trace.foot_geom_pos_w[lift, :, support]
        geom_mat = trace.foot_geom_mat_w[lift, :, support]
        com_margin = self.support_geometry.margin(
            trace.com_pos_w[lift, :, :2], geom_pos, geom_mat, support
        )
        cop_margin = self.support_geometry.margin(
            np.nan_to_num(trace.foot_cop_w[lift, :, support]), geom_pos, geom_mat, support
        )
        single_support = contacts[lift, :, support] & ~contacts[lift, :, swing]
        valid_cop = single_support & (trace.foot_normal_force_n[lift, :, support] > 1.0)
        com_penalty = np.sum(single_support * np.maximum(0.0, -com_margin) ** 2, axis=0)
        com_penalty /= np.maximum(single_support.sum(axis=0), 1)
        cop_penalty = np.sum(valid_cop * np.maximum(0.0, -cop_margin) ** 2, axis=0)
        cop_penalty /= np.maximum(valid_cop.sum(axis=0), 1)
        foot_gap = trace.foot_foot_distance_m[:period_steps]
        landing = period_steps - 1
        next_support_margin = self.support_geometry.margin(
            trace.com_pos_w[landing, :, :2],
            trace.foot_geom_pos_w[landing, :, swing],
            trace.foot_geom_mat_w[landing, :, swing],
            swing,
        )
        base_alignment = np.abs(np.sum(trace.base_quat_w[-1] * self.home_base_quat, axis=1))
        cost = (
            3.0 * np.maximum(0.0, 0.7 - airborne_fraction) ** 2
            + 10000.0 * np.mean(clearance_error**2, axis=0)
            + 3.0 * np.maximum(0.0, 0.7 - clear_fraction) ** 2
            + 6.0 * np.mean(~contacts[lift, :, support], axis=0)
            + 0.01 * np.mean(np.maximum(0.0, 10.0 - support_force) ** 2, axis=0)
            + 1000.0 * np.mean(np.sum(support_motion**2, axis=2), axis=0)
            + 1.0 * np.mean(~contacts[late, :, swing], axis=0)
            + 0.5 * np.mean(~contacts[late, :, support], axis=0)
            + 1000.0 * np.maximum(0.0, 0.22 - min_height) ** 2
            + 1000.0 * np.maximum(0.0, 0.22 - final_height) ** 2
            + 20.0 * np.mean(np.sum(xy_error**2, axis=2), axis=0)
            + 120.0 * np.sum(final_xy_error**2, axis=1)
            + 2.0 * (1.0 - base_alignment) ** 2
            + 0.01 * np.sum(qvel**2, axis=1)
            + 400.0 * com_penalty
            + 1000.0 * cop_penalty
            + 800.0 * np.maximum(0.0, -0.005 - next_support_margin) ** 2
            + 0.01 * np.maximum(0.0, 8.0 - trace.foot_normal_force_n[landing, :, swing]) ** 2
            + 3000.0 * np.mean(np.maximum(0.0, 0.01 - foot_gap) ** 2, axis=0)
        )
        return np.nan_to_num(cost, nan=1e6, posinf=1e6)

    def plan(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        ctrl: np.ndarray,
        state: StepState,
        time_s: float = 0.0,
    ) -> StepPlan:
        """Find the safest candidate from the measured state at a step boundary."""
        if state.elapsed_s >= state.target_period_s:
            raise ValueError("cannot plan a new step after its landing deadline")
        dt = self.rollout.control_dt
        remaining = state.target_period_s - state.elapsed_s
        period_steps = max(1, int(np.ceil(remaining / dt)))
        horizon = period_steps + int(np.ceil(self.settle_s / dt))
        times = state.elapsed_s + np.arange(horizon) * dt
        amplitude = self.amplitude(times, state.target_period_s)
        transfer_amplitude = self.transfer_amplitude(times, state.target_period_s)
        fast = state.target_period_s < 0.8
        warm = np.r_[self.warm_offsets[state.swing_foot], np.zeros(6)] if fast else self.warm_offsets[state.swing_foot]
        bounds = np.r_[np.full(6, 0.6), np.full(6, 0.4)] if fast else np.full(6, 0.6)
        previous_offset = self.previous_offsets[state.swing_foot]
        if previous_offset is not None and previous_offset.shape != warm.shape:
            previous_offset = None
        mean = (
            previous_offset.copy()
            if previous_offset is not None
            else warm.copy()
        )
        deviation = np.full(warm.shape, 0.15)
        elite_count = max(4, self.num_candidates // 16)
        best_cost = np.inf
        best_offset: np.ndarray | None = None
        for _ in range(self.iterations):
            offsets = np.clip(
                self.rng.normal(mean, deviation, (self.num_candidates, len(warm))),
                -bounds,
                bounds,
            )
            offsets[0] = mean
            if best_offset is not None:
                offsets[1] = best_offset
            offsets[2] = warm
            if previous_offset is not None:
                offsets[3] = previous_offset
            targets = self.stand[None, None, :] + offsets[:, None, :6] * amplitude[None, :, None]
            if fast:
                targets += offsets[:, None, 6:] * transfer_amplitude[None, :, None]
            self.rollout.set_state(qpos, qvel, ctrl, time_s=time_s)
            trace = self.rollout.rollout(targets, max_target_rate_rad_s=4.0)
            costs = self._cost(trace, state.swing_foot, period_steps)
            costs += 0.03 * np.sum(offsets[:, :6] ** 2, axis=1)
            if fast:
                costs += 0.02 * np.sum(offsets[:, 6:] ** 2, axis=1)
            minimum_gap = trace.foot_foot_distance_m[:period_steps].min(axis=0)
            foot_contact = trace.foot_foot_contact[:period_steps].any(axis=0)
            safe = (minimum_gap >= 0.002) & ~foot_contact
            costs += np.where(safe, 0.0, 1000.0 + 10000.0 * np.maximum(0.0, 0.002 - minimum_gap))
            costs += np.where(foot_contact, 1000.0, 0.0)
            order = np.argsort(costs)
            if safe[order[0]] and costs[order[0]] < best_cost:
                best_cost = float(costs[order[0]])
                best_offset = offsets[order[0]].copy()
            safe_count = int(safe.sum())
            elite = offsets[order[: min(elite_count, safe_count) if safe_count else elite_count]]
            mean = elite.mean(axis=0)
            deviation = np.maximum(0.015, elite.std(axis=0))
        if best_offset is None:
            raise RuntimeError("no MPC candidate maintained the 2 mm foot-foot clearance")
        if best_cost < 2.0:
            self.previous_offsets[state.swing_foot] = best_offset.copy()
        transfer_offset = best_offset[6:].copy() if fast else np.zeros(6)
        return StepPlan(state.swing_foot, state.target_period_s, best_offset[:6].copy(), transfer_offset, best_cost)
