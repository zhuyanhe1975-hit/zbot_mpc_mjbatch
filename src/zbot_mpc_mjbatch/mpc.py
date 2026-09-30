"""Step-level sampling MPC using parallel MuJoCo rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .cadence import MIN_SWING_CLEARANCE_M, MIN_SWING_CLEARANCE_S, StepState, landing_settle_s
from .model import standing_joint_targets
from .posture import support_posture_proposals
from .rl_prior import PPOProposal
from .rollout import BatchRollout, RolloutTrace
from .support import FootSupportGeometry
from .zmp import ZMPPreview, plan_zmp_preview

# About half of this model's 29.5 N weight; contact-found alone can be almost unloaded.
MIN_SINGLE_SUPPORT_FORCE_N = 15.0
MAX_LANDING_ROOT_SPEED = 1.5
MAX_LANDING_JOINT_SPEED_RAD_S = 0.8
MAX_LANDING_LINEAR_SPEED_M_S = 0.1
MAX_LANDING_ANGULAR_SPEED_RAD_S = 0.6
HANDOFF_PREVIEW_S = 0.1
LANDING_CONTACT_S = 0.06


def landing_state_is_safe(
    contacts: np.ndarray,
    forces: np.ndarray,
    velocities: np.ndarray,
    swing: int,
    period_s: float,
    quiet: bool = False,
) -> np.ndarray:
    """Check a contact history and its final measured state, with optional batching."""
    return (
        contacts.all(axis=(0, -1))
        & (forces.min(axis=-1) >= 4.0)
        & (forces[..., swing] >= 10.0)
        & (np.linalg.norm(velocities[..., :3], axis=-1) <= (0.03 if quiet else MAX_LANDING_LINEAR_SPEED_M_S))
        & (np.linalg.norm(velocities[..., 3:6], axis=-1) <= (0.15 if quiet else MAX_LANDING_ANGULAR_SPEED_RAD_S))
        & (np.linalg.norm(velocities[..., 6:], axis=-1) <= (0.2 if quiet else MAX_LANDING_JOINT_SPEED_RAD_S))
    )


@dataclass(frozen=True)
class StepPlan:
    """Transfer, swing, stance correction, and a persistent landing posture."""

    swing_foot: int
    period_s: float
    joint_offset: np.ndarray
    transfer_offset: np.ndarray
    hold_offset: np.ndarray
    cost: float
    rl_prior_selected: bool = False
    landing_offset: np.ndarray | None = None
    start_offset: np.ndarray | None = None


class StepMPC:
    """Replan at each contact-validated step boundary.

    Each candidate uses three six-joint pulses and a six-joint landing posture.
    The next plan starts at the measured control target for a continuous handoff.
    Candidates are evaluated for a complete step plus a settling interval; only
    the current step is executed.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        num_candidates: int = 256,
        iterations: int = 24,
        num_threads: int = 0,
        seed: int = 0,
        settle_s: float = 0.5,
        use_rl_prior: bool = False,
    ) -> None:
        if num_candidates < 8 or iterations < 1 or settle_s < HANDOFF_PREVIEW_S:
            raise ValueError(
                "at least eight candidates, one iteration, and 0.1 s settling required"
            )
        self.rollout = BatchRollout(
            model,
            num_candidates + int(use_rl_prior),
            substeps_per_control=1,
            num_threads=num_threads,
        )
        self.num_candidates = num_candidates
        self.iterations = iterations
        self.settle_s = settle_s
        self.stand = standing_joint_targets(model)
        self.rng = np.random.default_rng(seed)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
        mujoco.mj_forward(model, data)
        self.home_base_quat = data.xquat[model.body("base").id].copy()
        self.home_base_xy = data.xpos[model.body("base").id, :2].copy()
        self.standing_com_offset_xy = (
            data.subtree_com[model.body("foot_0").id, :2]
            - 0.5 * (data.body("foot_0").xpos[:2] + data.body("foot_1").xpos[:2])
        ).copy()
        self.support_geometry = FootSupportGeometry(model)
        policy_path = Path(__file__).parent / "assets" / "rl_walker_actor.npz"
        self.rl_proposal = PPOProposal(model, policy_path) if use_rl_prior else None
        # Feasible warm starts; the optimizer is free to move away from them.
        self.warm_offsets = np.array(
            [
                [-0.11, 0.20, -0.07, 0.12, 0.13, -0.05],
                [0.15, -0.10, -0.17, -0.15, 0.23, 0.05],
            ]
        )
        self.previous_offsets: list[np.ndarray | None] = [None, None]
        self.posture_proposals = np.pad(support_posture_proposals(model), ((0, 0), (0, 6)))

    @staticmethod
    def amplitude(elapsed_s: np.ndarray | float, period_s: float) -> np.ndarray:
        """Rise during lift, hold on one foot, and lower before touchdown."""
        t = np.asarray(elapsed_s)
        extra_s = max(0.0, period_s - 1.0)
        transfer_s = min(0.3, 0.3 * period_s) + 0.6 * extra_s
        rise_s = 0.12 + 0.18 * extra_s
        lower_s = 0.12 + 0.18 * extra_s
        return np.clip(
            np.minimum(
                (t - transfer_s) / rise_s, (period_s - landing_settle_s(period_s) - t) / lower_s
            ),
            0.0,
            1.0,
        )

    @staticmethod
    def transfer_amplitude(elapsed_s: np.ndarray | float, period_s: float) -> np.ndarray:
        """Build support load gradually and hold it during a slow swing."""
        t = np.asarray(elapsed_s)
        extra_s = max(0.0, period_s - 1.0)
        transfer_s = min(0.3, 0.3 * period_s) + 0.6 * extra_s
        lower_s = 0.12 + 0.18 * extra_s
        return np.clip(
            np.minimum(t / transfer_s, (period_s - landing_settle_s(period_s) - t) / lower_s),
            0.0,
            1.0,
        )

    @staticmethod
    def hold_amplitude(elapsed_s: np.ndarray | float, period_s: float) -> np.ndarray:
        """Mid-step correction that vanishes before touchdown."""
        t = np.asarray(elapsed_s)
        extra_s = max(0.0, period_s - 1.0)
        start = min(0.3, 0.3 * period_s) + 0.6 * extra_s + 0.12 + 0.18 * extra_s
        end = period_s - landing_settle_s(period_s) - (0.12 + 0.18 * extra_s)
        middle = 0.5 * (start + end)
        return np.clip(
            np.minimum((t - start) / (middle - start), (end - t) / (end - middle)), 0.0, 1.0
        )

    @staticmethod
    def landing_amplitude(elapsed_s: np.ndarray | float, period_s: float) -> np.ndarray:
        lower_s = 0.12 + 0.18 * max(0.0, period_s - 1.0)
        start = period_s - landing_settle_s(period_s) - lower_s
        return np.clip((np.asarray(elapsed_s) - start) / lower_s, 0.0, 1.0)

    @staticmethod
    def start_amplitude(elapsed_s: np.ndarray | float, period_s: float) -> np.ndarray:
        # Preserve the preceding balanced posture during transfer and swing;
        # replace it only as the new landing posture takes over.
        return 1.0 - StepMPC.landing_amplitude(elapsed_s, period_s)

    def target(self, plan: StepPlan, elapsed_s: float) -> np.ndarray:
        target = (
            self.stand
            + plan.joint_offset * self.amplitude(elapsed_s, plan.period_s)
            + plan.transfer_offset * self.transfer_amplitude(elapsed_s, plan.period_s)
            + plan.hold_offset * self.hold_amplitude(elapsed_s, plan.period_s)
        )
        if plan.landing_offset is not None:
            target += plan.landing_offset * self.landing_amplitude(elapsed_s, plan.period_s)
        if plan.start_offset is not None:
            target += plan.start_offset * self.start_amplitude(elapsed_s, plan.period_s)
        return target

    def _cost(
        self, trace: RolloutTrace, swing: int, period_steps: int, preview: ZMPPreview
    ) -> np.ndarray:
        base = trace.base_pos_w
        contacts = trace.foot_contacts
        support = 1 - swing
        period_s = period_steps * self.rollout.control_dt
        swing_profile = self.amplitude(np.arange(period_steps) * self.rollout.control_dt, period_s)
        plateau = np.flatnonzero(swing_profile >= 0.95)
        lift = slice(int(plateau[0]), int(plateau[-1]) + 1)
        late = slice(period_steps, None)
        airborne_fraction = np.mean(~contacts[lift, :, swing], axis=0)
        swing_clearance = trace.foot_clearance_m[lift, :, swing]
        clearance_error = np.maximum(0.0, 0.015 - swing_clearance)
        clear_fraction = np.mean(swing_clearance >= 0.004, axis=0)
        support_force = trace.foot_normal_force_n[lift, :, support]
        support_contact_loss = ~contacts[:period_steps, :, support]
        support_xy = trace.foot_pos_w[:period_steps, :, support, :2]
        support_motion_sq = np.sum((support_xy - support_xy[0:1]) ** 2, axis=2)
        xy_error = trace.com_pos_w[:, :, :2] - preview.com_xy[:, None, :]
        rest_center = trace.foot_pos_w[period_steps - 1, :, :, :2].mean(axis=1)
        xy_error[late] = trace.com_pos_w[late, :, :2] - (rest_center + self.standing_com_offset_xy)
        final_xy_error = xy_error[-1]
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
        cop_tracking = np.sum(
            (np.nan_to_num(trace.foot_cop_w[lift, :, support]) - preview.zmp_xy[lift, None]) ** 2,
            axis=2,
        )
        cop_tracking = np.sum(valid_cop * cop_tracking, axis=0) / np.maximum(
            valid_cop.sum(axis=0), 1
        )
        com_penalty = np.sum(single_support * np.maximum(0.0, -com_margin) ** 2, axis=0)
        com_penalty /= np.maximum(single_support.sum(axis=0), 1)
        cop_penalty = np.sum(valid_cop * np.maximum(0.0, -cop_margin) ** 2, axis=0)
        cop_penalty /= np.maximum(valid_cop.sum(axis=0), 1)
        foot_gap = trace.foot_foot_distance_m[:period_steps]
        landing = period_steps - 1
        landing_start = max(0, period_steps - max(2, int(np.ceil(0.08 / self.rollout.control_dt))))
        landing_velocity = trace.qvel[landing_start:period_steps]
        landing_joint_error = trace.joint_pos[landing_start:period_steps] - self.stand
        next_support_margin = self.support_geometry.margin(
            trace.com_pos_w[landing, :, :2],
            trace.foot_geom_pos_w[landing, :, swing],
            trace.foot_geom_mat_w[landing, :, swing],
            swing,
        )
        base_rotation_error = np.maximum(
            0.0, 1.0 - np.sum(trace.base_quat_w * self.home_base_quat, axis=2) ** 2
        )
        cost = (
            10.0 * np.maximum(0.0, 0.7 - airborne_fraction) ** 2
            + 1000000.0 * np.mean(clearance_error**2, axis=0)
            + 10.0 * np.maximum(0.0, 0.7 - clear_fraction) ** 2
            + 100.0 * np.mean(support_contact_loss, axis=0)
            + 100.0 * np.any(support_contact_loss[lift], axis=0)
            + 1.0 * np.mean(np.maximum(0.0, 18.0 - support_force) ** 2, axis=0)
            + 10000.0 * np.mean(support_motion_sq, axis=0)
            + 20000.0 * np.max(support_motion_sq, axis=0)
            + 200.0 * np.mean(~contacts[late, :, swing], axis=0)
            + 200.0 * np.mean(~contacts[late, :, support], axis=0)
            + 0.5
            * np.mean(
                np.maximum(0.0, 15.0 - trace.foot_normal_force_n[late, :, swing]) ** 2,
                axis=0,
            )
            + 1000.0 * np.maximum(0.0, 0.22 - min_height) ** 2
            + 1000.0 * np.maximum(0.0, 0.22 - final_height) ** 2
            + 1000.0 * np.mean(np.sum(xy_error**2, axis=2), axis=0)
            + 3000.0 * np.sum(final_xy_error**2, axis=1)
            + 3000.0
            * np.mean(np.sum((base[:period_steps, :, :2] - self.home_base_xy) ** 2, axis=2), axis=0)
            + 10000.0 * np.sum((base[landing, :, :2] - self.home_base_xy) ** 2, axis=1)
            + 300.0 * np.mean(base_rotation_error[:period_steps], axis=0)
            + 1000.0 * base_rotation_error[landing]
            + 0.01 * np.sum(qvel**2, axis=1)
            # The settling tail is prediction-only: the next plan starts at touchdown.
            # Penalize the state that is actually handed to that next plan.
            + 30.0 * np.mean(np.sum(landing_velocity[:, :, :6] ** 2, axis=2), axis=0)
            + 2.0 * np.mean(np.sum(landing_velocity[:, :, 6:] ** 2, axis=2), axis=0)
            + 40.0 * np.mean(np.sum(landing_joint_error**2, axis=2), axis=0)
            + 400.0 * com_penalty
            + 1000.0 * cop_penalty
            + 1000.0 * cop_tracking
            + 800.0 * np.maximum(0.0, -0.005 - next_support_margin) ** 2
            + 5.0 * np.maximum(0.0, 12.0 - trace.foot_normal_force_n[landing, :, swing]) ** 2
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
        """Try distinct initial postures before declaring a step infeasible."""
        previous = self.previous_offsets[state.swing_foot]
        error: RuntimeError | None = None
        for restart in range(3):
            if restart == 1:
                self.previous_offsets[state.swing_foot] = None
            elif restart == 2:
                self.previous_offsets[state.swing_foot] = (
                    0.5 * self.posture_proposals[state.swing_foot]
                )
            try:
                return self._plan_once(qpos, qvel, ctrl, state, time_s)
            except RuntimeError as candidate_error:
                error = candidate_error
        self.previous_offsets[state.swing_foot] = previous
        assert error is not None
        raise RuntimeError(f"{error}; exhausted three search initializations")

    def _plan_once(
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
        data = mujoco.MjData(self.rollout.model)
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(self.rollout.model, data)
        foot_xy = np.stack([data.body(f"foot_{foot}").xpos[:2] for foot in (0, 1)])
        com = data.subtree_com[self.rollout.model.body("foot_0").id]
        preview = plan_zmp_preview(
            com[:2],
            float(com[2]),
            foot_xy,
            state.swing_foot,
            state.target_period_s,
            state.elapsed_s,
            dt,
            horizon,
            transfer_s=min(0.3, 0.3 * state.target_period_s)
            + 0.6 * max(0.0, state.target_period_s - 1.0),
            lower_s=0.12 + 0.18 * max(0.0, state.target_period_s - 1.0),
            settle_s=landing_settle_s(state.target_period_s),
        )
        amplitude = self.amplitude(times, state.target_period_s)
        transfer_amplitude = self.transfer_amplitude(times, state.target_period_s)
        hold_amplitude = self.hold_amplitude(times, state.target_period_s)
        transfer_s = min(0.3, 0.3 * state.target_period_s) + 0.6 * max(
            0.0, state.target_period_s - 1.0
        )
        rl_prior = (
            np.r_[self.rl_proposal.propose(qpos, qvel, transfer_s), np.zeros(6)]
            if self.rl_proposal is not None
            else None
        )
        warm = np.r_[self.warm_offsets[state.swing_foot], np.zeros(18)]
        bounds = np.r_[np.full(6, 0.6), np.full(12, 0.4), np.full(6, 0.15)]
        start_offset = ctrl - self.stand
        start_amplitude = self.start_amplitude(times, state.target_period_s)
        landing_amplitude = self.landing_amplitude(times, state.target_period_s)
        previous_offset = self.previous_offsets[state.swing_foot]
        if previous_offset is not None and previous_offset.shape != warm.shape:
            previous_offset = None
        mean = previous_offset.copy() if previous_offset is not None else warm.copy()
        deviation = np.r_[np.full(18, 0.15), np.full(6, 0.04)]
        elite_count = max(4, self.num_candidates // 16)
        best_cost = np.inf
        best_offset: np.ndarray | None = None
        best_rl_prior_selected = False
        best_feasibility: dict[str, int] = {}
        for _ in range(self.iterations):
            noise = self.rng.standard_normal((self.num_candidates, len(warm)))
            offsets = mean + deviation * noise
            # Keep exploring independent basins when an inherited step is not
            # feasible from the new contact state; CEM otherwise collapses early.
            explore = max(1, self.num_candidates // 8)
            offsets[-explore:] = warm + 0.12 * noise[-explore:]
            offsets[-2 * explore : -explore] = (
                0.5 * self.posture_proposals[state.swing_foot]
                + 0.1 * noise[-2 * explore : -explore]
            )
            offsets[-2 * explore :, 18:] *= 0.4
            offsets = np.clip(offsets, -bounds, bounds)
            offsets[0] = mean
            if best_offset is not None:
                offsets[1] = best_offset
            offsets[2] = warm
            offsets[4] = np.clip(0.5 * self.posture_proposals[state.swing_foot], -bounds, bounds)
            offsets[5] = np.clip(0.65 * self.posture_proposals[state.swing_foot], -bounds, bounds)
            if previous_offset is not None:
                offsets[3] = previous_offset
            if rl_prior is not None:
                # PPO supplies a state-conditioned proposal. CEM's rollout and
                # hard contact filter decide whether it can influence the plan.
                offsets = np.vstack((offsets, rl_prior))
            targets = self.stand[None, None, :] + offsets[:, None, :6] * amplitude[None, :, None]
            targets += offsets[:, None, 6:12] * transfer_amplitude[None, :, None]
            targets += offsets[:, None, 12:18] * hold_amplitude[None, :, None]
            targets += offsets[:, None, 18:24] * landing_amplitude[None, :, None]
            targets += start_offset[None, None, :] * start_amplitude[None, :, None]
            self.rollout.set_state(qpos, qvel, ctrl, time_s=time_s)
            trace = self.rollout.rollout(targets, max_target_rate_rad_s=4.0)
            costs = self._cost(trace, state.swing_foot, period_steps, preview)
            costs += 0.03 * np.sum(offsets[:, :6] ** 2, axis=1)
            costs += 0.02 * np.sum(offsets[:, 6:] ** 2, axis=1)
            minimum_gap = trace.foot_foot_distance_m.min(axis=0)
            foot_contact = trace.foot_foot_contact.any(axis=0)
            stance_start = min(
                period_steps - 1, int(np.searchsorted(times[:period_steps], transfer_s))
            )
            support_contact = trace.foot_contacts[
                stance_start:period_steps, :, state.support_foot
            ].all(axis=0)
            support_loaded = (
                trace.foot_normal_force_n[stance_start:period_steps, :, state.support_foot].min(
                    axis=0
                )
                >= 1.0
            )
            contact_count = trace.foot_contacts[:period_steps].sum(axis=2)
            grounded = (contact_count >= 1).all(axis=0)
            single_support = contact_count == 1
            loaded_when_alone = np.all(
                ~single_support
                | (
                    trace.foot_normal_force_n[:period_steps].sum(axis=2)
                    >= MIN_SINGLE_SUPPORT_FORCE_N
                ),
                axis=0,
            )
            swing_above = (
                trace.foot_clearance_m[:period_steps, :, state.swing_foot] >= MIN_SWING_CLEARANCE_M
            )
            swing_above[:stance_start] = False
            clearance_run = np.zeros(self.rollout.num_candidates, dtype=np.int32)
            longest_clearance_run = clearance_run.copy()
            for above in swing_above:
                clearance_run = np.where(above, clearance_run + 1, 0)
                longest_clearance_run = np.maximum(longest_clearance_run, clearance_run)
            valid_swing = longest_clearance_run * dt >= MIN_SWING_CLEARANCE_S
            handoff_end = horizon
            stable_landing = landing_state_is_safe(
                trace.foot_contacts[
                    max(0, period_steps - int(np.ceil(LANDING_CONTACT_S / dt))) : handoff_end
                ],
                trace.foot_normal_force_n[handoff_end - 1],
                trace.qvel[handoff_end - 1],
                state.swing_foot,
                state.target_period_s,
                quiet=True,
            )
            stable_landing &= trace.foot_contacts[period_steps:handoff_end].all(axis=(0, 2))
            safe = (
                (minimum_gap >= 0.002)
                & ~foot_contact
                & grounded
                & support_contact
                & support_loaded
                & loaded_when_alone
                & valid_swing
                & stable_landing
                & (trace.base_pos_w[:, :, 2].min(axis=0) >= 0.22)
            )
            best_feasibility = {
                "support_contact": int(support_contact.sum()),
                "grounded": int(grounded.sum()),
                "support_loaded": int(support_loaded.sum()),
                "single_support_force": int(loaded_when_alone.sum()),
                "sustained_clearance": int(valid_swing.sum()),
                "stable_landing": int(stable_landing.sum()),
                "no_foot_collision": int((~foot_contact).sum()),
                "all_constraints": int(safe.sum()),
            }
            costs += np.where(safe, 0.0, 1000.0 + 10000.0 * np.maximum(0.0, 0.002 - minimum_gap))
            costs += np.where(foot_contact, 1000.0, 0.0)
            order = np.lexsort((costs, ~safe))
            if safe[order[0]] and costs[order[0]] < best_cost:
                best_cost = float(costs[order[0]])
                best_offset = offsets[order[0]].copy()
                best_rl_prior_selected = rl_prior is not None and order[0] == self.num_candidates
            safe_count = int(safe.sum())
            elite = offsets[order[: min(elite_count, safe_count) if safe_count else elite_count]]
            mean = elite.mean(axis=0)
            deviation = np.maximum(0.015, elite.std(axis=0))
        if best_offset is None:
            raise RuntimeError(
                f"no MPC candidate achieved sustained lift, loaded support, "
                f"stable landing, and 2 mm foot clearance at {time_s:.2f} s; "
                f"last-iteration feasible counts={best_feasibility} "
            )
        # Every retained plan passed the hard contact and clearance checks. Reuse it
        # for this side and seed the opposite side with the serial-chain mirror.
        self.previous_offsets[state.swing_foot] = best_offset.copy()
        other = 1 - state.swing_foot
        mirrored = np.concatenate([-best_offset[i : i + 6][::-1] for i in range(0, 24, 6)])
        if self.previous_offsets[other] is None:
            self.previous_offsets[other] = mirrored
        return StepPlan(
            state.swing_foot,
            state.target_period_s,
            best_offset[:6].copy(),
            best_offset[6:12].copy(),
            best_offset[12:18].copy(),
            best_cost,
            best_rl_prior_selected,
            best_offset[18:24].copy(),
            start_offset.copy(),
        )
