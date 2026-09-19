"""Step-level sampling MPC using parallel MuJoCo rollouts."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .cadence import StepState, phase_durations
from .model import standing_joint_targets
from .rollout import BatchRollout, RolloutTrace
from .support import FootSupportGeometry
from .warmstart import PLAN_BOUNDS, PLAN_DIM, WarmStartPolicy, make_warm_start_observation
from .zmp import ZMPPreview, plan_zmp_preview

# About half of this model's 29.5 N weight; contact-found alone can be almost unloaded.
MIN_SINGLE_SUPPORT_FORCE_N = 15.0
HOLD_KNOTS = 5
OPTIMIZATION_DIM = 12 + 6 * HOLD_KNOTS


@dataclass(frozen=True)
class StepPlan:
    """Transfer, swing, and stance-correction pulses for one step."""

    swing_foot: int
    period_s: float
    joint_offset: np.ndarray
    transfer_offset: np.ndarray
    hold_offset: np.ndarray
    hold_knots: np.ndarray
    cost: float
    observation: np.ndarray

    @property
    def parameter_vector(self) -> np.ndarray:
        """The 18 values a warm-start policy learns to propose."""
        return np.concatenate([self.joint_offset, self.transfer_offset, self.hold_offset])


class StepMPC:
    """Replan at each contact-validated step boundary.

    Each candidate uses three six-joint pulses: weight transfer, foot swing,
    and a correction during single-support hold.
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
        warm_start_policy: WarmStartPolicy | None = None,
        minimum_clearance_s: float = 0.1,
    ) -> None:
        if num_candidates < 8 or iterations < 1 or settle_s <= 0 or minimum_clearance_s <= 0:
            raise ValueError("at least eight candidates, one iteration, and positive settling time required")
        self.rollout = BatchRollout(model, num_candidates, num_threads=num_threads)
        self.num_candidates = num_candidates
        self.iterations = iterations
        self.settle_s = settle_s
        self.warm_start_policy = warm_start_policy
        self.minimum_clearance_s = minimum_clearance_s
        self.stand = standing_joint_targets(model)
        self.rng = np.random.default_rng(seed)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
        mujoco.mj_forward(model, data)
        self.home_base_quat = data.xquat[model.body("base").id].copy()
        self.support_geometry = FootSupportGeometry(model)
        # Feasible warm starts; the optimizer is free to move away from them.
        self.warm_offsets = np.array(
            [
                [-0.11, 0.20, -0.07, 0.12, 0.13, -0.05],
                [0.15, -0.10, -0.17, -0.15, 0.23, 0.05],
            ]
        )
        self.previous_offsets: list[np.ndarray | None] = [None, None]
        self.long_hold_offsets: list[np.ndarray | None] = [
            np.array(
                [
                    -0.054826223166908564, 0.05025794963501117,
                    -0.12913542322825938, 0.31415795977309874,
                    -0.04592000917223152, -0.2143227347971977,
                    0.13104319083589383, -0.05600945831550498,
                    0.050609518984139595, 0.29979952041740354,
                    0.0035368599978458787, -0.16117178797834628,
                    -0.372627535269236, 0.17247596840001478,
                    0.07019381207927092, 0.07888835957572372,
                    -0.21013388119146212, 0.1001917425342519,
                    0.2589369288214768, -0.05111865589654687,
                    0.08354364742783829, 0.0837928809074872,
                    -0.04036111755450695, -0.06158280454378629,
                    0.05366380214933694, -0.3802875155158198,
                    -0.044043252465604726, 0.05338140663961803,
                    0.017134127174259214, 0.07669925854967749,
                    0.11696285406221446, 0.05064938337102107,
                    -0.042898234830328175, 0.13147600109339913,
                    0.00037617008769937413, 0.010893509954526231,
                    -0.11966861928760572, -0.2170043262823404,
                    0.2622058959198863, -0.11170336527403821,
                    0.1475390356284375, 0.017844217409360645,
                ]
            ),
            np.array([0.2382907445927974, 0.037004851475820055, -0.16969022875436643, 0.06895666065630397, 0.15865748622176676, 0.08848380395413113, 0.05323381066646218, -0.021462270664937413, -0.046976505342298705, -0.054078189301692385, 0.11763684159729176, 0.10635810007521079, 0.00014054198717767218, 0.04588601883817467, 0.0041994968509491085, -0.04132799743834973, -0.03896661634233925, -0.020807656703101737, 0.2083821045959106, 0.16622464379128804, 0.05538422988238322, 0.09896493385136056, -0.05425913358950728, 0.13513070945029723, 0.22733063133431874, -0.01185345010986544, -0.01894704794077123, 0.06518838689893333, -0.07069904539807369, 0.09451571557140649, 0.1324240448339481, 0.04259816654416229, -0.05272466785236522, 0.02740565320896901, -0.015985336719804795, 0.1428487342085627, -0.021025657019679504, 0.0445992258473066, 0.036147980366766294, -0.027401630683370717, -0.1388897289672608, 0.14042682075880156]),
        ]

    @staticmethod
    def amplitude(elapsed_s: np.ndarray | float, period_s: float) -> np.ndarray:
        """Rise during lift, hold on one foot, and lower before touchdown."""
        t = np.asarray(elapsed_s)
        transfer_s, rise_s, lower_s = phase_durations(period_s)
        return np.clip(
            np.minimum((t - transfer_s) / rise_s, (period_s - t) / lower_s), 0.0, 1.0
        )

    @staticmethod
    def transfer_amplitude(elapsed_s: np.ndarray | float, period_s: float) -> np.ndarray:
        """Build support load gradually and hold it during a slow swing."""
        t = np.asarray(elapsed_s)
        transfer_s, _, lower_s = phase_durations(period_s)
        if period_s <= 1.0:
            peak_s = transfer_s * 0.5
            return np.clip(
                np.minimum(t / peak_s, (transfer_s - t) / peak_s), 0.0, 1.0
            )
        return np.clip(np.minimum(t / transfer_s, (period_s - t) / lower_s), 0.0, 1.0)

    @staticmethod
    def hold_amplitude(elapsed_s: np.ndarray | float, period_s: float) -> np.ndarray:
        """Grow a correction through hold, then remove it during lowering."""
        t = np.asarray(elapsed_s)
        transfer_s, lift_s, lower_s = phase_durations(period_s)
        start = transfer_s + lift_s
        end = period_s - lower_s
        return np.clip(
            np.minimum((t - start) / (end - start), (period_s - t) / lower_s), 0.0, 1.0
        )

    @staticmethod
    def hold_basis(elapsed_s: np.ndarray | float, period_s: float) -> np.ndarray:
        """Five piecewise-linear hold controls, with the last tapered to touchdown."""
        t = np.asarray(elapsed_s)
        transfer_s, lift_s, lower_s = phase_durations(period_s)
        start = transfer_s + lift_s
        end = period_s - lower_s
        centers = np.linspace(start, end, HOLD_KNOTS)
        basis = np.zeros(t.shape + (HOLD_KNOTS,), dtype=np.float64)
        basis[..., 0] = np.where(
            (t >= centers[0]) & (t <= centers[1]),
            (centers[1] - t) / (centers[1] - centers[0]),
            0.0,
        )
        for knot in range(1, HOLD_KNOTS - 1):
            left, center, right = centers[knot - 1 : knot + 2]
            basis[..., knot] = np.maximum(
                0.0, np.minimum((t - left) / (center - left), (right - t) / (right - center))
            )
        basis[..., -1] = np.where(
            (t >= centers[-2]) & (t <= centers[-1]),
            (t - centers[-2]) / (centers[-1] - centers[-2]),
            np.where(
                (t > centers[-1]) & (t <= period_s),
                (period_s - t) / lower_s,
                0.0,
            ),
        )
        return np.clip(basis, 0.0, 1.0)

    def target(self, plan: StepPlan, elapsed_s: float) -> np.ndarray:
        return (
            self.stand
            + plan.joint_offset * self.amplitude(elapsed_s, plan.period_s)
            + plan.transfer_offset * self.transfer_amplitude(elapsed_s, plan.period_s)
            + self.hold_basis(elapsed_s, plan.period_s) @ plan.hold_knots
        )

    def _cost(
        self, trace: RolloutTrace, swing: int, period_steps: int, preview: ZMPPreview
    ) -> np.ndarray:
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
        support_contact_loss = ~contacts[:period_steps, :, support]
        support_xy = trace.foot_pos_w[:period_steps, :, support, :2]
        support_motion_sq = np.sum((support_xy - support_xy[0:1]) ** 2, axis=2)
        xy_error = trace.com_pos_w[:, :, :2] - preview.com_xy[:, None, :]
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
        cop_tracking = np.sum(valid_cop * cop_tracking, axis=0) / np.maximum(valid_cop.sum(axis=0), 1)
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
            10.0 * np.maximum(0.0, 0.7 - airborne_fraction) ** 2
            + 1000000.0 * np.mean(clearance_error**2, axis=0)
            + 10.0 * np.maximum(0.0, 0.7 - clear_fraction) ** 2
            + 100.0 * np.mean(support_contact_loss, axis=0)
            + 100.0 * np.any(support_contact_loss[lift], axis=0)
            + 1.0 * np.mean(np.maximum(0.0, 18.0 - support_force) ** 2, axis=0)
            + 10000.0 * np.mean(support_motion_sq, axis=0)
            + 20000.0 * np.max(support_motion_sq, axis=0)
            + 20.0 * np.mean(~contacts[late, :, swing], axis=0)
            + 2.0 * np.mean(~contacts[late, :, support], axis=0)
            + 0.5 * np.mean(
                np.maximum(0.0, 15.0 - trace.foot_normal_force_n[late, :, swing]) ** 2,
                axis=0,
            )
            + 1000.0 * np.maximum(0.0, 0.22 - min_height) ** 2
            + 1000.0 * np.maximum(0.0, 0.22 - final_height) ** 2
            + 1000.0 * np.mean(np.sum(xy_error**2, axis=2), axis=0)
            + 3000.0 * np.sum(final_xy_error**2, axis=1)
            + 2.0 * (1.0 - base_alignment) ** 2
            + 0.01 * np.sum(qvel**2, axis=1)
            + 20.0 * np.sum(trace.qvel[landing] ** 2, axis=1)
            + 400.0 * com_penalty
            + 1000.0 * cop_penalty
            + 1000.0 * cop_tracking
            + 800.0 * np.maximum(0.0, -0.005 - next_support_margin) ** 2
            + 0.01 * np.maximum(0.0, 8.0 - trace.foot_normal_force_n[landing, :, swing]) ** 2
            + 3000.0 * np.mean(np.maximum(0.0, 0.01 - foot_gap) ** 2, axis=0)
        )
        if self.minimum_clearance_s >= 1.0:
            cost += 1000000.0 * np.maximum(0.0, 0.245 - min_height) ** 2
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
        data = mujoco.MjData(self.rollout.model)
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(self.rollout.model, data)
        observation: np.ndarray | None = None
        policy_offset: np.ndarray | None = None
        if self.warm_start_policy is not None:
            observation = make_warm_start_observation(
                self.rollout.model, qpos, qvel, ctrl, state
            )
            policy_offset = np.asarray(
                self.warm_start_policy.predict(observation), dtype=np.float64
            )
            if policy_offset.shape != (PLAN_DIM,) or not np.isfinite(policy_offset).all():
                raise ValueError("warm-start policy must return 18 finite values")
            policy_offset = np.clip(policy_offset, -PLAN_BOUNDS, PLAN_BOUNDS)
            policy_offset = np.r_[
                policy_offset[:12], np.tile(policy_offset[12:18], HOLD_KNOTS)
            ]
        foot_xy = np.stack([data.body(f"foot_{foot}").xpos[:2] for foot in (0, 1)])
        com = data.subtree_com[self.rollout.model.body("foot_0").id]
        preview = plan_zmp_preview(
            com[:2], float(com[2]), foot_xy, state.swing_foot,
            state.target_period_s, state.elapsed_s, dt, horizon,
            transfer_s=phase_durations(state.target_period_s)[0],
            lower_s=phase_durations(state.target_period_s)[2],
        )
        amplitude = self.amplitude(times, state.target_period_s)
        transfer_amplitude = self.transfer_amplitude(times, state.target_period_s)
        hold_basis = self.hold_basis(times, state.target_period_s)
        warm = np.r_[self.warm_offsets[state.swing_foot], np.zeros(6 + 6 * HOLD_KNOTS)]
        bounds = np.r_[np.full(6, 0.6), np.full(6 + 6 * HOLD_KNOTS, 0.5)]
        previous_offset = self.previous_offsets[state.swing_foot]
        if previous_offset is not None and previous_offset.shape != warm.shape:
            previous_offset = None
        long_hold_offset = (
            self.long_hold_offsets[state.swing_foot]
            if self.minimum_clearance_s >= 1.0 and state.target_period_s >= 3.0
            else None
        )
        if policy_offset is not None:
            mean = policy_offset.copy()
        elif long_hold_offset is not None:
            mean = long_hold_offset.copy()
        elif previous_offset is not None:
            mean = previous_offset.copy()
        else:
            mean = warm.copy()
        deviation = np.full(warm.shape, 0.05 if policy_offset is not None else 0.15)
        elite_count = max(4, self.num_candidates // 16)
        best_cost = np.inf
        best_offset: np.ndarray | None = None
        best_clearance_duration_s = 0.0
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
            if policy_offset is not None:
                offsets[4] = policy_offset
            if long_hold_offset is not None:
                offsets[5] = long_hold_offset
            targets = self.stand[None, None, :] + offsets[:, None, :6] * amplitude[None, :, None]
            targets += offsets[:, None, 6:12] * transfer_amplitude[None, :, None]
            hold_knots = offsets[:, 12:].reshape(self.num_candidates, HOLD_KNOTS, 6)
            targets += np.einsum("nkj,tk->ntj", hold_knots, hold_basis)
            self.rollout.set_state(qpos, qvel, ctrl, time_s=time_s)
            trace = self.rollout.rollout(targets, max_target_rate_rad_s=4.0)
            costs = self._cost(trace, state.swing_foot, period_steps, preview)
            costs += 0.03 * np.sum(offsets[:, :6] ** 2, axis=1)
            costs += 0.02 * np.sum(offsets[:, 6:] ** 2, axis=1)
            minimum_gap = trace.foot_foot_distance_m[:period_steps].min(axis=0)
            foot_contact = trace.foot_foot_contact[:period_steps].any(axis=0)
            support_contact = trace.foot_contacts[:period_steps, :, state.support_foot].all(axis=0)
            support_loaded = (
                trace.foot_normal_force_n[:period_steps, :, state.support_foot].min(axis=0)
                >= 1.0
            )
            single_support = ~trace.foot_contacts[:period_steps, :, state.swing_foot]
            loaded_when_alone = np.all(
                ~single_support
                | (
                    trace.foot_normal_force_n[:period_steps, :, state.support_foot]
                    >= MIN_SINGLE_SUPPORT_FORCE_N
                ),
                axis=0,
            )
            swing_above = trace.foot_clearance_m[:period_steps, :, state.swing_foot] >= 0.003
            clearance_run = np.zeros(self.num_candidates, dtype=np.int32)
            longest_clearance_run = clearance_run.copy()
            for above in swing_above:
                clearance_run = np.where(above, clearance_run + 1, 0)
                longest_clearance_run = np.maximum(longest_clearance_run, clearance_run)
            clearance_duration = longest_clearance_run * dt
            duration_deficit = np.maximum(0.0, self.minimum_clearance_s - clearance_duration)
            valid_swing = clearance_duration >= self.minimum_clearance_s
            base_safe = (
                (minimum_gap >= 0.002)
                & ~foot_contact
                & support_contact
                & support_loaded
                & loaded_when_alone
                & (trace.base_pos_w[:period_steps, :, 2].min(axis=0) >= 0.22)
            )
            if base_safe.any():
                best_clearance_duration_s = max(
                    best_clearance_duration_s, float(clearance_duration[base_safe].max())
                )
            if self.minimum_clearance_s >= 1.0:
                costs += 4000.0 * duration_deficit**2
            else:
                costs += np.where(base_safe, 5000.0 * duration_deficit**2, 0.0)
            safe = base_safe & valid_swing
            costs += np.where(safe, 0.0, 1000.0 + 10000.0 * np.maximum(0.0, 0.002 - minimum_gap))
            costs += np.where(foot_contact, 1000.0, 0.0)
            order = np.argsort(costs)
            safe_indices = np.flatnonzero(safe)
            if len(safe_indices):
                candidate = safe_indices[np.argmin(costs[safe_indices])]
                if costs[candidate] < best_cost:
                    best_cost = float(costs[candidate])
                    best_offset = offsets[candidate].copy()
            search_safe = base_safe & (clearance_duration >= min(0.1, self.minimum_clearance_s))
            search_indices = np.flatnonzero(search_safe)
            if len(search_indices):
                ranked = search_indices[np.argsort(costs[search_indices])]
                elite = offsets[ranked[:elite_count]]
            else:
                elite = offsets[order[:elite_count]]
            mean = elite.mean(axis=0)
            deviation = np.maximum(0.015, elite.std(axis=0))
        if best_offset is None:
            raise RuntimeError(
                f"no MPC candidate achieved {self.minimum_clearance_s:.2f} s sustained lift, "
                f"loaded support, and 2 mm foot clearance "
                f"at {time_s:.2f} s; best safe-support clearance run was "
                f"{best_clearance_duration_s:.2f} s"
            )
        if best_cost < 2.0:
            self.previous_offsets[state.swing_foot] = best_offset.copy()
        if observation is None:
            observation = make_warm_start_observation(
                self.rollout.model, qpos, qvel, ctrl, state
            )
        return StepPlan(
            state.swing_foot,
            state.target_period_s,
            best_offset[:6].copy(),
            best_offset[6:12].copy(),
            best_offset[12:].reshape(HOLD_KNOTS, 6).mean(axis=0),
            best_offset[12:].reshape(HOLD_KNOTS, 6).copy(),
            best_cost,
            observation.copy(),
        )
