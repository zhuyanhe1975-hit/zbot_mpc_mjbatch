"""Batch rollout primitive for offline trajectory search and future MPC."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from mjbatch import Batch

from .model import standing_joint_targets
from .support import center_of_pressure, mesh_pair_distance


@dataclass(frozen=True)
class RolloutTrace:
    base_pos_w: np.ndarray  # (horizon, candidates, xyz)
    base_quat_w: np.ndarray  # (horizon, candidates, wxyz)
    foot_pos_w: np.ndarray  # (horizon, candidates, foot, xyz)
    foot_contacts: np.ndarray  # (horizon, candidates, foot)
    joint_pos: np.ndarray  # (horizon, candidates, six joints)
    applied_targets: np.ndarray  # (horizon, candidates, six actuators)
    com_pos_w: np.ndarray  # (horizon, candidates, xyz)
    foot_geom_pos_w: np.ndarray  # (horizon, candidates, foot, xyz)
    foot_geom_mat_w: np.ndarray  # (horizon, candidates, foot, 9)
    foot_cop_w: np.ndarray  # (horizon, candidates, foot, xy); NaN when unloaded
    foot_normal_force_n: np.ndarray  # (horizon, candidates, foot)
    foot_clearance_m: np.ndarray  # (horizon, candidates, foot); signed geom-floor distance
    foot_foot_distance_m: np.ndarray  # (horizon, candidates); negative means overlap
    foot_foot_contact: np.ndarray  # (horizon, candidates)


class BatchRollout:
    def __init__(
        self,
        model: mujoco.MjModel,
        num_candidates: int,
        substeps_per_control: int = 4,
        num_threads: int = 0,
    ) -> None:
        if num_candidates <= 0 or substeps_per_control <= 0:
            raise ValueError("candidate and substep counts must be positive")
        if model.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST:
            raise ValueError("use load_model() to select the stable implicitfast integrator")
        self.model = model
        self.num_candidates = num_candidates
        self.substeps_per_control = substeps_per_control
        self.control_dt = model.opt.timestep * substeps_per_control
        self.batch = Batch(model, num_candidates, num_threads=num_threads, forward=True)
        # Bind derived fields before resetting so they are populated immediately.
        self.qpos = self.batch.bind("qpos")
        self.qvel = self.batch.bind("qvel")
        self.ctrl = self.batch.bind("ctrl")
        self.time = self.batch.bind("time")
        base_body = self.batch.body("base")
        self.base_pos_w = base_body.xpos
        self.base_quat_w = base_body.xquat
        self.com_pos_w = self.batch.bind("subtree_com")[:, model.body("foot_0").id]
        geom_pos = self.batch.bind("geom_xpos")
        geom_mat = self.batch.bind("geom_xmat")
        geom_ids = [model.geom(f"foot_{foot}_collisions").id for foot in (0, 1)]
        self.foot_geom_pos_w = tuple(geom_pos[:, geom_id] for geom_id in geom_ids)
        self.foot_geom_mat_w = tuple(geom_mat[:, geom_id] for geom_id in geom_ids)
        self.foot_pos_w = (
            self.batch.body("foot_0").xpos,
            self.batch.body("foot_1").xpos,
        )
        self.contacts = (
            self.batch.sensor("foot_0_contact"),
            self.batch.sensor("foot_1_contact"),
        )
        self.clearance = tuple(
            self.batch.sensor(f"foot_{foot}_clearance") for foot in (0, 1)
        )
        self.foot_foot_distance = self.batch.sensor("foot_foot_distance")
        self.foot_foot_fromto = self.batch.sensor("foot_foot_fromto")
        self.foot_foot_contact = self.batch.sensor("foot_foot_contact")
        self.net_contact_pos = tuple(
            self.batch.sensor(f"foot_{foot}_net_pos") for foot in (0, 1)
        )
        self.net_contact_force = tuple(
            self.batch.sensor(f"foot_{foot}_net_force") for foot in (0, 1)
        )
        self.net_contact_torque = tuple(
            self.batch.sensor(f"foot_{foot}_net_torque") for foot in (0, 1)
        )

    def reset_to_stand(self) -> None:
        self.batch.reset(keyframe=self.model.key("stand").id)
        self.ctrl[:] = standing_joint_targets(self.model)

    def set_state(
        self, qpos: np.ndarray, qvel: np.ndarray, ctrl: np.ndarray, time_s: float = 0.0
    ) -> None:
        if (
            qpos.shape != (self.model.nq,)
            or qvel.shape != (self.model.nv,)
            or ctrl.shape != (self.model.nu,)
        ):
            raise ValueError("state dimensions do not match the model")
        if not all(np.isfinite(array).all() for array in (qpos, qvel, ctrl)) or not np.isfinite(time_s):
            raise ValueError("state must be finite")
        self.qpos[:] = qpos
        self.qvel[:] = qvel
        self.ctrl[:] = ctrl
        self.time[:] = time_s
        self.batch.forward()

    def rollout(
        self, targets: np.ndarray, max_target_rate_rad_s: float | None = None
    ) -> RolloutTrace:
        if targets.ndim != 3 or targets.shape[0] != self.num_candidates or targets.shape[2] != self.model.nu:
            raise ValueError("targets must have shape (candidates, horizon, six actuators)")
        if not np.isfinite(targets).all():
            raise ValueError("targets must be finite")
        if targets.shape[1] == 0:
            raise ValueError("rollout horizon must be positive")
        if max_target_rate_rad_s is not None and (
            not np.isfinite(max_target_rate_rad_s) or max_target_rate_rad_s <= 0
        ):
            raise ValueError("max_target_rate_rad_s must be finite and positive")
        horizon = targets.shape[1]
        base = np.empty((horizon, self.num_candidates, 3))
        base_quat = np.empty((horizon, self.num_candidates, 4))
        feet = np.empty((horizon, self.num_candidates, 2, 3))
        contacts = np.empty((horizon, self.num_candidates, 2), dtype=np.bool_)
        joints = np.empty((horizon, self.num_candidates, self.model.nu))
        applied = np.empty((horizon, self.num_candidates, self.model.nu))
        com = np.empty((horizon, self.num_candidates, 3))
        geom_pos = np.empty((horizon, self.num_candidates, 2, 3))
        geom_mat = np.empty((horizon, self.num_candidates, 2, 9))
        cop = np.empty((horizon, self.num_candidates, 2, 2))
        normal_force = np.empty((horizon, self.num_candidates, 2))
        clearance = np.empty((horizon, self.num_candidates, 2))
        foot_foot_distance = np.empty((horizon, self.num_candidates))
        foot_foot_contact = np.empty((horizon, self.num_candidates), dtype=np.bool_)
        previous = self.ctrl.copy()
        for h in range(horizon):
            control = targets[:, h, :]
            if max_target_rate_rad_s is not None:
                limit = max_target_rate_rad_s * self.control_dt
                control = np.clip(control, previous - limit, previous + limit)
            self.ctrl[:] = control
            self.batch.step(nstep=self.substeps_per_control)
            base[h] = self.base_pos_w
            base_quat[h] = self.base_quat_w
            com[h] = self.com_pos_w
            feet[h, :, 0] = self.foot_pos_w[0]
            feet[h, :, 1] = self.foot_pos_w[1]
            for foot in (0, 1):
                geom_pos[h, :, foot] = self.foot_geom_pos_w[foot]
                geom_mat[h, :, foot] = self.foot_geom_mat_w[foot]
                cop[h, :, foot] = center_of_pressure(
                    self.net_contact_pos[foot],
                    self.net_contact_force[foot],
                    self.net_contact_torque[foot],
                )
                normal_force[h, :, foot] = self.net_contact_force[foot][:, 2]
                clearance[h, :, foot] = self.clearance[foot][:, 0]
            contacts[h, :, 0] = self.contacts[0][:, 0] > 0
            contacts[h, :, 1] = self.contacts[1][:, 0] > 0
            foot_foot_distance[h] = mesh_pair_distance(
                self.foot_foot_distance[:, 0], self.foot_foot_fromto
            )
            foot_foot_contact[h] = self.foot_foot_contact[:, 0] > 0
            joints[h] = self.qpos[:, 7:]
            applied[h] = control
            previous = control
        return RolloutTrace(
            base, base_quat, feet, contacts, joints, applied, com, geom_pos, geom_mat,
            cop, normal_force, clearance, foot_foot_distance, foot_foot_contact,
        )
