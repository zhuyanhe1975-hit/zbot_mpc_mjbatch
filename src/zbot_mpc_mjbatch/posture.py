"""Kinematic posture proposals; MPC must validate their contact dynamics."""

import mujoco
import numpy as np

from .model import standing_joint_targets
from .support import FootSupportGeometry


def support_posture_proposals(model: mujoco.MjModel) -> np.ndarray:
    """Find transfer/lift poses with COM near the fixed support sole center.

    These are only seeds for the sampler. Fixed-foot kinematics cannot certify
    contact forces, dynamic balance, or whether a transition can be executed.
    """
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
    mujoco.mj_forward(model, data)
    initial = data.qpos.copy()
    stand = standing_joint_targets(model)
    geometry = FootSupportGeometry(model)
    positions = [data.body(f"foot_{i}").xpos.copy() for i in (0, 1)]
    quats = [data.body(f"foot_{i}").xquat.copy() for i in (0, 1)]
    centers = [
        geometry.polygon_world(data.geom_xpos[gid], data.geom_xmat[gid], foot).mean(axis=0)[:2]
        for foot, gid in enumerate(geometry.geom_ids)
    ]
    proposals = np.zeros((2, 18))
    for support in (0, 1):
        swing = 1 - support

        def residual(
            offset: np.ndarray, height: float, support: int = support, swing: int = swing
        ) -> np.ndarray:
            data.qpos[:] = initial
            data.qpos[7:] = stand + offset
            mujoco.mj_forward(model, data)
            if support == 1:
                # Reposition the free root so foot_1 keeps its original pose.
                position = data.body("foot_1").xpos.copy()
                inverse = data.body("foot_1").xquat.copy()
                inverse[1:] *= -1
                rotation = np.empty(4)
                mujoco.mju_mulQuat(rotation, quats[1], inverse)
                matrix = np.empty(9)
                mujoco.mju_quat2Mat(matrix, rotation)
                data.qpos[:3] = positions[1] + matrix.reshape(3, 3) @ (data.qpos[:3] - position)
                quaternion = np.empty(4)
                mujoco.mju_mulQuat(quaternion, rotation, data.qpos[3:7])
                data.qpos[3:7] = quaternion
                mujoco.mj_forward(model, data)
            return np.r_[
                data.subtree_com[model.body("foot_0").id, :2] - centers[support],
                data.sensor(f"foot_{swing}_clearance").data[0] - height,
                0.2 * (data.body(f"foot_{swing}").xpos[:2] - positions[swing][:2]),
                0.005 * offset,
            ]

        poses = []
        for height in (0.0, 0.02):
            offset = np.zeros(6)
            for _ in range(60):
                error = residual(offset, height)
                jacobian = np.column_stack(
                    [
                        (residual(offset + np.eye(6)[i] * 1e-4, height) - error) / 1e-4
                        for i in range(6)
                    ]
                )
                delta = np.linalg.solve(
                    jacobian.T @ jacobian + 1e-5 * np.eye(6), -jacobian.T @ error
                )
                for scale in (1.0, 0.5, 0.25, 0.1, 0.01):
                    trial = np.clip(offset + scale * delta, -0.6, 0.6)
                    if np.linalg.norm(residual(trial, height)) < np.linalg.norm(error):
                        offset = trial
                        break
                else:
                    break
            poses.append(offset)
        proposals[swing, :6] = poses[1] - poses[0]
        proposals[swing, 6:12] = poses[0]
    return proposals
