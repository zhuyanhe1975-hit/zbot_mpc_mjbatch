"""NumPy inference for the existing PPO walker, used only as an MPC proposal."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


class PPOProposal:
    """Turn a PPO action into a bounded transfer-pulse warm-start candidate.

    This policy was trained for forward walking. Its action is therefore never
    applied to the simulated or executed robot directly: MPC evaluates the
    resulting pulse in MuJoCo and retains it only if every contact constraint
    passes.
    """

    def __init__(self, model: mujoco.MjModel, weights_path: Path) -> None:
        weights = np.load(weights_path)
        self.weights = tuple(
            (weights[f"w{i}"].astype(np.float64), weights[f"b{i}"].astype(np.float64))
            for i in range(3)
        )
        if tuple(self.weights[0][0].shape) != (128, 28):
            raise ValueError(f"unexpected PPO actor input shape at {weights_path}")
        self.model = model
        self.stand = np.array(
            [model.key("stand").qpos[model.joint(model.actuator(i).name).qposadr[0]]
             for i in range(model.nu)]
        )
        self.qpos_addresses = np.array(
            [model.joint(model.actuator(i).name).qposadr[0] for i in range(model.nu)]
        )
        self.qvel_addresses = np.array(
            [model.joint(model.actuator(i).name).dofadr[0] for i in range(model.nu)]
        )
        self.last_action = np.zeros(6, dtype=np.float64)

    def _actor(self, observation: np.ndarray) -> np.ndarray:
        value = observation
        for index, (weight, bias) in enumerate(self.weights):
            value = value @ weight.T + bias
            if index < 2:
                value = np.where(value > 0.0, value, np.expm1(value))
        return np.tanh(value)

    @staticmethod
    def _rotate(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
        xyz = quat[1:]
        uv = np.cross(xyz, vector)
        uuv = np.cross(xyz, uv)
        return vector + 2.0 * (quat[0] * uv + uuv)

    @classmethod
    def _rotate_inverse(cls, quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
        return cls._rotate(np.r_[quat[0], -quat[1:]], vector)

    def propose(
        self, qpos: np.ndarray, qvel: np.ndarray, transfer_duration_s: float
    ) -> np.ndarray:
        """Return an 18D MPC candidate; six entries are the transfer pulse."""
        data = mujoco.MjData(self.model)
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(self.model, data)
        base_quat = data.xquat[self.model.body("base").id].copy()
        gyro = data.sensor("gyro").data.copy()
        angular_world = self._rotate(base_quat, gyro)
        gravity_body = self._rotate_inverse(base_quat, np.array([0.0, 0.0, -1.0]))
        forward_body = np.cross(gravity_body, np.array([0.0, 0.0, 1.0]))
        forward_body /= max(np.linalg.norm(forward_body), 1e-8)
        forward_world = self._rotate(base_quat, forward_body)
        contacts = np.array(
            [data.sensor(f"foot_{i}_contact").data[0] > 0.0 for i in (0, 1)],
            dtype=bool,
        )
        flat_contact = np.zeros(2, dtype=np.float64)
        reference_normals = (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0]))
        for foot in (0, 1):
            if contacts[foot]:
                normal = self._rotate(
                    data.xquat[self.model.body(f"foot_{foot}").id], reference_normals[foot]
                )
                tilt = np.arccos(np.clip(normal[2], -1.0, 1.0))
                flat_contact[foot] = 1.0 - np.exp(-((tilt / np.deg2rad(15.0)) ** 2))
        observation = np.concatenate(
            (
                angular_world[2:3],
                gravity_body,
                -forward_world[1:2],
                qpos[self.qpos_addresses] - self.stand,
                0.1 * qvel[self.qvel_addresses],
                self.last_action,
                [2.0],
                flat_contact,
                contacts.astype(np.float64),
            )
        )
        action = self._actor(observation)
        self.last_action = action
        # The PPO walker integrates pi * 2 rad/s of target offset. Use the
        # transfer interval as a conservative proposal, clipped to the MPC's
        # existing pulse bounds. It is a search seed, not a control command.
        transfer = np.clip(np.pi * 2.0 * transfer_duration_s * action, -0.35, 0.35)
        proposal = np.zeros(18, dtype=np.float64)
        proposal[6:12] = transfer
        return proposal
