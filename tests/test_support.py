import mujoco
import numpy as np

from zbot_mpc_mjbatch.model import load_model, standing_joint_targets
from zbot_mpc_mjbatch.support import (
    FootSupportGeometry,
    center_of_pressure,
    mesh_pair_distance,
)


def test_foot_distance_preserves_penetration_and_repairs_zero_with_separated_witnesses():
    raw = np.array([-0.01, 0.0, 0.03])
    fromto = np.array(
        [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0.02, 0, 0], [0, 0, 0, 0.03, 0, 0]]
    )
    np.testing.assert_allclose(mesh_pair_distance(raw, fromto), [-0.01, 0.02, 0.03])


def test_foot_contact_sensor_detects_a_colliding_pose():
    model = load_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
    data.qpos[7:] = [-0.19, 0.45, -2.59, 2.46, -0.89, -0.78]
    mujoco.mj_forward(model, data)
    assert data.sensor("foot_foot_distance").data[0] < -0.01
    assert data.sensor("foot_foot_contact").data[0] > 0


def test_center_of_pressure_uses_the_complete_wrench_and_rejects_unloaded_foot():
    position = np.array([0.02, -0.03, 0.0])
    force = np.array([0.0, 0.0, 10.0])
    torque = np.array([0.1, -0.2, 0.0])
    np.testing.assert_allclose(center_of_pressure(position, force, torque), [0.04, -0.02])
    assert np.isnan(center_of_pressure(position, np.zeros(3), torque)).all()


def test_sole_polygon_margin_tracks_foot_pose():
    model = load_model()
    support = FootSupportGeometry(model)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
    mujoco.mj_forward(model, data)
    geom_id = support.geom_ids[0]
    center = data.body("foot_0").xpos[:2].copy()
    inside = support.margin(center, data.geom_xpos[geom_id], data.geom_xmat[geom_id], 0)
    outside = support.margin(
        center + [0.0, 0.08], data.geom_xpos[geom_id], data.geom_xmat[geom_id], 0
    )
    shifted = support.margin(
        center + [0.1, 0.0],
        data.geom_xpos[geom_id] + [0.1, 0.0, 0.0],
        data.geom_xmat[geom_id],
        0,
    )
    assert inside > 0.03
    assert outside < 0
    np.testing.assert_allclose(shifted, inside)


def test_netforce_sensors_match_all_ground_contacts_not_only_one_contact():
    model = load_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
    data.ctrl[:] = standing_joint_targets(model)
    for _ in range(6):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    floor_id = model.geom("floor").id
    for foot in (0, 1):
        foot_id = model.geom(f"foot_{foot}_collisions").id
        force = np.zeros(3)
        moment = np.zeros(3)
        for index in range(data.ncon):
            contact = data.contact[index]
            if {contact.geom1, contact.geom2} != {floor_id, foot_id}:
                continue
            contact_wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, contact_wrench)
            rotation = contact.frame.reshape(3, 3)
            contact_force = rotation.T @ contact_wrench[:3]
            contact_torque = rotation.T @ contact_wrench[3:]
            if contact.geom1 != floor_id:
                contact_force = -contact_force
                contact_torque = -contact_torque
            force += contact_force
            moment += np.cross(contact.pos, contact_force) + contact_torque

        position = data.sensor(f"foot_{foot}_net_pos").data
        sensor_force = data.sensor(f"foot_{foot}_net_force").data
        sensor_torque = data.sensor(f"foot_{foot}_net_torque").data
        np.testing.assert_allclose(sensor_force, force, atol=1e-8)
        np.testing.assert_allclose(np.cross(position, sensor_force) + sensor_torque, moment, atol=1e-8)
