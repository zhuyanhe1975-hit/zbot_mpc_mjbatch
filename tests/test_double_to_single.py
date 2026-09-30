import mujoco
import numpy as np

from zbot_mpc_mjbatch.double_to_single import simulate_raise
from zbot_mpc_mjbatch.model import load_model


def test_double_support_transitions_to_raised_single_support():
    run = simulate_raise(threads=8)
    quality = run.summary()

    assert run.stop_reason is None
    assert run.foot_contacts[0].all()
    assert quality["single_support_at_end"] is True
    assert quality["right_foot_off_ground_final_100ms"] is True
    assert quality["minimum_final_100ms_clearance_m"] >= 0.02
    assert quality["minimum_lift_support_force_n"] >= 15
    assert quality["minimum_single_support_zmp_margin_m"] >= 0
    assert quality["minimum_active_support_zmp_margin_m"] >= 0
    assert quality["raised_foot_height_m"] >= 0.1
    assert quality["straight_goal_reached"] is False
    assert quality["minimum_base_height_m"] >= 0.22
    assert quality["minimum_foot_foot_distance_m"] >= 0.002
    assert np.isfinite(run.qpos).all()


def test_straight_reference_points_raised_foot_upward():
    model = load_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("straight_up").id)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.qpos[7:], 0)
    assert data.body("foot_1").xpos[2] > 0.58
    assert data.body("foot_1").xmat.reshape(3, 3)[2, 2] > 0.99
