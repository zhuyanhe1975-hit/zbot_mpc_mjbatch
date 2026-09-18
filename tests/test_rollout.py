import mujoco
import numpy as np

from zbot_mpc_mjbatch.model import load_model, standing_joint_targets
from zbot_mpc_mjbatch.rollout import BatchRollout


def test_copied_zbot_model_stands_in_batched_mujoco():
    model = load_model()
    assert (model.nq, model.nv, model.nu) == (13, 12, 6)
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    simulator = BatchRollout(model, num_candidates=4, num_threads=2)
    simulator.reset_to_stand()
    targets = standing_joint_targets(model)
    trace = simulator.rollout(np.broadcast_to(targets, (4, 50, 6)).copy())

    assert trace.base_pos_w.shape == (50, 4, 3)
    assert trace.foot_pos_w.shape == (50, 4, 2, 3)
    assert trace.foot_contacts.shape == (50, 4, 2)
    assert np.isfinite(trace.base_pos_w).all()
    assert trace.base_pos_w[:, :, 2].min() > 0.2
    assert trace.foot_contacts[-1].all()
    drift = trace.base_pos_w[-1, :, :2] - trace.base_pos_w[0, :, :2]
    assert np.linalg.norm(drift, axis=1).max() < 0.05


def test_rollout_limits_position_target_change_per_control_step():
    model = load_model()
    simulator = BatchRollout(model, num_candidates=2, num_threads=2)
    simulator.reset_to_stand()
    stand = standing_joint_targets(model)
    targets = np.broadcast_to(stand + 1.0, (2, 5, 6)).copy()
    trace = simulator.rollout(targets, max_target_rate_rad_s=0.5)
    applied = np.concatenate([np.broadcast_to(stand, (1, 2, 6)), trace.applied_targets])
    assert np.abs(np.diff(applied, axis=0)).max() <= 0.5 * simulator.control_dt + 1e-12


def test_rollout_initializes_all_candidates_from_a_current_state():
    model = load_model()
    simulator = BatchRollout(model, num_candidates=3, num_threads=2)
    stand = standing_joint_targets(model)
    simulator.set_state(model.key("stand").qpos.copy(), np.zeros(model.nv), stand, time_s=0.3)
    np.testing.assert_allclose(
        simulator.qpos, np.broadcast_to(model.key("stand").qpos, simulator.qpos.shape)
    )
    np.testing.assert_allclose(simulator.ctrl, np.broadcast_to(stand, simulator.ctrl.shape))
    np.testing.assert_allclose(simulator.time, np.full(3, 0.3))
