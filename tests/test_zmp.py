import numpy as np

from zbot_mpc_mjbatch.zmp import plan_zmp_preview


def test_preview_moves_zmp_between_foot_centers_before_each_swing():
    feet = np.array([[0.0, -0.06], [0.0, 0.06]])
    preview = plan_zmp_preview(np.zeros(2), 0.15, feet, 1, 1.0, 0.0, 0.02, 110)
    np.testing.assert_allclose(preview.zmp_xy[14], feet[0], atol=1e-12)
    np.testing.assert_allclose(preview.zmp_xy[64], feet[1], atol=1e-12)
    assert np.isfinite(preview.com_xy).all()
    assert np.max(np.abs(preview.com_xy[:, 1])) <= 0.06


def test_preview_stays_at_constant_support_center():
    feet = np.array([[0.0, 0.02], [0.0, 0.02]])
    preview = plan_zmp_preview(np.array([0.0, 0.02]), 0.15, feet, 0, 1.0, 0.0, 0.02, 60)
    expected = np.tile(feet[0], (60, 1))
    np.testing.assert_allclose(preview.zmp_xy, expected, atol=1e-12)
    np.testing.assert_allclose(preview.com_xy, expected, atol=1e-12)
