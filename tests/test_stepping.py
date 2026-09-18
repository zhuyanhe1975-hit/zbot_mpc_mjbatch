import mujoco
import numpy as np
import pytest

from zbot_mpc_mjbatch.model import load_model
from zbot_mpc_mjbatch.stepping import simulate_steps


def test_mpc_completes_alternating_steps_without_falling_or_walking_away():
    run = simulate_steps(seconds=4.3, num_threads=8)

    assert len(run.landing_times_s) >= 4
    np.testing.assert_allclose(np.diff(run.landing_times_s[:4]), 1.0, atol=0.06)
    assert run.base_pos_w[:, 2].min() > 0.24
    drift = np.linalg.norm(run.base_pos_w[:, :2] - run.base_pos_w[0, :2], axis=1)
    assert drift.max() < 0.08
    quality = run.summary()
    peak = quality["minimum_step_peak_clearance_m"]
    clear_run = quality["minimum_step_3mm_clearance_run_s"]
    support_fraction = quality["minimum_step_support_contact_fraction"]
    foot_gap = quality["minimum_foot_foot_distance_m"]
    assert isinstance(peak, float) and peak >= 0.005
    assert isinstance(clear_run, float) and clear_run >= 0.1
    assert isinstance(support_fraction, float) and support_fraction >= 0.9
    assert isinstance(foot_gap, float) and foot_gap >= 0.002
    assert quality["foot_foot_contact_frames"] == 0

    model = load_model()
    data = mujoco.MjData(model)
    assert run.qpos.shape == (len(run.times_s), model.nq)
    assert run.qvel.shape == (len(run.times_s), model.nv)
    assert run.ctrl.shape == (len(run.times_s), model.nu)
    for i in (0, len(run.times_s) // 2, len(run.times_s) - 1):
        data.qpos[:] = run.qpos[i]
        data.qvel[:] = run.qvel[i]
        data.ctrl[:] = run.ctrl[i]
        data.time = float(run.times_s[i])
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.body("base").xpos, run.base_pos_w[i], atol=1e-9)
        np.testing.assert_allclose(
            data.subtree_com[model.body("foot_0").id], run.com_pos_w[i], atol=1e-9
        )
        for foot in (0, 1):
            np.testing.assert_allclose(
                data.sensor(f"foot_{foot}_net_force").data[2],
                run.foot_normal_force_n[i, foot],
                atol=1e-8,
            )

    single = run.foot_contacts[np.arange(len(run.times_s)), 1 - run.swing_feet] & ~run.foot_contacts[
        np.arange(len(run.times_s)), run.swing_feet
    ]
    assert np.isfinite(run.support_com_margin_m[single]).all()
    assert np.isfinite(run.support_cop_margin_m[single]).all()
    assert run.support_cop_margin_m[single].min() > -0.005

    # Each completed step has a real airborne interval, a planted support foot,
    # and measured contact on both feet at its boundary.
    start = 0
    for landing_time in run.landing_times_s[:4]:
        end = int(round(landing_time / 0.02))
        swing = int(run.swing_feet[start])
        active = slice(start + 10, end - 5)
        assert (~run.foot_contacts[active, swing]).sum() >= 8
        assert run.foot_contacts[active, 1 - swing].mean() > 0.7
        assert run.foot_contacts[end - 1].all()
        start = end


@pytest.mark.parametrize(
    ("frequency_hz", "seconds", "minimum_steps"),
    [(0.5, 4.2, 2), (2.0, 2.2, 3)],
)
def test_cadence_endpoints_complete_contact_validated_steps(
    frequency_hz, seconds, minimum_steps
):
    run = simulate_steps(frequency_hz, seconds, num_threads=8)
    assert len(run.landing_times_s) >= minimum_steps
    assert run.base_pos_w[:, 2].min() > 0.24
    interval = np.diff(run.landing_times_s)
    np.testing.assert_allclose(interval, 1.0 / frequency_hz, atol=0.18)
    assert run.summary()["foot_foot_contact_frames"] == 0
