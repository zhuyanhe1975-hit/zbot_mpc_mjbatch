import numpy as np
import pytest

from zbot_mpc_mjbatch.single_leg import simulate_single_leg_stand


@pytest.mark.parametrize("support_foot", [0, 1])
def test_robot_transitions_from_two_feet_and_holds_single_leg(support_foot):
    run = simulate_single_leg_stand(
        support_foot=support_foot,
        hold_seconds=3.0,
        num_threads=8,
    )
    swing_foot = 1 - support_foot
    hold = run.times_s >= run.hold_start_s

    assert run.stop_reason is None
    assert run.foot_contacts[0].all()
    assert run.foot_contacts[-1, support_foot]
    assert not run.foot_contacts[-1, swing_foot]
    assert (run.foot_clearance_m[hold, swing_foot] >= 0.003).all()
    assert (run.foot_normal_force_n[hold, support_foot] >= 15.0).all()
    assert run.base_pos_w[:, 2].min() > 0.24
    assert abs(run.foot_pos_w[-1, 1, 2] - run.foot_pos_w[-1, 0, 2]) > 0.55
    assert np.abs(run.qpos[-1, 7:]).max() < 1e-3
    assert np.abs(run.ctrl[-1]).max() < 1e-12
    assert run.foot_foot_distance_m.min() >= 0.002
    assert not run.foot_foot_contact.any()
    assert np.isfinite(run.qpos).all()

    quality = run.summary()
    assert quality["final_single_support"] is True
    assert quality["achieved_continuous_hold_seconds"] >= 3.0
    assert quality["minimum_hold_clearance_m"] >= 0.003
    assert quality["minimum_hold_support_force_n"] >= 15.0
    # A successful stand must also settle: the previous 5 ms physics had
    # ~0.027 generalized-velocity RMS and ~1 mm column rocking here.
    assert quality["last_second_hold_qvel_rms"] < 0.012
    assert quality["last_second_hold_base_motion_m"] < 0.0008
    dt = run.times_s[0]
    acceleration = np.diff(run.ctrl, n=2, axis=0) / dt**2
    entering_hold = (
        (run.times_s[2:] >= run.hold_start_s - 0.15)
        & (run.times_s[2:] <= run.hold_start_s + 0.05)
    )
    assert np.abs(acceleration[entering_hold]).max() < 0.15


@pytest.mark.parametrize(
    ("support_foot", "hold_seconds", "unfold_seconds"),
    [
        (-1, 1.0, 6.0),
        (2, 1.0, 6.0),
        (0, 0.0, 6.0),
        (0, float("nan"), 6.0),
        (0, 1.0, 0.0),
    ],
)
def test_single_leg_rejects_invalid_requests(support_foot, hold_seconds, unfold_seconds):
    with pytest.raises(ValueError):
        simulate_single_leg_stand(
            support_foot=support_foot,
            hold_seconds=hold_seconds,
            unfold_seconds=unfold_seconds,
        )
