import pytest

from zbot_mpc_mjbatch.cadence import CadenceStateMachine, Phase


def test_frequency_changes_single_support_hold_time():
    long_hold = CadenceStateMachine(0.4, minimum_clearance_s=1.0)
    slow = CadenceStateMachine(0.5)
    fast = CadenceStateMachine(2.0)
    assert slow.target_period_s == 2.0
    assert fast.target_period_s == 0.5
    assert slow.current_transfer_s == pytest.approx(0.9)
    assert slow.current_lift_s == pytest.approx(0.3)
    assert slow.current_lower_s == pytest.approx(0.3)
    assert slow.hold_s == pytest.approx(0.5)
    assert fast.hold_s == pytest.approx(0.11)
    assert long_hold.target_period_s == pytest.approx(2.5)
    assert long_hold.current_transfer_s == pytest.approx(0.9)
    assert long_hold.current_lift_s == pytest.approx(0.3)
    assert long_hold.current_lower_s == pytest.approx(0.3)
    assert long_hold.hold_s == pytest.approx(1.0)


def test_contact_guards_switch_and_frequency_change_waits_for_landing():
    gait = CadenceStateMachine(0.5, first_swing_foot=0)
    gait.request_frequency(2.0)
    state = gait.advance(0.5, (False, True))
    assert state.phase == Phase.TRANSFER
    assert state.swing_foot == 0
    assert state.target_period_s == 2.0

    state = gait.advance(1.51, (False, True))
    assert state.phase == Phase.WAIT_CONTACT
    assert state.touchdown_late_s == pytest.approx(0.01)
    assert state.swing_foot == 0

    state = gait.advance(0.02, (True, True))
    assert state.phase == Phase.TRANSFER
    assert state.swing_foot == 1
    assert state.support_foot == 0
    assert state.target_period_s == 0.5
    assert state.elapsed_s == 0.0


def test_normal_landing_preserves_fractional_period_remainder():
    gait = CadenceStateMachine(1.5)
    for _ in range(33):
        gait.advance(0.02, (False, True))
    state = gait.advance(0.02, (True, True))
    assert state.swing_foot == 1
    assert state.elapsed_s == pytest.approx(0.68 - 1.0 / 1.5)


def test_standing_still_does_not_count_as_an_alternating_step():
    gait = CadenceStateMachine(1.0)
    for _ in range(50):
        state = gait.advance(0.02, (True, True))
    assert state.phase == Phase.WAIT_CONTACT
    assert state.swing_foot == 0


def test_real_clearance_must_persist_before_landing_counts():
    gait = CadenceStateMachine(2.0)
    for _ in range(25):
        state = gait.advance(0.02, (False, True), 0.002)
    assert state.phase == Phase.WAIT_CONTACT
    assert state.swing_foot == 0
    for _ in range(5):
        gait.advance(0.02, (False, True), 0.004)
    assert gait.advance(0.02, (True, True), 0.0).swing_foot == 1


def test_one_second_clearance_requirement_blocks_early_landing():
    gait = CadenceStateMachine(0.4, minimum_clearance_s=1.0)
    for _ in range(125):
        gait.advance(0.02, (False, True), 0.004)
    assert gait.advance(0.02, (True, True), 0.0).swing_foot == 1

    interrupted = CadenceStateMachine(0.4, minimum_clearance_s=1.0)
    for i in range(125):
        clearance = 0.0 if i % 40 == 39 else 0.004
        interrupted.advance(0.02, (False, True), clearance)
    assert interrupted.advance(0.02, (True, True), 0.0).swing_foot == 0


def test_invalid_frequency_and_phase_durations_are_rejected():
    with pytest.raises(ValueError):
        CadenceStateMachine(0.0)
    with pytest.raises(ValueError):
        CadenceStateMachine(2.0, transfer_s=0.3, lift_s=0.25, lower_s=0.2)
