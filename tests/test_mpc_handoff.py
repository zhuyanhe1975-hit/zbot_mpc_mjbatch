import numpy as np
import pytest

from zbot_mpc_mjbatch.cadence import CadenceStateMachine, Phase
from zbot_mpc_mjbatch.mpc import StepMPC, StepPlan, landing_state_is_safe
from zbot_mpc_mjbatch.zmp import plan_zmp_preview


def test_touching_at_one_instant_is_not_a_loaded_handoff():
    contacts = np.ones((3, 3, 2), dtype=bool)
    forces = np.full((3, 2), 14.7)
    velocities = np.zeros((3, 12))
    contacts[0, 1, 1] = False  # Momentary contact at touchdown is insufficient.
    forces[2, 1] = 5.0  # The foot becoming support must actually carry load.
    np.testing.assert_array_equal(
        landing_state_is_safe(contacts, forces, velocities, swing=1, period_s=2.0),
        [True, False, False],
    )


def test_cadence_waits_for_measured_handoff_even_after_swing_clearance():
    gait = CadenceStateMachine(1.0)
    for _ in range(50):
        gait.advance(0.02, (False, True), 0.01)
    waiting = gait.advance(0.02, (True, True), 0.0, landing_ready=False)
    assert waiting.phase == Phase.WAIT_CONTACT
    assert waiting.swing_foot == 0
    assert gait.advance(0.02, (True, True), 0.0, landing_ready=True).swing_foot == 1


def test_optimized_landing_posture_is_preserved_between_steps():
    mpc = StepMPC.__new__(StepMPC)
    mpc.stand = np.arange(6, dtype=float)
    start = np.full(6, 0.03)
    end = np.full(6, -0.04)
    plan = StepPlan(
        1, 2.0, np.ones(6), np.ones(6), np.ones(6), 0.0, landing_offset=end, start_offset=start
    )
    np.testing.assert_allclose(mpc.target(plan, 0.0), mpc.stand + start)
    np.testing.assert_allclose(mpc.target(plan, 2.0), mpc.stand + end)
    np.testing.assert_allclose(mpc.target(plan, 2.4), mpc.stand + end)
    assert StepMPC.start_amplitude(1.0, 2.0) == 1.0


@pytest.mark.parametrize("index, speed", [(0, 0.11), (3, 0.61), (6, 0.81)])
def test_slow_steps_do_not_relax_landing_speed_limits(index, speed):
    velocity = np.zeros(12)
    velocity[index] = speed
    assert not landing_state_is_safe(
        np.ones((3, 2), dtype=bool), np.full(2, 14.7), velocity, 1, 4.0
    )


def test_unloading_during_transfer_does_not_count_as_the_planned_swing():
    gait = CadenceStateMachine(0.5)
    for _ in range(100):
        gait.advance(0.005, (False, True), 0.02)
    for _ in range(300):
        state = gait.advance(0.005, (True, True), 0.0)
    assert state.swing_foot == 0


def test_planned_double_support_dwell_is_executed_before_switching():
    gait = CadenceStateMachine(1.0, handoff_hold_s=0.5)
    for i in range(200):
        airborne = 60 <= i < 140
        state = gait.advance(0.005, (not airborne, True), 0.01 if airborne else 0.0)
    assert state.swing_foot == 0
    assert state.phase == Phase.SETTLE
    for _ in range(100):
        state = gait.advance(0.005, (True, True), 0.0)
    assert state.swing_foot == 1


def test_zmp_reference_returns_to_double_support_during_settling():
    feet = np.array([[0.0, -0.06], [0.0, 0.06]])
    preview = plan_zmp_preview(
        np.zeros(2), 0.2, feet, 1, 2.0, 0.0, 0.02, 100, transfer_s=0.9, lower_s=0.3, settle_s=0.3
    )
    np.testing.assert_allclose(preview.zmp_xy[-15:], 0.0, atol=1e-12)


@pytest.mark.parametrize("frequency", [0.25, 0.5, 1.0, 2.0])
def test_transfer_persists_through_swing_and_targets_settle_before_handoff(frequency):
    gait = CadenceStateMachine(frequency)
    period = gait.target_period_s
    hold_middle = gait.current_transfer_s + gait.current_lift_s + gait.hold_s / 2
    assert StepMPC.transfer_amplitude(hold_middle, period) == pytest.approx(1.0)
    assert StepMPC.amplitude(hold_middle, period) == pytest.approx(1.0)
    assert StepMPC.amplitude(period - 0.04, period) == 0
    assert StepMPC.transfer_amplitude(period - 0.04, period) == 0
    assert StepMPC.hold_amplitude(period - 0.04, period) == 0
    gait.elapsed_s = period - 0.04
    assert gait.state().phase == Phase.SETTLE
