import pytest

from zbot_mpc_mjbatch.stepping import periodic_search_budget, simulate_periodic_steps


@pytest.mark.parametrize("frequency_hz", [0.5, 1.0, 1.5, 2.0])
def test_periodic_mpc_completes_eight_safe_alternating_steps(frequency_hz):
    run = simulate_periodic_steps(frequency_hz, steps=8, num_threads=8)
    quality = run.summary()

    assert run.stop_reason is None
    assert quality["completed_steps"] == 8
    assert [plan.swing_foot for plan in run.plans] == [1, 0, 1, 0, 1, 0, 1, 0]
    assert quality["support_contact_loss_frames"] == 0
    assert quality["weak_single_support_frames"] == 0
    assert quality["minimum_single_support_force_n"] >= 15.0
    assert quality["foot_foot_contact_frames"] == 0
    assert quality["minimum_foot_foot_distance_m"] >= 0.002
    assert quality["both_feet_contact_at_end"] is True
    assert 0.75 <= quality["cadence_tracking_ratio"] <= 1.1


def test_periodic_budget_covers_only_validated_frequency_range():
    assert periodic_search_budget(0.5) == (64, 4)
    assert periodic_search_budget(2.0) == (128, 8)
    with pytest.raises(ValueError):
        periodic_search_budget(0.49)
    with pytest.raises(ValueError):
        periodic_search_budget(2.01)
