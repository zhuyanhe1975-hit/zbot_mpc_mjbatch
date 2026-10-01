from dataclasses import replace

import numpy as np
import pytest

from zbot_mpc_mjbatch.double_leg import ReturnMPC, simulate_double_leg_stand
from zbot_mpc_mjbatch.model import load_model, standing_joint_targets
from zbot_mpc_mjbatch.single_leg import simulate_single_leg_stand


@pytest.fixture(scope="module", params=[0, 1])
def column(request):
    return simulate_single_leg_stand(support_foot=request.param, hold_seconds=1, num_threads=8)


def test_reverse_mpc_returns_measured_column_to_loaded_double_support(column):
    run = simulate_double_leg_stand(
        support_foot=column.support_foot,
        source_run=column,
        hold_seconds=3,
        num_threads=8,
    )
    support = column.support_foot
    alone = ~run.foot_contacts[:, 1 - support]
    hold = run.times_s >= run.hold_start_s
    stand = standing_joint_targets(load_model())

    assert run.stop_reason is None
    assert run.foot_contacts[0, support]
    assert not run.foot_contacts[0, 1 - support]
    assert np.abs(run.qpos[0, 7:]).max() < 0.001
    assert np.linalg.norm(run.base_pos_w[0] - column.base_pos_w[-1]) < 0.002
    assert run.foot_contacts[:, support].all()
    assert (run.foot_normal_force_n[alone, support] >= 15).all()
    assert run.base_pos_w[:, 2].min() >= 0.22
    assert run.foot_foot_distance_m.min() >= 0.002
    assert not run.foot_foot_contact.any()
    assert run.foot_contacts[hold].all()
    assert (run.foot_normal_force_n[hold] >= 1).all()
    assert np.abs(run.qpos[-1, 7:] - stand).max() < 0.025
    assert np.isfinite(run.qpos).all()
    applied = np.vstack((column.ctrl[-1], run.ctrl))
    assert np.abs(np.diff(applied, axis=0)).max() <= 0.8 * 0.02 + 1e-12
    summary = run.summary()
    assert summary["final_double_support"]
    assert summary["achieved_continuous_double_support_seconds"] >= 3
    assert summary["last_second_hold_qvel_rms"] < 0.02
    assert summary["mpc_replans"] == 2
    assert summary["safe_candidate_evaluations"] > 0


def test_reverse_rejects_failed_or_wrong_support_source(column):
    with pytest.raises(ValueError, match="successful column"):
        simulate_double_leg_stand(support_foot=1 - column.support_foot, source_run=column)
    with pytest.raises(ValueError, match="successful column"):
        simulate_double_leg_stand(
            support_foot=column.support_foot,
            source_run=replace(column, stop_reason="failed"),
        )


def test_search_never_executes_an_unsafe_reference(column, monkeypatch):
    model = load_model()
    model.opt.timestep = 0.001
    optimizer = ReturnMPC(model, num_candidates=8, iterations=1, num_threads=2)
    # Force every rollout to fail the common safety predicate. Even the seed
    # reference must be rejected rather than returned as an unchecked fallback.
    monkeypatch.setattr(
        "zbot_mpc_mjbatch.double_leg._safety",
        lambda trace, support: np.zeros(trace.foot_foot_distance_m.shape, dtype=bool),
    )
    with pytest.raises(RuntimeError, match="no safe reverse MPC candidate"):
        optimizer.plan(
            column.qpos[-1],
            column.qvel[-1],
            column.ctrl[-1],
            np.zeros((2, 6)),
            column.support_foot,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"support_foot": -1},
        {"support_foot": 2},
        {"hold_seconds": 0},
        {"hold_seconds": float("nan")},
        {"num_candidates": 7},
        {"iterations": 0},
    ],
)
def test_reverse_rejects_invalid_requests(kwargs):
    with pytest.raises(ValueError):
        simulate_double_leg_stand(**kwargs)


def test_failed_landing_replan_returns_completed_motion_without_fallback(column, monkeypatch):
    original = ReturnMPC.plan
    calls = 0

    def fail_second_plan(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("no safe reverse MPC candidate for two-foot landing")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ReturnMPC, "plan", fail_second_plan)
    run = simulate_double_leg_stand(
        support_foot=column.support_foot,
        source_run=column,
        num_candidates=8,
        iterations=1,
        num_threads=2,
    )
    assert run.stop_reason is not None
    assert "no safe reverse MPC candidate" in run.stop_reason
    assert run.times_s[-1] == pytest.approx(column.unfold_s)
    assert len(run.plans) == 1
    assert run.summary()["terminated_early"]
    assert run.summary()["achieved_continuous_double_support_seconds"] == 0


def test_actual_terminal_motion_is_checked_after_planning(column, monkeypatch):
    from zbot_mpc_mjbatch.rollout import BatchRollout

    original = BatchRollout.rollout
    frames = 0
    hold_seconds = 1.0
    expected = int((column.times_s < column.hold_start_s).sum()) + 1 + 51 + 50

    def inject_final_velocity(self, *args, **kwargs):
        nonlocal frames
        trace = original(self, *args, **kwargs)
        if self.num_candidates == 1:
            frames += 1
            if frames == expected:
                self.qvel[:] = 2.0
        return trace

    monkeypatch.setattr(BatchRollout, "rollout", inject_final_velocity)
    run = simulate_double_leg_stand(
        support_foot=column.support_foot,
        source_run=column,
        hold_seconds=hold_seconds,
        num_candidates=8,
        iterations=1,
        num_threads=2,
    )
    assert run.foot_contacts[-1].all()
    assert run.stop_reason == "measured terminal velocity RMS exceeded 0.02"
    assert run.summary()["terminated_early"]
