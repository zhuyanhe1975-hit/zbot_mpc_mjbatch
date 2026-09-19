import numpy as np
import pytest

from zbot_mpc_mjbatch.cadence import CadenceStateMachine
from zbot_mpc_mjbatch.model import load_model, standing_joint_targets
from zbot_mpc_mjbatch.mpc import StepMPC
from zbot_mpc_mjbatch.warmstart import (
    PLAN_BOUNDS,
    PLAN_DIM,
    LinearWarmStartPolicy,
    WarmStartDataset,
    make_warm_start_observation,
)


def test_warm_start_observation_is_finite_and_ignores_world_xy_translation():
    model = load_model()
    state = CadenceStateMachine(0.5, first_swing_foot=1).state()
    qpos = model.key("stand").qpos.copy()
    qvel = np.zeros(model.nv)
    ctrl = standing_joint_targets(model)

    first = make_warm_start_observation(model, qpos, qvel, ctrl, state)
    qpos[:2] += [3.0, -4.0]
    translated = make_warm_start_observation(model, qpos, qvel, ctrl, state)

    assert first.ndim == 1
    assert np.isfinite(first).all()
    np.testing.assert_allclose(translated, first, atol=1e-7)


def test_linear_policy_fits_saves_and_clips_predictions(tmp_path):
    rng = np.random.default_rng(3)
    observations = rng.normal(size=(80, 40))
    observations[:40, 31] = -1.0
    observations[40:, 31] = 1.0
    mappings = rng.normal(scale=0.003, size=(2, 41, PLAN_DIM))
    augmented = np.column_stack([observations, np.ones(len(observations))])
    targets = np.empty((len(observations), PLAN_DIM))
    targets[:40] = augmented[:40] @ mappings[0]
    targets[40:] = augmented[40:] @ mappings[1]
    policy = LinearWarmStartPolicy.fit(observations, targets, ridge=1e-8)

    np.testing.assert_allclose(policy.predict(observations[0]), targets[0], atol=2e-4)
    np.testing.assert_allclose(policy.predict(observations[-1]), targets[-1], atol=2e-4)
    path = tmp_path / "warm-start.npz"
    policy.save(path)
    restored = LinearWarmStartPolicy.load(path)
    np.testing.assert_allclose(restored.predict(observations[0]), policy.predict(observations[0]))

    extreme = restored.predict(np.full(40, 1e9))
    assert np.all(np.abs(extreme) <= PLAN_BOUNDS + 1e-12)


def test_mpc_rejects_malformed_policy_output_before_rollout():
    class BadPolicy:
        def predict(self, observation):
            return np.zeros(PLAN_DIM - 1)

    model = load_model()
    mpc = StepMPC(model, num_candidates=8, iterations=1, warm_start_policy=BadPolicy())
    qpos = model.key("stand").qpos.copy()
    state = CadenceStateMachine(0.5, first_swing_foot=1).state()

    with pytest.raises(ValueError, match="18 finite values"):
        mpc.plan(qpos, np.zeros(model.nv), standing_joint_targets(model), state)


def test_expert_dataset_round_trips_and_trains_policy(tmp_path):
    observations = np.arange(160, dtype=np.float32).reshape(4, 40)
    observations[:2, 31] = -1.0
    observations[2:, 31] = 1.0
    targets = np.tile(np.linspace(-0.1, 0.1, PLAN_DIM, dtype=np.float32), (4, 1))
    dataset = WarmStartDataset(observations, targets, np.arange(4, dtype=float))
    path = tmp_path / "expert-data.npz"
    dataset.save(path)

    restored = WarmStartDataset.load(path)
    np.testing.assert_array_equal(restored.observations, observations)
    np.testing.assert_array_equal(restored.targets, targets)
    policy = LinearWarmStartPolicy.fit(restored.observations, restored.targets)
    np.testing.assert_allclose(policy.predict(observations[0]), targets[0], atol=1e-6)
