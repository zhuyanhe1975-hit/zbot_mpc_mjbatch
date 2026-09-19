"""Learned proposal interface for safety-filtered step MPC warm starts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import mujoco
import numpy as np

from .cadence import StepState
from .model import standing_joint_targets

PLAN_DIM = 18
PLAN_BOUNDS = np.r_[np.full(6, 0.6), np.full(12, 0.4)]
SWING_SIGN_INDEX = 31


class WarmStartPolicy(Protocol):
    """A policy that proposes an MPC search center, never a direct command."""

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Return the 18 transfer, swing, and hold pulse parameters."""


class ExpertPlan(Protocol):
    observation: np.ndarray
    cost: float

    @property
    def parameter_vector(self) -> np.ndarray: ...


@dataclass(frozen=True)
class WarmStartDataset:
    """MPC expert labels used to bootstrap a learned search center."""

    observations: np.ndarray
    targets: np.ndarray
    costs: np.ndarray

    def __post_init__(self) -> None:
        observations = np.asarray(self.observations, dtype=np.float32)
        targets = np.asarray(self.targets, dtype=np.float32)
        costs = np.asarray(self.costs, dtype=np.float64)
        if observations.ndim != 2 or targets.shape != (len(observations), PLAN_DIM):
            raise ValueError("expected observations (N, D) and targets (N, 18)")
        if observations.shape[1] <= SWING_SIGN_INDEX:
            raise ValueError("observations are missing the swing-foot feature")
        if costs.shape != (len(observations),):
            raise ValueError("costs must contain one scalar per sample")
        if not all(np.isfinite(value).all() for value in (observations, targets, costs)):
            raise ValueError("dataset values must be finite")
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "costs", costs)

    @classmethod
    def from_plans(cls, plans: Sequence[ExpertPlan]) -> WarmStartDataset:
        if not plans:
            raise ValueError("cannot build a warm-start dataset without plans")
        return cls(
            np.stack([np.asarray(plan.observation) for plan in plans]),
            np.stack([np.asarray(plan.parameter_vector) for plan in plans]),
            np.asarray([float(plan.cost) for plan in plans]),
        )

    @classmethod
    def concatenate(cls, datasets: list[WarmStartDataset]) -> WarmStartDataset:
        if not datasets:
            raise ValueError("at least one dataset is required")
        return cls(
            np.concatenate([dataset.observations for dataset in datasets]),
            np.concatenate([dataset.targets for dataset in datasets]),
            np.concatenate([dataset.costs for dataset in datasets]),
        )

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            format_version=np.array(1, dtype=np.int64),
            observations=self.observations,
            targets=self.targets,
            costs=self.costs,
        )

    @classmethod
    def load(cls, path: str | Path) -> WarmStartDataset:
        with np.load(path, allow_pickle=False) as archive:
            version = int(archive["format_version"])
            if version != 1:
                raise ValueError(f"unsupported warm-start dataset format version {version}")
            return cls(archive["observations"], archive["targets"], archive["costs"])


def make_warm_start_observation(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray,
    ctrl: np.ndarray,
    state: StepState,
) -> np.ndarray:
    """Build a translation-invariant observation at a step boundary."""
    if qpos.shape != (model.nq,) or qvel.shape != (model.nv,) or ctrl.shape != (model.nu,):
        raise ValueError("qpos, qvel, and ctrl do not match the MuJoCo model")
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)

    progress = np.clip(state.elapsed_s / state.target_period_s, 0.0, 1.0)
    angle = 2.0 * np.pi * progress
    contacts = np.array(
        [data.sensor(f"foot_{foot}_contact").data[0] > 0 for foot in (0, 1)],
        dtype=np.float64,
    )
    normal_force = np.array(
        [data.sensor(f"foot_{foot}_net_force").data[2] for foot in (0, 1)],
        dtype=np.float64,
    )
    weight = max(float(model.body_mass.sum()) * abs(float(model.opt.gravity[2])), 1e-6)
    root = model.body("foot_0").id
    support = model.body(f"foot_{state.support_foot}").id
    com_from_support = data.subtree_com[root] - data.xpos[support]
    observation = np.concatenate(
        [
            qpos[2:],  # remove unobservable/global horizontal translation
            qvel,
            ctrl - standing_joint_targets(model),
            [np.sin(angle), np.cos(angle)],
            [1.0 if state.swing_foot else -1.0, state.target_period_s / 2.0],
            contacts,
            normal_force / weight,
            com_from_support,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(observation).all():
        raise ValueError("warm-start observation contains non-finite values")
    return observation.astype(np.float32)


class LinearWarmStartPolicy:
    """Small dependency-free policy suitable for MPC-teacher bootstrapping.

    It is intentionally an inference primitive, not a safety mechanism. Every
    prediction is clipped to the MPC parameter box and must still pass MPC's
    rollout-based hard constraints.
    """

    def __init__(
        self,
        weights: np.ndarray,
        observation_mean: np.ndarray,
        observation_scale: np.ndarray,
    ) -> None:
        weights = np.asarray(weights, dtype=np.float64)
        observation_mean = np.asarray(observation_mean, dtype=np.float64)
        observation_scale = np.asarray(observation_scale, dtype=np.float64)
        observation_dim = len(observation_mean)
        expected = (2, observation_dim + 1, PLAN_DIM)
        if weights.shape == expected[1:]:
            weights = np.stack([weights, weights])
        if weights.shape != expected:
            raise ValueError("weights must contain two foot-specific observation-to-plan maps")
        if observation_dim <= SWING_SIGN_INDEX:
            raise ValueError("observation is missing the swing-foot feature")
        if observation_scale.shape != (observation_dim,) or np.any(observation_scale <= 0):
            raise ValueError("observation_scale must be positive and match observation_mean")
        if not all(np.isfinite(value).all() for value in (weights, observation_mean, observation_scale)):
            raise ValueError("policy parameters must be finite")
        self.weights = weights.copy()
        self.observation_mean = observation_mean.copy()
        self.observation_scale = observation_scale.copy()

    @classmethod
    def fit(
        cls,
        observations: np.ndarray,
        targets: np.ndarray,
        ridge: float = 1e-4,
    ) -> LinearWarmStartPolicy:
        """Fit an MPC-teacher policy with ridge-regularized least squares."""
        observations = np.asarray(observations, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        if observations.ndim != 2 or targets.shape != (len(observations), PLAN_DIM):
            raise ValueError("expected observations (N, D) and targets (N, 18)")
        if len(observations) < 2 or ridge < 0:
            raise ValueError("at least two samples and a non-negative ridge are required")
        if not np.isfinite(observations).all() or not np.isfinite(targets).all():
            raise ValueError("training data must be finite")
        mean = observations.mean(axis=0)
        # A floor prevents tiny teacher-state differences from becoming huge
        # normalized changes when a small bootstrap dataset is used.
        scale = np.maximum(observations.std(axis=0), 0.05)
        design = np.column_stack([(observations - mean) / scale, np.ones(len(observations))])
        regularizer = ridge * np.eye(design.shape[1])
        regularizer[-1, -1] = 0.0
        shared = np.linalg.solve(design.T @ design + regularizer, design.T @ targets)
        weights = np.stack([shared.copy(), shared.copy()])
        for foot, selected in enumerate(
            (observations[:, SWING_SIGN_INDEX] < 0, observations[:, SWING_SIGN_INDEX] > 0)
        ):
            if not selected.any():
                continue
            local = design[selected]
            # Keep each sparse foot-specific head near the shared solution in
            # directions not constrained by that foot's expert samples.
            prior = max(ridge, 1e-3)
            matrix = local.T @ local + prior * np.eye(design.shape[1])
            right = local.T @ targets[selected] + prior * shared
            weights[foot] = np.linalg.solve(matrix, right)
        return cls(weights, mean, scale)

    def predict(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float64)
        if observation.shape != self.observation_mean.shape or not np.isfinite(observation).all():
            raise ValueError(
                f"expected {len(self.observation_mean)} finite warm-start observation values"
            )
        normalized = (observation - self.observation_mean) / self.observation_scale
        foot = int(observation[SWING_SIGN_INDEX] > 0)
        prediction = np.r_[normalized, 1.0] @ self.weights[foot]
        return np.clip(prediction, -PLAN_BOUNDS, PLAN_BOUNDS)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            format_version=np.array(2, dtype=np.int64),
            weights=self.weights,
            observation_mean=self.observation_mean,
            observation_scale=self.observation_scale,
        )

    @classmethod
    def load(cls, path: str | Path) -> LinearWarmStartPolicy:
        with np.load(path, allow_pickle=False) as archive:
            version = int(archive["format_version"])
            if version not in (1, 2):
                raise ValueError(f"unsupported warm-start policy format version {version}")
            return cls(
                archive["weights"], archive["observation_mean"], archive["observation_scale"]
            )
