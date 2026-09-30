"""Save measured stepping trajectories for replay without running MPC again."""

import json
from dataclasses import fields
from pathlib import Path

import numpy as np

from .cadence import Phase
from .mpc import StepPlan
from .stepping import StepRun

_METADATA_FIELDS = {"phases", "plans", "wall_seconds", "stop_reason"}
_PLAN_ARRAYS = {"joint_offset", "transfer_offset", "hold_offset", "landing_offset", "start_offset"}


def save_step_run(run: StepRun, path: Path) -> None:
    """Write arrays and JSON metadata; the archive contains no pickled objects."""
    arrays = {
        field.name: getattr(run, field.name)
        for field in fields(StepRun)
        if field.name not in _METADATA_FIELDS
    }
    plans = []
    for plan in run.plans:
        values = {}
        for field in fields(StepPlan):
            value = getattr(plan, field.name)
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, np.generic):
                value = value.item()
            values[field.name] = value
        plans.append(values)
    metadata = {
        "version": 1,
        "plans": plans,
        "wall_seconds": run.wall_seconds,
        "stop_reason": run.stop_reason,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        np.savez_compressed(
            file,
            **arrays,
            phases=np.array([phase.value for phase in run.phases]),
            metadata=np.array(json.dumps(metadata)),
        )


def load_step_run(path: Path) -> StepRun:
    """Read a measured trajectory without executing serialized Python objects."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if metadata.get("version") != 1:
            raise ValueError("unsupported stepping trajectory version")
        arrays = {
            field.name: archive[field.name].copy()
            for field in fields(StepRun)
            if field.name not in _METADATA_FIELDS
        }
        phases = tuple(Phase(str(value)) for value in archive["phases"])
    count = len(arrays["times_s"])
    if (
        count == 0
        or len(phases) != count
        or not np.isfinite(arrays["times_s"]).all()
        or np.any(np.diff(arrays["times_s"]) <= 0)
    ):
        raise ValueError("trajectory must contain increasing times and a phase for every frame")
    for name, columns in (("qpos", 13), ("qvel", 12), ("ctrl", 6)):
        if arrays[name].shape != (count, columns) or not np.isfinite(arrays[name]).all():
            raise ValueError(f"invalid ZBot trajectory field: {name}")
    plans = tuple(
        StepPlan(
            **{
                key: np.asarray(value, dtype=float)
                if key in _PLAN_ARRAYS and value is not None
                else value
                for key, value in plan.items()
            }
        )
        for plan in metadata["plans"]
    )
    return StepRun(
        **arrays,
        phases=phases,
        plans=plans,
        wall_seconds=float(metadata["wall_seconds"]),
        stop_reason=metadata["stop_reason"],
    )
