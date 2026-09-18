"""LIPM ZMP preview for alternating in-place support phases."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ZMPPreview:
    com_xy: np.ndarray
    zmp_xy: np.ndarray


def _zmp_schedule(
    times_s: np.ndarray,
    foot_xy: np.ndarray,
    first_swing_foot: int,
    period_s: float,
    transfer_s: float,
    lower_s: float,
) -> np.ndarray:
    step = np.floor(times_s / period_s).astype(int)
    local = times_s - step * period_s
    support = np.where(step % 2 == 0, 1 - first_swing_foot, first_swing_foot)
    center = foot_xy.mean(axis=0)
    support_center = foot_xy[support]
    weight = np.minimum(
        np.clip(local / transfer_s, 0.0, 1.0),
        np.clip((period_s - local) / lower_s, 0.0, 1.0),
    )
    return center + weight[:, None] * (support_center - center)


def plan_zmp_preview(
    com_xy: np.ndarray,
    com_height_m: float,
    foot_xy: np.ndarray,
    swing_foot: int,
    period_s: float,
    elapsed_s: float,
    control_dt: float,
    horizon: int,
    transfer_s: float = 0.3,
    lower_s: float = 0.12,
) -> ZMPPreview:
    """Preview future support ZMP, then integrate the bounded LIPM DCM backward."""
    if com_height_m <= 0 or horizon < 1 or control_dt <= 0:
        raise ValueError("positive COM height, horizon, and control interval required")
    omega = np.sqrt(9.81 / com_height_m)
    extra = int(np.ceil(2.0 * period_s / control_dt))
    grid = elapsed_s + np.arange(horizon + extra + 1) * control_dt
    zmp = _zmp_schedule(grid, foot_xy, swing_foot, period_s, transfer_s, lower_s)
    decay = np.exp(-omega * control_dt)
    dcm = np.empty_like(zmp)
    dcm[-1] = zmp[-1]
    for i in range(len(zmp) - 2, -1, -1):
        dcm[i] = decay * dcm[i + 1] + (1.0 - decay) * zmp[i]
    com = np.empty((horizon + 1, 2))
    com[0] = com_xy
    for i in range(horizon):
        com[i + 1] = decay * com[i] + (1.0 - decay) * dcm[i]
    return ZMPPreview(com[1:], zmp[1 : horizon + 1])
