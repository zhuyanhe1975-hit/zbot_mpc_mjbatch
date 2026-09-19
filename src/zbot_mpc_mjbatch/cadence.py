"""Contact-guarded timing for alternating steps at a commanded landing rate."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class Phase(str, Enum):
    TRANSFER = "transfer"
    LIFT = "lift"
    HOLD = "hold"
    LOWER = "lower"
    WAIT_CONTACT = "wait_contact"


def phase_durations(
    period_s: float,
    transfer_s: float = 0.3,
    lift_s: float = 0.12,
    lower_s: float = 0.12,
) -> tuple[float, float, float]:
    """Stretch transitions up to the proven 2 s gait, then add hold time."""
    stretch = min(max(0.0, period_s - 1.0), 1.0)
    transfer = min(transfer_s, 0.3 * period_s) + 0.6 * stretch
    return transfer, lift_s + 0.18 * stretch, lower_s + 0.18 * stretch


@dataclass(frozen=True)
class StepState:
    phase: Phase
    swing_foot: int
    support_foot: int
    elapsed_s: float
    target_period_s: float
    phase_progress: float
    touchdown_late_s: float


class CadenceStateMachine:
    """Schedule one alternating landing every 1/f seconds.

    A requested frequency takes effect only after both feet make contact at
    the step boundary. Joint targets and balance control are supplied elsewhere.
    """

    def __init__(
        self,
        frequency_hz: float,
        first_swing_foot: int = 0,
        transfer_s: float = 0.3,
        lift_s: float = 0.12,
        lower_s: float = 0.12,
        minimum_clearance_s: float = 0.1,
    ) -> None:
        if first_swing_foot not in (0, 1):
            raise ValueError("first_swing_foot must be 0 or 1")
        if not all(
            math.isfinite(value) and value > 0
            for value in (transfer_s, lift_s, lower_s, minimum_clearance_s)
        ):
            raise ValueError("phase durations must be finite and positive")
        if min(transfer_s, 0.15) + lift_s + lower_s >= 0.5:
            raise ValueError("fixed phases must fit inside a 2 Hz step period")
        self.transfer_s = transfer_s
        self.lift_s = lift_s
        self.lower_s = lower_s
        self.minimum_clearance_s = minimum_clearance_s
        self.swing_foot = first_swing_foot
        self.elapsed_s = 0.0
        self._swing_airborne_seen = False
        self._clearance_run = 0
        self.frequency_hz = self._check_frequency(frequency_hz)
        self.pending_frequency_hz = self.frequency_hz

    @staticmethod
    def _check_frequency(value: float) -> float:
        if not math.isfinite(value) or not 1.0 / 3.0 <= value <= 2.0:
            raise ValueError("frequency must be between 1/3 and 2.0 Hz")
        return float(value)

    @property
    def target_period_s(self) -> float:
        return 1.0 / self.frequency_hz

    @property
    def hold_s(self) -> float:
        return (
            self.target_period_s
            - self.current_transfer_s
            - self.current_lift_s
            - self.current_lower_s
        )

    @property
    def current_transfer_s(self) -> float:
        return phase_durations(
            self.target_period_s, self.transfer_s, self.lift_s, self.lower_s
        )[0]

    @property
    def current_lift_s(self) -> float:
        return phase_durations(
            self.target_period_s, self.transfer_s, self.lift_s, self.lower_s
        )[1]

    @property
    def current_lower_s(self) -> float:
        return phase_durations(
            self.target_period_s, self.transfer_s, self.lift_s, self.lower_s
        )[2]

    def request_frequency(self, frequency_hz: float) -> None:
        self.pending_frequency_hz = self._check_frequency(frequency_hz)

    def state(self) -> StepState:
        t = self.elapsed_s
        period = self.target_period_s
        transfer = self.current_transfer_s
        lift = self.current_lift_s
        lower = self.current_lower_s
        intervals = (
            (Phase.TRANSFER, 0.0, transfer),
            (Phase.LIFT, transfer, transfer + lift),
            (Phase.HOLD, transfer + lift, period - lower),
            (Phase.LOWER, period - lower, period),
        )
        for phase, start, end in intervals:
            if t < end:
                return StepState(
                    phase=phase,
                    swing_foot=self.swing_foot,
                    support_foot=1 - self.swing_foot,
                    elapsed_s=t,
                    target_period_s=period,
                    phase_progress=max(0.0, (t - start) / (end - start)),
                    touchdown_late_s=0.0,
                )
        return StepState(
            phase=Phase.WAIT_CONTACT,
            swing_foot=self.swing_foot,
            support_foot=1 - self.swing_foot,
            elapsed_s=t,
            target_period_s=period,
            phase_progress=1.0,
            touchdown_late_s=t - period,
        )

    def advance(
        self,
        dt: float,
        contacts: tuple[bool, bool],
        swing_clearance_m: float | None = None,
    ) -> StepState:
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if len(contacts) != 2:
            raise ValueError("contacts must contain two feet")
        self.elapsed_s += dt
        if swing_clearance_m is None:
            if not contacts[self.swing_foot]:
                self._swing_airborne_seen = True
        else:
            self._clearance_run = (
                self._clearance_run + 1 if swing_clearance_m >= 0.003 else 0
            )
            if self._clearance_run * dt >= self.minimum_clearance_s:
                self._swing_airborne_seen = True
        period = self.target_period_s
        if self.elapsed_s >= period and all(contacts) and self._swing_airborne_seen:
            overshoot = self.elapsed_s - period
            self.elapsed_s = overshoot if overshoot <= dt else 0.0
            self.swing_foot = 1 - self.swing_foot
            self._swing_airborne_seen = False
            self._clearance_run = 0
            self.frequency_hz = self.pending_frequency_hz
        return self.state()
