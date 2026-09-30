"""Contact-guarded timing for alternating steps at a commanded landing rate."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

LANDING_SETTLE_S = 0.08
MIN_SWING_CLEARANCE_M = 0.005
MIN_SWING_CLEARANCE_S = 0.1


def landing_settle_s(period_s: float) -> float:
    """Allow slow steps to settle on both feet before reversing support."""
    return min(0.3, 0.16 * period_s)


class Phase(str, Enum):
    TRANSFER = "transfer"
    LIFT = "lift"
    HOLD = "hold"
    LOWER = "lower"
    SETTLE = "settle"
    WAIT_CONTACT = "wait_contact"


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
        handoff_hold_s: float = 0.0,
    ) -> None:
        if first_swing_foot not in (0, 1):
            raise ValueError("first_swing_foot must be 0 or 1")
        if not all(math.isfinite(value) and value > 0 for value in (transfer_s, lift_s, lower_s)):
            raise ValueError("phase durations must be finite and positive")
        if min(transfer_s, 0.15) + lift_s + lower_s + LANDING_SETTLE_S >= 0.5:
            raise ValueError("fixed phases must fit inside a 2 Hz step period")
        self.transfer_s = transfer_s
        self.lift_s = lift_s
        self.lower_s = lower_s
        if not math.isfinite(handoff_hold_s) or handoff_hold_s < 0:
            raise ValueError("handoff hold must be finite and nonnegative")
        self.handoff_hold_s = handoff_hold_s
        self.swing_foot = first_swing_foot
        self.elapsed_s = 0.0
        self._swing_airborne_seen = False
        self._clearance_run = 0
        self.frequency_hz = self._check_frequency(frequency_hz)
        self.pending_frequency_hz = self.frequency_hz

    @staticmethod
    def _check_frequency(value: float) -> float:
        if not math.isfinite(value) or not 0.25 <= value <= 2.0:
            raise ValueError("frequency must be between 0.25 and 2.0 Hz")
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
            - landing_settle_s(self.target_period_s)
        )

    @property
    def current_transfer_s(self) -> float:
        period = self.target_period_s
        return min(self.transfer_s, 0.3 * period) + 0.6 * max(0.0, period - 1.0)

    @property
    def current_lift_s(self) -> float:
        return self.lift_s + 0.18 * max(0.0, self.target_period_s - 1.0)

    @property
    def current_lower_s(self) -> float:
        return self.lower_s + 0.18 * max(0.0, self.target_period_s - 1.0)

    def request_frequency(self, frequency_hz: float) -> None:
        self.pending_frequency_hz = self._check_frequency(frequency_hz)

    def state(self) -> StepState:
        t = self.elapsed_s
        period = self.target_period_s
        transfer = self.current_transfer_s
        lift = self.current_lift_s
        lower = self.current_lower_s
        settle = landing_settle_s(period)
        intervals = (
            (Phase.TRANSFER, 0.0, transfer),
            (Phase.LIFT, transfer, transfer + lift),
            (Phase.HOLD, transfer + lift, period - lower - settle),
            (Phase.LOWER, period - lower - settle, period - settle),
            (Phase.SETTLE, period - settle, period + self.handoff_hold_s),
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
            touchdown_late_s=t - period - self.handoff_hold_s,
        )

    def advance(
        self,
        dt: float,
        contacts: tuple[bool, bool],
        swing_clearance_m: float | None = None,
        landing_ready: bool = True,
    ) -> StepState:
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if len(contacts) != 2:
            raise ValueError("contacts must contain two feet")
        lift_allowed = self.elapsed_s + 1e-9 >= self.current_transfer_s
        self.elapsed_s += dt
        if swing_clearance_m is None:
            if self.elapsed_s >= self.current_transfer_s and not contacts[self.swing_foot]:
                self._swing_airborne_seen = True
        else:
            self._clearance_run = (
                self._clearance_run + 1
                if lift_allowed and swing_clearance_m >= MIN_SWING_CLEARANCE_M
                else 0
            )
            if self._clearance_run * dt >= MIN_SWING_CLEARANCE_S:
                self._swing_airborne_seen = True
        period = self.target_period_s + self.handoff_hold_s
        if (
            self.elapsed_s + 1e-9 >= period
            and all(contacts)
            and self._swing_airborne_seen
            and landing_ready
        ):
            overshoot = max(0.0, self.elapsed_s - period)
            self.elapsed_s = overshoot if overshoot <= dt else 0.0
            self.swing_foot = 1 - self.swing_foot
            self._swing_airborne_seen = False
            self._clearance_run = 0
            self.frequency_hz = self.pending_frequency_hz
        return self.state()
