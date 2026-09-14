#  The exogenous driver, the severity dial, and the declared class constants.
#
#  ---------------------------------------------------------------------------
#  Which cell of the classification this is
#  ---------------------------------------------------------------------------
#  NS_design_guide.md sorts non-stationarity three ways:
#
#      (A) learning-induced   co-learners keep changing        vanishes if
#                             policies                          partners frozen
#      (B) exogenous          the world drifts on its own       a LONE agent
#                                                               feels it
#      (C) interaction-       an exogenous driver exists but    a lone agent
#          mediated           reaches i ONLY through others     feels NOTHING
#
#  and PACT_NS_SPEC II.6 cuts (C) again by whether the harm has an inverse.
#
#  This module is (C, INVERTIBLE), the same cell as ``simple_ns``: the
#  disturbance is ADDITIVE in the agent's own action space -- reactive power,
#  in MVAr -- so a correct estimate cancels it exactly and the method may claim
#  identification AND compensation.  ``ns_direct: true`` moves the identical
#  driver, scale and reward into cell (B) as the control.
#
#  ---------------------------------------------------------------------------
#  The mechanism, and why it is not a gremlin
#  ---------------------------------------------------------------------------
#  MAPDN's inverters are pure setpoint followers: the agent asks for q MVAr and
#  the power flow is run with exactly q.  Real inverters are not.  Every
#  IEEE 1547-2018 inverter ships with an AUTONOMOUS volt-var function in its
#  firmware -- a droop on its own terminal voltage -- and utilities (California
#  Rule 21, HECO, AS/NZS 4777.2) require it enabled.  A dispatched setpoint is
#  applied on top of that curve, so what the inverter DELIVERS is
#
#      q_delivered  =  q_dispatched  +  q_firmware( V_terminal )
#
#  The terminal voltage is moved by every other inverter on the feeder through
#  the network's reactance.  So the firmware term reacting to what the PEERS
#  injected is a disturbance in the agent's own action space that is caused by
#  the other agents: the fleet's volt-var curves react to each other's VARs.
#  Distribution engineers know it as smart-inverter control interaction
#  ("volt-var hunting"), and it is why fleets under central dispatch are so
#  often put into constant-Q mode -- the autonomous curves fight the dispatch.
#
#  How strongly the fleet reacts is not constant over the day.  At night the
#  feeder's voltage profile sits inside the curves' deadband (IEEE 1547 default
#  0.98-1.02 pu) and the firmware is inert; through the solar day the PV-driven
#  rise puts the fleet on the slope of its curves.  That is the exogenous driver
#  A(t): a function of observable time that no agent controls, and it reaches an
#  agent only by scaling what its NEIGHBOURS' VARs do to it.  With one inverter
#  on the feeder there are no neighbours and the term is identically zero at
#  every severity -- structurally, because the sum runs over j != i.
#
#  Every clause is a requirement:
#
#  * INTERACTION-MEDIATED.  The reaction you feel is to the voltage the OTHER
#    inverters imposed.  Your own firmware's reaction to your own injection is a
#    fixed self-gain of your own plant, calibrated at commissioning, and is not
#    part of the disturbance (it is exactly the kind of term that would make a
#    lone agent feel the drift -- category B -- which is why ``ns_direct``
#    exists as the control and not as the claim).
#  * EXOGENOUS DRIVER.  The solar day.  Ask a distribution engineer what makes
#    voltage control harder on some days and "the middle of a sunny day" is the
#    unprompted answer.  Half of every cycle is EXACTLY dark, so the placebo is
#    free (NS-2.5).
#  * NEVER A REWARD TERM.  It corrupts the delivered VARs.  The reward function
#    is inherited from the stock host untouched; the voltages are simply worse.
#  * INVERTIBLE, WITH A REAL SATURATION.  Dispatching extra to cover the
#    firmware's reaction is what a DERMS already does; it stops working at the
#    curve's own saturation (Q1 = 0.44 pu of nameplate) and at the inverter's
#    capability sqrt(S^2 - P^2), which is where sigma* comes from rather than
#    from a number we chose.
#  * THE CLASSES ARE REAL.  The elements are the feeder's line segments and the
#    class is the conductor size printed on the pole.  What is public is the
#    feeder's own reactance map -- which segments a neighbour's VARs cross to
#    reach you.  What is not handed over is what a unit of VAR on each class of
#    segment costs you TODAY, which drifts with the driver.
#  * ANCHORED.  sigma = 1 is the IEEE 1547-2018 Category B DEFAULT volt-var
#    slope, 0.44 pu-Q per 0.06 pu-V = 7.33.  The steepest curve the standard
#    allows (V4 - V3 = 0.02) is sigma = 3.0; the Category A default is
#    sigma = 0.34.  Everything above 3 is a beyond-physical stress test and must
#    be labelled as one wherever it appears.
#
#  torch only.  No pandapower.

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "DialParams",
    "driver_A",
    "class_constants",
    "beta_star",
    "cycle_mean_A",
    "K_1547_CAT_B",
    "K_1547_MAX",
    "K_1547_CAT_A",
    "Q_SAT_1547_CAT_B",
]

#  IEEE 1547-2018, Table 8 (Category B default volt-var settings).  Q in pu of
#  nameplate apparent power, V in pu.  These are published constants.
K_1547_CAT_B: float = 0.44 / 0.06  # 7.333 pu-Q / pu-V, the default slope
K_1547_MAX: float = 0.44 / 0.02    # 22.0, the steepest curve the standard allows
K_1547_CAT_A: float = 0.25 / 0.10  # 2.5, the Category A default
Q_SAT_1547_CAT_B: float = 0.44     # Q1 = -Q4: where the default curve saturates


@dataclass(frozen=True)
class DialParams:
    """Declared constants.  ``severity`` is the only experimental variable."""

    severity: float = 1.0
    """sigma.  Multiplies the firmware slope: 0 makes the disturbance EXACTLY
    zero at every driver value so the stock task is recovered bit for bit; 1 is
    the IEEE 1547-2018 Category B default curve; 3 is the steepest compliant
    curve; above 3 is beyond-physical."""

    # -- the driver -----------------------------------------------------------
    driver: str = "solar"
    """``solar`` -- a smooth clear-sky day, ``A = sin^2(pi (h - rise)/(set -
    rise))`` between ``day_start_hour`` and ``day_end_hour`` and EXACTLY zero
    otherwise.  The same bump shape ``road_ns`` and ``simple_ns`` use, so the
    ladders read against each other.

    ``schedule`` -- the literal alternative: the DERMS enables the fleet's
    autonomous curves on a clock (IEEE 2030.5 DER control schedules), ``A = 1``
    over the same window and exactly zero outside it.  Kept for the ablation;
    both have the exact-zero placebo."""
    day_start_hour: float = 6.0
    day_end_hour: float = 18.0
    """The window over which the fleet's firmware is on its slope.  The other
    half of the day is EXACTLY inert, which is NS-2.5's placebo regime."""

    # -- the anchor (NS-2.4) --------------------------------------------------
    k_slope: float = K_1547_CAT_B
    """The firmware's volt-var slope at sigma = 1, pu-Q per pu-V.  A published
    constant (IEEE 1547-2018 Category B default), not a number we chose."""
    q_sat: float = Q_SAT_1547_CAT_B
    """Where the curve saturates, pu of nameplate.  The relief valve: a physical
    bound on a physical quantity, not a tuning knob."""

    rho: float = 0.0
    """Channel memory.  The firmware's response time (1-10 s) is far below the
    3-minute step, so the default is memoryless; a leak is kept on the PUBLIC
    channels only (never on the private disturbance) so the model stays exactly
    linear in quantities the agent can compute -- see ``coupling.Coupling``."""

    # -- the declared classes (P-1.1, P-1.2) ----------------------------------
    n_classes: int = 3
    """``r``: number of conductor classes the feeder's segments are sorted into.
    INDEPENDENT of the number of agents and of the number of lines: adding an
    inverter or a lateral adds no parameters."""

    send_spread: float = 0.8
    """Spread of the per-class SENDER gain ``send_m``.  This is beta*, the
    quantity the estimator has to identify -- what a unit of peer VAR crossing a
    class-m segment costs you today -- and it is NOT handed to the agent.  It
    is declared (a deterministic fan around 1, as in ``simple_ns``) so that
    beta* is known exactly and beta-against-truth can be scored."""

    y_clip: float = 10.0
    """P-2.1's declared outlier bound on the sensor."""


# ---------------------------------------------------------------------------
#  NS-1.3 -- the exogenous driver
# ---------------------------------------------------------------------------


def driver_A(hour: Tensor, p: DialParams) -> Tensor:
    """``A(t) in [0, 1]`` from the hour of day alone.

    A function of observable time.  No agent's action can influence it, and it
    reaches the agents only by scaling the cross-agent term -- never by adding a
    term to the loss.  It is EXACTLY zero outside the daylight window, returned
    as a literal zero rather than reached through arithmetic, so the placebo is
    provable rather than merely small.
    """
    h = torch.as_tensor(hour, dtype=torch.float32) % 24.0
    rise, set_ = float(p.day_start_hour), float(p.day_end_hour)
    if set_ <= rise:
        return torch.zeros_like(h)
    phi = (h - rise) / (set_ - rise)
    inside = (phi >= 0.0) & (phi < 1.0)
    if p.driver == "solar":
        bump = torch.sin(math.pi * phi.clamp(0.0, 1.0)) ** 2
    elif p.driver == "schedule":
        bump = torch.ones_like(h)
    else:
        raise ValueError(f"unknown driver {p.driver!r}; expected 'solar' or 'schedule'")
    return torch.where(inside, bump, torch.zeros_like(h))


def cycle_mean_A(p: DialParams, steps_per_day: int = 480) -> float:
    """Mean driver value over one day.  I.6 wants the level reported, not just
    the peak: half the cycle is exactly dark, so this is well under 0.5."""
    h = torch.arange(steps_per_day, dtype=torch.float32) * (24.0 / steps_per_day)
    return float(driver_A(h, p).mean())


# ---------------------------------------------------------------------------
#  the declared class constants
# ---------------------------------------------------------------------------


def class_constants(p: DialParams) -> Tensor:
    """``send``, ``(r,)`` with mean exactly 1.

    Deterministic in the class index alone -- no RNG, no run data -- so it is
    identical across arms, seeds and severities, and the operator is *declared*
    rather than fitted (NS-1.2).  A cosine fan is used simply because it is
    reproducible and spreads the values without a magic table.

    ``send`` is NOT public: it is how much a unit of peer VAR crossing a class-m
    segment actually costs today, and it is what the estimator must recover.
    The public side of the split -- which segments a neighbour's VARs cross,
    and their reactance -- lives in ``coupling.Coupling`` and is read straight
    off the network file.
    """
    r = int(p.n_classes)
    if r < 1:
        raise ValueError(f"n_classes must be >= 1, got {r}")
    if r == 1:
        return torch.ones(1)
    idx = torch.arange(r, dtype=torch.float32)
    fan = torch.cos(math.pi * idx / (r - 1))  # +1 .. -1, deterministic
    send = 1.0 + p.send_spread * fan
    return send / send.mean()


def beta_star(a: Tensor, p: DialParams) -> Tensor:
    """The true per-class transmission gain right now.  ``(B, r)``.

        beta*_m(t) = - sigma * k_slope * A(t) * send_m

    in units of (pu of nameplate Q) per (pu of peer-induced terminal voltage).
    The sign is the physics: the firmware ABSORBS VARs when the peers raise its
    voltage.  This is the whole of what drifts, and the whole of what the
    estimator is asked to track.  Three properties hold by construction:

    * ``sigma = 0`` makes it exactly ``0.0`` at every driver value, so the
      disturbance is exactly zero and the stock task is recovered bit for bit
      (NS-2.1).
    * monotone in sigma at every driver value, since it is linear in sigma with
      a fixed-sign coefficient (NS-2.2).
    * exactly zero wherever ``A(t)`` is exactly zero, which is half of every
      day (NS-2.5).
    """
    send = class_constants(p)
    scale = -float(p.severity) * float(p.k_slope)
    a = torch.as_tensor(a, dtype=torch.float32)
    return scale * a.reshape(-1, 1) * send.reshape(1, -1).to(a.device)
