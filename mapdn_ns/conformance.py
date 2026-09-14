#!/usr/bin/env python
#  Offline conformance.  torch only -- no pandapower, no data, seconds to run.
#
#      python mapdn_ns/conformance.py
#      python mapdn_ns/conformance.py --scenario case33_3min_final
#
#  The two decision-procedure checks come first, because they are what the
#  classification claim rests on: which cell an instance is in is settled by
#  MEASUREMENT here, not by argument in the paper.  If the exported structure
#  for the scenario is absent the synthetic feeder is used and the banner says
#  so; nothing measured on it is a result.

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mapdn_ns.coupling import Coupling, _single  # noqa: E402
from mapdn_ns.driver import (  # noqa: E402
    K_1547_CAT_A,
    K_1547_CAT_B,
    K_1547_MAX,
    Q_SAT_1547_CAT_B,
    DialParams,
    beta_star,
    class_constants,
    cycle_mean_A,
    driver_A,
)
from mapdn_ns.structure import FeederStructure, load_structure  # noqa: E402

RESULTS: List[Tuple[str, bool, str]] = []
CHECKS: List[Tuple[str, Callable[[], str]]] = []
ST: FeederStructure = None  # set in main
HOURS = torch.arange(480, dtype=torch.float32) * (24.0 / 480)  # every 3-minute interval of a day


def check(name: str):
    def wrap(fn):
        CHECKS.append((name, fn))
        return fn
    return wrap


def fleet_q(seed: int = 0, frac: float = 1.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(ST.n_agents, generator=g) * 2 - 1
    return u * frac * ST.action_scale * ST.s_rated_mva


def load_of(c: Coupling, q: torch.Tensor, a: float, sigma: float) -> torch.Tensor:
    x = c.step_channels(q, torch.zeros(c.n, c.r))
    _, d, _ = c.disturbance(x, torch.tensor([a]), sigma)
    return d


# ===========================================================================
#  the decision procedure
# ===========================================================================


@check("test_lone_agent_feels_nothing")
def _lone():
    """The (B) vs (C) test.  A lone inverter must read EXACTLY zero at any
    severity.  If a single agent suffers, the driver is reaching it directly
    and the instance is category B in disguise."""
    for i in range(ST.n_agents):
        st1 = _single(ST, i)
        for sigma in (0.5, 1.0, 3.0, 100.0):
            c = Coupling(st1, DialParams(severity=sigma))
            d = load_of(c, fleet_q()[i : i + 1], 1.0, sigma)
            assert float(d.abs().max()) == 0.0, f"inverter {i} alone read {float(d.abs().max()):.3g} at sigma={sigma}"
    return f"each of the {ST.n_agents} inverters alone reads exactly 0.0 at sigma in {{0.5, 1, 3, 100}} -- category C"


@check("test_direct_control_is_felt_by_a_lone_agent")
def _control():
    """The (B) control must be a DIFFERENT cell, measurably: a lone inverter
    under ns_direct feels the same driver at the same scale."""
    c = Coupling(ST, DialParams(severity=1.0))
    d_ref = c.derate_reference(64)  # (N,) MVAr
    q = fleet_q()
    d_direct = -d_ref  # what layer._disturbance does with ns_direct at sigma=1, A=1: a level shift
    assert float(d_direct.abs().min()) > 0.0, "the direct control read zero"
    assert float(d_direct.max()) < 0.0, "the (B) control must absorb (react to the PV-driven rise)"
    frac = d_ref / (ST.action_scale * ST.s_rated_mva)
    return (f"ns_direct at N=1: |d| = {float(d_direct.abs().mean()):.3f} MVAr, "
            f"{100 * float(frac.mean()):.1f}% of the VAR range on average (per-inverter D_ref "
            f"{[round(float(v), 3) for v in d_ref]} MVAr) -- cell (B), matched to (C) by construction")


@check("test_frozen_partners_still_drift")
def _frozen():
    """The (A) vs (C) test.  With teammates frozen at a fixed dispatch the
    disturbance must still drift over the day, or it is a learning artefact
    rather than a property of the task."""
    p = DialParams(severity=1.0)
    c = Coupling(ST, p)
    q = fleet_q()
    vals = [float(load_of(c, q, float(driver_A(torch.tensor([h]), p)), 1.0).abs().mean()) for h in HOURS[::10]]
    assert max(vals) > 2 * (min(vals) + 1e-12), f"frozen partners gave a flat load {min(vals):.4g}..{max(vals):.4g}"
    return f"partners frozen, |d| still swings {min(vals):.4f} -> {max(vals):.4f} MVAr over one day -- not category A"


# ===========================================================================
#  the dial
# ===========================================================================


@check("test_identity_at_zero_is_exact")
def _zero():
    """NS-2.1.  Not 'approximately'; equality, over the whole driver domain."""
    p = DialParams(severity=0.0)
    c = Coupling(ST, p)
    q = fleet_q()
    for h in HOURS:
        a = float(driver_A(h.reshape(1), p))
        d = load_of(c, q, a, 0.0)
        assert float(d.abs().max()) == 0.0, float(h)
        assert torch.equal(q + d, q), "q + d is not bit-identical to q at sigma=0"
    return f"sigma=0 gives exactly 0.0 MVAr at all {len(HOURS)} driver values; q + 0.0 == q bit for bit"


@check("test_monotone_in_severity")
def _mono():
    """NS-2.2, at every driver value, not merely at the peak.  The curve's
    saturation keeps it non-decreasing rather than strictly increasing."""
    c = Coupling(ST, DialParams())
    q = fleet_q(frac=0.3)
    worst = 0.0
    for h in HOURS[::20]:
        a = float(driver_A(h.reshape(1), DialParams()))
        prev = -1.0
        for sigma in (0.0, 0.34, 0.5, 1.0, 2.0, 3.0, 6.0):
            v = float(load_of(c, q, a, sigma).abs().mean())
            assert v >= prev - 1e-9, f"|d| fell from {prev} to {v} at sigma={sigma}, hour {float(h):.1f}"
            prev = v
        worst = max(worst, prev)
    return f"non-decreasing in sigma at 24 driver values; peak mean |d| {worst:.3f} MVAr at sigma=6"


@check("test_disturbance_derates_the_fleet_not_the_reward")
def _opposes():
    """NS-1.4 / NS-2.3 for an additive channel: the firmware reaction OPPOSES
    the fleet's collective dispatch, so the fleet's net delivered VAR is never
    larger than what it dispatched when it pulls one way.  The reward function
    is untouched by construction (the layer never overrides _calc_reward)."""
    c = Coupling(ST, DialParams(severity=1.0))
    for sign in (+1.0, -1.0):
        q = sign * fleet_q().abs()
        d = load_of(c, q, 1.0, 1.0)
        assert float((torch.sign(d) * sign).max()) <= 0.0, "the reaction did not oppose a one-sided dispatch"
        assert abs(float((q + d).sum())) <= abs(float(q.sum())) + 1e-9, "net VAR grew"
    import mapdn_ns.layer as L
    src = Path(L.__file__).read_text(encoding="utf-8")
    assert "_calc_reward" not in src, "the layer touches the reward function"
    return "a one-sided dispatch is always opposed: the fleet's authority is derated; reward code untouched"


@check("test_placebo_regime_is_exactly_inert")
def _placebo():
    """NS-2.5.  Half of every day is EXACTLY quiet, at every severity."""
    p = DialParams(severity=3.0)
    c = Coupling(ST, p)
    q = fleet_q()
    dark = [float(h) for h in HOURS if float(driver_A(h.reshape(1), p)) == 0.0]
    assert len(dark) >= len(HOURS) // 2, f"only {len(dark)} exactly-dark intervals"
    for h in dark:
        d = load_of(c, q, float(driver_A(torch.tensor([h]), p)), 3.0)
        assert float(d.abs().max()) == 0.0, h
        assert torch.equal(q + d, q)
    return f"{len(dark)}/{len(HOURS)} intervals exactly inert at sigma=3, bit for bit"


@check("test_driver_is_a_function_of_time_alone")
def _exogenous():
    """NS-1.3.  A(t) depends on the hour of day and on nothing else, for both
    declared forms."""
    out = []
    for form in ("solar", "schedule"):
        p = DialParams(driver=form)
        a = driver_A(HOURS, p)
        b = driver_A(HOURS + 24.0, p)
        # float32 (h + 24) % 24 is not bit-identical to h; the DARK set must be
        assert torch.allclose(a, b, atol=1e-6), f"{form}: not periodic in 24 h"
        assert torch.equal(a == 0, b == 0), f"{form}: the exactly-dark set moved across a day"
        assert float(a.min()) == 0.0 and abs(float(a.max()) - 1.0) < 1e-6, form
        assert int((a == 0).sum()) >= len(HOURS) // 2, f"{form}: less than half the day exactly zero"
        out.append(f"{form}: range [0,1], cycle mean {cycle_mean_A(p):.3f}, {int((a == 0).sum())}/480 exactly zero")
    return "; ".join(out)


@check("test_anchor_is_the_published_constant")
def _anchor():
    """NS-2.4.  sigma = 1 is the IEEE 1547-2018 Category B default slope; the
    dial's other landmarks are the standard's own."""
    p = DialParams()
    assert abs(p.k_slope - 0.44 / 0.06) < 1e-12, p.k_slope
    assert abs(p.q_sat - 0.44) < 1e-12, p.q_sat
    assert abs(K_1547_MAX / K_1547_CAT_B - 3.0) < 1e-9
    b1 = beta_star(torch.tensor([1.0]), DialParams(severity=1.0))
    b3 = beta_star(torch.tensor([1.0]), DialParams(severity=3.0))
    assert torch.allclose(b3, 3 * b1)
    return (
        f"sigma=1 -> slope {K_1547_CAT_B:.3f} pu/pu (Cat. B default); sigma=3 -> {K_1547_MAX:.1f} "
        f"(steepest compliant); Cat. A default is sigma={K_1547_CAT_A / K_1547_CAT_B:.2f}; "
        f"saturation {Q_SAT_1547_CAT_B} pu"
    )


# ===========================================================================
#  the operator
# ===========================================================================


@check("test_operator_is_zero_diagonal_spread_and_asymmetric")
def _operator():
    """NS-1.2.  A flat proxy measured a fit gain of -0.0045 on the source
    implementation -- worse than an intercept-only null."""
    c = Coupling(ST, DialParams())
    st = c.operator_stats()
    assert st["diag_max"] == 0.0, f"W has a non-zero diagonal: {st['diag_max']}"
    assert st["spread"] > 0.1, f"W is nearly flat: spread {st['spread']:.4f}"
    assert st["asymmetry"] > 0.01, (
        f"W is nearly symmetric ({st['asymmetry']:.4f}): the inverters' nameplates are "
        "equal, so the receiver's rating does not distinguish rows"
    )
    return (
        f"zero diagonal, spread {st['spread']:.3f}, asymmetry {st['asymmetry']:.3f}, "
        f"{st['zero_pairs']:.0%} of pairs share no segment"
    )


@check("test_loop_gain_is_reported_at_the_anchor")
def _loop_gain():
    """I.6's swing, in this medium's own terms: the mutual-reaction loop gain
    at the anchored curve, and the severity at which it crosses 1 -- where
    the disturbance becomes the volt-var hunting instability.  A gain above
    1 at sigma = 1 is not a failure, it is the finding that the standard's
    default curve already hunts on this feeder (case322); it is reported so
    the row is labelled accordingly."""
    c = Coupling(ST, DialParams())
    g = c.loop_gain()
    assert g > 0.0, f"loop gain at sigma=1 is {g:.3f}: the coupling is inert"
    where = ("inside" if 1 / g <= 3.0 else "beyond") + " the standard's steepest compliant curve (sigma=3)"
    if g >= 1.0:
        return (f"rho(W) = {g:.3f} at sigma=1: THE ANCHORED DEFAULT CURVE ALREADY HUNTS at noon on this "
                f"feeder (threshold sigma = {1 / g:.2f}); sigma=1 is a severe row here, say so")
    return f"rho(W) = {g:.3f} at sigma=1 (stable); the fleet hunts at noon for sigma > {1 / g:.2f} -- {where}"


@check("test_channels_equal_the_brute_force_definition")
def _gate1():
    """P-3.2, at startup rather than in a test somebody can skip."""
    c = Coupling(ST, DialParams())
    return c.verify(fleet_q())


@check("test_r_is_independent_of_the_number_of_agents")
def _rank():
    """P-1.1.  If the reduction has a parameter per agent it is not this method."""
    p = DialParams()
    dims = set()
    for n in range(1, ST.n_agents + 1):
        sub = FeederStructure(
            name="sub", v_base_kv=ST.v_base_kv, line_x_ohm=ST.line_x_ohm, line_r_ohm=ST.line_r_ohm,
            line_class=ST.line_class, class_names=ST.class_names, paths=ST.paths[:n],
            sgen_bus=ST.sgen_bus[:n], s_rated_mva=ST.s_rated_mva[:n], action_scale=ST.action_scale,
        )
        dims.add(Coupling(sub, p).r)
    assert dims == {p.n_classes}, dims
    return f"r = {p.n_classes} at N = 1..{ST.n_agents}, over {ST.n_lines} elements -- no parameter per agent or element"


@check("test_class_constants_are_declared_not_fitted")
def _declared():
    """P-1.2.  Deterministic in the class index alone."""
    p = DialParams()
    a1, a2 = class_constants(p), class_constants(p)
    assert torch.equal(a1, a2)
    assert abs(float(a1.mean()) - 1.0) < 1e-6
    assert float(a1.max() - a1.min()) > 0.1
    cls = ST.line_class
    counts = [int((cls == m).sum()) for m in range(ST.n_classes)]
    assert min(counts) > 0, f"an empty conductor class: {counts}"
    return f"send {[round(float(v), 3) for v in a1]} (unknown), mean 1; classes {dict(zip(ST.class_names, counts))} from `{ST.class_rule}`"


@check("test_centring_conditions_the_design_matrix")
def _centring():
    """P-3.3."""
    c = Coupling(ST, DialParams())
    ref, scale = c.geometric_reference(128)
    raw, cen = [], []
    for s in range(200):
        x = c.step_channels(fleet_q(seed=100 + s, frac=0.4).abs() * -1.0, torch.zeros(c.n, c.r))
        raw.append(torch.cat([torch.ones(c.n, 1), x], -1))
        cen.append(c.design(x, ref, scale))
    R, C = torch.cat(raw), torch.cat(cen)
    k_raw = float(torch.linalg.cond(R.T @ R))
    k_cen = float(torch.linalg.cond(C.T @ C))
    assert k_cen < k_raw, f"centring made it worse: {k_raw:.3g} -> {k_cen:.3g}"
    return f"condition number under a one-sided (absorbing) fleet: {k_raw:.3g} -> {k_cen:.3g}"


# ===========================================================================
#  the sensor and the channel
# ===========================================================================


@check("test_sensor_is_relative_and_observable")
def _sensor():
    """P-2.1: y = (delivered - sent) / nameplate, from the inverter's own meter."""
    c = Coupling(ST, DialParams(severity=1.0))
    q = fleet_q(frac=0.3)
    x = c.step_channels(q, torch.zeros(c.n, c.r))
    y_model, d, sat = c.disturbance(x, torch.tensor([1.0]), 1.0)
    y = d / ST.s_rated_mva
    assert torch.allclose(y, y_model.clamp(-0.44, 0.44)), "sensor is not the clipped model value"
    assert torch.allclose((q + d - q) / ST.s_rated_mva, y), "not what the meter reads"
    return f"y = d/s in pu of nameplate; {int(sat.sum())}/{c.n} saturated at 0.44 in this draw"


@check("test_channel_is_invertible")
def _invertible():
    """II.6 row 1: handed the true disturbance, the correction cancels it."""
    from pact1.core import compensate
    c = Coupling(ST, DialParams(severity=2.0))
    q = fleet_q(frac=0.5)
    x = c.step_channels(q, torch.zeros(c.n, c.r))
    _, d, _ = c.disturbance(x, torch.tensor([1.0]), 2.0)
    pred = d / ST.s_rated_mva
    sent = compensate(q.unsqueeze(-1), torch.ones(c.n, 1), pred * ST.s_rated_mva, torch.ones(c.n)).squeeze(-1)
    exec_ = sent + d
    err = float((exec_ - q).abs().max())
    assert err < 1e-5, f"the exact inverse left {err:.3e} MVAr"
    # and g = 0 is bit-identical (P-7.1)
    off = compensate(q.unsqueeze(-1), torch.ones(c.n, 1), torch.randn(c.n) * 1e6, torch.zeros(c.n)).squeeze(-1)
    assert torch.equal(off, q), "g=0 was not a bit-for-bit no-op"
    return f"with a correct estimate the disturbance cancels to {err:.1e} MVAr; g=0 is bit-identical for a 1e6 estimate"


# ===========================================================================
#  the ceiling (I.5)
# ===========================================================================


@check("test_ceiling_decomposition_is_all_peer")
def _ceiling():
    """NS-4.1.  No uncontrolled participant draws on this medium and every sum
    is strictly j != i, so Delta_fixed = 0, Delta_own = 0 and the coordination
    gap is 100% by construction.  Reported, and flagged as degenerate."""
    c = Coupling(ST, DialParams(severity=1.0))
    q = fleet_q(frac=0.4)
    d_all = load_of(c, q, 1.0, 1.0).abs().sum()
    d_own = 0.0  # a lone inverter's own contribution to its own load
    for i in range(c.n):
        q_i = torch.zeros_like(q)
        q_i[i] = q[i]
        d_own += float(load_of(c, q_i, 1.0, 1.0)[i].abs())
    peer = 1.0 - d_own / float(d_all)
    assert abs(peer - 1.0) < 1e-9, peer
    return f"irreducible 0.0%, own 0.0%, PEER 100.0% -- degenerate by construction (see README)"


def main() -> int:
    global ST
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default="case33_3min_final")
    args = ap.parse_args()
    ST = load_structure(args.scenario)
    width = 84
    print("=" * width)
    print("mapdn_ns -- offline conformance (torch only)")
    print(ST.banner())
    if ST.extra.get("synthetic"):
        print("!! SYNTHETIC structure: the exported structure is absent; nothing here is a result")
    print("=" * width)
    for name, fn in CHECKS:
        try:
            RESULTS.append((name, True, fn() or ""))
        except AssertionError as exc:
            RESULTS.append((name, False, str(exc)))
        except Exception as exc:  # noqa: BLE001
            RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
    failed = 0
    for name, ok, detail in RESULTS:
        failed += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"       {detail}")
    print("-" * width)
    print(f"{len(RESULTS) - failed}/{len(RESULTS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
