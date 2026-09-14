#!/usr/bin/env python
#  PACT-1 offline self-test on the feeder's own basis.  Build-order step 5.
#
#      python mapdn_ns/estimator_selftest.py
#
#  Seconds, torch only, no simulator.  If beta is not recovered here the
#  arithmetic is wrong, not the domain -- and that is worth knowing before any
#  compute is spent.  The method objects are imported from pact1/core.py; the
#  basis is mapdn_ns.coupling on the exported (or synthetic) structure.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pact1.core import PactParams, RLS, compensate, confidence, trust_from_logit  # noqa: E402
from mapdn_ns.coupling import Coupling  # noqa: E402
from mapdn_ns.driver import DialParams, beta_star, driver_A  # noqa: E402
from mapdn_ns.structure import load_structure  # noqa: E402

RESULTS: List[Tuple[str, bool, str]] = []
ST = load_structure()
P = PactParams(mu=0.98)
DIAL = DialParams(severity=1.0)
C = Coupling(ST, DIAL)
N, R = C.n, C.r
DIM = 1 + R
REF, SCALE = C.geometric_reference(256)


def check(name: str):
    def wrap(fn: Callable[[], str]):
        try:
            RESULTS.append((name, True, fn() or ""))
        except AssertionError as exc:
            RESULTS.append((name, False, str(exc)))
        except Exception as exc:  # noqa: BLE001
            RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
        return fn
    return wrap


def dispatch(seed: int, frac: float = 0.5) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(N, generator=g) * 2 - 1) * frac * ST.action_scale * ST.s_rated_mva


def psi_of(q: torch.Tensor) -> torch.Tensor:
    return C.design(C.step_channels(q, torch.zeros(N, R)), REF, SCALE)


def target_beta(a: float, sigma: float = 1.0) -> torch.Tensor:
    """beta* expressed in the CENTRED coordinates the RLS sees."""
    b = beta_star(torch.tensor([a]), DialParams(severity=sigma))[0]
    return torch.cat([(b * REF).sum().reshape(1), b * SCALE])


LIVE = torch.cat([torch.ones(N, 1, dtype=torch.bool), C.channel_liveness()], dim=-1).float()  # (N, d)


def beta_score(beta_hat: torch.Tensor, true: torch.Tensor):
    """Relative error and cosine on each agent's LIVE columns (P-3.4)."""
    b = beta_hat * LIVE
    t = true.reshape(1, -1) * LIVE
    err = ((b - t).norm(dim=-1) / t.norm(dim=-1).clamp_min(1e-12)).mean()
    cos = ((b * t).sum(-1) / (b.norm(dim=-1) * t.norm(dim=-1)).clamp_min(1e-12)).mean()
    return float(err), float(cos)


# ===========================================================================


@check("test_basis_zero_diagonal_and_N1")
def _zero_diag():
    x = C.step_channels(dispatch(0), torch.zeros(N, R))
    assert float(x.abs().max()) > 0.0, "the channels are inert with the whole fleet"
    from mapdn_ns.coupling import _single
    one = Coupling(_single(ST, 0), DIAL)
    x1 = one.step_channels(dispatch(0)[:1], torch.zeros(1, R))
    assert float(x1.abs().max()) == 0.0, "a lone inverter read non-zero peer load"
    return f"N=1 reads exactly 0 on all {R} channels; N={N} is live"


@check("test_basis_waveform_arithmetic")
def _bruteforce():
    for seed in (0, 1, 2):
        C.verify(dispatch(seed))
    return "vectorised == brute-force definition over 3 fleets (gate 1)"


@check("test_rls_recovers_known_beta")
def _recover():
    true = target_beta(1.0)
    r = RLS(N, DIM, P)
    for t in range(3000):
        psi = psi_of(dispatch(t))
        r.update(psi, (psi * true.unsqueeze(0)).sum(-1))
    err, cos = beta_score(r.beta[0], true)
    assert err < 0.15 and cos > 0.95, f"beta relerr {err:.3f} cos {cos:.3f} on live columns; true={true.tolist()}"
    dead = int((LIVE[:, 1:] == 0).sum())
    return (f"beta relerr {err:.3f}, cos {cos:.3f} on live columns over 3000 rows, mu={P.mu} "
            f"({dead} structurally dead agent-channels of {N * R}, e.g. an inverter sharing only the trunk head)")


@check("test_end_to_end_recovers_the_firmware_law")
def _known_law():
    """The layer's own law, plus meter noise, over the real basis."""
    true = target_beta(1.0)
    r = RLS(N, DIM, P)
    gen = torch.Generator().manual_seed(3)
    sse_full = sse_null = 0.0
    ybar = torch.zeros(N)
    for t in range(4000):
        psi = psi_of(dispatch(10_000 + t))
        y = (psi * true.unsqueeze(0)).sum(-1) + 0.005 * torch.randn(N, generator=gen)
        resid = r.update(psi, y)[0]
        if t > 500:
            sse_full += float(resid.pow(2).sum())
            ybar = 0.99 * ybar + 0.01 * y
            sse_null += float((y - ybar).pow(2).sum())
    fit_gain = 1.0 - sse_full / max(sse_null, 1e-12)
    err, cos = beta_score(r.beta[0], true)
    assert fit_gain > 0.5, f"fit gain over an intercept-only null is only {fit_gain:.3f}"
    assert err < 0.25 and cos > 0.9, f"beta relerr {err:.3f} cos {cos:.3f} with meter noise"
    return f"fit_gain over the null = {fit_gain:.3f}, beta relerr {err:.3f} cos {cos:.3f} on live columns"


@check("test_rls_tracks_the_solar_day")
def _drift():
    """beta* follows A(t) through a day; the estimator must follow it, per
    agent over its live columns exactly as the layer runs it.  This is the mu
    question in miniature -- 0.999 (a 1000-step memory) averages the day away
    -- and it is scored on PREDICTION error (trap #10), beta error being
    dominated by the poorly excited directions of a radial feeder.  With the
    driver's shape in the basis (pact_known_driver) beta* is stationary and
    the error should vanish: that is the ceiling of the ablation."""
    live = LIVE.bool()

    def run(mu: float, known: bool):
        rls = [RLS(1, int(live[i].sum()) + (1 if known else 0), PactParams(mu=mu)) for i in range(N)]
        perr = pnull = 0.0
        for t in range(3 * 480):
            h = (t % 480) * (24.0 / 480)
            a = float(driver_A(torch.tensor([h]), DIAL))
            true = target_beta(a)
            psi = psi_of(dispatch(50_000 + t, frac=0.3))
            y = (psi * true).sum(-1)
            for i in range(N):
                row = psi[i][live[i]]
                if known:
                    row = torch.cat([row[:1], torch.tensor([a]), a * row[1:]])
                row = row.reshape(1, 1, -1)
                ahead = rls[i].predict(row)[0, 0]
                rls[i].update(row, y[i].reshape(1, 1))
                if t >= 480 and a > 0.3:
                    perr += float((ahead - y[i]).abs())
                    pnull += float(y[i].abs())
        return perr / max(pnull, 1e-12)

    out = {mu: run(mu, False) for mu in (0.999, 0.98, 0.9)}
    known = run(0.98, True)
    assert out[0.98] < out[0.999] and out[0.9] < out[0.98], f"prediction error did not fall with mu: {out}"
    assert out[0.9] < 0.2, f"mu=0.9 one-step prediction error on the live half-day: {out[0.9]:.3f}"
    assert known < 0.05, f"with the driver's shape in the basis the error should vanish, got {known:.3f}"
    return (
        f"one-step prediction error / |y| on the live half-day: mu=0.999 -> {out[0.999]:.3f}, "
        f"0.98 -> {out[0.98]:.3f}, 0.9 -> {out[0.9]:.3f}; with pact_known_driver at 0.98 -> {known:.3f}"
    )


@check("test_rls_dead_row_does_not_inflate_covariance")
def _dead_row():
    r_dead = RLS(4, 3, P)
    tr0 = float(r_dead.P.diagonal(dim1=-2, dim2=-1).sum(-1).max())
    dead = torch.zeros(4, 3)
    dead[:, 0] = 1.0  # an intercept of exactly 1 and NO channel content
    for _ in range(2000):
        r_dead.update(dead, torch.zeros(4))
    tr1 = float(r_dead.P.diagonal(dim1=-2, dim2=-1).sum(-1).max())
    assert abs(tr1 - tr0) < 1e-6, f"2000 dead rows changed tr(P) {tr0:.4f} -> {tr1:.4f}"
    assert float(r_dead.n_skipped.max()) == 2000
    return f"tr(P) held at {tr1:.3f} over 2000 intercept-only rows (trap #8: liveness on the CHANNELS)"


@check("test_covariance_windup_is_bounded")
def _windup():
    r = RLS(N, DIM, PactParams(mu=0.9, p_trace_max=100.0))
    frozen = psi_of(dispatch(7))
    true = target_beta(1.0)
    for _ in range(3000):
        r.update(frozen, (frozen * true.unsqueeze(0)).sum(-1))
    tr = float(r.P.diagonal(dim1=-2, dim2=-1).sum(-1).max())
    cap = 100.0 * 10.0 * DIM
    assert tr <= cap * (1 + 1e-6), f"tr(P) = {tr:.3g} exceeds the bound {cap:.3g}"
    assert torch.isfinite(r.predict(frozen)).all(), "the prediction overflowed"
    return f"mu=0.9, excitation frozen 3000 steps: tr(P) = {tr:.3g} <= bound {cap:.3g}; bounded {int(r.n_bounded.max())} times"


@check("test_trust_prior_is_inverted")
def _prior():
    g0 = float(trust_from_logit(torch.zeros(1), P))
    assert g0 > 0.85, f"w=0 gives trust {g0:.3f}"
    return f"w=0 -> trust {g0:.3f} of g_max (bias {P.trust_bias})"


@check("test_confidence_cold_to_warm")
def _conf():
    r = RLS(N, DIM, P)
    psi0 = psi_of(dispatch(0))
    cold = float(confidence(psi0, r.P, P, DIM).mean())
    true = target_beta(1.0)
    for t in range(1500):
        psi = psi_of(dispatch(30_000 + t))
        r.update(psi, (psi * true.unsqueeze(0)).sum(-1))
    warm = float(confidence(psi0, r.P, P, DIM).mean())
    assert warm > cold, f"confidence did not rise: {cold:.4f} -> {warm:.4f}"
    return f"prediction confidence {cold:.4f} (cold) -> {warm:.4f} (warm)"


@check("test_pred_confidence_survives_dead_excitation")
def _conf_survives():
    r = RLS(N, DIM, P)
    frozen = psi_of(dispatch(7))
    true = target_beta(1.0)
    for _ in range(4000):
        r.update(frozen, (frozen * true.unsqueeze(0)).sum(-1))
    trP = float(r.P.diagonal(dim1=-2, dim2=-1).sum(-1).mean())
    trace_gate = 1.0 / (1.0 + trP / P.p0)
    pred_gate = float(confidence(frozen, r.P, P, DIM).mean())
    assert pred_gate > 0.9, f"prediction gate fell to {pred_gate:.4f}"
    assert trace_gate < pred_gate, "the trace gate did not reproduce its failure"
    return f"excitation frozen 4000 steps: prediction gate {pred_gate:.4f} (armed) vs trace gate {trace_gate:.4f}"


@check("test_floor_property_is_exact")
def _floor():
    gen = torch.Generator().manual_seed(11)
    for _ in range(200):
        q = torch.randn(N, generator=gen)
        pred = torch.randn(N, generator=gen) * 1e6
        out = compensate(q.unsqueeze(-1), torch.ones(N, 1), pred, torch.zeros(N)).squeeze(-1)
        assert torch.equal(out, q), "g=0 was not a bit-for-bit no-op"
    return "g=0 is bit-identical to the untouched command over 200 draws with a 1e6-scale estimate"


@check("test_oracle_cancels_and_off_is_blind")
def _arms():
    """The two identities the arms rest on, on the layer's own arithmetic."""
    q = dispatch(5)
    x = C.step_channels(q, torch.zeros(N, R))
    _, d, _ = C.disturbance(x, torch.tensor([1.0]), 2.0)
    pred = d / ST.s_rated_mva
    exec_oracle = compensate(q.unsqueeze(-1), torch.ones(N, 1), pred * ST.s_rated_mva, torch.full((N,), 0.9)).squeeze(-1) + d
    exec_off = compensate(q.unsqueeze(-1), torch.ones(N, 1), pred * ST.s_rated_mva, torch.zeros(N)).squeeze(-1) + d
    left = float((exec_oracle - q).abs().max() / d.abs().max().clamp_min(1e-12))
    assert abs(left - 0.1) < 1e-4, f"oracle at trust 0.9 should leave exactly 10% of |d|, left {left:.4f}"
    assert torch.equal(exec_off, q + d), "trust 0 is not exactly the blind arm"
    return f"oracle at trust 0.9 leaves {left:.3f} of |d|; trust 0 == blind bit for bit"


def main() -> int:
    width = 84
    print("=" * width)
    print("mapdn_ns -- PACT-1 estimator self-test on the feeder basis (torch only)")
    print(ST.banner())
    print("=" * width)
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
