#  The declared coupling operator and the PACT basis.
#
#  NS-1.2 and P-1.1/P-1.2/P-3.1/P-3.2/P-3.3.
#
#  ---------------------------------------------------------------------------
#  The reduction
#  ---------------------------------------------------------------------------
#  The unknown quantity is how much of each neighbour's VAR actually reaches
#  you today, through the feeder, as a firmware reaction.  It is projected onto
#  ``r`` declared CONDUCTOR CLASSES of the feeder's segments, so the number of
#  parameters is ``r``, independent of the number of inverters and of the number
#  of segments (P-1.1).  What is declared is the GEOMETRY -- which segments a
#  neighbour's VARs cross to reach you, and their reactance, read off the
#  network file.  What is not declared is ``beta*``: what a unit of VAR on a
#  class-m segment costs, which drifts with the driver.
#
#      S_m[i, j] = sum over segments a on path(i) AND path(j), class(a) = m,
#                  of  X_a / V_base^2                      (r, N, N), S_m[i,i] = 0
#      x_m,i(t)  = rho x_m,i(t-1) + (1-rho) sum_{j != i} S_m[i, j] q_j(t-1)
#                                                            pu voltage, PUBLIC
#      psi_i     = [1, x_1,i, ..., x_r,i]
#      d_i       = s_i * clip( beta*(t) . x_i , +-q_sat )   MVAr, PRIVATE
#
#  Three things this buys, and each one is a requirement rather than a
#  convenience:
#
#  * The model is EXACTLY linear in psi below the curve's saturation.  The
#    channels carry the whole public geometry; the private part is r numbers.
#  * The channels are in the SENSOR's own units.  y_i = d_i / s_i is the
#    fractional VAR shortfall the inverter's own meter reports, in pu of its
#    nameplate, so beta* has no per-agent gain to un-normalise -- the same
#    coefficient for every inverter, which is what makes r independent of N.
#  * Every sum is strictly over ``j != i`` (P-3.1), so at N=1 every channel is
#    exactly zero, psi is [1, 0, ..., 0], and the disturbance is exactly zero
#    at any severity.  Category C, structurally.
#
#  torch only.  No pandapower.

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor

from mapdn_ns.driver import DialParams, beta_star, class_constants
from mapdn_ns.structure import FeederStructure

__all__ = ["Coupling"]


class Coupling:
    """The declared operator, the filtered channels, and the basis."""

    def __init__(self, struct: FeederStructure, p: DialParams, device=None) -> None:
        if struct.n_classes != int(p.n_classes):
            raise ValueError(
                f"the structure declares {struct.n_classes} conductor classes but the "
                f"dial says n_classes={p.n_classes}; these must agree (P-3.4: keep every "
                "index aligned)"
            )
        self.st = struct
        self.p = p
        self.device = device
        self.n = struct.n_agents
        self.r = struct.n_classes
        self.send = class_constants(p).to(device)
        self.s_rated = struct.s_rated_mva.to(device)
        self.k = float(p.k_slope)

        L = struct.n_lines
        inc = torch.zeros(self.n, L, device=device)
        for i, path in enumerate(struct.paths):
            if len(path):
                inc[i, torch.as_tensor(path, dtype=torch.long, device=device)] = 1.0
        self.incidence = inc
        xpu = struct.line_x_pu.to(device)
        S = torch.empty(self.r, self.n, self.n, device=device)
        eye = torch.eye(self.n, device=device, dtype=torch.bool)
        for m in range(self.r):
            mask = (struct.line_class.to(device) == m).to(torch.float32)
            S[m] = (inc * (xpu * mask).reshape(1, -1)) @ inc.transpose(0, 1)
            # The zero diagonal is ASSERTED, not argued: it is what makes the
            # estimated quantity a coupling rather than a self-effect, and what
            # makes a lone inverter read exactly zero at any severity.
            S[m] = S[m].masked_fill(eye, 0.0)
        self.S = S  # (r, N, N)
        self._eye = eye

    # ------------------------------------------------------------------
    #  NS-1.2 -- the declared operator
    # ------------------------------------------------------------------

    def W(self) -> Tensor:
        """``W[i, j]`` -- MVAr of firmware reaction at i per MVAr injected by j,
        at sigma = 1 and A = 1.  ``(N, N)``, zero diagonal.

            W_ij = k * s_i * sum_m send_m S_m[i, j]

        Asymmetric because the RECEIVER's nameplate scales its reaction: a
        1 MVAr push from a small unit onto a large one produces a large
        reaction, and not the reverse.  Spread by the feeder's own geometry --
        a neighbour on the same lateral shares many segments with you, one on
        the far side of the substation shares none.  A flat proxy was measured
        on the URB instance at a fit gain of -0.0045, worse than an
        intercept-only null, which is why both are load-bearing.
        """
        Ssum = (self.send.reshape(-1, 1, 1) * self.S).sum(0)
        return self.k * self.s_rated.reshape(-1, 1) * Ssum

    def loop_gain(self) -> float:
        """Spectral radius of ``W`` at sigma = 1, A = 1.

        The fleet's reactions feed each other: ``q_exec(t) = q_cmd(t) - sigma A W
        q_exec(t-1)``, so the mutual reaction is a discrete-time loop whose
        gain is ``sigma * A * rho(W)``.  Above 1 the fleet HUNTS -- the volt-var
        interaction instability IEEE 1547 bounds with its response-time
        requirement -- until the curves saturate.  ``1 / rho(W)`` is therefore
        the severity at which the anchored curve, made steeper, turns a
        disturbance into an instability at noon; it is a property of the
        feeder and the fleet, not a number anyone chose.
        """
        return float(torch.linalg.eigvals(self.W()).abs().max())

    # ------------------------------------------------------------------
    #  the channels
    # ------------------------------------------------------------------

    def step_channels(self, q_prev: Tensor, x_prev: Tensor) -> Tensor:
        """Advance the public channels one step.  ``(N,) , (N, r) -> (N, r)``.

        ``q_prev`` is the VAR each peer actually DELIVERED last step, in MVAr --
        the executed action, which P-4.1 allows (a connected fleet broadcasts
        it anyway) -- and never any peer's residual.
        """
        raw = torch.einsum("mij,j->im", self.S, q_prev)  # (N, r), j != i built in
        rho = float(self.p.rho)
        if rho > 0.0:
            return rho * x_prev + (1.0 - rho) * raw
        return raw

    def disturbance(self, x: Tensor, a: Tensor, severity: float) -> Tuple[Tensor, Tensor, Tensor]:
        """The private disturbance from the public channels.

        Returns ``(y_model, d_mvar, saturated)``: the model value in pu of
        nameplate ``(N,)``, the MVAr actually added to each command after the
        curve's own saturation, and a bool mask of where it saturated.

        At ``severity = 0`` beta* is exactly 0.0 so ``y_model`` is exactly 0.0,
        the clip of 0.0 is 0.0 and ``d`` is exactly 0.0 -- the identity NS-2.1
        demands, bit for bit.
        """
        p = DialParams(**{**self.p.__dict__, "severity": float(severity)})
        beta = beta_star(a, p).to(x.device)  # (1, r)
        y_model = (beta * x).sum(-1)  # (N,)
        lim = float(self.p.q_sat)
        y_sat = y_model.clamp(-lim, lim)
        sat = y_sat != y_model
        return y_model, y_sat * self.s_rated, sat

    def channel_liveness(self) -> Tensor:
        """``(N, r)`` bool: whether ANY peer's VAR can reach inverter i through a
        class-m segment.  P-3.4, per agent: an inverter whose only shared
        segment with the rest of the fleet is the trunk head has exactly-zero
        channels for the other classes, structurally.  Those columns are dead
        for that agent -- the RLS never moves them and the prediction never uses
        them -- and beta must be scored against the truth on the LIVE columns
        only, or an unobservable component reads as an estimation failure."""
        return self.S.abs().sum(-1).transpose(0, 1) > 0  # (N, r)

    def design(self, x: Tensor, ref: Tensor, scale: Tensor) -> Tensor:
        """``psi = [1, (x - ref) / scale]``.  ``(N, 1 + r)``.

        P-3.3: centre and scale on a geometric reference.  Raw channels carry a
        common mean against an intercept column of 1; measured on the source
        implementation that gave a design-matrix condition number of ~1.3e5, at
        which the intercept and the class channels trade off and the per-class
        split is unidentifiable even though prediction is fine.
        """
        centred = (x - ref.reshape(1, -1)) / scale.reshape(1, -1).clamp_min(1e-9)
        ones = torch.ones_like(centred[:, :1])
        return torch.cat([ones, centred], dim=-1)

    # ------------------------------------------------------------------
    #  P-3.3 -- the geometric reference
    # ------------------------------------------------------------------

    def reference_dispatch(self, samples: int, seed: int) -> Tensor:
        """``(samples, N)`` MVAr: every inverter dispatching uniformly at random
        over its OWN nameplate VAR range at full capability, ``[-scale s_i,
        +scale s_i]``.  A function of structure only -- no run data enters.

        The range is the host's own (``action_scale`` times the capability at
        zero active power), not a uniform draw over some arena: normalising to
        the wrong reference is trap #3/#4 of the porting notes, measured at 14x
        on ``balance``."""
        gen = torch.Generator().manual_seed(seed)
        u = torch.rand(samples, self.n, generator=gen) * 2.0 - 1.0
        return (u * self.st.action_scale * self.s_rated.cpu().reshape(1, -1)).to(self.S.device)

    def geometric_reference(self, samples: int = 512, seed: int = 0) -> Tuple[Tensor, Tensor]:
        """``(ref, scale)``, each ``(r,)``: the channel each inverter would see if
        every peer dispatched uniformly at random from the host's own range.
        Computed with an explicit CPU generator so it cannot consume the run's
        RNG stream and cannot differ between arms."""
        Q = self.reference_dispatch(samples, seed)
        x = torch.zeros(self.n, self.r, device=self.S.device)
        vals = []
        for k in range(samples):
            # the filtered channel at steady state under a sustained dispatch
            for _ in range(3 if self.p.rho > 0 else 1):
                x = self.step_channels(Q[k], x)
            vals.append(x)
        V = torch.stack(vals).reshape(-1, self.r)
        return V.mean(0), V.std(0, unbiased=False).clamp_min(1e-9)

    def derate_reference(self, samples: int = 512, seed: int = 7) -> Tensor:
        """``D_ref``, ``(N,)`` MVAr: the mean |d_i| the coupling produces on each
        inverter at sigma = 1, A = 1 under the reference dispatch.

        This is what the (B) control derates every inverter's OWN command by
        (times sigma and A), so the two cells share a scale per inverter and
        their ladders can be read against each other: at a given sigma a lone
        inverter under ``ns_direct`` loses, on average, exactly what the fleet's
        mutual reaction removes from it on average.  Per inverter, not a fleet
        average -- a 3.6 MVA unit reacts more than a 0.33 MVA one in (C), and
        (B) matches that.
        """
        Q = self.reference_dispatch(samples, seed)
        acc = torch.zeros(self.n, device=self.S.device)
        x = torch.zeros(self.n, self.r, device=self.S.device)
        one = torch.ones(1, device=self.S.device)
        for k in range(samples):
            for _ in range(3 if self.p.rho > 0 else 1):
                x = self.step_channels(Q[k], x)
            _, d, _ = self.disturbance(x, one, 1.0)
            acc += d.abs()
        ref = acc / samples
        if not float(ref.sum()) > 0:
            raise RuntimeError(
                "the reference disturbance came out non-positive; the coupling is "
                "inert and sigma would have no meaning"
            )
        return ref

    # ------------------------------------------------------------------
    #  P-3.2 -- gate 1: the vectorised form must equal the definition
    # ------------------------------------------------------------------

    def channels_bruteforce(self, q_prev: Tensor) -> Tensor:
        """The definition, written straight out as loops (rho = 0)."""
        out = torch.zeros(self.n, self.r)
        xpu = self.st.line_x_pu
        for i in range(self.n):
            Ei = set(self.st.paths[i])
            for j in range(self.n):
                if j == i:
                    continue  # P-3.1, written out
                Ej = set(self.st.paths[j])
                for a in Ei & Ej:
                    out[i, int(self.st.line_class[a])] += float(xpu[a]) * float(q_prev[j])
        return out

    def verify(self, q_prev: Tensor, tol: float = 1e-5) -> str:
        """Check ``step_channels`` against the brute-force loop and abort on
        mismatch.  Index order and self-exclusion are exactly the kind of wiring
        bug that leaves every diagnostic looking healthy, so this runs at
        startup rather than in a test file somebody can skip."""
        fast = self.step_channels(q_prev.to(self.S.device), torch.zeros(self.n, self.r, device=self.S.device))
        if self.p.rho > 0:
            fast = fast / (1.0 - self.p.rho)
        slow = self.channels_bruteforce(q_prev.cpu())
        err = float((fast.cpu() - slow).abs().max())
        ref = float(slow.abs().max())
        if err > tol * max(1.0, ref):
            raise RuntimeError(
                f"vectorised channels differ from the brute-force definition by "
                f"{err:.3e}. Index order or self-exclusion is wrong; every "
                "downstream diagnostic would still look healthy."
            )
        if float(self.S.diagonal(dim1=-2, dim2=-1).abs().max()) != 0.0:
            raise RuntimeError("the operator has a non-zero diagonal")
        # a lone inverter must read exactly zero on every channel
        one = Coupling(_single(self.st, 0), self.p)
        x1 = one.step_channels(q_prev[:1].cpu(), torch.zeros(1, self.r))
        if float(x1.abs().max()) != 0.0:
            raise RuntimeError(
                "a lone inverter read a non-zero channel; the sum is not strictly "
                "over j != i and this is category B in disguise"
            )
        return (
            f"channels == definition to {err:.2e}; N=1 reads exactly zero on "
            f"all {self.r} channels"
        )

    # ------------------------------------------------------------------
    #  reporting
    # ------------------------------------------------------------------

    def operator_stats(self) -> Dict[str, float]:
        """NS-1.2's three properties, measured rather than asserted."""
        W = self.W()
        off = W[~self._eye]
        off_pos = off[off > 0]
        num = (W - W.T).abs()
        den = W + W.T
        mask = (~self._eye) & (den > 0)
        return {
            "diag_max": float(W.diagonal().abs().max()),
            "spread": float(off_pos.std(unbiased=False) / off_pos.mean().clamp_min(1e-30))
            if off_pos.numel() > 1 else 0.0,
            "asymmetry": float((num[mask] / den[mask]).mean()) if bool(mask.any()) else 0.0,
            "zero_pairs": float((off == 0).float().mean()),
            "w_max": float(off.max()),
            "w_mean": float(off.mean()),
        }

    def banner(self) -> str:
        st = self.operator_stats()
        return (
            f"coupling        N={self.n} r={self.r} classes={self.st.class_names}\n"
            f"                send (unknown) {[round(float(v), 3) for v in self.send]}\n"
            f"                W (MVAr/MVAr @ sigma=1, A=1): mean {st['w_mean']:.4f} max {st['w_max']:.4f} "
            f"spread {st['spread']:.3f} asym {st['asymmetry']:.3f} zero-pairs {st['zero_pairs']:.0%}\n"
            f"                k_slope={self.k:.3f} q_sat={self.p.q_sat} rho={self.p.rho}\n"
            f"                loop gain rho(W)={self.loop_gain():.3f} at sigma=1 -> the fleet hunts at noon "
            f"for sigma > {1.0 / max(self.loop_gain(), 1e-9):.2f}"
        )


def _single(st: FeederStructure, i: int) -> FeederStructure:
    """The same feeder with only inverter ``i`` on it."""
    return FeederStructure(
        name=st.name + "/N1",
        v_base_kv=st.v_base_kv,
        line_x_ohm=st.line_x_ohm,
        line_r_ohm=st.line_r_ohm,
        line_class=st.line_class,
        class_names=st.class_names,
        paths=[list(st.paths[i])],
        sgen_bus=[st.sgen_bus[i]],
        s_rated_mva=st.s_rated_mva[i : i + 1],
        action_scale=st.action_scale,
        class_rule=st.class_rule,
        extra=dict(st.extra),
    )
