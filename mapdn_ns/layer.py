#  The layer, stacked onto MAPDN's ``VoltageControl``.
#
#      MultiAgentEnv
#        +-- VoltageControl                <- the stock host, untouched
#              +-- NsMixin                 <- I.4: EVERY arm gets this, unmodified
#                    +-- PactMixin         <- adds the compensator only
#
#  NS-3.1: the dial sits BELOW the method in the hierarchy and is read from the
#  task configuration (``mapdn_ns/conf/<scenario>.yaml`` through the launcher).
#  A dial only the method's arm experienced is worthless as evidence.
#
#  ---------------------------------------------------------------------------
#  Where the disturbance is applied, and why there
#  ---------------------------------------------------------------------------
#  In ``_take_action``, the single place the host converts an action into a
#  reactive-power injection and runs the power flow.  That is I.4's corollary:
#  apply the harm where every host reads an action, not in a training loop
#  that only some algorithms have.  ``step``'s reward, ``done`` and the stock
#  observation are inherited untouched -- the inverter is paid exactly what it
#  was paid before and earns less only because it physically delivered
#  different VARs (NS-1.4).
#
#  The host has exactly one environment (no batch dimension), so every tensor
#  here is ``(N, ...)`` and the RLS runs with ``batch = 1``.

from __future__ import annotations

import atexit
import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

from pact1.core import PactParams, RLS, compensate, confidence
from mapdn_ns.coupling import Coupling
from mapdn_ns.driver import DialParams, beta_star, driver_A
from mapdn_ns.structure import FeederStructure, build_structure, load_net, structure_path

__all__ = ["NsMixin", "PactMixin", "NS_KWARGS", "PACT_KWARGS", "InertLayerError"]


NS_KWARGS = (
    "ns_severity",
    "ns_direct",
    "ns_driver",
    "ns_day_start_hour",
    "ns_day_end_hour",
    "ns_k_slope",
    "ns_q_sat",
    "ns_rho",
    "ns_n_classes",
    "ns_send_spread",
    "ns_y_clip",
    "ns_observe_residual",
    "ns_observe_channels",
    "ns_observe_driver",
    "ns_structure",
)

PACT_KWARGS = (
    "pact_enabled",
    "pact_trust",
    "pact_mu",
    "pact_p0",
    "pact_warmup",
    "pact_channels",
    "pact_oracle",
    "pact_corr_clip",
    "pact_p_trace_max",
    "pact_known_driver",
)


def _ratio(num, den) -> float:
    """sum(num) / sum(den), NaN when the denominator is negligible (a dark
    episode has no disturbance to explain or cancel)."""
    if not num or not den:
        return float("nan")
    d = float(np.sum(den))
    return float(np.sum(num)) / d if d > 1e-6 else float("nan")


def _ratio_where(num, den, cond_vals, pred) -> float:
    if not num or not den or not cond_vals:
        return float("nan")
    n = [v for v, c in zip(num, cond_vals) if pred(c)]
    d = [v for v, c in zip(den, cond_vals) if pred(c)]
    return _ratio(n, d)


def _mean_where(vals, cond_vals, pred) -> float:
    if not vals or not cond_vals:
        return float("nan")
    sel = [v for v, c in zip(vals, cond_vals) if pred(c)]
    return float(np.mean(sel)) if sel else float("nan")


class InertLayerError(RuntimeError):
    """NS-3.3: the dial was on and never reached the physics."""


def _pop(cfg: Dict[str, Any], keys) -> Dict[str, Any]:
    return {k: cfg.pop(k) for k in keys if k in cfg}


def _as_dict(kwargs) -> Dict[str, Any]:
    if isinstance(kwargs, dict):
        return dict(kwargs)
    if hasattr(kwargs, "_asdict"):
        return dict(kwargs._asdict())
    return dict(vars(kwargs))


class NsMixin:
    """Interaction-mediated, invertible non-stationarity on ``VoltageControl``.

    Overrides ``__init__``, ``reset``, ``manual_reset``, ``_take_action``,
    ``step``, ``get_obs`` and -- a loader repair only, see
    ``structure.load_net`` -- ``_load_network``.  Nothing else.
    """

    def _load_network(self):
        import os

        return self._create_basenet(load_net(os.path.join(self.data_path, "model.p")))

    # ------------------------------------------------------------------
    #  construction
    # ------------------------------------------------------------------

    def __init__(self, kwargs) -> None:
        cfg = _as_dict(kwargs)
        raw = _pop(cfg, NS_KWARGS)
        self._pact_raw = _pop(cfg, PACT_KWARGS)
        self.ns = DialParams(
            severity=float(raw.get("ns_severity", 1.0)),
            driver=str(raw.get("ns_driver", "solar")),
            day_start_hour=float(raw.get("ns_day_start_hour", 6.0)),
            day_end_hour=float(raw.get("ns_day_end_hour", 18.0)),
            k_slope=float(raw.get("ns_k_slope", DialParams.k_slope)),
            q_sat=float(raw.get("ns_q_sat", DialParams.q_sat)),
            rho=float(raw.get("ns_rho", 0.0)),
            n_classes=int(raw.get("ns_n_classes", 3)),
            send_spread=float(raw.get("ns_send_spread", 0.8)),
            y_clip=float(raw.get("ns_y_clip", 10.0)),
        )
        # The (B) CONTROL.  See _disturbance for what it changes and why the
        # comparison between it and the default is the paper's central pair.
        self._direct = bool(raw.get("ns_direct", False))
        self._observe_residual = bool(raw.get("ns_observe_residual", True))
        self._observe_channels = bool(raw.get("ns_observe_channels", True))
        #  The realised driver level, published to every arm -- as a DLR system
        #  publishes ratings to the control room.  Off by default: the driver
        #  is inferable from the PV column already in the stock observation.
        self._observe_driver = bool(raw.get("ns_observe_driver", False))
        self._structure_override = raw.get("ns_structure", None)
        self._built = False
        self._debug_csv = os.environ.get("MAPDN_NS_DEBUG_CSV", "")
        self._episode_index = 0
        self._arm_label = "blind"
        #  "train" | "eval" | "test": the launcher flips this around the
        #  model's evaluation() and in test.py, so the debug rows of the greedy
        #  evaluation episodes are separable from the exploring training ones
        self.phase = "train"
        self._episode_phase = "train"
        super().__init__(cfg)
        if getattr(self, "history", 1) > 1:
            raise NotImplementedError("mapdn_ns does not support obs history > 1")
        atexit.register(self._at_exit)

    def _scenario_name(self) -> str:
        return Path(str(self.data_path)).name

    def _ensure_built(self) -> None:
        if self._built:
            return
        scen = self._scenario_name()
        path = Path(self._structure_override) if self._structure_override else structure_path(scen)
        if path.exists():
            st = FeederStructure.from_json(path)
        else:
            # Built from the host's OWN network file and nameplate rule
            # (s_max = 1.2 p_max), then written so the torch-only tools can read
            # the same structure the run used.
            st = build_structure(
                self.base_powergrid, self.s_max, float(self.args.action_scale),
                name=scen, n_classes=self.ns.n_classes,
            )
            try:
                st.to_json(path)
            except OSError:
                pass
        if st.n_agents != self.n_agents:
            raise RuntimeError(
                f"structure has {st.n_agents} inverters but the host has {self.n_agents}; "
                "the exported structure belongs to a different scenario"
            )
        if self.args.mode != "distributed":
            raise NotImplementedError("mapdn_ns supports the distributed mode only (one inverter per agent)")
        self.structure = st
        self.coupling = Coupling(st, self.ns)
        self.n_ag = st.n_agents
        N, r = self.n_ag, self.ns.n_classes
        self._s_rated = st.s_rated_mva.clone()  # (N,) MVA
        # P-3.3, on the host's OWN dispatch range -- see Coupling.reference_dispatch
        self._ref, self._scale = self.coupling.geometric_reference()
        # the matched scale for the (B) control -- see Coupling.derate_reference.
        # An inert coupling (every peer's curve switched off, as smoke.py does
        # to put one inverter alone on the medium) has no reference; that is
        # only an error if the (B) control then asks for it.
        try:
            self._d_ref = self.coupling.derate_reference()  # (N,) MVAr
        except RuntimeError as exc:
            if self._direct:
                raise
            print(f"ns layer        NOTE: {exc}")
            self._d_ref = torch.full((N,), float("nan"))

        self._x = torch.zeros(N, r)
        self._x_prev = torch.zeros(N, r)
        self._u_prev = torch.zeros(N)
        self._d = torch.zeros(N)
        self._y_model = torch.zeros(N)
        self._y = torch.zeros(N)
        self._y_prev = torch.zeros(N)
        self._sat = torch.zeros(N, dtype=torch.bool)
        self._clipped = torch.zeros(N, dtype=torch.bool)
        self._A = 0.0
        self._hour = float("nan")
        self._q_cmd = torch.zeros(N)
        self._q_sent = torch.zeros(N)
        self._q_exec = torch.zeros(N)

        # NS-3.3: count what the layer actually touched.
        self._n_seen = 0
        self._n_hit = 0
        self._ep_acc: Dict[str, List[float]] = {}

        self._on_built()
        self._built = True
        print(self.structure.banner())
        print(self.coupling.banner())
        print(
            f"ns layer        sigma={self.ns.severity} driver={self.ns.driver} "
            f"day=[{self.ns.day_start_hour:g},{self.ns.day_end_hour:g}) N={self.n_ag} "
            f"D_ref={float((self._d_ref / (self.structure.action_scale * self._s_rated)).mean()):.3f} of the VAR range "
            f"obs+=(residual={self._observe_residual}, "
            f"channels={self._observe_channels}, driver={self._observe_driver}) "
            f"channel={'DIRECT (B, control)' if self._direct else 'COUPLED (C)'}",
            flush=True,
        )

    def _on_built(self) -> None:
        """Hook for the layer above the dial."""
        return

    # ------------------------------------------------------------------
    #  lifecycle
    # ------------------------------------------------------------------

    def _reset_layer(self) -> None:
        # Peers' EXECUTED VARs at the start of the episode are the host's own
        # reset dispatch (random, or zero), which is what the medium carries
        # into the first step.
        self._u_prev = torch.as_tensor(
            self.powergrid.sgen["q_mvar"].to_numpy(copy=True), dtype=torch.float32
        )
        for t in (self._x, self._x_prev, self._d, self._y_model, self._y, self._y_prev,
                  self._q_cmd, self._q_sent, self._q_exec):
            t.zero_()
        self._sat.zero_()
        self._clipped.zero_()
        self._A = 0.0
        # latch the phase the episode is PLAYED in: the row is flushed at the
        # next reset, by which time the launcher may have flipped it
        self._episode_phase = self.phase
        self._on_reset()

    def _on_reset(self) -> None:
        return

    def reset(self, reset_time: bool = True):
        self._ensure_built()
        if self._built and self._n_seen > 0:
            self._flush_episode()
        super().reset(reset_time)
        self._reset_layer()
        # NS-3.4: the driver's clock is the DATASET's clock -- the episode start
        # hour the host drew -- and it is not reset by us.
        return self.get_obs(), self.get_state()

    def manual_reset(self, day, hour, interval):
        self._ensure_built()
        if self._built and self._n_seen > 0:
            self._flush_episode()
        super().manual_reset(day, hour, interval)
        self._reset_layer()
        return self.get_obs(), self.get_state()

    def _hour_now(self) -> float:
        """Hour of day of the interval being dispatched, from the host's own
        episode clock (start hour, start interval, step counter)."""
        delta = float(getattr(self, "time_delta", 3))
        start_min = float(self._episode_start_hour) * 60.0 + float(self._episode_start_interval) * delta
        minutes = start_min + float(self.steps - 1) * delta
        return (minutes / 60.0) % 24.0

    # ------------------------------------------------------------------
    #  the disturbance
    # ------------------------------------------------------------------

    def _disturbance(self, q_cmd: Tensor) -> None:
        """Compute this step's additive disturbance for the whole fleet."""
        self._hour = self._hour_now()
        a = driver_A(torch.tensor([self._hour]), self.ns)
        self._A = float(a)

        # the public channels, from peers' executed VARs of the last interval
        self._x_prev = self._x
        self._x = self.coupling.step_channels(self._u_prev, self._x_prev)

        if self._direct:
            #  THE (B) CONTROL -- exogenous, not interaction-mediated.
            #
            #  Same firmware, same driver, same severity, same scale, same
            #  reward, same ladder.  The ONLY change is WHAT the curve reacts
            #  to: the exogenous, PV-driven rise of the solar day itself rather
            #  than the voltage the PEERS impose.  Every inverter's curve then
            #  absorbs a fixed amount, sigma * A(t) * D_ref_i, whatever it or
            #  anyone else was dispatched, with no sum over j != i anywhere.
            #  D_ref_i is the mean |d_i| the coupling produces on THAT inverter
            #  under the reference dispatch (Coupling.derate_reference), so
            #  sigma means the same severity in both cells, per inverter.
            #
            #  A LONE inverter now feels it, so the N=1 test separates the two
            #  cells by measurement.  PACT's PEER CHANNELS carry no information
            #  about it: it is a level shift that drifts with the clock, the
            #  regression puts it in the INTERCEPT and the class channels go to
            #  zero, so fit gain over an intercept-only null collapses to ~0.
            #  That is what `pact_channels=intercept` isolates, and the pair is
            #  what the classification rests on.
            #
            #  (An earlier form opposed each inverter's OWN command instead,
            #  -sign(q_cmd) * mag.  That is not a level shift: its sign flips
            #  with the agent's dispatch, which on a coherent fleet correlates
            #  with the peers' dispatch, so the peer channels predicted it by
            #  correlation and "PACT beat intercept on (B)" -- a confound of
            #  the control, not a property of the cell.)
            d = -(self.ns.severity * self._A) * self._d_ref
            self._y_model = d / self._s_rated
            self._sat = torch.zeros_like(self._sat)
            self._d = d
        else:
            y_model, d, sat = self.coupling.disturbance(self._x, a, self.ns.severity)
            self._y_model, self._d, self._sat = y_model, d, sat

        self._after_disturbance()

    def _after_disturbance(self) -> None:
        return

    def _correction(self) -> Tensor:
        """The compensator's correction, MVAr, ``(N,)``.  The dial never
        produces one; the layer above (PACT) overrides this."""
        return torch.zeros(self.n_ag)

    def _take_action(self, actions):
        """The single hook.  Returns the host's own solvability flag."""
        self._ensure_built()
        p = self.powergrid.sgen["p_mw"]
        # the stock host's own conversion, bit for bit: capability * fraction
        q_cmd_series = self._clip_reactive_power(actions, p)
        #  The action path stays in the host's own float64: at sigma = 0 the
        #  value that reaches the power flow must be the stock value bit for
        #  bit, and a float32 round trip was measured to move the reward by
        #  2e-9.  The estimator's own arithmetic is float32 and is cast at the
        #  boundary.
        q_cmd = torch.as_tensor(np.asarray(q_cmd_series, dtype=np.float64))
        cap = torch.as_tensor(
            np.sqrt(np.asarray(self.s_max, dtype=np.float64) ** 2 - np.asarray(p, dtype=np.float64) ** 2)
        )

        self._disturbance(q_cmd.to(torch.float32))
        corr = self._correction().to(torch.float64)
        d = self._d.to(torch.float64)
        # II.6 row 1: the exact inverse.  `compensate` is pact1's own channel,
        # with the 1-D action's direction the unit scalar.
        q_sent = compensate(q_cmd.unsqueeze(-1), torch.ones_like(q_cmd).unsqueeze(-1),
                            corr, torch.ones_like(corr)).squeeze(-1)
        # NS-1.4: an unmodelled term added to what the inverter was asked for.
        # At sigma = 0 the disturbance is exactly 0.0, so this is q + 0.0.
        q_exec = q_sent + d
        # the inverter's physical capability, |q| <= sqrt(S^2 - P^2)
        q_clip = torch.nan_to_num(q_exec, nan=0.0, posinf=0.0, neginf=0.0).clamp(-cap, cap)
        self._clipped = q_clip != q_exec
        q_exec = q_clip

        # P-2.1 proprioception: the inverter knows what it sent and meters what
        # it delivered, so the fractional shortfall in pu of its own nameplate
        # is directly observable.  Nothing privileged is used.
        self._y = ((q_exec - q_sent) / self._s_rated.to(torch.float64)).to(torch.float32)
        self._q_cmd, self._q_sent, self._q_exec = q_cmd.to(torch.float32), q_sent.to(torch.float32), q_exec.to(torch.float32)

        if self.ns.severity > 0:
            self._n_hit += int((self._d.abs() > 0).sum())
        self._n_seen += self.n_ag

        self.powergrid.sgen["q_mvar"] = q_exec.numpy()
        solvable = self._run_pf()
        # Peers' EXECUTED VARs, which is what the medium transmits and what
        # P-4.1 lets an agent see.  Broadcast for the next interval.
        self._u_prev = q_exec.to(torch.float32)
        self._y_prev = self._y.clone()
        return solvable

    def _run_pf(self) -> bool:
        import pandapower as pp
        from pandapower import ppException

        try:
            pp.runpp(self.powergrid)
            return True
        except ppException:
            print("The power flow for the reactive power penetration cannot be solved.")
            return False

    # ------------------------------------------------------------------
    #  sensor and read-out
    # ------------------------------------------------------------------

    def get_obs(self):
        obs = super().get_obs()
        if not self._built:
            return obs
        if not (self._observe_residual or self._observe_channels or self._observe_driver):
            return obs
        psi = self.coupling.design(self._x, self._ref, self._scale)  # (N, 1+r)
        out = []
        for i, o in enumerate(obs):
            extra = []
            if self._observe_residual:
                # ONE STEP STALE, and clipped to a declared bound (P-2.1).
                extra.append(float(self._y_prev[i].clamp(-1.0, self.ns.y_clip)))
            if self._observe_channels:
                extra.extend(psi[i, 1:].tolist())
            if self._observe_driver:
                extra.append(float(self._A))
            out.append(np.concatenate([np.asarray(o, dtype=np.float64), np.asarray(extra, dtype=np.float64)]))
        return out

    def step(self, actions, add_noise: bool = True):
        reward, terminated, info = super().step(actions, add_noise)
        info = dict(info)
        info.update(self._layer_info())
        for k, v in info.items():
            self._ep_acc.setdefault(k, []).append(float(v))
        self._ep_acc.setdefault("reward", []).append(float(reward))
        return reward, terminated, info

    def _layer_info(self) -> Dict[str, float]:
        s = self._s_rated
        q_range = self.structure.action_scale * s
        return {
            "ns_A": float(self._A),
            "ns_hour": float(self._hour),
            "ns_load": float((self._d.abs() / s).mean()),
            "ns_load_frac": float((self._d.abs() / q_range).mean()),
            "ns_y": float(self._y.mean()),
            "ns_sat": float(self._sat.float().mean()),
            "ns_clipped": float(self._clipped.float().mean()),
            "ns_x_std": float(self._x.sum(-1).std(unbiased=False)) if self.n_ag > 1 else 0.0,
            "ns_q_exec_abs": float(self._q_exec.abs().mean()),
            "ns_q_cmd_abs": float(self._q_cmd.abs().mean()),
            "ns_resid_after": float(((self._q_exec - self._q_cmd).abs() / s).mean()),
        }

    # ------------------------------------------------------------------
    #  NS-3.3 -- fail loudly when the layer is inert
    # ------------------------------------------------------------------

    def severity_report(self) -> str:
        return (
            f"ns layer: agent-steps seen {self._n_seen}, disturbed {self._n_hit} "
            f"(sigma={self.ns.severity}, {'direct' if self._direct else 'coupled'})"
        )

    def assert_layer_fired(self, min_seen: int = 480) -> None:
        if self.ns.severity <= 0 or self._n_seen < min_seen:
            return
        if self._n_hit == 0:
            raise InertLayerError(
                f"ns_severity={self.ns.severity} but NOT ONE action was disturbed "
                f"over {self._n_seen} agent-steps. The layer is not reaching the "
                "physics; this is a wiring bug, not a null result."
            )

    def _at_exit(self) -> None:
        if not self._built:
            return
        try:
            self._flush_episode()  # the last episode has no reset after it
        except Exception:  # noqa: BLE001
            pass
        print(self.severity_report(), flush=True)
        if self.ns.severity > 0 and self._n_hit == 0 and self._n_seen >= 480 * self.n_ag:
            print(
                "REFUSING to report this run as a severity arm: the dial never "
                "reached the physics (NS-3.3).",
                flush=True,
            )

    # ------------------------------------------------------------------
    #  the per-episode debug row (II.10)
    # ------------------------------------------------------------------

    def _panel_row(self) -> Dict[str, float]:
        acc = self._ep_acc
        m = lambda k: float(np.mean(acc[k])) if k in acc and acc[k] else float("nan")  # noqa: E731
        row = {
            "episode": self._episode_index,
            "phase": self._episode_phase,
            "arm": self._arm_label,
            "sigma": self.ns.severity,
            "direct": int(self._direct),
            # did the dial fire, and how big was it
            "A": m("ns_A"),
            "load": m("ns_load"),
            "load_frac": m("ns_load_frac"),
            "sat": m("ns_sat"),
            "clipped": m("ns_clipped"),
            # is there anything to identify
            "x_std": m("ns_x_std"),
            "q_cmd_abs": m("ns_q_cmd_abs"),
            "q_exec_abs": m("ns_q_exec_abs"),
        }
        row.update(self._panel_extra())
        row.update({
            "resid_after": m("ns_resid_after"),
            # the domain metric, in the benchmark's own units
            "reward": m("reward"),
            "v_out": m("percentage_of_v_out_of_control"),
            "cr": m("totally_controllable_ratio"),
            "q_loss": m("q_loss"),
            "destroy": m("destroy"),
            "steps": len(acc.get("reward", [])),
        })
        return row

    def _panel_extra(self) -> Dict[str, float]:
        return {}

    def _flush_episode(self) -> None:
        if not self._ep_acc:
            return
        row = self._panel_row()
        self._episode_index += 1
        self._ep_acc = {}
        if self._debug_csv:
            path = Path(self._debug_csv)
            path.parent.mkdir(parents=True, exist_ok=True)
            new = not path.exists()
            with path.open("a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(row.keys()))
                if new:
                    w.writeheader()
                w.writerow(row)


class PactMixin(NsMixin):
    """PACT-1 on top of the dial, in II.6's INVERTIBLE row.

    The disturbance is additive in the inverter's own action space, so a
    correct scalar estimate cancels it exactly.  The method may therefore claim
    identification **and compensation** here.  Everything except the channel
    is the same object as in ``road_ns`` and ``simple_ns``: the RLS with its
    dead-row skip and covariance bound, the inverted trust prior, and the
    prediction-based confidence gate all come from ``pact1.core`` unchanged.

    P-3.4, per agent.  On a radial feeder an inverter whose only shared segment
    with the rest of the fleet is the trunk head has exactly-zero channels for
    the other conductor classes, structurally.  Feeding those columns to a
    batched estimator lets P grow by 1/mu per step in the dead directions until
    the windup bound shrinks the LIVE directions with them (measured: the bound
    fired on 1178 of 1440 steps and slowed tracking).  So each inverter runs
    its own ``RLS`` over its own live columns, and every downstream index is
    re-aligned through ``_live_cols``.
    """

    def _on_built(self) -> None:
        raw = self._pact_raw
        self.pact_enabled = bool(raw.get("pact_enabled", False))
        self.pact_params = PactParams(
            mu=float(raw.get("pact_mu", 0.999)),
            p0=float(raw.get("pact_p0", 10.0)),
            p_trace_max=float(raw.get("pact_p_trace_max", 100.0)),
            y_clip=self.ns.y_clip,
        )
        self._trust_const = float(raw.get("pact_trust", 0.9))
        self._warmup = int(raw.get("pact_warmup", 40))
        #  A DECLARED bound on the correction, as a fraction of the inverter's
        #  own VAR range.  A guard rail, not a knob: measured on simple_ns, a
        #  diverged estimator (error 1e34) clamped by the action box scored
        #  BETTER than a correct one.  Bounding the correction closes that
        #  route, and it is what makes selecting mu on prediction error safe.
        self._corr_clip = float(raw.get("pact_corr_clip", 0.6))
        #  The free-answer controller as an ARM inside the layer, not a wrapper:
        #  computed outside the env it is one step stale, which understated the
        #  ceiling enough that PACT appeared to BEAT it (trap #5).
        self._oracle = bool(raw.get("pact_oracle", False))
        self._channels = str(raw.get("pact_channels", "full"))
        if self._channels not in ("full", "intercept"):
            raise ValueError(f"pact_channels must be 'full' or 'intercept', got {self._channels!r}")
        #  ABLATION: the realised driver level in the basis, psi = [1, A, A z].
        #  beta* is then constant over the day and the estimator has no drift
        #  to track -- the ceiling of what identification can do when the
        #  driver's shape is public.  With the default (false) the estimator
        #  must TRACK beta*(t) through the day with forgetting, which is the
        #  spec's tracking floor and what the headline row reports.  Pair it
        #  with ns_observe_driver so the baselines are information-matched.
        self._known_driver = bool(raw.get("pact_known_driver", False))

        N, r = self.n_ag, self.ns.n_classes
        live = self.coupling.channel_liveness()  # (N, r)
        self._live = live
        cols: List[Tensor] = []
        for i in range(N):
            c = [0]  # the intercept
            if self._known_driver:
                c.append(1)
            if self._channels == "full":
                off = 2 if self._known_driver else 1
                c.extend(off + m for m in range(r) if bool(live[i, m]))
            cols.append(torch.as_tensor(c, dtype=torch.long))
        self._live_cols = cols
        self._full_dim = (2 if self._known_driver else 1) + r
        self.rls: List[RLS] = [RLS(1, int(c.numel()), self.pact_params, batch=1) for c in cols]

        self._psi_prev = torch.zeros(N, self._full_dim)
        self._have_prev = False
        self._pred = torch.zeros(N)
        self._conf = torch.zeros(N)
        self._trust = torch.zeros(N)
        self._corr = torch.zeros(N)
        self._n_diverged = 0
        # II.10's instrument panel, maintained online as decaying sums
        self._fit_sse = torch.zeros(N)
        self._null_sse = torch.zeros(N)
        self._ybar = torch.zeros(N)
        self._fit_gain = torch.zeros(N)
        self._beta_cos = torch.zeros(N)
        self._beta_err = torch.zeros(N)
        self._gram = [torch.zeros(int(c.numel()), int(c.numel())) for c in cols]
        self._cond = torch.full((N,), float("nan"))

        if self.pact_enabled:
            if self._oracle:
                self._arm_label = "oracle"
            elif self._trust_const == 0.0:
                self._arm_label = "pactoff"
            elif self._channels == "intercept":
                self._arm_label = "intercept"
            else:
                self._arm_label = "pact"

        # II.9 gates 1 and 3, at startup, before a single episode is simulated.
        gen = torch.Generator().manual_seed(0)
        q = (torch.rand(N, generator=gen) * 2 - 1) * self.structure.action_scale * self._s_rated
        print("PACT gate 1/3   " + self.coupling.verify(q))
        print(
            f"PACT            enabled={self.pact_enabled} arm={self._arm_label} "
            f"trust={self._trust_const} mu={self.pact_params.mu} channels={self._channels} "
            f"known_driver={self._known_driver} dims={[int(c.numel()) for c in cols]} "
            f"warmup={self._warmup} corr_clip={self._corr_clip} "
            f"channel=EXACT INVERSE (II.6 row 1)",
            flush=True,
        )

    def _on_reset(self) -> None:
        if hasattr(self, "_corr"):
            self._corr.zero_()
            # the medium is re-initialised at reset (a new day is drawn), so the
            # pairing of psi(t-1) with y(t-1) does not cross the reset
            self._have_prev = False

    # -- the basis, per agent ---------------------------------------------

    def _full_psi(self) -> Tensor:
        """``(N, full_dim)``: [1, (A,) centred channels]; per-agent live columns
        are selected from this by ``_live_cols``."""
        psi = self.coupling.design(self._x, self._ref, self._scale)  # (N, 1+r)
        if self._known_driver:
            a = torch.full((self.n_ag, 1), float(self._A))
            psi = torch.cat([psi[:, :1], a, a * psi[:, 1:]], dim=-1)
        return psi

    def _rows(self, psi: Tensor, i: int) -> Tensor:
        return psi[i][self._live_cols[i]].reshape(1, 1, -1)

    def _n_updates_min(self) -> int:
        return min(int(r.n_updates.min()) for r in self.rls)

    def _after_disturbance(self) -> None:
        if not self.pact_enabled:
            self._corr = torch.zeros(self.n_ag)
            return
        psi = self._full_psi()

        # II.2: the target is the agent's own residual, ONE STEP STALE, and it
        # never sees another agent's residual (P-4.1).
        y = self._y_prev.clamp(-self.pact_params.y_clip, self.pact_params.y_clip)

        #  PAIR THE TARGET WITH THE ROW THAT PRODUCED IT (trap #6): y(t-1) was
        #  generated by psi(t-1).  The prediction is scored BEFORE the update,
        #  so fit_gain is an honest one-step-ahead number.
        if self._have_prev:
            ahead = torch.stack([self.rls[i].predict(self._rows(self._psi_prev, i))[0, 0] for i in range(self.n_ag)])
            self._update_panel(ahead, y, self._psi_prev)
            for i in range(self.n_ag):
                self.rls[i].update(self._rows(self._psi_prev, i), y[i].reshape(1, 1))
        self._psi_prev = psi.clone()
        self._have_prev = True

        pred = torch.zeros(self.n_ag)
        conf = torch.zeros(self.n_ag)
        for i in range(self.n_ag):
            row = self._rows(psi, i)
            pred[i] = self.rls[i].predict(row)[0, 0]
            conf[i] = confidence(row, self.rls[i].P, self.pact_params, int(row.shape[-1]))[0, 0]
        self._pred, self._conf = pred, conf

        #  P-7.1, enforced: a non-finite estimate is treated as no information.
        bad = ~(torch.isfinite(self._pred) & torch.isfinite(self._conf))
        if bool(bad.any()):
            self._pred = torch.where(bad, torch.zeros_like(self._pred), self._pred)
            self._conf = torch.where(bad, torch.zeros_like(self._conf), self._conf)
            self._n_diverged += int(bad.sum())

        if self._oracle:
            # the TRUE disturbance, after the curve's own saturation, in the
            # sensor's units -- what a perfect estimator would output
            self._pred = (self._d / self._s_rated).clone()
            self._conf = torch.ones_like(self._conf)

        # P-5.1's inverted prior lives in pact_trust: near full reliance, not
        # half.  P-5.2 gates it on PREDICTION uncertainty, never on tr(P).
        ready = self._oracle or self._n_updates_min() >= self._warmup
        self._trust = (self._trust_const if ready else 0.0) * self._conf

        corr = self._trust * self._pred * self._s_rated  # MVAr
        corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        if self._corr_clip > 0.0:
            lim = self._corr_clip * self.structure.action_scale * self._s_rated
            corr = corr.clamp(-lim, lim)
        self._corr = corr

    def _correction(self) -> Tensor:
        if not self.pact_enabled:
            return torch.zeros(self.n_ag)
        return self._corr

    # ------------------------------------------------------------------
    #  II.10 -- the instrument panel
    # ------------------------------------------------------------------

    def _target_beta(self) -> Tensor:
        """beta* in the estimator's own coordinates, ``(full_dim,)``.

        y = beta*(t) . x = beta*(t) . (ref + scale z) = (beta* . ref) + (beta* scale) . z
        With the driver in the basis, beta*(t) = A beta_1 and the coefficients
        sit on the A and A z columns instead, with a zero intercept.
        """
        if self._known_driver:
            b1 = beta_star(torch.tensor([1.0]), self.ns)[0]
            return torch.cat([torch.zeros(1), (b1 * self._ref).sum().reshape(1), b1 * self._scale])
        bt = beta_star(torch.tensor([self._A]), self.ns)[0]
        return torch.cat([(bt * self._ref).sum().reshape(1), bt * self._scale])

    def _update_panel(self, ahead: Tensor, y: Tensor, psi_prev: Tensor, decay: float = 0.99) -> None:
        """Score the one-step-ahead prediction against an INTERCEPT-ONLY null,
        and beta against the TRUTH on each agent's live columns.  Scored only
        where the driver is LIVE: beta* is exactly zero for half of every day
        and there is nothing to recover there."""
        d = decay
        self._ybar = d * self._ybar + (1 - d) * y
        self._fit_sse = d * self._fit_sse + (1 - d) * (y - ahead) ** 2
        self._null_sse = d * self._null_sse + (1 - d) * (y - self._ybar) ** 2
        self._fit_gain = 1.0 - self._fit_sse / self._null_sse.clamp_min(1e-12)
        # gate 7: the design matrix's conditioning, per agent, as a decaying Gram
        for i in range(self.n_ag):
            row = self._rows(psi_prev, i)[0, 0]
            self._gram[i] = d * self._gram[i] + (1 - d) * torch.outer(row, row)
            if row.numel() > 1:
                try:
                    self._cond[i] = float(torch.linalg.cond(self._gram[i]))
                except RuntimeError:
                    self._cond[i] = float("nan")
            else:
                self._cond[i] = 1.0

        if self._direct:
            return  # no beta* to compare against in the (B) control
        if self._A < 0.1 and not self._known_driver:
            return  # the driver is (nearly) dark: beta* -> 0 and there is nothing to recover
        tgt = self._target_beta()
        if float(tgt.norm()) <= 1e-9:
            return
        for i in range(self.n_ag):
            t = tgt[self._live_cols[i]]
            b = self.rls[i].beta[0, 0]
            tn = float(t.norm())
            if tn <= 1e-12:
                continue
            bn = float(b.norm())
            self._beta_cos[i] = float((b * t).sum() / (bn * tn)) if bn > 1e-12 else 0.0
            self._beta_err[i] = float((b - t).norm() / tn)

    def _layer_info(self) -> Dict[str, float]:
        info = super()._layer_info()
        if not self.pact_enabled:
            return info
        y_true = self._d / self._s_rated
        info.update({
            "pact_trust_pol": float(self._trust_const if (self._oracle or self._n_updates_min() >= self._warmup) else 0.0),
            "pact_trust_app": float(self._trust.mean()),
            "pact_conf": float(self._conf.mean()),
            "pact_pred_err": float((y_true - self._pred).abs().mean()),
            "pact_corr_abs": float((self._corr.abs() / self._s_rated).mean()),
            "pact_fit_gain": float(self._fit_gain.mean()),
            "pact_cond_psi": float(torch.nan_to_num(self._cond, nan=0.0, posinf=1e12).clamp(max=1e12).median()),
            "pact_beta_cos": float(self._beta_cos.mean()),
            "pact_beta_relerr": float(self._beta_err.mean()),
            "pact_updates": float(self._n_updates_min()),
            "pact_skipped": float(max(int(r.n_skipped.max()) for r in self.rls)),
            "pact_bounded": float(max(int(r.n_bounded.max()) for r in self.rls)),
            "pact_diverged": float(self._n_diverged),
        })
        return info

    def _panel_extra(self) -> Dict[str, float]:
        if not self.pact_enabled:
            return {}
        acc = self._ep_acc
        m = lambda k: float(np.mean(acc[k])) if k in acc and acc[k] else float("nan")  # noqa: E731
        return {
            # does the reduction hold, and is beta being recovered?
            "fit_gain": m("pact_fit_gain"),
            "cond_psi": m("pact_cond_psi"),
            "beta_cos": m("pact_beta_cos"),
            "beta_relerr": m("pact_beta_relerr"),
            "pred_err": m("pact_pred_err"),
            # is the estimator healthy, is trust armed
            "updates": m("pact_updates"),
            "skipped": m("pact_skipped"),
            "bounded": m("pact_bounded"),
            "diverged": m("pact_diverged"),
            "conf": m("pact_conf"),
            "trust_pol": m("pact_trust_pol"),
            "trust_app": m("pact_trust_app"),
            # how much was cancelled: episode sums, not a mean of per-step ratios
            # (the ratio explodes wherever the driver is dark and |d| -> 0)
            # over the LIVE steps only (A > 0): what was cancelled, and how much
            # of the one-step disturbance the estimate explained
            "corr_frac": _ratio_where(acc.get("pact_corr_abs"), acc.get("ns_load"), acc.get("ns_A"), lambda a: a > 0.0),
            "explained": 1.0 - _ratio_where(acc.get("pact_pred_err"), acc.get("ns_load"), acc.get("ns_A"), lambda a: a > 0.0),
            # the phantom: what the compensator applied while the driver was
            # DARK and there was nothing to cancel (pu of nameplate, mean/step)
            "corr_dark": _mean_where(acc.get("pact_corr_abs"), acc.get("ns_A"), lambda a: a == 0.0),
        }
