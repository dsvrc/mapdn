#!/usr/bin/env python
#  In-simulator integration smoke test.  RUN THIS BEFORE THE SWEEP.
#
#      python mapdn_ns/smoke.py
#      python mapdn_ns/smoke.py --steps 40
#
#  conformance.py proves the ARITHMETIC offline.  This proves the WIRING: that
#  the dial reaches the physics, that the identities the spec gates survive
#  contact with the real host, and that the arms differ only where they are
#  supposed to.  Every check corresponds to a requirement whose violation would
#  silently produce plausible numbers.
#
#  Needs pandapower and the data.  Does NOT need a trained model.

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mapdn_ns.driver import driver_A  # noqa: E402
from mapdn_ns.hosts import build_env  # noqa: E402

CHECKS: List[Tuple[str, Callable[[], str]]] = []
RESULTS: List[Tuple[str, bool, str]] = []
ARGS = None
DAY, NOON, NIGHT = 400, 10, 20  # a fixed dataset day; a daytime and a night start hour


def check(name: str):
    def wrap(fn):
        CHECKS.append((name, fn))
        return fn
    return wrap


def build(arm: str, sigma: float, seed: int = 0, steps: int = None, **kw):
    over = kw.pop("overrides", {})
    over.setdefault("pact_warmup", 10)
    steps = ARGS.steps if steps is None else steps
    return build_env(arm=arm, sigma=sigma, seed=seed, episode_limit=steps + 2, overrides=over, **kw)


def inert_peers_structure(env) -> str:
    """The same feeder with every inverter but the first given a ZERO nameplate
    curve -- legacy units with volt-var disabled -- so the first one is alone
    on the medium.  Written to a temp file and passed through ns_structure."""
    import json
    import tempfile

    from mapdn_ns.structure import structure_path

    d = json.loads(Path(structure_path(env._scenario_name())).read_text(encoding="utf-8"))
    d["s_rated_mva"] = [d["s_rated_mva"][0]] + [0.0] * (len(d["s_rated_mva"]) - 1)
    d["name"] += "/inert-peers"
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(d, f)
    f.close()
    return f.name


def drive(env, steps: int, hour: int = NOON, seed: int = 99, actions=None):
    """Identical action stream and identical data (no noise) for every arm."""
    rng = np.random.default_rng(seed)
    # the host draws its reset dispatch from the GLOBAL numpy stream, which every
    # construction reseeds; pin it so two arms start from the same reset VARs
    np.random.seed(seed)
    obs, _ = env.manual_reset(DAY, hour, 0)
    obs_l, rew_l, info_l = [np.stack(obs)], [], []
    for t in range(steps):
        a = actions(t, env) if actions is not None else rng.uniform(-0.8, 0.8, env.n_agents)
        r, done, info = env.step(a, add_noise=False)
        obs = env.get_obs()
        obs_l.append(np.stack(obs))
        rew_l.append(r)
        info_l.append(info)
    return np.stack(obs_l), np.asarray(rew_l), info_l


# ===========================================================================


@check("test_the_layer_reaches_the_physics")
def _fires():
    """NS-3.3.  A silently inert disturbance is the one failure mode
    indistinguishable from a clean null result."""
    env = build("blind", 2.0)
    _, r, info = drive(env, ARGS.steps)
    env.assert_layer_fired(min_seen=env.n_agents * ARGS.steps)
    assert np.isfinite(r).all(), "non-finite reward"
    load = np.mean([i["ns_load_frac"] for i in info])
    return f"{env.severity_report()}; mean |d| {load:.4f} of the VAR range at sigma=2, daytime"


@check("test_records_carry_the_disturbance")
def _records():
    """NS-3.2: harm the RECORDS as well as the rewards."""
    env = build("pact", 2.0)
    obs, r, info = drive(env, ARGS.steps)
    for k in ("ns_load", "ns_A", "ns_hour", "ns_y", "ns_sat", "ns_clipped", "pact_fit_gain", "pact_trust_app",
              "pact_pred_err", "pact_beta_cos"):
        assert k in info[-1], f"info is missing {k}"
        assert np.isfinite(info[-1][k]), f"{k} is not finite"
    assert np.isfinite(obs).all(), "observation is not finite"
    stock = build("blind", 0.0, stock=True)
    extra = obs.shape[-1] - stock.get_obs_size()
    assert extra == 1 + env.ns.n_classes, f"expected {1 + env.ns.n_classes} extra obs dims, got {extra}"
    return f"info carries the II.10 panel; obs = stock {stock.get_obs_size()} + residual + {env.ns.n_classes} channels"


@check("test_driver_clock_is_the_dataset_clock")
def _clock():
    """NS-1.3 / NS-3.4: A(t) is a function of the host's own hour of day."""
    env = build("blind", 1.0)
    _, _, info = drive(env, ARGS.steps, hour=NOON)
    hours = np.array([i["ns_hour"] for i in info])
    a = np.array([i["ns_A"] for i in info])
    expect = driver_A(torch.tensor(hours, dtype=torch.float32), env.ns).numpy()
    assert abs(hours[0] - NOON) < 1e-6, hours[0]
    assert np.allclose(np.diff(hours), 0.05), "the clock does not advance 3 minutes per step"
    assert np.allclose(a, expect, atol=1e-6), "ns_A is not driver_A(hour)"
    return f"hour {hours[0]:.2f} -> {hours[-1]:.2f}, A {a[0]:.3f} -> {a[-1]:.3f} == driver_A(hour)"


@check("test_sigma_zero_is_bit_identical_to_the_stock_host")
def _sigma0():
    """NS-2.1.  At sigma=0 the wrapped host -- blind AND pact -- must produce
    the stock host's rewards and the stock part of its observation bit for
    bit, on the same data and the same action stream."""
    stock = build("blind", 0.0, stock=True)
    o0, r0, _ = drive(stock, ARGS.steps)
    for arm in ("blind", "pact"):
        env = build(arm, 0.0)
        o1, r1, _ = drive(env, ARGS.steps)
        assert np.array_equal(r0, r1), f"{arm}: rewards differ by {np.abs(r0 - r1).max():.3e}"
        assert np.array_equal(o0, o1[..., : o0.shape[-1]]), f"{arm}: the stock part of the observation differs"
        assert float(env._n_hit) == 0
    return f"{ARGS.steps} steps: blind and pact at sigma=0 reproduce the stock host bit for bit"


@check("test_placebo_is_bit_identical_across_arms")
def _placebo():
    """NS-2.5.  A night episode at sigma=6: the dial is provably inert and
    every arm agrees with the stock host."""
    stock = build("blind", 0.0, stock=True)
    o0, r0, _ = drive(stock, ARGS.steps, hour=NIGHT)
    for arm in ("blind", "pact", "oracle"):
        env = build(arm, 6.0)
        o1, r1, info = drive(env, ARGS.steps, hour=NIGHT)
        assert max(i["ns_load"] for i in info) == 0.0, f"{arm}: the placebo half was not exactly inert"
        assert np.array_equal(r0, r1), f"{arm}: rewards differ at night"
        assert np.array_equal(o0, o1[..., : o0.shape[-1]])
    return f"sigma=6 at {NIGHT}:00, {ARGS.steps} steps: load exactly 0.0, all arms == stock"


@check("test_floor_property_in_the_simulator")
def _floor():
    """P-7.1.  trust forced to 0 must be bit-identical to the blind arm --
    same wrapper, same observation, same seed, any estimate."""
    a_env = build("blind", 2.0)
    b_env = build("pactoff", 2.0)
    oa, ra, _ = drive(a_env, ARGS.steps)
    ob, rb, _ = drive(b_env, ARGS.steps)
    assert np.array_equal(ra, rb), f"rewards differ by {np.abs(ra - rb).max():.3e}"
    assert np.array_equal(oa, ob), "observations differ"
    return f"the pactoff arm is provably the blind arm over {ARGS.steps} steps at sigma=2"


@check("test_lone_actor_is_untouched_in_the_simulator")
def _lone():
    """I.2, on the real host: with every peer silent (zero VAR, from a zero
    reset), the one inverter that acts reads EXACTLY zero disturbance at a
    severity far past anything the paper quotes -- and its peers feel it."""
    rng = np.random.default_rng(5)

    def acts(t, e):
        a = np.zeros(e.n_agents)
        a[0] = rng.uniform(-0.8, 0.8)
        return a

    # with live peers, silent peers still REACT (their firmware answers the
    # actor's VARs) and the reaction reaches the actor: that is the coupling
    env = build("blind", 20.0, env_overrides={"reset_action": False})
    drive(env, ARGS.steps, actions=acts)
    d_peer = float(env._d[1:].abs().max())
    d0_live = float(env._d[0].abs())
    assert d_peer > 0.0, "the peers did not feel the lone actor's VARs"
    # with the peers' curves inert the actor is alone on the medium: exactly 0
    env = build("blind", 20.0, env_overrides={"reset_action": False},
                overrides={"ns_structure": inert_peers_structure(env)})
    worst = 0.0
    obs, _ = env.manual_reset(DAY, NOON, 0)
    for t in range(ARGS.steps):
        env.step(acts(t, env), add_noise=False)
        # the first interval carries the HOST's reset dispatch (the net file's
        # 0.05 MVAr on every sgen) into the channels; from the second on, the
        # peers have executed their own zero and are silent
        if t >= 1:
            worst = max(worst, float(env._d[0].abs()))
    assert worst == 0.0, f"the lone actor read a disturbance of {worst:.3g} at sigma=20"
    return (f"sigma=20: with live peers the actor reads {d0_live:.3f} MVAr of their reaction (peers up to "
            f"{d_peer:.3f}); with the peers' curves inert it reads exactly 0.0 over {ARGS.steps} steps")


@check("test_direct_control_is_felt_by_a_lone_actor")
def _control():
    """The (B) control must be a DIFFERENT cell, and measurably so: with
    every peer's curve inert and every peer silent, the one inverter still
    feels the firmware's reaction to the solar day."""
    env = build("blind", 2.0, direct=True, env_overrides={"reset_action": False})
    env = build("blind", 2.0, direct=True, env_overrides={"reset_action": False},
                overrides={"ns_structure": inert_peers_structure(env)})
    rng = np.random.default_rng(5)

    def acts(t, e):
        a = np.zeros(e.n_agents)
        a[0] = rng.uniform(-0.8, 0.8)
        return a

    drive(env, ARGS.steps, actions=acts)
    d0 = float(env._d[0].abs())
    assert d0 > 0.0, "ns_direct=True still read zero for the lone actor; it is not cell (B)"
    assert float(env._d[0]) < 0.0, "the (B) reaction must absorb VARs (react to the PV-driven rise)"
    return f"ns_direct at sigma=2, peers inert and silent: the lone actor reads {d0:.3f} MVAr -- cell (B), as intended"


@check("test_the_channel_is_invertible")
def _invertible():
    """II.6 row 1: handed the TRUE disturbance, the correction cancels it, so
    the executed VAR equals the commanded one wherever the inverter has not
    saturated."""
    env = build("oracle", 2.0, overrides={"pact_warmup": 0, "pact_trust": 1.0})
    worst = 0.0
    frac = []
    rng = np.random.default_rng(3)
    obs, _ = env.manual_reset(DAY, NOON, 0)
    for _ in range(ARGS.steps):
        env.step(rng.uniform(-0.8, 0.8, env.n_agents), add_noise=False)
        unsat = ~env._clipped
        err = ((env._q_exec - env._q_cmd).abs() / env._s_rated)[unsat]
        worst = max(worst, float(err.max()) if err.numel() else 0.0)
        frac.append(float(env._clipped.float().mean()))
    assert worst < 1e-5, f"the oracle did not cancel the disturbance: {worst:.3e} pu"
    return f"with the true disturbance the executed VAR equals the command to {worst:.1e} pu ({100 * np.mean(frac):.1f}% steps clipped)"


@check("test_estimator_is_live_and_learns")
def _estimator():
    """II.9 gate 5 and the point of the exercise: the prediction must beat
    predicting zero, and the peer channels must beat an intercept-only null."""
    env = build("pact", 2.0, steps=max(ARGS.steps, 120))
    _, _, info = drive(env, max(ARGS.steps, 120))
    upd = env._n_updates_min()
    assert upd > 0, "the estimator was never updated"
    assert max(int(r.n_skipped.max()) for r in env.rls) == 0, "rows were skipped -- dead regressor"
    tail = info[len(info) // 2:]
    err = np.mean([i["pact_pred_err"] for i in tail])
    null = np.mean([i["ns_load"] for i in tail])
    fit = np.mean([i["pact_fit_gain"] for i in tail])
    assert null > 0, "no disturbance to predict"
    assert err < 0.5 * null, f"the estimator did not beat predicting zero: |d|={null:.4f} vs error={err:.4f}"
    assert fit > 0.3, f"fit gain over the intercept-only null is only {fit:.3f}"
    trust = np.mean([i["pact_trust_app"] for i in tail])
    return (f"|d| {null:.4f} vs prediction error {err:.4f} pu ({100 * (1 - err / null):.0f}% explained); "
            f"fit_gain {fit:.3f}, {upd} updates, applied trust {trust:.3f}")


@check("test_severity_moves_the_domain_metric")
def _bites():
    """A severity that does not move the metric is a dead experiment.  This
    runs the droop controller for one daytime episode; use calibrate.py for
    the ladder."""
    from mapdn_ns.controllers import DroopController

    out = {}
    for sg in (0.0, 3.0):
        env = build("blind", sg)
        ctl = DroopController()
        obs, _ = env.manual_reset(DAY, NOON, 0)
        ctl.reset()
        rs, crs, loads = [], [], []
        for _ in range(ARGS.steps):
            r, _, info = env.step(ctl(env, obs), add_noise=False)
            obs = env.get_obs()
            rs.append(r)
            crs.append(info["totally_controllable_ratio"])
            loads.append(info["ns_load_frac"])
        out[sg] = (float(np.mean(rs)), float(np.mean(crs)), float(np.mean(loads)))
    assert out[3.0][2] > out[0.0][2], "the load did not rise with sigma"
    return "  ".join(f"sigma={k:g}: reward {v[0]:+.4f} CR {v[1]:.2f} |d| {v[2]:.3f}" for k, v in out.items())


def main() -> int:
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=60)
    ARGS = ap.parse_args()
    width = 84
    print("=" * width)
    print(f"mapdn_ns integration smoke   steps={ARGS.steps}  day={DAY} noon={NOON} night={NIGHT}")
    print("=" * width, flush=True)
    for name, fn in CHECKS:
        try:
            RESULTS.append((name, True, fn() or ""))
        except AssertionError as exc:
            RESULTS.append((name, False, str(exc)))
        except Exception as exc:  # noqa: BLE001
            RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
        ok, detail = RESULTS[-1][1], RESULTS[-1][2]
        print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       {detail}", flush=True)
    failed = sum(0 if ok else 1 for _, ok, _ in RESULTS)
    print("-" * width)
    print(f"{len(RESULTS) - failed}/{len(RESULTS)} checks passed")
    if failed:
        print("\nDo NOT start the sweep.  Each of these gates a requirement whose "
              "violation produces plausible numbers.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
