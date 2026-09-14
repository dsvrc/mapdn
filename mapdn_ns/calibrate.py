#!/usr/bin/env python
#  Pick the operating point BEFORE spending training compute.
#
#      python mapdn_ns/calibrate.py                          # droop controller, sigma ladder
#      python mapdn_ns/calibrate.py --mu-sweep              # mu on PREDICTION error
#      python mapdn_ns/calibrate.py --controller policy --save-path runs \
#             --alg matd3 --log-name var_voltage_control-case33_3min_final-distributed-matd3-l1-b0-blind-s0-seed0
#
#  "At severity sigma, could ANYTHING recover?  Not 'can our algorithm learn
#  it' -- can any controller, even one handed the answer for free, get back to
#  normal?  If the answer is no, a failed training run tells you nothing."
#
#  So three arms are run down the same severity ladder on the same episodes
#  (identical days and start hours, no data noise), with a controller that was
#  actually trying:
#
#      blind      the controller, disturbed, no compensation      -> how far it falls
#      oracle     the controller, handed the TRUE disturbance     -> the ceiling
#      pact       the controller, with PACT estimating online     -> what is earned
#
#  What to read off it:
#    * blind vs B0 -- how much headroom there is to recover.  If it does not
#      move there is nothing to show, whatever the method does.
#    * oracle vs B0 -- the ceiling.  If the oracle cannot recover, sigma is
#      past sigma* and the row is about the environment, not the method.
#    * pact between the two -- what an online estimator earns of that ceiling.
#
#  Rewards in MAPDN are negative (a loss), so the table reports the LOSS as a
#  multiple of B0's (1.00 = B0, larger = worse) and the paper's own metric, the
#  controllable ratio CR (fraction of steps with every bus inside 0.95-1.05).
#
#  Needs pandapower and the data.  Does NOT need a trained model unless
#  --controller policy.

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mapdn_ns.controllers import controller_for  # noqa: E402
from mapdn_ns.hosts import build_env  # noqa: E402


def episodes_for(n: int, seed: int, env) -> List[tuple]:
    """Identical episodes for every arm: (day, hour, interval) drawn once.
    Start hours span the day so the driver's whole cycle is covered, and the
    placebo half is in every ladder."""
    rng = np.random.default_rng(seed)
    n_days = (env.pv_data.index[-1] - env.pv_data.index[0]).days - 3
    hours = [6, 9, 12, 15, 18, 0]
    return [(int(rng.integers(0, n_days)), hours[k % len(hours)], int(rng.integers(0, 20))) for k in range(n)]


def roll(env, ctl, episodes, steps: int, noise: bool = False) -> Dict[str, float]:
    acc: Dict[str, List[float]] = {}
    for day, hour, interval in episodes:
        # the host's reset dispatch comes from the global numpy stream; pin it
        # per episode so every arm starts each episode from the same VARs
        np.random.seed(1000003 * day + 17 * hour + interval)
        obs, _ = env.manual_reset(day, hour, interval)
        ctl.reset()
        for _ in range(steps):
            a = ctl(env, obs)
            r, done, info = env.step(a, add_noise=noise)
            acc.setdefault("reward", []).append(float(r))
            for k, v in info.items():
                acc.setdefault(k, []).append(float(v))
            obs = env.get_obs()
            if done:
                break
    m = lambda k: float(np.mean(acc[k])) if k in acc else float("nan")  # noqa: E731
    return dict(
        reward=m("reward"), cr=m("totally_controllable_ratio"), v_out=m("percentage_of_v_out_of_control"),
        q_loss=m("q_loss"), load=m("ns_load_frac"), sat=m("ns_sat"), clipped=m("ns_clipped"),
        pred_err=m("pact_pred_err"), corr_frac=(float(np.sum(acc["pact_corr_abs"])) / max(float(np.sum(acc["ns_load"])), 1e-12)) if "pact_corr_abs" in acc else float("nan"), fit_gain=m("pact_fit_gain"),
        beta_cos=m("pact_beta_cos"), trust=m("pact_trust_app"), resid_after=m("ns_resid_after"),
        A=m("ns_A"), load_pu=m("ns_load"), destroy=m("destroy"), steps=len(acc.get("reward", [])),
        explained=(1.0 - float(np.sum(acc["pact_pred_err"])) / max(float(np.sum(acc["ns_load"])), 1e-12)) if "pact_pred_err" in acc else float("nan"),
    )


def make(args, arm: str, sigma: float, extra: Optional[dict] = None, direct: bool = False):
    over = {"pact_warmup": args.warmup}
    if args.mu is not None:
        over["pact_mu"] = float(args.mu)
    if args.known_driver:
        over["pact_known_driver"] = True
        over["ns_observe_driver"] = True
    if extra:
        over.update(extra)
    return build_env(scenario=args.scenario, arm=arm, sigma=sigma, seed=args.seed, barrier=args.barrier,
                     episode_limit=args.steps + 2, overrides=over, direct=direct)


def controller(args, env):
    if args.controller == "droop":
        return controller_for("droop", env, k=args.droop_k, alpha=args.droop_alpha)
    return controller_for("policy", env, save_path=args.save_path, alg=args.alg, log_name=args.log_name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="case33_3min_final")
    ap.add_argument("--barrier", default="l1")
    ap.add_argument("--severities", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 3.0, 6.0])
    ap.add_argument("--arms", nargs="+", default=["blind", "oracle", "pact"],
                    choices=["blind", "oracle", "pact", "pactoff", "intercept"])
    ap.add_argument("--direct", action="store_true", help="run the ladder in the (B) control cell")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--mu", type=float, default=None, help="override pact_mu for the ladder")
    ap.add_argument("--controller", choices=["droop", "policy"], default="droop")
    ap.add_argument("--droop-k", type=float, default=7.333)
    ap.add_argument("--droop-alpha", type=float, default=0.5)
    ap.add_argument("--save-path", default="runs")
    ap.add_argument("--alg", default="matd3")
    ap.add_argument("--log-name", default=None, help="model_save/<log_name>/model.pt for --controller policy")
    ap.add_argument("--known-driver", action="store_true",
                    help="ablation: pact_known_driver=true (psi = [1, A, A z]) and ns_observe_driver=true")
    ap.add_argument("--mu-sweep", action="store_true", help="sweep pact_mu at one sigma on prediction error")
    ap.add_argument("--mus", type=float, nargs="+", default=[0.999, 0.99, 0.98, 0.95, 0.9])
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    t0 = time.time()
    probe = make(args, "blind", 0.0)
    eps = episodes_for(args.episodes, args.seed, probe)
    print("=" * 100)
    print(f"mapdn_ns -- operating point on {args.scenario}, controller={args.controller}, "
          f"{'DIRECT (B)' if args.direct else 'COUPLED (C)'}")
    print(f"episodes={args.episodes} x {args.steps} steps, starts (day,hour,interval)={eps}")
    print("=" * 100, flush=True)

    rows = []
    if args.mu_sweep:
        sigma = args.severities[-1] if len(args.severities) == 1 else 1.0
        print(f"\n-- mu sweep at sigma={sigma}: select on PREDICTION error, never on return --")
        print(f"  {'mu':>6} {'pred_err':>9} {'|d|':>7} {'expl%':>6} {'fit_gain':>9} {'beta_cos':>9} {'trust':>6} {'loss xB0':>9} {'CR':>6}")
        base = roll(probe, controller(args, probe), eps, args.steps)  # blind, sigma = 0
        for mu in args.mus:
            env = make(args, "pact", sigma, extra={"pact_mu": mu}, direct=args.direct)
            r = roll(env, controller(args, env), eps, args.steps)
            expl = 100.0 * r["explained"]
            rows.append(dict(kind="mu", mu=mu, sigma=sigma, **r))
            print(f"  {mu:>6g} {r['pred_err']:>9.4f} {r['load_pu']:>7.4f} {expl:>6.1f} {r['fit_gain']:>9.3f} "
                  f"{r['beta_cos']:>9.3f} {r['trust']:>6.3f} {r['reward'] / base['reward']:>9.3f} {r['cr']:>6.3f}", flush=True)
        print("  (|d| and pred_err in pu of nameplate; expl% = 1 - sum|err| / sum|d| over the ladder's episodes)")
    else:
        head = f"  {'sigma':>6} {'arm':>9} {'loss xB0':>9} {'CR':>6} {'v_out%':>7} {'destroy':>7} {'|d|frac':>8} {'sat':>5} {'clip':>5} {'A':>5} {'pred_err':>9} {'corr%':>6} {'fit':>6} {'b_cos':>6} {'trust':>6}"
        print(head)
        #  B0 is ALWAYS the blind arm at sigma = 0, whatever arms and
        #  severities were asked for -- otherwise a ladder run for one
        #  ablation arm normalises against itself and reads 1.000 at its own
        #  first row (which is what happened to the first intercept ladder).
        r0 = roll(probe, controller(args, probe), eps, args.steps)
        base = r0["reward"]
        rows.append(dict(kind="ladder", sigma=0.0, arm="blind", b0=base, **r0))
        print(f"  {0.0:>6g} {'blind':>9} {1.0:>9.3f} {r0['cr']:>6.3f} {100 * r0['v_out']:>7.2f} {r0['destroy']:>7.3f} {r0['load']:>8.4f} "
              f"{r0['sat']:>5.2f} {r0['clipped']:>5.2f} {r0['A']:>5.2f} {'':>9} {'':>6} {'':>6} {'':>6} {'':>6}", flush=True)
        for sg in args.severities:
            for arm in args.arms:
                if sg == 0.0:
                    continue
                env = make(args, arm, sg, direct=args.direct)
                r = roll(env, controller(args, env), eps, args.steps)
                rows.append(dict(kind="ladder", sigma=sg, arm=arm, b0=base, **r))
                ratio = r["reward"] / base if abs(base) > 1e-12 else float("nan")
                print(f"  {sg:>6g} {arm:>9} {ratio:>9.3f} {r['cr']:>6.3f} {100 * r['v_out']:>7.2f} {r['destroy']:>7.3f} {r['load']:>8.4f} "
                      f"{r['sat']:>5.2f} {r['clipped']:>5.2f} {r['A']:>5.2f} {r['pred_err']:>9.4f} "
                      f"{100 * r['corr_frac']:>6.1f} {r['fit_gain']:>6.3f} {r['beta_cos']:>6.3f} {r['trust']:>6.3f}", flush=True)
        print("\nRead it like this:")
        print("  blind loss xB0 near 1.00 -> nothing to recover at this sigma; raise it.")
        print("  oracle far above 1.00   -> past sigma*; the ROW is about the environment, not the method.")
        print("  pact between the two    -> what an online estimator earned of the ceiling.  That fraction is the result.")
        print("  CR is the paper's own metric (1 = every bus inside 0.95-1.05 at every step).")
        print("  destroy > 0 -> the power flow diverged on that fraction of steps (a -200 penalty and an")
        print("     early termination in MAPDN); the loss ratio is then dominated by the penalty -- read CR.")
    print(f"\n{time.time() - t0:.0f} s")
    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
