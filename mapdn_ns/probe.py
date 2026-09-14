#!/usr/bin/env python
#  Build-order step 7: the maximum-excitation probe.
#
#      python mapdn_ns/probe.py
#      python mapdn_ns/probe.py --sigma 2 --episodes 4 --mu 0.95
#
#  Uniformly random actions are the best case for identification -- maximum
#  excitation -- so a fit gain near zero HERE is decisive, and it costs a
#  fiftieth of a training run.  Prints the II.10 panel per episode and a
#  summary; every column is the layer's own, read off `info`.
#
#  Needs pandapower and the data.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mapdn_ns.hosts import build_env  # noqa: E402

COLS = ("ns_A", "ns_load", "ns_sat", "ns_x_std", "pact_fit_gain", "pact_cond_psi", "pact_beta_cos",
        "pact_beta_relerr", "pact_pred_err", "pact_conf", "pact_trust_app", "pact_corr_abs",
        "pact_updates", "pact_skipped", "pact_bounded", "pact_diverged", "pact_resets", "ns_resid_after")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default="case33_3min_final")
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mu", type=float, default=None)
    ap.add_argument("--direct", action="store_true")
    ap.add_argument("--known-driver", action="store_true")
    ap.add_argument("--hours", type=int, nargs="+", default=[6, 9, 12])
    args = ap.parse_args()

    over = {"pact_warmup": 20}
    if args.mu is not None:
        over["pact_mu"] = args.mu
    if args.known_driver:
        over["pact_known_driver"] = True
    env = build_env(scenario=args.scenario, arm="pact", sigma=args.sigma, seed=args.seed,
                    episode_limit=args.steps + 2, overrides=over, direct=args.direct)
    rng = np.random.default_rng(args.seed)
    n_days = (env.pv_data.index[-1] - env.pv_data.index[0]).days - 3
    print("=" * 100)
    print(f"maximum-excitation probe  sigma={args.sigma} {'DIRECT (B)' if args.direct else 'COUPLED (C)'} "
          f"mu={env.pact_params.mu} known_driver={args.known_driver} episodes={args.episodes} x {args.steps}")
    print("=" * 100)
    hdr = f"{'ep':>3} {'hour':>4} " + " ".join(f"{c.replace('pact_', '').replace('ns_', ''):>10}" for c in COLS)
    print(hdr)
    summary = {c: [] for c in COLS}
    for e in range(args.episodes):
        day = int(rng.integers(0, n_days))
        hour = args.hours[e % len(args.hours)]
        np.random.seed(1000 + e)
        env.manual_reset(day, hour, 0)
        acc = {c: [] for c in COLS}
        for _ in range(args.steps):
            _, done, info = env.step(rng.uniform(-0.8, 0.8, env.n_agents), add_noise=False)
            for c in COLS:
                acc[c].append(float(info.get(c, float("nan"))))
            if done:
                break
        tail = {c: float(np.mean(v[len(v) // 2:])) for c, v in acc.items()}
        for c in COLS:
            summary[c].append(tail[c])
        print(f"{e:>3} {hour:>4} " + " ".join(f"{tail[c]:>10.4f}" for c in COLS), flush=True)
    print("-" * len(hdr))
    print(f"{'avg':>8} " + " ".join(f"{np.mean(summary[c]):>10.4f}" for c in COLS))
    fg = float(np.mean(summary["pact_fit_gain"]))
    pe = float(np.mean(summary["pact_pred_err"]))
    ld = float(np.mean(summary["ns_load"]))
    print()
    print(f"fit_gain over the intercept-only null: {fg:.3f}   (gate 6 floor: 0.3)")
    print(f"prediction error {pe:.4f} vs |d| {ld:.4f} pu of nameplate -> {100 * (1 - pe / max(ld, 1e-9)):.0f}% of the disturbance explained")
    print(f"beta cos {np.mean(summary['pact_beta_cos']):.3f}, cond(psi) {np.mean(summary['pact_cond_psi']):.3g}  "
          f"(gate 7: warn if high -- beta can be USED, not decomposed)")
    if fg < 0.3:
        print("\nFIT GAIN NEAR ZERO UNDER MAXIMUM EXCITATION: a full run cannot help.  Do not queue it.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
