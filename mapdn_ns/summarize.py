#!/usr/bin/env python
#  Tabulate a sweep: one line per (alg, arm, sigma, cell, seed), then the
#  paired means, from the per-episode debug rows every run writes.
#
#      python mapdn_ns/summarize.py                      # runs/ns_logs
#      python mapdn_ns/summarize.py --root runs/paper --last 40
#
#  Two tables.  The first is the domain metric over the LAST `--last`
#  training episodes (reward, CR, v_out) -- read against the sigma=0 blind row
#  of the same algorithm and seed.  The second is the II.10 panel over the same
#  window, in the order the README says to read it: did the dial fire, is there
#  anything to identify, does the reduction hold, is beta recovered, is the
#  estimator healthy, is trust armed, how much was cancelled.  A run whose
#  first "no" is in the panel is not a result about the method.
#
#  Test-time (test.py --test-mode batch) rows are summarised the same way from
#  pact_debug_test.csv when present.

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PANEL = ("A", "load", "x_std", "fit_gain", "cond_psi", "beta_cos", "beta_relerr", "explained",
         "updates", "skipped", "bounded", "diverged", "resets", "trust_pol", "trust_app", "corr_frac",
         "corr_dark", "resid_after")
METRIC = ("reward", "cr", "v_out", "q_loss", "destroy")


def load_run(d: Path, which: str):
    cfg = yaml.safe_load((d / "ns_config.yaml").read_text(encoding="utf-8"))
    f = d / ("pact_debug.csv" if which == "train" else "pact_debug_test.csv")
    if not f.exists():
        return None
    with f.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    return cfg, rows


def window(rows: List[dict], last: int) -> Dict[str, float]:
    sel = rows[-last:] if last > 0 else rows
    out: Dict[str, float] = {"episodes": len(rows), "window": len(sel)}
    for k in METRIC + PANEL:
        vals = [float(r[k]) for r in sel if k in r and r[k] not in ("", "nan")]
        vals = [v for v in vals if np.isfinite(v)]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def fmt(v: float, w: int = 8, p: int = 3) -> str:
    return f"{v:>{w}.{p}f}" if np.isfinite(v) else f"{'-':>{w}}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="runs/ns_logs")
    ap.add_argument("--last", type=int, default=40, help="episodes in the window (0 = all)")
    ap.add_argument("--which", choices=["train", "test"], default="train")
    ap.add_argument("--phase", choices=["train", "eval", "all"], default="eval",
                    help="which episodes of a training run to window: the exploring training "
                         "episodes, the greedy evaluation episodes (default), or both")
    args = ap.parse_args()

    root = Path(args.root)
    runs = []
    for d in sorted(root.glob("*")):
        if not (d / "ns_config.yaml").exists():
            continue
        got = load_run(d, args.which)
        if got is None:
            continue
        cfg, rows = got
        if args.which == "train" and args.phase != "all":
            rows = [r for r in rows if r.get("phase", "train") == args.phase]
            if not rows:
                continue
        ns = cfg["ns"]
        key = dict(alg=cfg["alg"], arm=cfg["arm"], sigma=float(ns["ns_severity"]),
                   cell="B" if ns.get("ns_direct") else "C", seed=int(cfg["seed"]), name=d.name)
        runs.append((key, window(rows, args.last)))
    if not runs:
        print(f"no runs with {args.which} rows under {root}")
        return 1

    # B0 per (alg, seed): the blind arm at sigma 0
    b0 = {(k["alg"], k["seed"]): w["reward"] for k, w in runs if k["arm"] == "blind" and k["sigma"] == 0.0}

    print("=" * 118)
    print(f"{args.which}/{args.phase} window = last {args.last} episodes    loss xB0 = reward / (blind, sigma=0, same alg & seed)")
    print("=" * 118)
    print(f"{'alg':>7} {'cell':>4} {'sigma':>5} {'arm':>9} {'seed':>4} {'eps':>4} {'reward':>9} {'loss xB0':>9} {'CR':>6} {'v_out%':>7} {'q_loss':>7}")
    groups: Dict[tuple, List[float]] = defaultdict(list)
    for k, w in sorted(runs, key=lambda kw: (kw[0]["alg"], kw[0]["cell"], kw[0]["sigma"], kw[0]["arm"], kw[0]["seed"])):
        base = b0.get((k["alg"], k["seed"]), float("nan"))
        ratio = w["reward"] / base if np.isfinite(base) and abs(base) > 1e-12 else float("nan")
        groups[(k["alg"], k["cell"], k["sigma"], k["arm"])].append(ratio)
        print(f"{k['alg']:>7} {k['cell']:>4} {k['sigma']:>5g} {k['arm']:>9} {k['seed']:>4} {w['episodes']:>4} "
              f"{fmt(w['reward'], 9, 4)} {fmt(ratio, 9)} {fmt(w['cr'], 6)} {fmt(100 * w['v_out'], 7, 2)} {fmt(w['q_loss'], 7)}")

    print("\n-- paired means over seeds (loss xB0; std in brackets) --")
    for g, vals in sorted(groups.items()):
        v = np.array([x for x in vals if np.isfinite(x)])
        if v.size:
            print(f"{g[0]:>7} {g[1]:>4} {g[2]:>5g} {g[3]:>9}   {v.mean():.3f} [{v.std():.3f}]  n={v.size}")

    print("\n" + "=" * 118)
    print("II.10 panel, same window -- read left to right; the first column that answers 'no' is the one to fix")
    print("=" * 118)
    hdr = f"{'alg':>7} {'cell':>4} {'sigma':>5} {'arm':>9} {'seed':>4} " + " ".join(f"{c[:9]:>9}" for c in PANEL)
    print(hdr)
    for k, w in sorted(runs, key=lambda kw: (kw[0]["alg"], kw[0]["cell"], kw[0]["sigma"], kw[0]["arm"], kw[0]["seed"])):
        if k["arm"] == "blind":
            continue
        print(f"{k['alg']:>7} {k['cell']:>4} {k['sigma']:>5g} {k['arm']:>9} {k['seed']:>4} "
              + " ".join(fmt(w[c], 9, 3 if c not in ("cond_psi", "updates") else 0) for c in PANEL))
    return 0


if __name__ == "__main__":
    sys.exit(main())
