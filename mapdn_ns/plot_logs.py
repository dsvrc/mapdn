#!/usr/bin/env python
#  Plot training curves straight from the printed logs (stdout of run.py /
#  train.py), with no CSVs needed.
#
#      python mapdn_ns/plot_logs.py case141_matd3_blind_sig0.log case141_matd3_blind_sig1.log \
#             case141_matd3_pact_sig1.log --out case141.png
#      python mapdn_ns/plot_logs.py runs/paper/logs/*.log --keys mean_test_reward mean_test_totally_controllable_ratio
#
#  Two things are parsed from each log:
#    * every "Episode terminated at time: T with return: R." line -> the
#      per-episode return, in order (training AND evaluation episodes, as
#      the log prints them); plotted with a moving average;
#    * every "Episode: N" block -> the stats MAPDN prints every
#      save_model_freq episodes (mean_train_* / mean_test_*); plotted at
#      episode N.
#  Labels default to the file stem.

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RET = re.compile(r"Episode terminated at time:\s*(\d+)\s+with return:\s*(-?\d+(?:\.\d+)?)")
EPI = re.compile(r"^Episode:\s*(\d+)\s*$")
KV = re.compile(r"^([A-Za-z_][\w]*):\s*(-?[\d.]+(?:e[-+]?\d+)?)\s*$")


def parse(path: Path):
    returns, lengths = [], []
    blocks = {}  # episode -> {key: value}
    cur = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = RET.search(line)
        if m:
            lengths.append(int(m.group(1)))
            returns.append(float(m.group(2)))
            continue
        m = EPI.match(line.strip())
        if m:
            cur = int(m.group(1))
            blocks[cur] = {}
            continue
        if cur is not None:
            m = KV.match(line.strip())
            if m:
                blocks[cur][m.group(1)] = float(m.group(2))
            elif line.strip() == "" or line.startswith("The model"):
                cur = None
    return np.array(returns), np.array(lengths), blocks


def smooth(y, w):
    if w <= 1 or len(y) < w:
        return y
    k = np.ones(w) / w
    return np.convolve(y, k, mode="valid")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--labels", nargs="*", default=None, help="one per log (default: file stem)")
    ap.add_argument("--keys", nargs="*", default=["mean_test_reward", "mean_test_totally_controllable_ratio",
                                                  "mean_test_percentage_of_v_out_of_control", "mean_test_destroy"],
                    help="stats from the 'Episode: N' blocks to plot")
    ap.add_argument("--window", type=int, default=20, help="moving-average window for per-episode returns")
    ap.add_argument("--out", default="training_curves.png")
    args = ap.parse_args()

    labels = args.labels or [Path(p).stem for p in args.logs]
    runs = [(lab, parse(Path(p))) for lab, p in zip(labels, args.logs)]

    n = 1 + len(args.keys)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 4 * rows), squeeze=False)
    axes = axes.ravel()

    ax = axes[0]
    for lab, (ret, _, _) in runs:
        if len(ret) == 0:
            continue
        ax.plot(ret, alpha=0.25, lw=0.8)
        s = smooth(ret, args.window)
        ax.plot(np.arange(len(s)) + args.window - 1, s, lw=1.8, label=f"{lab} (n={len(ret)})")
    ax.set_title(f"per-episode return (moving avg {args.window})")
    ax.set_xlabel("episode (as printed, train + eval)")
    ax.set_ylabel("return")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    for i, key in enumerate(args.keys, start=1):
        ax = axes[i]
        any_data = False
        for lab, (_, _, blocks) in runs:
            xs = sorted(e for e in blocks if key in blocks[e])
            if not xs:
                continue
            any_data = True
            ax.plot(xs, [blocks[e][key] for e in xs], marker="o", ms=3, label=lab)
        ax.set_title(key)
        ax.set_xlabel("training episode")
        ax.grid(alpha=0.3)
        if any_data:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")
    for lab, (ret, _, blocks) in runs:
        last = blocks[max(blocks)] if blocks else {}
        print(f"{lab}: {len(ret)} episodes, {len(blocks)} stat blocks; last block: "
              + ", ".join(f"{k}={last[k]:.4f}" for k in args.keys if k in last))


if __name__ == "__main__":
    main()
