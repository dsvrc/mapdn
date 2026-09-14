#!/usr/bin/env python
#  The single entry point.  P-10.1: every arm launches through THIS, with the
#  severity supplied from outside the method, so the learners are bit-for-bit
#  the ones MAPDN ships and the only new object is the environment they run in.
#
#      python mapdn_ns/run.py train --alg matd3 --arm blind --sigma 0   --alias b0
#      python mapdn_ns/run.py train --alg matd3 --arm blind --sigma 1.0 --alias ns
#      python mapdn_ns/run.py train --alg matd3 --arm pact  --sigma 1.0 --alias ns
#      python mapdn_ns/run.py test  --alg matd3 --arm pact  --sigma 1.0 --alias ns --test-mode batch
#
#  It substitutes the wrapped environment class for ``VoltageControl`` in the
#  module ``train.py`` / ``test.py`` import from, then runs that script
#  unmodified with the remaining arguments.  Run from the repository root.

from __future__ import annotations

import argparse
import os
import random
import runpy
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

from mapdn_ns.hosts import ARMS, arm_kwargs, load_task_config, make_env_class  # noqa: E402


def _parse(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("script", choices=["train", "test"])
    ap.add_argument("--alg", default="matd3")
    ap.add_argument("--scenario", default="case33_3min_final")
    ap.add_argument("--mode", default="distributed")
    ap.add_argument("--voltage-barrier-type", "--barrier", dest="barrier", default="l1")
    ap.add_argument("--save-path", default="runs")
    ap.add_argument("--alias", default="ns")
    ap.add_argument("--seed", type=int, default=0, help="torch/numpy/random AND the env's seed")
    ap.add_argument("--arm", choices=ARMS, default="blind")
    ap.add_argument("--sigma", type=float, default=None, help="ns_severity; default from the task yaml")
    ap.add_argument("--direct", action="store_true", help="the (B) control: ns_direct=true")
    ap.add_argument("--ns-config", default=None, help="task yaml (default mapdn_ns/conf/<case>.yaml)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override any ns_*/pact_* key, e.g. --set pact_mu=0.97")
    ap.add_argument("--no-obs-channels", action="store_true", help="ablation: hide the public channels from every arm")
    ap.add_argument("--no-obs-residual", action="store_true", help="ablation: hide the residual from every arm")
    ap.add_argument("--episode-limit", type=int, default=None, help="override the host's episode_limit")
    ap.add_argument("--train-episodes", type=int, default=None, help="override train_episodes_num (train only)")
    ap.add_argument("--load-alias", default=None,
                    help="test only: the full alias of the checkpoint to load (e.g. paper-blind-s0-seed0) when it "
                         "differs from this evaluation's own arm/sigma/seed -- e.g. a sigma=0 policy evaluated "
                         "under the disturbance, zero-shot")
    args, passthrough = ap.parse_known_args(argv)
    return args, passthrough


def _coerce(v: str):
    lv = v.lower()
    if lv in ("true", "false"):
        return lv == "true"
    if lv in ("null", "none"):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def main(argv=None) -> int:
    args, passthrough = _parse(sys.argv[1:] if argv is None else argv)

    over = {}
    if args.sigma is not None:
        over["ns_severity"] = float(args.sigma)
    if args.direct:
        over["ns_direct"] = True
    if args.no_obs_channels:
        over["ns_observe_channels"] = False
    if args.no_obs_residual:
        over["ns_observe_residual"] = False
    for kv in args.set:
        if "=" not in kv:
            raise SystemExit(f"--set expects KEY=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        over[k.strip()] = _coerce(v.strip())
    over.update(arm_kwargs(args.arm))
    cfg = load_task_config(args.scenario, over, path=args.ns_config)
    pact = bool(cfg.get("pact_enabled", False))

    sigma = float(cfg["ns_severity"])
    alias = f"{args.alias}-{args.arm}-s{sigma:g}{'-B' if cfg.get('ns_direct') else ''}-seed{args.seed}"
    log_name = "-".join(["var_voltage_control", args.scenario, args.mode, args.alg, args.barrier, alias])
    save = Path(args.save_path)
    save.mkdir(parents=True, exist_ok=True)

    # one debug row per episode, next to the run it describes (II.10).  NOT in
    # the tensorboard folder: train.py empties that folder on start.
    ns_dir = save / "ns_logs" / log_name
    ns_dir.mkdir(parents=True, exist_ok=True)
    debug_csv = ns_dir / ("pact_debug.csv" if args.script == "train" else "pact_debug_test.csv")
    if debug_csv.exists():
        debug_csv.rename(debug_csv.with_suffix(".prev.csv"))  # never append across schema changes
    os.environ["MAPDN_NS_DEBUG_CSV"] = str(debug_csv)
    with open(ns_dir / "ns_config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(dict(ns=cfg, arm=args.arm, seed=args.seed, alg=args.alg,
                            scenario=args.scenario, barrier=args.barrier), fh, sort_keys=False)

    # seeds: the launcher owns them, so two arms on one seed see one world
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env_over = {"seed": int(args.seed)}
    if args.episode_limit is not None:
        env_over["episode_limit"] = int(args.episode_limit)

    # --- the substitution ---------------------------------------------------
    import environments.var_voltage_control.voltage_control_env as host_module

    cls = make_env_class(cfg, pact=pact)
    _orig_init = cls.__init__

    def _init(self, kwargs):
        d = dict(kwargs) if isinstance(kwargs, dict) else dict(kwargs._asdict())
        d.update(env_over)
        _orig_init(self, d)

    cls.__init__ = _init
    host_module.VoltageControl = cls

    # mark the greedy evaluation episodes (Model.evaluation, every eval_freq
    # training episodes) so the debug rows separate exploring from greedy play
    import models.model as _model_module

    _orig_eval = _model_module.Model.evaluation

    def _eval(self, stat, trainer):
        env = trainer.env
        prev = getattr(env, "phase", "train")
        env.phase = "eval"
        try:
            return _orig_eval(self, stat, trainer)
        finally:
            env.phase = prev

    _model_module.Model.evaluation = _eval
    if args.script == "test":
        _orig_init2 = cls.__init__

        def _init_test(self, kwargs):
            _orig_init2(self, kwargs)
            self.phase = "test"

        cls.__init__ = _init_test

    if args.train_episodes is not None:
        # train.py reads args/default.yaml; override through a shim on yaml.safe_load
        _orig_load = yaml.safe_load

        def _patched(stream, *a, **k):
            doc = _orig_load(stream, *a, **k)
            if isinstance(doc, dict) and "train_episodes_num" in doc:
                doc["train_episodes_num"] = int(args.train_episodes)
            return doc

        yaml.safe_load = _patched

    print("=" * 78)
    print(f"mapdn_ns launcher  script={args.script} alg={args.alg} arm={args.arm} "
          f"sigma={sigma:g} direct={bool(cfg.get('ns_direct'))} seed={args.seed} "
          f"scenario={args.scenario} barrier={args.barrier}")
    print(f"log_name           {log_name}")
    print(f"debug csv          {os.environ['MAPDN_NS_DEBUG_CSV']}")
    print("=" * 78, flush=True)

    script = "train.py" if args.script == "train" else "test.py"
    script_alias = alias
    if args.script == "test" and args.load_alias:
        script_alias = args.load_alias
        print(f"loading checkpoint   {script_alias}  (evaluated as {alias})", flush=True)
    sys.argv = [script, "--alg", args.alg, "--alias", script_alias, "--mode", args.mode,
                "--scenario", args.scenario, "--voltage-barrier-type", args.barrier,
                "--save-path", str(save)] + passthrough
    before = set(_ROOT.glob("test_record_*.pickle"))
    runpy.run_path(str(_ROOT / script), run_name="__main__")
    if args.script == "test":
        # test.py drops its record pickle in the cwd; file it with the run
        for f in set(_ROOT.glob("test_record_*.pickle")) - before:
            f.replace(ns_dir / f.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
