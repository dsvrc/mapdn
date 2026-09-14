#!/usr/bin/env python
#  Config-plumbing consistency.  Run this FIRST, always.
#
#      python mapdn_ns/check_plumbing.py
#
#  torch only -- no pandapower, no data -- so it runs before anything heavy is
#  imported and catches the failures that otherwise only appear minutes into a
#  launch: a yaml key the layer never pops (a knob nothing reads, which looks
#  exactly like a setting that works), a dataclass field with no yaml value
#  (silently the dataclass default, so the run is not the run you think it
#  is), a declared constant that drifted between the yaml and the code, an
#  arm that sets a key nothing reads, and a ``pact1/core.py`` that is no
#  longer the same object the sibling instances use.

from __future__ import annotations

import ast
import hashlib
import re
import sys
from dataclasses import fields
from pathlib import Path
from typing import Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mapdn_ns.driver import DialParams  # noqa: E402
from pact1.core import PactParams  # noqa: E402

LAYER = ROOT / "mapdn_ns" / "layer.py"
HOSTS = ROOT / "mapdn_ns" / "hosts.py"
CONF = ROOT / "mapdn_ns" / "conf"
CORE = ROOT / "pact1" / "core.py"
#  sha256 of BenchMARL-main/pact1/core.py at the time of the port.  If the
#  sibling checkout is on this machine it is compared byte for byte as well.
CORE_SHA256 = "e1df1d03bd1a3064"
SIBLING_CORE = Path(r"C:/Users/chinnu/iclr/BenchMARL-main/BenchMARL-main/pact1/core.py")

problems: List[str] = []


def report(label: str, items) -> None:
    items = sorted(items)
    print(f"  {label}: {items if items else 'none'}")
    if items:
        problems.append(label)


def tuple_keys(name: str, text: str) -> set:
    m = re.search(name + r"\s*=\s*\((.*?)\n\)", text, re.S)
    return set(re.findall(r'"([a-z0-9_]+)"', m.group(1))) if m else set()


def same(want, got) -> bool:
    if isinstance(want, bool) or isinstance(got, (bool, str)) or isinstance(want, str):
        return want == got
    if got is None or want is None:
        return got is want
    return abs(float(got) - float(want)) < 1e-9


layer = LAYER.read_text(encoding="utf-8")
NS_K = tuple_keys("NS_KWARGS", layer)
PACT_K = tuple_keys("PACT_KWARGS", layer)

# ---------------------------------------------------------------------------
#  0. the method is the shared object
# ---------------------------------------------------------------------------
print("== pact1/core.py ==")
digest = hashlib.sha256(CORE.read_bytes()).hexdigest()[:16]
report("pact1/core.py hash differs from the ported copy", [] if digest == CORE_SHA256 else [digest])
if SIBLING_CORE.exists():
    report("pact1/core.py differs from the sibling checkout",
           [] if CORE.read_bytes() == SIBLING_CORE.read_bytes() else [str(SIBLING_CORE)])
else:
    print("  sibling checkout not present; hash check only")

# ---------------------------------------------------------------------------
#  1. every key the layer pops is read somewhere, and every field is popped
# ---------------------------------------------------------------------------
print("\n== layer <-> dataclasses ==")
dial_fields = {f.name for f in fields(DialParams)}
report("DialParams fields with no ns_ key in the layer",
       {f for f in dial_fields if f"ns_{f}" not in NS_K})
popped_in_layer = set(re.findall(r'raw\.get\("(ns_[a-z0-9_]+)"', layer)) | set(re.findall(r'raw\.get\("(pact_[a-z0-9_]+)"', layer))
report("NS_KWARGS keys the layer never reads", NS_K - popped_in_layer)
report("PACT_KWARGS keys the layer never reads", PACT_K - popped_in_layer)
report("keys the layer reads but never pops", popped_in_layer - (NS_K | PACT_K))
pact_mapped = set(re.findall(r"PactParams\((.*?)\)", layer, re.S)[0].split(",") if "PactParams(" in layer else [])
pact_fields_used = {s.strip().split("=")[0] for s in pact_mapped if "=" in s}
report("PactParams fields the layer sets that do not exist",
       pact_fields_used - {f.name for f in fields(PactParams)})

# ---------------------------------------------------------------------------
#  2. the arms only set known keys
# ---------------------------------------------------------------------------
print("\n== arms ==")
tree = ast.parse(HOSTS.read_text(encoding="utf-8"))
arm_keys = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "dict":
        for kw in node.keywords:
            if kw.arg and kw.arg.startswith(("ns_", "pact_")):
                arm_keys.add(kw.arg)
report("arm keys the layer never pops", arm_keys - (NS_K | PACT_K))

# ---------------------------------------------------------------------------
#  3. the substitution target still exists
# ---------------------------------------------------------------------------
print("\n== substitution target ==")
for script in ("train.py", "test.py"):
    text = (ROOT / script).read_text(encoding="utf-8")
    ok = "from environments.var_voltage_control.voltage_control_env import VoltageControl" in text
    report(f"{script} no longer imports VoltageControl from the patched module", [] if ok else [script])

# ---------------------------------------------------------------------------
#  4. every task yaml
# ---------------------------------------------------------------------------
DECLARED_OVERRIDES = {"ns_severity", "pact_mu", "pact_enabled", "pact_trust", "pact_warmup",
                      "pact_corr_clip", "pact_channels", "pact_oracle", "pact_p0",
                      "pact_p_trace_max", "ns_direct", "ns_observe_residual",
                      "ns_observe_channels", "ns_structure"}
yamls = sorted(CONF.glob("*.yaml"))
report("no task yaml found", [] if yamls else [str(CONF)])
for y in yamls:
    print(f"\n== {y.name} ==")
    doc = yaml.safe_load(y.read_text(encoding="utf-8"))
    ns = dict(doc.get("ns", {}))
    keys = set(ns)
    report("yaml keys the layer never pops", keys - (NS_K | PACT_K))
    report("keys the layer pops with no yaml value", (NS_K | PACT_K) - keys)
    mismatch = []
    for f in fields(DialParams):
        k = f"ns_{f.name}"
        if k in DECLARED_OVERRIDES or k not in ns:
            continue
        want = getattr(DialParams(), f.name)
        if not same(want, ns[k]):
            mismatch.append(f"{k}: DialParams.{f.name}={want!r} yaml={ns[k]!r}")
    report("declared constants that disagree with DialParams", mismatch)
    pm = []
    for name in ("p0", "p_trace_max"):
        k = f"pact_{name}"
        if k in ns and not same(getattr(PactParams(), name), ns[k]):
            pm.append(f"{k}: PactParams.{name}={getattr(PactParams(), name)!r} yaml={ns[k]!r}")
    report("pact constants that disagree with PactParams (declared, not tuned)", pm)
    # the structure, if exported, must agree on r
    for sj in CONF.glob("structure_*.json"):
        import json
        d = json.loads(sj.read_text(encoding="utf-8"))
        if len(d["class_names"]) != int(ns.get("ns_n_classes", 3)):
            problems.append("structure classes")
            print(f"  {sj.name}: {len(d['class_names'])} classes but yaml says {ns.get('ns_n_classes')}")
        else:
            print(f"  {sj.name}: r={len(d['class_names'])} agrees; N={len(d['paths'])} inverters")

print()
print("PLUMBING OK" if not problems else f"PROBLEMS in {len(problems)} place(s)")
raise SystemExit(1 if problems else 0)
