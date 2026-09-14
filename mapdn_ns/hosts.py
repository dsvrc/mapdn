#  The host classes and the task configuration.
#
#  Nothing in the stock MAPDN tree is modified.  ``train.py`` and ``test.py``
#  import ``VoltageControl`` from ``environments.var_voltage_control.
#  voltage_control_env`` and build ``VoltageControl(env_config_dict)``; the
#  launcher (``mapdn_ns/run.py``) replaces that module attribute with one of the
#  classes below BEFORE running the script, so the baselines are bit-for-bit the
#  ones the authors shipped and the only new object is the environment they run
#  in (P-10.1).
#
#  The task configuration is ``mapdn_ns/conf/<scenario>.yaml``: the story, the
#  committed operating point, and every declared constant.  ``load_task_config``
#  merges it with command-line overrides and returns the flat ``ns_*`` /
#  ``pact_*`` dict the layer pops.

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from environments.var_voltage_control.voltage_control_env import VoltageControl
from mapdn_ns.layer import NS_KWARGS, PACT_KWARGS, NsMixin, PactMixin

__all__ = ["NsVoltageControl", "PactVoltageControl", "ARMS", "arm_kwargs",
           "load_task_config", "make_env_class", "task_config_path"]

_HERE = Path(__file__).resolve().parent


class NsVoltageControl(NsMixin, VoltageControl):
    """Stock host + the dial.  Every baseline runs on this."""


class PactVoltageControl(PactMixin, VoltageControl):
    """Stock host + the dial + PACT-1."""


#  II.11 / the porting notes: five arms through the identical wrapper.
ARMS = ("blind", "pact", "pactoff", "oracle", "intercept")


def arm_kwargs(arm: str) -> Dict[str, Any]:
    """What each arm sets, and nothing else."""
    if arm == "blind":
        return dict(pact_enabled=False)
    if arm == "pact":
        return dict(pact_enabled=True, pact_oracle=False, pact_channels="full")
    if arm == "pactoff":
        # trust forced to 0: same wrapper, same observation, same seed, and
        # provably bit-identical to blind (P-7.1) -- smoke.py checks it
        return dict(pact_enabled=True, pact_trust=0.0, pact_oracle=False, pact_channels="full")
    if arm == "oracle":
        # handed the true disturbance: the ceiling, not a competitor
        return dict(pact_enabled=True, pact_oracle=True, pact_channels="full")
    if arm == "intercept":
        # peer channels deleted, everything else identical
        return dict(pact_enabled=True, pact_oracle=False, pact_channels="intercept")
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")


def task_config_path(scenario: str) -> Path:
    short = scenario.split("_")[0]  # case33_3min_final -> case33
    return _HERE / "conf" / f"{short}.yaml"


def load_task_config(scenario: str, overrides: Optional[Dict[str, Any]] = None,
                     path: Optional[str] = None) -> Dict[str, Any]:
    """The flat ``ns_*`` / ``pact_*`` dict for ``scenario``, from the task yaml
    plus overrides.  Unknown keys are an error: a knob nothing reads looks
    exactly like a setting that works."""
    p = Path(path) if path else task_config_path(scenario)
    with open(p, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    cfg = dict(doc.get("ns", {}))
    known = set(NS_KWARGS) | set(PACT_KWARGS)
    for k, v in (overrides or {}).items():
        if v is None:
            continue
        cfg[k] = v
    unknown = sorted(set(cfg) - known)
    if unknown:
        raise KeyError(f"unknown task keys {unknown}; the layer pops only {sorted(known)}")
    return cfg


def make_env_class(ns_cfg: Dict[str, Any], pact: bool):
    """A ``VoltageControl`` replacement with the task configuration bound.

    ``train.py`` builds ``VoltageControl(env_config_dict)`` from ITS yaml; the
    bound class merges the task keys in before the layer pops them, so the
    severity is supplied from outside the method and outside the script.
    """
    base = PactVoltageControl if pact else NsVoltageControl

    class BoundVoltageControl(base):
        _bound_ns_cfg = dict(ns_cfg)

        def __init__(self, kwargs):
            d = dict(kwargs) if isinstance(kwargs, dict) else dict(kwargs._asdict())
            for k, v in self._bound_ns_cfg.items():
                d.setdefault(k, v)
            super().__init__(d)

    BoundVoltageControl.__name__ = base.__name__
    BoundVoltageControl.__qualname__ = base.__qualname__
    return BoundVoltageControl


def build_env(scenario: str = "case33_3min_final", arm: str = "blind", sigma: Optional[float] = None,
              seed: int = 0, barrier: str = "l1", episode_limit: Optional[int] = None,
              overrides: Optional[Dict[str, Any]] = None, direct: bool = False,
              env_overrides: Optional[Dict[str, Any]] = None, stock: bool = False):
    """A wrapped host, built exactly the way ``train.py`` builds it -- the same
    env yaml, the same scenario setup -- for the smoke, calibration and probe
    scripts.  ``arm`` and ``sigma`` come from outside, as in the launcher."""
    from utilities.util import setup_voltage_control_scenario

    root = Path(__file__).resolve().parents[1]
    with open(root / "args" / "env_args" / "var_voltage_control.yaml", "r", encoding="utf-8") as fh:
        env_cfg = yaml.safe_load(fh)["env_args"]
    env_cfg["data_path"] = str(root / "environments" / "var_voltage_control" / "data" / scenario).replace("\\", "/")
    setup_voltage_control_scenario(env_cfg, scenario)
    env_cfg["mode"] = "distributed"
    env_cfg["voltage_barrier_type"] = barrier
    env_cfg["seed"] = int(seed)
    if episode_limit is not None:
        env_cfg["episode_limit"] = int(episode_limit)
    env_cfg.update(env_overrides or {})
    _install_csv_cache()
    if stock:
        # the untouched host, for the bit-identity checks in smoke.py
        return VoltageControl(env_cfg)

    over = dict(overrides or {})
    if sigma is not None:
        over["ns_severity"] = float(sigma)
    if direct:
        over["ns_direct"] = True
    over.update(arm_kwargs(arm))
    ns_cfg = load_task_config(scenario, over)
    cls = make_env_class(ns_cfg, pact=bool(ns_cfg.get("pact_enabled", False)))
    env_cfg.update(ns_cfg)
    return cls(env_cfg)


_CSV_CACHE: Dict[str, Any] = {}


def _install_csv_cache() -> None:
    """The host re-reads three ~350 MB CSVs on every construction (~20 s).  The
    calibration and smoke scripts build a dozen hosts per run, so the frames
    are cached per process; each caller still gets its own copy."""
    import pandas as pd

    if getattr(pd.read_csv, "_mapdn_ns_cached", False):
        return
    orig = pd.read_csv

    def cached(path, *a, **k):
        key = str(path)
        if key.endswith((".csv",)) and "var_voltage_control" in key.replace("\\", "/") and not a and list(k) == ["index_col"]:
            if key not in _CSV_CACHE:
                _CSV_CACHE[key] = orig(path, *a, **k)
            return _CSV_CACHE[key].copy()
        return orig(path, *a, **k)

    cached._mapdn_ns_cached = True
    pd.read_csv = cached
