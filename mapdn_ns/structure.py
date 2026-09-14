#  The feeder's structure: the elements (line segments), their reactance, their
#  conductor class, and each inverter's path to the substation.
#
#  NS-1.2 / P-1.2: everything here is read off the network file BEFORE any run.
#  Nothing is fitted.  ``build_structure`` needs pandapower and the host's own
#  ``model.p``; it writes a small JSON so that every torch-only module
#  (``coupling``, ``conformance``, ``estimator_selftest``, ``check_plumbing``)
#  can load the structure without a simulator.  A synthetic radial feeder with
#  the same shape is provided for offline checks on a machine that has no data.
#
#  The medium.  A radial distribution feeder is a tree rooted at the substation.
#  An inverter's VARs flow along its path to the root, and the voltage rise they
#  cause at another inverter's terminal is, in the linearised DistFlow
#  (Baran & Wu 1989),
#
#      dV_i / dQ_j  =  sum over segments a on BOTH paths of  X_a / V_base^2
#
#  -- incidence over reactance, the distribution-grid twin of "incidence over
#  capacity" (URB) and of the PTDF (transmission).  It is the declared operator.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor

__all__ = ["FeederStructure", "load_structure", "synthetic_structure", "build_structure"]

CLASS_RULE = "conductor size: tercile of r_ohm_per_km (heavy trunk / medium / light lateral)"


@dataclass
class FeederStructure:
    name: str
    v_base_kv: float
    line_x_ohm: Tensor              # (L,) total series reactance of each element
    line_r_ohm: Tensor              # (L,) total series resistance (reported, unused by W)
    line_class: Tensor              # (L,) long in [0, r)
    class_names: List[str]
    paths: List[List[int]]          # per agent: element ids from its bus to the root
    sgen_bus: List[int]
    s_rated_mva: Tensor             # (N,) inverter nameplate apparent power
    action_scale: float             # the host's own action range, a fraction of capability
    class_rule: str = CLASS_RULE
    extra: Dict[str, object] = field(default_factory=dict)
    line_vn_kv: Optional[Tensor] = None   # (L,) the voltage level each element sits at

    # -- derived -----------------------------------------------------------
    @property
    def n_agents(self) -> int:
        return len(self.paths)

    @property
    def n_lines(self) -> int:
        return int(self.line_x_ohm.shape[0])

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def line_x_pu(self) -> Tensor:
        """Reactance in (pu voltage) per MVAr: ``X[ohm] / V_n[kV]^2`` at the
        element's OWN voltage level -- a feeder with transformers has lines at
        several levels, and a 110 kV base applied to a 20 kV lateral would
        understate its coupling 30-fold."""
        vn = self.line_vn_kv if self.line_vn_kv is not None else torch.full_like(self.line_x_ohm, float(self.v_base_kv))
        return self.line_x_ohm / (vn ** 2)

    def to_json(self, path: Path) -> None:
        d = dict(
            name=self.name,
            v_base_kv=self.v_base_kv,
            line_x_ohm=self.line_x_ohm.tolist(),
            line_r_ohm=self.line_r_ohm.tolist(),
            line_class=self.line_class.tolist(),
            line_vn_kv=(self.line_vn_kv.tolist() if self.line_vn_kv is not None else None),
            class_names=list(self.class_names),
            paths=[list(map(int, p)) for p in self.paths],
            sgen_bus=list(map(int, self.sgen_bus)),
            s_rated_mva=self.s_rated_mva.tolist(),
            action_scale=self.action_scale,
            class_rule=self.class_rule,
            extra=self.extra,
        )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(d, indent=1), encoding="utf-8")

    @staticmethod
    def from_json(path: Path) -> "FeederStructure":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return FeederStructure(
            name=d["name"],
            v_base_kv=float(d["v_base_kv"]),
            line_x_ohm=torch.tensor(d["line_x_ohm"], dtype=torch.float32),
            line_r_ohm=torch.tensor(d["line_r_ohm"], dtype=torch.float32),
            line_class=torch.tensor(d["line_class"], dtype=torch.long),
            line_vn_kv=(torch.tensor(d["line_vn_kv"], dtype=torch.float32) if d.get("line_vn_kv") is not None else None),
            class_names=list(d["class_names"]),
            paths=[list(map(int, p)) for p in d["paths"]],
            sgen_bus=list(map(int, d["sgen_bus"])),
            s_rated_mva=torch.tensor(d["s_rated_mva"], dtype=torch.float32),
            action_scale=float(d["action_scale"]),
            class_rule=str(d.get("class_rule", CLASS_RULE)),
            extra=dict(d.get("extra", {})),
        )

    def banner(self) -> str:
        cls_counts = [int((self.line_class == m).sum()) for m in range(self.n_classes)]
        return (
            f"feeder          {self.name}: {self.n_lines} elements, {self.n_agents} inverters, "
            f"V_base={self.v_base_kv} kV\n"
            f"                classes {dict(zip(self.class_names, cls_counts))}  ({self.class_rule})\n"
            f"                path lengths {[len(p) for p in self.paths]}\n"
            f"                s_rated (MVA) {[round(float(s), 3) for s in self.s_rated_mva]}"
        )


# ---------------------------------------------------------------------------
#  loading
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent


def structure_path(scenario: str) -> Path:
    return _HERE / "conf" / f"structure_{scenario}.json"


def load_structure(scenario: str = "case33_3min_final", allow_synthetic: bool = True) -> FeederStructure:
    """The exported structure for ``scenario``, or -- offline, with no data on
    the machine -- a synthetic radial feeder of the same shape so that the
    torch-only checks can still run.  The banner says which one you got."""
    path = structure_path(scenario)
    if path.exists():
        return FeederStructure.from_json(path)
    if not allow_synthetic:
        raise FileNotFoundError(
            f"{path} not found. Run `python mapdn_ns/structure.py --scenario "
            f"{scenario}` with pandapower and the data installed."
        )
    return synthetic_structure()


def synthetic_structure(n_agents: int = 6, n_lines: int = 32, seed: int = 0) -> FeederStructure:
    """A radial 33-bus-shaped feeder: one trunk of 18 segments with three
    laterals, reactances in the IEEE 33-bus range, six inverters on the
    laterals and near the trunk end.  Used ONLY when the exported structure is
    absent; nothing measured on it is a result."""
    gen = torch.Generator().manual_seed(seed)
    # bus 0 is the root.  Trunk 0-1-...-17; laterals hang off trunk buses.
    parent: List[int] = [-1]
    for b in range(1, 18):
        parent.append(b - 1)
    # three laterals: from bus 1 (len 4), bus 2 (len 5), bus 5 (len 6)
    for root, length in ((1, 4), (2, 5), (5, 6)):
        prev = root
        for _ in range(length):
            parent.append(prev)
            prev = len(parent) - 1
    n_bus = len(parent)
    line_from = []  # element a connects bus (a+1) to parent[a+1]
    x = []
    r = []
    for b in range(1, n_bus):
        line_from.append(parent[b])
        x.append(float(0.3 + 1.4 * torch.rand(1, generator=gen)))
        r.append(float(0.3 + 1.2 * torch.rand(1, generator=gen)))
    x_t, r_t = torch.tensor(x), torch.tensor(r)
    cls = _tercile_classes(r_t, 3)
    # inverters: spread over trunk end and laterals
    sgen_bus = [17, 13, 21, 26, 32, 9][:n_agents]
    paths = [_path_to_root(b, parent) for b in sgen_bus]
    return FeederStructure(
        name="synthetic33",
        v_base_kv=12.66,
        line_x_ohm=x_t,
        line_r_ohm=r_t,
        line_class=cls,
        class_names=["heavy", "medium", "light"],
        paths=paths,
        sgen_bus=sgen_bus,
        s_rated_mva=torch.tensor([1.75, 1.75, 1.75, 1.75, 1.75, 1.75][:n_agents]),
        action_scale=0.8,
        extra={"synthetic": True},
    )


def _path_to_root(bus: int, parent: Sequence[int]) -> List[int]:
    out = []
    b = bus
    while parent[b] >= 0:
        out.append(b - 1)  # element id: bus b is connected to its parent by element b-1
        b = parent[b]
    return out


def _tercile_classes(values: Tensor, r: int) -> Tensor:
    """Class = rank tercile of a public per-element property, LARGEST first.
    Deterministic, no RNG."""
    order = torch.argsort(values, descending=True)
    cls = torch.zeros_like(values, dtype=torch.long)
    n = values.shape[0]
    for k, idx in enumerate(order.tolist()):
        cls[idx] = min(int(k * r / n), r - 1)
    return cls


# ---------------------------------------------------------------------------
#  building from the host's own network file (needs pandapower)
# ---------------------------------------------------------------------------


def build_structure(net, s_rated_mva: Sequence[float], action_scale: float,
                    name: str, n_classes: int = 3) -> FeederStructure:
    """Read the declared operator's ingredients off a pandapower net.

    Elements are the lines AND the transformers (a transformer's reactance is
    on every path below it, and every inverter's VARs cross it).  Paths are the
    tree path from each sgen's bus to the ext_grid bus.
    """
    import networkx as nx  # pandapower depends on it

    lines = net.line
    trafos = getattr(net, "trafo", None)
    vn = float(net.bus.loc[net.ext_grid.bus.iloc[0], "vn_kv"])

    g = nx.Graph()
    x_ohm: List[float] = []
    r_ohm: List[float] = []
    vn_kv: List[float] = []
    kind: List[str] = []
    r_per_km: List[float] = []
    bus_vn = net.bus["vn_kv"]
    for idx, row in lines.iterrows():
        if hasattr(row, "in_service") and not bool(row.in_service):
            continue
        eid = len(x_ohm)
        x_ohm.append(float(row.x_ohm_per_km) * float(row.length_km) / float(getattr(row, "parallel", 1) or 1))
        r_ohm.append(float(row.r_ohm_per_km) * float(row.length_km) / float(getattr(row, "parallel", 1) or 1))
        vn_kv.append(float(bus_vn.loc[int(row.from_bus)]))
        r_per_km.append(float(row.r_ohm_per_km))
        kind.append("line")
        g.add_edge(int(row.from_bus), int(row.to_bus), eid=eid)
    if trafos is not None and len(trafos):
        for idx, row in trafos.iterrows():
            if hasattr(row, "in_service") and not bool(row.in_service):
                continue
            eid = len(x_ohm)
            # short-circuit reactance referred to the LV side, in ohm
            vlv = float(row.vn_lv_kv)
            z = float(row.vk_percent) / 100.0 * vlv ** 2 / float(row.sn_mva)
            rr = float(row.vkr_percent) / 100.0 * vlv ** 2 / float(row.sn_mva)
            xx = max(z ** 2 - rr ** 2, 0.0) ** 0.5
            x_ohm.append(xx)
            r_ohm.append(rr)
            vn_kv.append(vlv)  # referred to the LV side, so the LV level is its base
            r_per_km.append(float("nan"))
            kind.append("trafo")
            g.add_edge(int(row.hv_bus), int(row.lv_bus), eid=eid)

    root = int(net.ext_grid.bus.iloc[0])
    sgen_bus = [int(b) for b in net.sgen.bus.tolist()]
    paths: List[List[int]] = []
    for b in sgen_bus:
        nodes = nx.shortest_path(g, source=b, target=root)
        paths.append([int(g.edges[nodes[k], nodes[k + 1]]["eid"]) for k in range(len(nodes) - 1)])

    x_t = torch.tensor(x_ohm, dtype=torch.float32)
    r_t = torch.tensor(r_ohm, dtype=torch.float32)
    rk = torch.tensor([v if v == v else -1.0 for v in r_per_km])  # trafo -> heaviest
    is_line = torch.tensor([k == "line" for k in kind])
    cls = torch.zeros(len(x_ohm), dtype=torch.long)
    if int(is_line.sum()) > 0:
        # tercile over LINES by r_ohm_per_km, largest R/km (lightest conductor) = last class
        order_vals = rk.clone()
        order_vals[~is_line] = float("-inf")  # transformers: heaviest class (0)
        # _tercile_classes puts the LARGEST value in class 0, so pass -R/km:
        # the smallest R/km (heaviest conductor, and any transformer) is class 0
        cls = _tercile_classes(-order_vals, n_classes)
    names = ["heavy", "medium", "light"][:n_classes] if n_classes == 3 else [f"class{m}" for m in range(n_classes)]
    return FeederStructure(
        name=name,
        v_base_kv=vn,
        line_x_ohm=x_t,
        line_r_ohm=r_t,
        line_class=cls,
        line_vn_kv=torch.tensor(vn_kv, dtype=torch.float32),
        class_names=names,
        paths=paths,
        sgen_bus=sgen_bus,
        s_rated_mva=torch.tensor([float(s) for s in s_rated_mva], dtype=torch.float32),
        action_scale=float(action_scale),
        extra={"kind": kind, "root_bus": root, "n_bus": int(len(net.bus)),
               "voltage_levels_kv": sorted(set(round(v, 3) for v in vn_kv))},
    )


def load_net(path: str):
    """``pp.from_pickle`` with one repair: the shipped ``case322`` pickle
    (pandapower 2.7 format) lacks ``sgen.type``, which pandapower 3.3's format
    converter indexes, so the stock loader raises ``KeyError: 'type'`` on it.
    The column is filled with ``"PV"`` -- what ``case33`` and ``case141``
    carry -- before conversion.  Purely a loader fix; nothing electrical."""
    import pandapower as pp
    import pandapower.file_io as fio
    from pandapower.auxiliary import pandapowerNet

    try:
        return pp.from_pickle(path)
    except KeyError as exc:
        if "type" not in str(exc):
            raise
    net = pandapowerNet(fio.get_raw_data_from_pickle(path))
    fio.transform_net_with_df_and_geo(net, ["bus_geodata"], ["line_geodata"])
    if "type" not in net.sgen.columns:
        net.sgen["type"] = "PV"
    pp.convert_format(net)
    return net


def _main() -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="export the feeder structure JSON from the host's model.p")
    ap.add_argument("--scenario", default="case33_3min_final")
    ap.add_argument("--data-root", default=str(_HERE.parent / "environments" / "var_voltage_control" / "data"))
    args = ap.parse_args()

    sys.path.insert(0, str(_HERE.parent))
    import numpy as np
    import pandapower as pp
    import pandas as pd

    root = Path(args.data_root) / args.scenario
    net = load_net(str(root / "model.p"))
    pv = pd.read_csv(root / "pv_active.csv", index_col=None).iloc[:, 1:]
    p_max = pv.to_numpy().max(axis=0)
    s_rated = 1.2 * p_max  # the host's own rule: VoltageControl._set_reactive_power_boundary
    scale = {"case33_3min_final": 0.8, "case141_3min_final": 0.6, "case322_3min_final": 0.8}[args.scenario]
    st = build_structure(net, s_rated, scale, name=args.scenario)
    out = structure_path(args.scenario)
    st.to_json(out)
    print(st.banner())
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
