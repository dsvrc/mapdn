#  Competent controllers, for CALIBRATION and PROBING only.
#
#  The severity has to be chosen against a controller that was actually
#  trying (porting notes, trap #1): a random policy's return can RISE with
#  severity.  Two instruments:
#
#    DroopController   the domain's own scripted baseline -- a volt-var droop
#                      closed on each inverter's OWN terminal voltage, with the
#                      IEEE 1547 Category B slope and a first-order response.
#                      MAPDN ships droop control as its traditional baseline
#                      (traditional_control/pf_droop_matpower_all.m, in Matlab);
#                      this is the same idea in the host's action space.
#    PolicyController  a trained MAPDN checkpoint acting greedily, exactly as
#                      test.py drives it -- for calibrating against the trained
#                      B0 when the scripted controller is not competent enough
#                      (trap #2: first check the controller can do the task at
#                      sigma = 0).
#
#  These are NOT baselines and must never be reported as one.  The droop reads
#  bus voltages off the power-flow result directly, which no policy may do.

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

__all__ = ["DroopController", "PolicyController", "controller_for"]


class DroopController:
    """``a_i = clip( q_i / cap_i )`` with ``q_i <- (1-alpha) q_i + alpha * (-k s_i (V_i - V_ref))``.

    ``k`` is the IEEE 1547-2018 Category B default slope in pu-Q per pu-V (the
    same constant the disturbance is anchored to -- here used as a CONTROLLER on
    the agent's own voltage, closed each step).  ``alpha`` is the first-order
    response the standard's open-loop response time implies, and it is what
    keeps a fleet of droops from hunting each other at the 3-minute step.
    Gains are declared here and never moved with sigma.
    """

    def __init__(self, k: float = 7.333, alpha: float = 0.5, v_ref: float = 1.0,
                 deadband: float = 0.0) -> None:
        self.k, self.alpha, self.v_ref, self.deadband = k, alpha, v_ref, deadband
        self._q: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._q = None

    def __call__(self, env, obs=None) -> np.ndarray:
        v_all = env.powergrid.res_bus["vm_pu"].sort_index().to_numpy(copy=True)
        buses = env.powergrid.sgen["bus"].to_numpy()
        v = v_all[buses]
        s = np.asarray(env.s_max, dtype=np.float64)
        p = env.powergrid.sgen["p_mw"].to_numpy(dtype=np.float64)
        cap = np.sqrt(np.maximum(s ** 2 - p ** 2, 1e-12))
        dv = v - self.v_ref
        dv = np.sign(dv) * np.maximum(np.abs(dv) - self.deadband, 0.0)
        target = -self.k * s * dv
        if self._q is None:
            self._q = target
        else:
            self._q = (1.0 - self.alpha) * self._q + self.alpha * target
        hi = float(env.args.action_scale)
        return np.clip(self._q / cap, -hi, hi)


class PolicyController:
    """A trained checkpoint, driven as test.py drives it (greedy, no noise)."""

    def __init__(self, save_path: str, alg: str, log_name: str, env, root: Optional[Path] = None) -> None:
        root = root or Path(__file__).resolve().parents[1]
        from models.model_registry import Model
        from utilities.util import convert

        with open(root / "args" / "default.yaml", "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        with open(root / "args" / "alg_args" / f"{alg}.yaml", "r", encoding="utf-8") as fh:
            alg_cfg = yaml.safe_load(fh)["alg_args"]
        cfg.update(alg_cfg)
        cfg["action_scale"] = float(env.args.action_scale)
        cfg["action_bias"] = float(env.args.action_bias)
        cfg["agent_num"] = env.get_num_of_agents()
        cfg["obs_size"] = env.get_obs_size()
        cfg["action_dim"] = env.get_total_actions()
        cfg["cuda"] = False
        self.args = convert(cfg)
        model = Model[alg]
        if self.args.target:
            target = model(self.args)
            self.net = model(self.args, target)
        else:
            self.net = model(self.args)
        path = Path(save_path) / "model_save" / log_name / "model.pt"
        ck = torch.load(str(path), map_location="cpu")
        self.net.load_state_dict(ck["model_state_dict"])
        self.net.eval()
        self.env = env
        self._hid = None

    def reset(self) -> None:
        self._hid = self.net.policy_dicts[0].init_hidden()

    def __call__(self, env, obs=None) -> np.ndarray:
        from utilities.util import prep_obs, translate_action

        if self._hid is None:
            self.reset()
        if obs is None:
            obs = env.get_obs()
        state = prep_obs(obs).contiguous().view(1, self.args.agent_num, self.args.obs_size)
        with torch.no_grad():
            action, _, _, _, hid = self.net.get_actions(
                state, status="test", exploration=False,
                actions_avail=torch.tensor(env.get_avail_actions()), target=False, last_hid=self._hid,
            )
        self._hid = hid
        _, actual = translate_action(self.args, action, env)
        return np.asarray(actual, dtype=np.float64).reshape(-1)


def controller_for(name: str, env, **kw):
    if name == "droop":
        return DroopController(**kw)
    if name == "policy":
        return PolicyController(kw["save_path"], kw["alg"], kw["log_name"], env)
    raise ValueError(f"unknown controller {name!r}")
