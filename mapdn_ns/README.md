# Coupling Under Drift on MAPDN — PV inverters on one feeder

One layer under the stock `VoltageControl`, no edits to MAPDN, every learner
bit-for-bit the one the authors shipped.  PACT is `pact1/core.py`, a
byte-identical copy of the object `road_ns` and `simple_ns` use.  Three
feeders: `case33` (6 inverters, the development host), `case141` (22, the
headline) and `case322` (38, exported, uncalibrated).

```bash
conda activate mapdn                     # the repo's own environment
python mapdn_ns/check_plumbing.py        # config consistency, torch only      ~1 s
python mapdn_ns/conformance.py           # 18 offline checks, torch only       ~5 s
python mapdn_ns/estimator_selftest.py    # build-order step 5, torch only      ~30 s
python mapdn_ns/smoke.py                 # 11 in-simulator checks              ~8 min
python mapdn_ns/probe.py                 # step 7, maximum-excitation probe    ~3 min
python mapdn_ns/calibrate.py             # the sigma ladder, blind/oracle/pact ~50 min
```

The first three run with torch alone — no pandapower, no data.  Run them in
that order; do not start a sweep on a red gate.

Data: `environments/var_voltage_control/data/<scenario>/` (the MAPDN
download).  The feeder structure the layer reads is exported once to
`mapdn_ns/conf/structure_<scenario>.json` by `python mapdn_ns/structure.py
--scenario <scenario>`; the layer exports it itself on first use if it is
missing.  Every tool takes `--scenario case141_3min_final` (default `case33`).

---

## Which cell this is

| | invertible channel | no inverse |
|---|---|---|
| **(C) interaction-mediated** — a lone agent feels *nothing* | **this package**, and `simple_ns` | `road_ns` |
| **(B) exogenous** — a lone agent feels it | `--direct`, the control | — |

The disturbance is **additive in the inverter's own action space** (reactive
power, MVAr), so a correct estimate cancels it exactly: II.6 row 1,
**identification and compensation**.  Every sum is strictly `j != i`, so one
inverter alone on the feeder reads exactly zero at any severity: category C
structurally, and measured in the simulator at σ = 20.

## The mechanism

MAPDN's inverters follow their setpoint exactly.  Real ones do not.  Every
IEEE 1547-2018 inverter ships with an **autonomous volt-var curve** in its
firmware, utilities (California Rule 21, HECO, AS/NZS 4777.2) require it
enabled, and a dispatched setpoint is applied on top of it.  So what an
inverter delivers is

```
q_delivered = q_dispatched + q_firmware(V_terminal)
```

and its terminal voltage is moved by every **other** inverter's VARs through
the feeder's reactance.  The fleet's curves react to each other's dispatch —
distribution engineers call it smart-inverter control interaction, and it is
why fleets under central dispatch are so often forced into constant-Q mode.

```
S_m[i,j] = sum over segments a on path(i) AND path(j), class(a) = m, of X_a / V_base^2
x_m,i(t) = sum_{j != i} S_m[i,j] q_j(t-1)                 pu voltage, PUBLIC
d_i(t)   = s_i * clip( -sigma * k * A(t) * sum_m send_m x_m,i , +-0.44 )   MVAr, PRIVATE
q_exec   = clip( q_cmd - g * pred_i * s_i + d_i , +-sqrt(S_i^2 - P_i^2) )
y_i      = (q_exec - q_sent) / s_i                        the inverter's own meter
```

* **The medium** is the radial feeder.  `S` is the linearised DistFlow
  voltage sensitivity — incidence over reactance, read off `model.p` before
  any run (`structure.py`).  Zero diagonal asserted; measured spread 1.51,
  asymmetry 0.46 (the receiver's nameplate scales its reaction; the six
  units are 2.2 / 2.2 / 2.1 / 3.6 / 0.33 / 0.33 MVA).
* **The classes** are the conductor size of the feeder's segments (terciles
  of R/km, `r = 3`), a property painted on the pole.  Public: which segments
  a neighbour's VARs cross to reach you.  Unknown: what a unit of VAR on
  each class costs you today (`send_m`, the declared β\* fan, as in `simple_ns`).
* **The driver** `A(t)` is the solar day — `sin²` between 06:00 and 18:00 on
  the **dataset's own clock**, and *exactly* zero at night, when the profile
  sits inside the curves' deadband and the firmware is inert.  Half of every
  day is a placebo.  `ns_driver: schedule` is the literal on/off alternative
  (a DERMS enabling the fleet's curves on a schedule, IEEE 2030.5).
* **The anchor.**  σ = 1 is the IEEE 1547-2018 **Category B default** slope,
  0.44 pu-Q / 0.06 pu-V = 7.33.  σ = 3 is the steepest curve the standard
  allows (V4 − V3 = 0.02); σ = 0.34 is the Category A default; the 0.44
  saturation is the curve's own Q1/Q4.  Above σ = 3 is beyond-physical and
  must be labelled as such wherever it appears.
* **The loop gain.**  The reactions feed each other —
  `q_exec(t) = q_cmd(t) − σ·A(t)·W·q_exec(t−1)` — so the fleet is a
  discrete-time loop with gain `σ·A·ρ(W)`.  Measured on this feeder,
  `ρ(W) = 0.378` at σ = 1: the anchored default curve is a **stable, mild**
  interaction (a peer's MVAr costs at most 0.34 MVAr of reaction, between the
  two zone-1 units), and the fleet **hunts** at noon for σ > 2.65 — still
  inside the standard's steepest compliant curve.  That is the volt-var
  hunting instability IEEE 1547 bounds with its response-time requirement,
  and it is what the ladder's σ = 3 row measures: blind fleets oscillate
  against the curves' saturation; a cancelled reaction has no loop to
  destabilise.  `conformance.py` reports both numbers.
* **The reward is never touched.**  The layer overrides `_take_action`, the
  single place the host turns an action into an injection; `reward`, `done`
  and the stock observation are inherited.  σ = 0 reproduces the stock host
  bit for bit (rewards and observations, checked in `smoke.py`).

Every arm sees its own one-step-stale residual and the public channels in
its observation (`ns_observe_residual`, `ns_observe_channels`, both default
on) — the baselines are information-matched; PACT's advantage has to be the
mechanism.

## The (B) control

`--direct` (`ns_direct: true`) keeps the firmware, the driver, the
severity, the reward and the ladder and changes only *what the curve reacts
to*: the exogenous, PV-driven rise of the solar day itself instead of the
voltage the peers impose.  Every inverter's curve then absorbs a fixed
`sigma * A(t) * D_ref_i`, whatever anyone was dispatched, where `D_ref_i` is
the mean |d_i| the coupling produces on that inverter under the reference
dispatch (uniformly random VARs over the host's own range), so σ means the
same severity in both cells, per inverter.  It is a level shift that drifts
with the clock.  A lone inverter feels it (measured
in `smoke.py`), and PACT's peer channels carry no information about it — the
regression puts it in the intercept.  `--arm intercept` is the ablation that
carries the (B)-vs-(C) claim: on (B) it should lose nothing against `pact`;
on (C) it should lose most of the recovery.  Measured (RESULTS.md): on (C)
the intercept closes 1 % of the gap and PACT 98 %; on (B) the intercept
closes 73 % by itself, and with the clock in the basis (`pact_known_driver`)
`intercept` and `pact` are identical to three decimals.

## The arms

| arm | what it is |
|---|---|
| `blind` | the dial, no compensator — every baseline runs here |
| `pact` | PACT-1: per-inverter RLS on `[1, x]`, trust 0.9, exact inverse |
| `pactoff` | trust forced to 0 — **bit-identical to blind** (P-7.1, checked) |
| `oracle` | handed the true disturbance — the ceiling, not a competitor |
| `intercept` | peer channels deleted, everything else identical |

## Commands — the three runs

MATD3 is the strongest learner in the MAPDN paper on this case; MAPPO is
the one usually named.  Both are launched **unmodified** through the same
entry point, with the severity supplied from outside (P-10.1).  Run from the
repository root.

```bash
# 1. no NS: the stock task, B0                        (blind, sigma = 0)
python mapdn_ns/run.py train --alg matd3 --arm blind --sigma 0   --alias paper --seed 0

# 2. NS, the best existing algorithm                  (blind, sigma = sigma* = 3)
python mapdn_ns/run.py train --alg matd3 --arm blind --sigma 3   --alias paper --seed 0

# 3. NS with PACT on top of the same algorithm        (pact,  sigma = sigma* = 3)
python mapdn_ns/run.py train --alg matd3 --arm pact  --sigma 3   --alias paper --seed 0
```

Swap `--alg mappo` for the MAPPO rows, and add `--scenario case141_3min_final
--sigma 1` for the 22-inverter feeder at the anchor.  Then, per seed, the
ablation arms at the same σ — `--arm oracle`, `--arm pactoff`, `--arm
intercept` — and the (B) control, `--direct`, for `blind`, `pact` and
`intercept`.  Evaluate a run the way
MAPDN does:

```bash
python mapdn_ns/run.py test --alg matd3 --arm pact --sigma 3 --alias paper --seed 0 --test-mode batch
```

`--load-alias <alias>` evaluates a checkpoint under a configuration other
than the one it was trained in — the σ = 0 policy under the disturbance,
zero-shot, is `--arm blind --sigma 3 --load-alias paper-blind-s0-seed0`.

Each run writes `runs/ns_logs/<log_name>/pact_debug.csv`, one row per
episode (`phase` = `train` for the exploring episodes, `eval` for the greedy
evaluation episodes MAPDN runs every 20, `test` under `test.py`), and
`ns_config.yaml` with every constant it ran with.  `python mapdn_ns/summarize.py
--root runs/ns_logs --last 40` tabulates a sweep: loss ×B0 and CR per
(algorithm, cell, σ, arm, seed) against the σ = 0 blind run of the same
algorithm and seed, the paired means over seeds, and the II.10 panel.  Tensorboard
gets the same panel as `data/mean_train_ns_*` and `data/mean_train_pact_*`.
Read the CSV in column order: did the dial fire (`A`, `load`), is there
anything to identify (`x_std`), does the reduction hold (`fit_gain`,
`cond_psi`), is β recovered (`beta_cos`, `beta_relerr`), is the estimator
healthy (`updates`, `skipped`, `bounded`, `diverged`), is trust armed
(`trust_pol` vs `trust_app`), how much was cancelled (`corr_frac`,
`explained`, `resid_after`), what it did in the dark (`corr_dark`) — and only
then the domain metric (`reward`, `cr`, `v_out`).

`--set key=value` overrides any `ns_*` / `pact_*` key; `--no-obs-channels`
and `--no-obs-residual` are the observation ablations; `--train-episodes`
and `--episode-limit` shorten a run.  `mapdn_ns/scripts/sweep.sh <alg>
<sigma> "<seeds>" <alias>` runs the whole paired sweep (the three rows, the
ablation arms and the (B) control, per seed) and `scripts/evaluate.sh` the
matching `test.py --test-mode batch` evaluations.

## The operating points

Two feeders carry the claim; the full ladders are in [`RESULTS.md`](RESULTS.md)
(`calibrate.py`, the scripted volt-var controller of `controllers.py`,
episodes whose start hours span the day).

**case141 (22 inverters) — the headline, at the anchor itself, σ = 1.**
At the standard's *default* curve the droop fleet's mutual reaction collapses
the power flow (MAPDN's −200 "destroy" penalty, two of three daytime
episodes); PACT restores B0's controllable ratio exactly and the oracle
recovers B0; deleting the peer channels collapses like blind.  No stress
test anywhere in the row.

| case141, σ = 1 | blind | intercept | PACT | oracle |
|---|---|---|---|---|
| loss ×B0 | 43.2 | 43.5 | **1.028** | 0.998 |
| CR | 0.646 | 0.608 | **0.994** | 0.994 |

**case33 (6 inverters) — the small-feeder row, σ = 3.**  σ = 1 is a real but
*mild* interaction on six inverters (loop gain 0.38); σ = 3, the steepest
curve the standard allows, makes the droop fleet hunt at noon: it loses 87 %
more reward, the oracle recovers B0 exactly, PACT recovers 98 % of the gap.
A *learned* dispatcher feels it far less (it does not re-command against its
own reaction the way a droop does): on the trained B0, σ = 3 costs 3 % of
reward and 3.5 CR points and PACT recovers to B0 — so on case33 expect a
modest gap between a blind learner and PACT, and a large one for the
domain's own controller.

| case33 | blind loss ×B0 / CR | oracle | PACT |
|---|---|---|---|
| σ = 1 (anchor), droop | 0.998 / 0.978 | 1.000 / 0.978 | 1.000 / 0.979 |
| **σ = 3, droop** | **1.867 / 0.731** | 0.999 / 0.978 | **1.013 / 0.973** |
| σ = 3, trained B0 policy | 1.030 / 0.838 | 0.995 / 0.873 | 0.997 / 0.864 |
| σ = 6 (beyond-physical), trained B0 | 1.334 / 0.667 | 0.991 / 0.873 | 1.014 / 0.850 |

The loop gain of the mutual reaction rises with the number of inverters
sharing segments — 0.38 (case33) → 0.68 (case141) → 1.19 (case322, where the
default curve already hunts) — computed from the network files before any
run.  That is the N-scaling prediction of the design, and it is why the
anchored physics needs no stress test on the larger feeders.

Run the three case141 rows with `--scenario case141_3min_final --sigma 1`;
each episode costs ~5× case33's (0.6 s/step).  Confirm any operating point
on the trained B0 with `calibrate.py --controller policy --save-path runs
--alg matd3 --log-name <log_name>` before quoting it.

## What is shared, and what is not

`pact1/core.py` is the *same object* `road_ns` and `simple_ns` use (the hash
is checked by `check_plumbing.py`): the RLS with its dead-row skip and
covariance bound, the inverted trust prior, the prediction-based confidence
gate, and `compensate` — the exact inverse.  Only the basis is this
package's: the DistFlow operator over conductor classes, with **per-inverter
pruning** (P-3.4): an inverter whose only segment shared with the rest of the
fleet is the trunk head has structurally zero channels for the other
classes, and feeding those dead columns to one batched estimator let the
covariance wind up until the trace bound slowed the live directions
(measured: the bound fired on 1178 of 1440 steps).  Each inverter therefore
runs its own RLS over its own live columns.

## Traps this build paid for

* **float32 on the action path.**  A float32 round trip of the commanded VAR
  moved the σ = 0 reward by 2e-9 — not bit-identical.  The action path is
  float64, the host's own; the estimator casts at the boundary.
* **The host's reset dispatch is drawn from the global numpy stream**, which
  every construction reseeds.  Two arms built in one process then start
  each episode from different VARs and the paired comparison measures the
  interpreter's history.  `smoke.py` and `calibrate.py` pin it per episode.
* **Silent peers are not absent peers.**  Command every neighbour to zero
  and their firmware still answers the one inverter that acts, and that
  reaction reaches it.  The in-simulator N = 1 test switches the neighbours'
  curves off (nameplate 0) — legacy units with volt-var disabled.
* **`corr_clip` below the curve's own saturation.**  At 0.5 of the range
  (0.40 pu) the guard rail clipped a *correct* estimate of a saturated
  reaction (0.44 pu) and the oracle left 0.04 pu uncancelled.  It is 0.6
  (0.48 pu): a correct estimate is never clipped, a diverged one still is.
* **β scored on unobservable columns.**  Scored against the full β\*, an
  inverter with a dead class channel read as an estimation failure.  β is
  scored on each inverter's live columns (`Coupling.channel_liveness`).
* **One centring scale shared across inverters** (case141).  The pooled
  P‑3.3 reference gave per-inverter design-matrix condition numbers of
  1e4–1e6 on 22 inverters with paths of 5–31 segments and nameplates of
  0.9–12 MVA; the float32 RLS lost positive-definiteness at episode 31 of a
  training run, 89 756 predictions went non-finite and the confidence gate
  disarmed the compensator for the rest of the run (`trust_app` 0.24).  The
  estimator now centres each inverter on its **own** reference (`(N, r)`),
  holds its state in float64, and re-initialises any inverter's RLS whose
  state goes non-finite (`resets` column).  The observation keeps the pooled
  reference, so blind/B0 runs are unaffected.  Probe on case141 at σ = 3
  after the fix: `cond_psi` 720, 0 diverged, `fit_gain` 0.66, β cos 0.75.
* **The ratios explode in the dark.**  `|corr| / |d|` and `pred_err / |d|`
  averaged per step blow up wherever `A(t) = 0`; they are episode sums, and
  NaN when there was no disturbance to explain.

## The tracking floor, and the dark phantom

β\* follows `A(t)` through the day, and the RLS must track it with
forgetting.  Measured offline on the feeder's own basis, one-step prediction
error over the live half of the day: μ = 0.999 → 66 %, 0.98 → 40 %,
0.9 → 14 % of the disturbance; with the driver's shape in the basis
(`pact_known_driver`, ψ = `[1, A, A·z]`) → 0.0 %.  Two consequences:

* **μ is selected on prediction error, never on return** (`calibrate.py
  --mu-sweep`), and the value in the yaml is a swept value, not a default.
* **At dusk the compensator applies a phantom.**  β̂ learned by day decays
  over ~1/(1−μ) steps after `A(t)` reaches zero, while the channels stay
  live, so PACT corrects a disturbance that is no longer there.  The
  `corr_dark` column measures it.  P-7.1 protects the arm only at g = 0;
  this is the price of tracking, and `pact_known_driver` (paired with
  `ns_observe_driver` so the baselines are information-matched) is the
  ablation that removes it.

## Not implemented, and what it costs the claim

* **P-6.2, trust through the objective.**  The correction is a
  deterministic transform of the sampled action, so `∂ log π / ∂g = 0` and
  a trust head gets no gradient.  `off` and `fixed` (0.9) exist; `learned`
  does not.  Since P-5.1 starts near full reliance, `fixed` is close to the
  intended operating point.
* **The per-class decomposition is weak on this feeder.**  Conductor classes
  along one radial trunk are nearly collinear (per-inverter design-matrix
  condition numbers 6e2–2e3 under random dispatch); prediction is fine
  (`fit_gain` 0.8–0.95) and β's cosine to truth is 0.8–1.0 on live columns,
  but the relative error stays large.  Claim compensation, not a
  per-channel decomposition, unless `beta_relerr` in your run says
  otherwise (gate 7 warns; it does not abort).
* **NS-4.1/4.2, the ceiling decomposition, is degenerate.**  No uncontrolled
  participant draws on this medium and every sum is `j != i`, so the
  coordination gap is 100 % by construction and the controllable-share sweep
  cannot be run.  Making it informative needs legacy inverters with fixed
  curves as background participants — not done.
* **II.9 gates 5–7 do not abort during training.**  `fit_gain` and
  `cond_psi` are in every debug row; nothing stops a run on them.
* **Decentralised mode** (one agent per zone) is not supported; the layer
  asserts `distributed`.
* **`case322`** is exported and smoke-tested (38 inverters, two voltage
  levels, 0.9 s/step) but not calibrated.  Its shipped `model.p` is a
  pandapower-2.7 pickle without `sgen.type`, which pandapower 3.3's format
  converter indexes; the layer's `_load_network` fills the column before
  conversion (`structure.load_net`) — the stock host cannot open the file
  in the shipped conda environment at all.
* **`case141` is calibrated on the droop controller only** (3 short
  episodes); confirm on its trained B0 before quoting, and budget ~5×
  case33's compute per episode.
