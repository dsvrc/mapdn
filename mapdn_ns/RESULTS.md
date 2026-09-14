# Calibration results — `case33`, distributed, `l1` barrier

Everything here was produced by `mapdn_ns/calibrate.py` on 2026-09-13 with
the committed constants in `conf/case33.yaml`, six 12-hour episodes whose
start hours span the day (06, 09, 12, 15, 18, 00; random dataset days; no
data noise; identical episodes and reset dispatch for every arm).  Raw rows
are in `runs/calib/*.csv`.

`loss ×B0` is the mean per-step loss (−reward) divided by the σ = 0 blind
value: 1.00 = B0, larger = worse.  `CR` is MAPDN's own metric, the fraction
of steps with every bus inside 0.95–1.05 pu.  `|d|` is the mean disturbance
in fractions of the inverter's VAR range, `sat` the fraction of agent-steps
at the curve's 0.44 saturation.

## I.6 — what the dial does, before any controller

Under the reference dispatch (every inverter uniformly random over its own
VAR range, the one-shot reaction before any feedback):

| σ | what it is | mean \|d\| over a day | at noon | saturated | loop gain at noon |
|---|---|---|---|---|---|
| 0.34 | IEEE 1547 Cat. A default | 0.8 % | 3.1 % | 0.0 % | 0.13 |
| **1** | **IEEE 1547 Cat. B default (the anchor)** | 2.3 % | 9.2 % | 0.0 % | 0.38 |
| 2 | compliant, steeper | 4.5 % | 17.7 % | 0.7 % | 0.76 |
| **3** | **the steepest compliant curve** | 6.4 % | 23.6 % | 2.8 % | **1.13 — hunts** |
| 6 | beyond-physical | 9.8 % | 32.4 % | 8.8 % | 2.27 |

Placebo: 241 of 480 intervals per day exactly dark; cycle mean of `A` 0.250.
Loop gain `ρ(W) = 0.378` at σ = 1: the fleet's mutual reaction becomes an
instability at noon for σ > 2.65.  Coordination gap: 100 % by construction
(degenerate, see README).

## The σ ladder — scripted volt-var controller (C cell)

`controllers.DroopController`, IEEE 1547 slope, first-order response 0.5.
Competent at σ = 0: CR 0.978, 0.24 % of bus-steps out of band (doing nothing
gives CR 0.86; random actions CR 0.39).

| σ | blind loss ×B0 | blind CR | oracle loss ×B0 | oracle CR | **PACT loss ×B0** | **PACT CR** | blind \|d\| | PACT \|d\| | recovered |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.998 | 0.978 | 1.000 | 0.978 | 1.000 | 0.979 | 0.3 % | 0.3 % | — (inert) |
| 1 | 0.998 | 0.978 | 1.000 | 0.978 | 1.000 | 0.979 | 0.5 % | 0.5 % | — (inert) |
| 2 | 1.442 | 0.837 | 0.999 | 0.978 | **1.004** | **0.976** | 5.1 % | 1.1 % | 99 % |
| **3** | **1.867** | **0.731** | 0.999 | 0.978 | **1.013** | **0.973** | 9.7 % | 1.7 % | **98 %** |
| 6 | 2.333 | 0.642 | 0.998 | 0.979 | 1.119 | 0.902 | 14.5 % | 6.2 % | 91 % |

Read it like this:

* **σ ≤ 1 is inert for a droop-controlled fleet on this feeder.**  The
  Category B default curve has loop gain 0.38 and the controller dispatches
  small, smooth VARs; the reaction is half a percent of the range.  The
  anchored curve is real physics and a mild disturbance — say so.
* **σ = 2 and 3 bite, and the oracle recovers B0 exactly** — the channel is
  invertible, nothing here is past σ\*.  At σ = 3 the fleet hunts at noon
  (12 % of agent-steps at the curve's saturation) and the controller loses
  87 % more reward and a quarter of its in-band steps.
* **PACT recovers 98 % of the gap at σ = 3** with an online estimate that
  explains only ~35 % of the one-step disturbance: partial cancellation
  breaks the mutual-reaction loop, and the closed-loop disturbance falls
  from 9.7 % to 1.7 % of the range.  The domain metric is the result; the
  explained fraction understates it.
* **σ = 6 (beyond-physical) is still recoverable by the oracle** and PACT
  gets 91 %; the curve's saturation bounds what a fixed estimator can track.

**Committed operating point: σ = 3** — the steepest curve the standard
allows, the largest severity that is still compliant physics, and the row
where blind falls furthest while the ceiling is still full recovery.  σ = 2
is the secondary row.

## μ — selected on prediction error, never on return

Scripted controller, σ = 2 (the σ = 1 sweep is on an inert dial and says
nothing).  `expl%` = 1 − Σ|error| / Σ|d| over the ladder's episodes.

| μ | memory (steps) | pred_err (pu) | expl% | fit_gain | β cos | loss ×B0 | CR |
|---|---|---|---|---|---|---|---|
| 0.999 | 1000 | 0.0108 | −26 | −4.5 | 0.30 | 1.009 | 0.977 |
| **0.98** | **50** | **0.0057** | **34** | −1.6 | 0.11 | 1.004 | 0.976 |
| 0.95 | 20 | 0.0056 | 34 | −2.2 | 0.02 | 1.004 | 0.976 |
| 0.9 | 10 | 0.0057 | 34 | −3.0 | 0.02 | 1.004 | 0.976 |

μ = 0.999 (URB's value) averages the solar day away; everything from 0.98
down is equivalent on prediction error.  0.98 is kept: the longest memory
that pays no prediction penalty.  `fit_gain` is negative under this
controller — its dispatch is smooth and small, so the channels barely vary
within an episode and an intercept null tracks the residual as well as the
channels do (II.9 gate 5: excitation, not the reduction).  β is not
identified here; under random actions (`probe.py`) the same estimator
reaches `fit_gain` 0.83 and β cosine 0.83.  Re-run the sweep on the trained
B0 (`--controller policy`), whose exploration noise provides excitation.

## Offline, on the feeder's own basis (`estimator_selftest.py`)

One-step prediction error over the live half of the day, tracking β\*(t)
through the solar day with per-inverter live-column RLS: μ = 0.999 → 66 %,
0.98 → 40 %, 0.9 → 14 % of the disturbance; with the driver's shape in the
basis (`pact_known_driver`) → 0.0 %.  Known-law recovery: β relative error
0.001, cosine 1.000 on live columns.

## The (B)-vs-(C) pair — the experiment the classification rests on

Same firmware, same driver, same σ, same reward, same controller, same
episodes.  In (B) the curves react to the exogenous PV-driven rise of the
day itself — a level shift `−σ·A(t)·D_ref_i` that a lone inverter feels —
instead of to the peers' VARs; `D_ref_i` matches the two cells per inverter.
`intercept` is PACT with the peer channels deleted: a per-inverter adaptive
bias that knows nothing about who its neighbours are.  Loss ×B0 against the
true B0; "gap closed" = (blind − arm) / (blind − oracle).

**Scripted droop controller** (`ladder_droop_l1.csv`, `ladder_droop_l1_intercept.csv`,
`ladder_droop_l1_direct.csv`):

| cell | σ | blind | intercept | PACT | oracle | gap closed: intercept / PACT |
|---|---|---|---|---|---|---|
| **(C) coupled** | 2 | 1.442 | 1.460 | **1.004** | 0.999 | **−4 % / 99 %** |
| **(C) coupled** | 3 | 1.867 | 1.862 | **1.013** | 0.999 | **1 % / 98 %** |
| (B) direct | 2 | 1.070 | 1.020 | 1.022 | 1.004 | 76 % / 73 % |
| (B) direct | 3 | 1.130 | 1.039 | 1.047 | 1.006 | 73 % / 67 % |

**Trained B0 policy, 80-episode MATD3** (`ladder_policy_matd3_C.csv`,
`ladder_policy_matd3_B.csv`):

| cell | σ | blind | intercept | PACT | oracle | gap closed: intercept / PACT |
|---|---|---|---|---|---|---|
| **(C) coupled** | 3 | 1.030 | 1.038 | **0.997** | 0.995 | **−23 % / 94 %** |
| **(C) coupled** | 6 | 1.334 | 1.428 | **1.014** | 0.991 | **−27 % / 93 %** |
| (B) direct | 2 | 1.100 | 1.011 | 0.986 | 0.983 | 76 % / 97 % |
| (B) direct | 3 | 1.267 | 1.072 | 1.005 | 0.976 | 67 % / 90 % |

* **On (C) the peer channels are load-bearing.**  Delete them and the
  recovery disappears — `intercept` is blind or worse on both controllers
  (`explained` ≈ 0: an intercept cannot represent a disturbance that differs
  per inverter and flips with the neighbours' dispatch).  Keep them and
  93–99 % of the gap comes back.
* **On (B) a per-inverter bias does most of the job by itself** — 67–76 % of
  the gap on both controllers, against 1 % or less on (C).  Nothing about
  (B) needs the neighbours.
* What PACT adds on (B) beyond the intercept (nothing on the droop, 20–30
  points on the policy) is not peer information: the level shift drifts
  with the clock, and on a coherent fleet the channels *correlate* with the
  clock (everyone absorbs more at noon), so they help the RLS track `A(t)`
  faster than forgetting alone.  The known-driver ablation
  (`pact_known_driver`, ψ = `[1, A, A·z]`, with `A` also published to every
  arm) removes that route; see below.
* The (B) oracle lands at or below B0 (0.976–1.006): at trust 0.9 it leaves
  10 % of a pure absorption, which on the policy trims slightly
  over-dispatched VARs and lowers the reward's `q_loss` term.

In the (B) cell a lone inverter feels the disturbance (`smoke.py`); in the
(C) cell it reads exactly zero.

**With the driver's shape in the basis** (`calibrate.py --known-driver`:
`pact_known_driver`, ψ = `[1, A, A·z]`, and `A` published to every arm),
droop controller, σ = 3 (`ladder_droop_l1_knowndriver.csv`,
`ladder_droop_l1_direct_knowndriver.csv`):

| cell | blind | intercept | PACT | fit_gain (PACT) | β cos (PACT) |
|---|---|---|---|---|---|
| **(C) coupled** | 1.867 | 1.874 | **1.002** | 0.94 | **0.89** |
| (B) direct | 1.130 | 1.008 | 1.010 | 0.99 | — |

This is the pair in its cleanest form.  On (B) `intercept` and `pact` are
identical to three decimals — the peer channels add exactly nothing once
the clock is in the basis, and a per-inverter bias on `[1, A]` does the
whole job.  On (C) the same bias is blind and the peer channels close
99.8 % of the gap.  And with β\* stationary in these coordinates the
estimator *identifies* it (cosine 0.89, against 0.1 when it has to track
the day with forgetting), so on (C) the claim can be identification *and*
compensation; without the clock in the basis it is compensation by
tracking, and the per-class decomposition is not earned.

## The trained B0 policy (case33, C cell)

`calibrate.py --controller policy` on an 80-episode MATD3 checkpoint trained
at σ = 0 (B0: CR 0.873, greedy; the full 400-episode B0 will be tighter).
Same six episodes, `ladder_policy_matd3_C.csv`.

| σ | blind loss ×B0 / CR | intercept | PACT | oracle | \|d\| blind → PACT |
|---|---|---|---|---|---|
| 1 | 0.989 / 0.873 | 0.997 / 0.874 | 0.998 / 0.873 | 0.998 / 0.873 | 0.5 % → 0.6 % (inert) |
| 2 | 0.984 / 0.873 | 0.995 / 0.876 | 0.997 / 0.872 | 0.996 / 0.873 | 0.9 % → 1.1 % (inert) |
| **3** | **1.030 / 0.838** | 1.038 / 0.842 | **0.997 / 0.864** | 0.995 / 0.873 | 2.1 % → 1.5 % |
| 6 | 1.334 / 0.667 | 1.428 / 0.659 | **1.014 / 0.850** | 0.991 / 0.873 | 8.6 % → 3.2 % |

**The learned dispatcher feels the same physics far less than the droop
fleet does**, and this is the single most important thing to know before
budgeting training runs on case33.  The reaction feeds on *coherent* VAR
dispatch and is amplified by a controller that closes the loop on voltage:
the droop re-commands against every reaction and hunts (loop gain 1.13σ/3
at noon), while an open-loop RL dispatcher takes only the one-shot reaction
(gain 0.38σ/3) and the partially trained policy still had slack to absorb
it.  On the trained policy σ = 3 costs 3 % of reward and 3.5 CR points,
which PACT recovers to B0; σ = 6 (beyond-physical) costs 33 % / 21 CR
points and PACT recovers 95 % of it.  `intercept` never helps and hurts at
σ = 6 — the (C) signature again.

For the *training* experiment on case33 this means the gap between a blind
learner and PACT at σ = 3 will be modest and the disturbance is a function
of observables (channels + PV level) that a learner can in principle absorb
into its policy; the effect size is what the runs will measure.  The
22-inverter feeder is where the anchored physics bites a learner without a
stress test — see the next section.

## N-scaling: case141 and case322

The loop gain of the fleet's mutual reaction grows with the number of
inverters sharing segments, computed from the network files alone
(`conformance.py`, before any run):

| feeder | inverters | voltage levels | ρ(W) at σ = 1 | hunts at noon for | D_ref (mean, of range) |
|---|---|---|---|---|---|
| case33 | 6 | 12.66 kV | 0.378 | σ > 2.65 | 9.4 % |
| case141 | 22 | 12.5 kV | 0.675 | σ > 1.48 | 9.9 % |
| case322 | 38 | 20 kV + 0.4 kV | **1.186** | **σ > 0.84 — the default curve already hunts** | 10.7 % |

The σ ladder on case141 with the droop controller (3 episodes × 120 steps,
start hours 06 / 09 / 12; `ladder_droop_case141.csv`):

| σ | arm | loss ×B0 | CR | v_out | power flow diverged |
|---|---|---|---|---|---|
| 0 | blind (B0) | 1.000 | 0.994 | 0.43 % | 0 / 3 episodes |
| **1 (the anchor)** | **blind** | **43.2** | **0.646** | **29.8 %** | **2 / 3 episodes** |
| 1 | intercept | 43.5 | 0.608 | 33.5 % | 2 / 3 |
| 1 | **PACT** | **1.028** | **0.994** | **0.44 %** | 0 / 3 |
| 1 | oracle | 0.998 | 0.994 | 0.44 % | 0 / 3 |
| 2 | blind | 181.8 | 0.720 | 22.6 % | 3 / 3 |
| 2 | intercept | 183.8 | 0.770 | 18.1 % | 3 / 3 |
| 2 | PACT | 1.360 | 0.881 | 5.7 % | 0 / 3 |
| 2 | oracle | 0.996 | 0.994 | 0.44 % | 0 / 3 |

**At the standard's default curve, the 22-inverter droop fleet collapses
the power flow** (MAPDN's −200 "destroy" penalty and an early termination,
in two of three daytime episodes) — the volt-var interaction instability,
with no stress test.  PACT restores B0's controllable ratio exactly
(0.994) at a 3 % reward cost; the oracle recovers B0; deleting the peer
channels (`intercept`) collapses like blind.  `fit_gain` 0.79 and β cosine
0.28 on this feeder (the per-class decomposition is again weak; the
prediction is not).  At σ = 2 PACT still prevents every divergence but
loses 36 % of reward; the oracle is unaffected, so σ = 2 is inside σ\* and
the residual is the estimator's tracking floor.

Committed operating point for case141: **σ = 1, the anchor itself** (see
`conf/case141.yaml`).  It needs ~5× case33's compute per episode
(0.6 s/step).  case322 (38 inverters, 0.9 s/step, two voltage levels) is
exported and smoke-tested but not calibrated.
