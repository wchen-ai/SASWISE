# Consistency-Loss Balancing — Problem Statement

> Design rationale for the `ConsistencyBalance` mechanism in
> [`src/adapters/finetune.py`](../src/adapters/finetune.py). This
> document describes *why* picking a consistency-loss weight is hard;
> the *how* lives in the source code and in the
> [README](../README.md#how-λ-the-consistency-weight-is-balanced).

## Context

The SASWISE adopter fine-tunes a family of sub-models with a two-term
objective:

```
L_total = L_task + λ · L_cons
```

where

- **`L_task`** is the user's primary objective on a single assembled
  sub-model — cross-entropy, MSE, Dice, the Cox partial likelihood,
  whatever. Its magnitude is set by the *task*.
- **`L_cons`** is the disagreement between two sub-models built from
  different block configurations sampled at each step — KL of
  softmaxes, MSE of feature maps, JS divergence of probabilities.
  Its magnitude is set by *how diverse the sub-models currently are*.
- **`λ`** is a scalar that balances the two terms.

We need to choose `λ`. This document states why that is hard.

## The problem in one sentence

> **Given two losses with incommensurable units, time-varying
> magnitudes, asymmetric failure modes, no closed-form target metric,
> and a user requirement that they kick in at different times, choose
> a single scalar `λ(t)` that is robust across tasks, datasets, and
> architectures we have not seen.**

That sentence packs together at least seven distinct difficulties.
Each one alone would be enough to make a constant `λ` wrong; together
they make even a per-task hand-tuned `λ` wrong.

## 1. The two losses do not share units

`L_task` and `L_cons` measure different things:

- `L_task` is whatever the user picked: cross-entropy on logits, MSE
  on pixel values, Dice on segmentation masks. Its magnitude is set
  by the *task*.
- `L_cons` is the disagreement between two sub-models: KL of
  softmaxes, MSE of feature maps, JS divergence. Its magnitude is set
  by *how diverse the sub-models currently are*, which has nothing to
  do with the task.

Adding the two scalars together is dimensional nonsense in the same
way that *velocity + temperature* is dimensional nonsense. There is
no first-principles unit conversion. `λ` has to do the conversion
empirically, and the conversion factor is whatever the user wants the
*relative importance* of the two terms to be — which is itself not
given by the math.

## 2. The magnitudes change as training progresses

Even if you somehow guessed the right `λ` for step 0, it would be
wrong by step 1000:

- `L_task` shrinks as the model fits the data.
- `L_cons` may shrink (consistency loss is being minimised) or grow
  (variants are diverging from the consistency pull) — and *which* of
  those happens depends on what `λ` was, which is the thing we are
  trying to choose.

A constant `λ` therefore implements a *time-varying contribution
ratio* of consistency to total — even though the user specified
neither the schedule nor the contribution ratio. The implicit
schedule is determined by the loss dynamics, not by the user's intent.

## 3. There is no clean objective function for `λ` itself

For most hyperparameters you can in principle grid-search and look at
a downstream metric (val accuracy, calibration error, etc.). For `λ`
this is fundamentally weak:

- The thing we actually care about is **ensemble uncertainty quality**
  — calibration of predictive entropy, correlation of `std` with
  error, expected calibration error on held-out data. That metric is
  expensive (requires assembling many sub-models on the validation
  set and aggregating) and noisy.
- The cheap-to-observe quantities (`L_task`, `L_cons`) are *proxies*
  for what we want. They tell you how hard the model is fighting,
  not whether the resulting ensemble is well-calibrated.
- There is no analytical relationship between `λ` and downstream
  calibration that we can differentiate.

So you cannot just optimise `λ` directly. You have to choose it based
on *some other* property — magnitude, ratio, gradient norm, etc. —
and hope that property is a reasonable proxy.

## 4. Both extremes are catastrophic, in different and asymmetric ways

| `λ` is too small | `λ` is too large |
|---|---|
| Consistency gradient is dwarfed by the task gradient | Task gradient is dwarfed by the consistency gradient |
| Sub-models drift apart freely | Sub-models collapse onto one another |
| Ensemble diversity becomes uncontrolled randomness | Ensemble has no diversity at all |
| Uncertainty estimates are noise, not signal | Uncertainty estimates are zero everywhere |
| Per-variant accuracy is fine | Per-variant accuracy is poor (model never fit the data) |
| **You can't tell from `L_task` alone that anything is wrong** | **You can't tell from `L_cons` alone that anything is wrong** |

Note the asymmetry in the bottom row: each failure mode is *invisible
from inside the other loss term*. So a mechanism that monitors only
one loss can't detect either failure.

## 5. The user has a temporal-ordering requirement that the math does not naturally express

A common (and correct) intuition is *"task loss is the main part;
after that we need consistency"*. This is a statement about the
**time evolution** of training, not the steady state. A naive
constant `λ` cannot encode "first this, then that" because a constant
has no time dependence.

So the strategy must answer two questions, not one:

1. *What* should the steady-state balance be?
2. *When* does the steady-state regime begin, and how does training
   transition into it from the cold-start regime?

These are independent. You can have the right steady-state balance
with the wrong transition (model never fits the data because
consistency hits hard at step 0), or the right transition with the
wrong steady-state balance (warm-up ends and `λ` jumps to a value
that is way too small/large for the actual loss scales).

## 6. The cold-start chicken-and-egg

At step 0:

- You have zero observations of `L_task` and `L_cons`.
- Any data-driven choice of `λ` requires observations.
- Any observation requires a forward+backward pass.
- Any forward+backward pass requires a `λ`.

So the very first step must use a `λ` chosen *without* any data.
After step 0 you have a single-sample estimate of each loss, which
is high-variance. After step 100 you have a smoother estimate. The
strategy has to decay gracefully from "no-data" to "lots-of-data"
without producing a discontinuity.

## 7. Task and architecture dependence

The "correct" balance depends on:

- **Loss function choice.** KL-divergence of softmax `L_cons` is
  `~1000×` smaller than MSE-on-logits `L_cons` for the same model.
  Same task, same architecture, completely different `λ`.
- **Model size.** A larger model has finer-grained gradients; the
  same `L_cons` produces a different update magnitude.
- **Number of blocks `B` and variants `V` in the warehouse.** More
  blocks → each variant covers fewer parameters → `L_cons`
  magnitude shifts.
- **Dataset difficulty.** A model on MNIST reaches `L_task ≈ 0.05`
  quickly; a model on ImageNet sits at `L_task ≈ 2.0` for a long
  time. Same target ratio, very different `λ`.
- **Whether the user supplied a custom `consistency_fn`** or fell
  through to the default MSE.

There is no universal constant `λ`, and worse, there is no universal
*function* `λ(L_task, L_cons)` that works for every combination of
these axes — unless that function expresses a property that is
itself invariant to all of them.

## The combined difficulty

The problem is therefore not "pick a number". It is:

> Choose a `λ(t)` whose value at every step is determined by **a
> property of the two losses that is invariant to scale,
> architecture, dataset, task, and time** — and that **transitions
> from a cold-start regime into a steady-state regime in a way the
> user can anchor to a single intuitive setting** ("task first, then
> consistency") — and that **does not blow up under degenerate
> observations** (one loss approaches zero, one loss spikes, etc.) —
> and that **lets the user override every default** when their
> problem deviates from the assumptions.

Every part of the strategy implemented in
[`src/adapters/finetune.py`](../src/adapters/finetune.py) — the
ratio formulation, the EMAs, the linear warm-up, the lag-1 update,
the clipping bounds, the `noise_std` knob at `init` time — exists to
address one of these difficulties specifically. Skipping any one of
them makes the mechanism break on a particular class of inputs.

## Final ratio answer

If we follow the recipe **"task loss first, then consistency"**, the
question of *what the long-run loss ratio actually is* depends on what
happens **after** the transition out of the warm-up regime. There are
two clean interpretations.

### Case 1 — Keep both losses with a target ratio `r`

The total loss stays

$$
L_{\text{total}} \;=\; L_{\text{task}} \;+\; \lambda_t \cdot L_{\text{cons}}
$$

and `λ_t` is chosen so that the weighted consistency share approaches
a fixed target ratio `r`. After long training the asymptotic
contributions are

$$
\frac{\lambda_t \cdot L_{\text{cons}}}{L_{\text{task}} + \lambda_t \cdot L_{\text{cons}}} \;\longrightarrow\; r,
\qquad
\frac{L_{\text{task}}}{L_{\text{task}} + \lambda_t \cdot L_{\text{cons}}} \;\longrightarrow\; 1 - r.
$$

So the final ratio is

$$
\text{task} : \text{consistency} \;=\; (1 - r) : r.
$$

Examples:

- `r = 0.1`  →  final ratio is **90 % task / 10 % consistency** *(default)*.
- `r = 0.2`  →  final ratio is **80 % task / 20 % consistency**.
- `r = 0.5`  →  final ratio is **50 % task / 50 % consistency**.

In this interpretation, *"accuracy first, then consistency"* only
determines the **transition** out of the cold-start regime — *not*
the steady-state ratio. The steady-state ratio is whatever the user
chose for `r`.

This is what `ConsistencyBalance(adaptive=True, target_ratio=r)`
implements. The closed-form derivation of `λ_t = (r / (1-r)) ·
(EMA[L_task] / EMA[L_cons])` is exactly the value that makes the
contribution ratio above hold in expectation.

### Case 2 — Hard switch to consistency only

If we *stop* optimising the task loss after the warm-up and train
only with the consistency loss, then asymptotically

$$
\text{task} : \text{consistency} \;=\; 0 : 1.
$$

This is the degenerate end of the spectrum: the model is no longer
fitting the data, only enforcing agreement between sub-models. Useful
in principle as a separate "agreement-tuning" phase, but it forfeits
the data-fitting work and is not what we want by default.

### What we use, and why

| Interpretation | Final ratio | Behaviour |
|---|---|---|
| **Case 1**, `r = 0.1` *(default)* | `(1 - r) : r` = `0.9 : 0.1` | Task loss stays the dominant signal forever; consistency contributes a small but non-vanishing fraction. |
| **Case 1**, `r = 0.5` | `0.5 : 0.5` | Equal weighting; reasonable when consistency is as important as fit (e.g. very small base ensemble). |
| **Case 2**, hard switch | `0 : 1` | Only enforces agreement; data-fitting is frozen. |

The default `r = 0.1` is chosen so that the **task loss is always at
least 90 %** of the total — matching the natural intuition that the
accuracy objective is the main thing and consistency is a *regulariser
on top of it*, not a competing objective. The steady-state ratio is
`0.9 : 0.1`; the warm-up controls only how the model gets there.

If you want something more aggressive (e.g. consistency contributing
20 % or 30 %), set `--target-ratio 0.2` or `--target-ratio 0.3`. If
you want a hard switch instead, train in two passes: first with
`--target-ratio 0.0001` (effectively task-only), then re-run with
`--target-ratio 0.999 --warmup-steps 0` (effectively consistency-only).

## See also

- [`src/adapters/finetune.py`](../src/adapters/finetune.py) — the
  `ConsistencyBalance` dataclass and `_ConsistencyBalancer` class
  that implement the strategy. The `λ` formula in
  `_ConsistencyBalancer.current_weight()` is the closed-form
  derivation of the Case 1 ratio above.
- [README — How `λ` is balanced](../README.md#how-λ-the-consistency-weight-is-balanced)
  — user-facing description of the mechanism, its defaults, and its
  CLI flags.
- [`scripts/adopt_checkpoint.py`](../scripts/adopt_checkpoint.py) —
  CLI surface that exposes every balancing knob (`--target-ratio`,
  `--warmup-steps`, `--ema-decay`, `--min-weight`, `--max-weight`,
  `--consistency-mode`).
