# Track A — a classical stability surrogate for the three-body problem

**Question.** Can a cheap machine-learning surrogate predict the *stability* of a
periodic three-body orbit — its maximum Floquet multiplier `λ_max` — from its
low-dimensional parameters `(a, c, L)`, so that expensive variational
integrations can be avoided or prioritised?

**Short answer.** Yes within the explored region (R²≈0.996), and as a *triage*
tool it saves ~9× the expensive analysis. Two honest negatives bound it — active
learning gives no advantage over random, and it does not extrapolate to novel
families — but features learned from the *dynamics* (not the raw parameters)
recover most of the cross-family gap. This report is deliberately even-handed
about what did and did not work.

---

## Setup

- **Target:** `y = log10(λ_max)`, where `λ_max` is the largest Floquet multiplier
  magnitude of a periodic orbit. `λ_max ≈ 1` ⇒ (marginally) stable; larger ⇒ more
  unstable. Values span **1 → 2.7×10⁵** (log10 0 → 5.4).
- **Features:** `(a, c, L)` plus cheap derived terms (`a·c`, `L−a·c`, `sign c`).
- **Labels:** each label is one variational integration of the 156-component
  extended state (orbit + 12×12 monodromy). Expensive — the whole point of a
  surrogate is to spend fewer of them.
- **Data engine:** *continuation*, not random sampling. Random `(a,c,L)` refine
  to a valid orbit only **~6 %** of the time, and the failures are the expensive
  (near-collision) ones. Continuation walks a known family, emitting one
  on-manifold labelled point per cheap step. Pool: **~2,400 points across 10
  families.**

---

## Result 1 — the surrogate interpolates extremely well

5-fold cross-validation on the full pool (target std = 1.63):

| Model | RMSE on `log10(λ_max)` |
|---|---|
| RandomForest | **0.099** |
| Gaussian Process (isotropic Matérn) | 0.104 |
| HistGradientBoosting | 0.109 |
| GP with per-feature (ARD) length-scales | 0.183 (worse — derived features are collinear) |

RMSE ~0.10 against a spread of 1.63 is **R² ≈ 0.996**. Within the region covered
by known families, `(a,c,L)` is almost fully predictive of stability. As a
*triage* tool — rank/skip candidates before spending integrations — this works.

## Result 2 (negative) — active learning does **not** beat random

The original plan was an active-learning loop: fit a GP, label where it is most
uncertain, repeat, and beat a uniform baseline on labels-to-accuracy.

On a *single* family it looked spectacular (~4× fewer labels). **On the diverse
10-family pool that advantage vanished** — active learning is indistinguishable
from (slightly worse than) random selection:

![active vs random on the diverse pool](figures/al_vs_random_diverse_pool.png)

Why: uncertainty-only acquisition over-samples family *endpoints* (highest
variance, least representative), while random naturally covers the manifold. For
*global* regression accuracy, random is a strong baseline — a known result in the
active-learning literature. The single-curve win was an artifact of one smooth,
dominant family, and the diversity test was built specifically to expose that.

## Result 3 (negative) — extrapolation to novel families fails

Leave-one-family-out (train on 9 families, predict the 10th):

| Held-out family | RandomForest RMSE |
|---|---|
| `seed50`, `seed60` (near training regions) | 0.10 – 0.16 |
| `seed1`, `seed20`, `seed30` | 0.17 – 0.30 |
| `seed10`, `seed40` (complex bifurcations) | 0.45 – 0.98 |
| `jankovic2_b3` (out-of-distribution, `λ_max` to 2.7×10⁵) | **3.18** |
| **mean** | **0.62** |

Extrapolation is 6–10× worse than interpolation, and *worse than a mean-predictor*
for the out-of-distribution family. The surrogate learns the local
`(a,c,L)→stability` law near families it has seen; it does **not** generalise to
genuinely novel topologies. It is triage-grade, not a discovery oracle.

---

## Result 4 — do transferable physics features help extrapolation? (a little)

Before reaching for deep learning, the cheap thing to try is *hand-crafted*
transferable features. Adding energy `E` and period `T` (both available per
point, non-leaky) to the leave-one-family-out test:

| feature set | mean extrapolation RMSE |
|---|---|
| `(a,c,L)` + derived | 0.657 |
| + energy `E` | 0.626 |
| + energy `E`, period `T` | **0.611** |

A real but marginal ~7 % gain, with solid improvements on *near-distribution*
families (seed20 0.20→0.06, seed60 0.26→0.08). But the out-of-distribution
family `jankovic2_b3` stays catastrophic (~3.5). **Hand-engineered scalars help
at the margin and cannot fix novel-topology extrapolation** — which is the
evidence that the real lever is a representation learned from the dynamics, not
more feature engineering.

## Result 5 — dynamics features *do* improve extrapolation (the positive result)

Scalars weren't enough (Result 4), so we extracted **rotation/scale-invariant
features from each orbit's trajectory** (`data/build_traj_features.py`):
close-approach severity (`d_min`), extent, size-pulsation (moment-of-inertia
CV), and shape-sphere path/topology descriptors (equator crossings, path
spread). These describe the orbit's intrinsic geometry rather than the `(a,c,L)`
that merely index it.

Leave-one-family-out, RandomForest:

| feature set | interpolation (CV) | extrapolation (mean) | typical novel family* |
|---|---|---|---|
| `(a,c,L)` + derived | 0.102 | 0.656 | 0.350 |
| **+ trajectory features** | **0.065** | **0.523** | **0.195** |

\*mean over the 9 in-distribution families (excludes `jankovic2_b3`).

Per-family the gains are large — seed70 0.57→0.14, seed60 0.26→0.04, seed20
0.20→0.05; 6 of 10 families now extrapolate below 0.15 RMSE. **Representation, not
parameters, is what transfers.**

The exception is honest and instructive: `jankovic2_b3` stays broken (~3.5)
because its `λ_max` reaches 2.7×10⁵ — a *target* range 100× beyond any other
family. That is out-of-distribution in the label, a data-coverage limit no
feature representation can fix. It motivates the deep-learning thread (learned
embeddings of the dynamics) *and* bounds what any surrogate can promise.

## Result 6 — as a triage tool it is genuinely useful (within the explored region)

RMSE was never the real objective; *ranking* is. Reframed as triage — rank orbits
by predicted stability so expensive exact (variational-monodromy) analysis is
spent on the promising (least-unstable) ones:

| features | interpolation Spearman | extrapolation Spearman |
|---|---|---|
| `(a,c,L)` | 0.988 | 0.219 |
| + trajectory | 0.994 | −0.045 |

**Interpolation triage is excellent:** analysing only the **~10 %** of candidates
the surrogate ranks most promising catches **90 %** of the truly-interesting
(bottom-decile `λ_max`) orbits — a ~9× saving in expensive analysis (see
`reports/figures/triage_efficiency.png`).

**Extrapolation triage fails**, and instructively: trajectory features cut the
error *magnitude* across families (Result 5) but do **not** fix cross-family
*rank order* (Spearman ≈ 0), and the out-of-distribution `jankovic2_b3` dominates
the global statistic. So the honest operating envelope is: **triage the rest of a
region after labelling part of it; do not triage genuinely novel regions.**

## Honest conclusion

A cheap classical surrogate predicts three-body orbital stability to R²≈0.996
**within known family regions**, and as a triage tool catches 90 % of the
interesting orbits while analysing only ~10 % — a ~9× saving. Two things it does
*not* do: active learning gives no advantage over random at these budgets (the
early single-curve 4× was an artifact), and it does not extrapolate to unseen
families from parameters alone.

The lever for that last gap is **representation, not parameters**: features
describing the orbit's *dynamics* (close-approach, size-pulsation, shape-sphere
path/topology) cut mean cross-family error from 0.66 → 0.52 (0.35 → 0.20 off the
one out-of-distribution family) — validating that learned embeddings of the
dynamics are the right next thread (the deep-learning / Neural-ODE track). The
residual failure is a *data-coverage* limit (one family's `λ_max` is 100× beyond
the rest), which no representation can fix — only more data can.

*Reproduce:* `python -m experiments.exp02_active_vs_random` (AL benchmark),
`python -m experiments.exp03_cross_family` (extrapolation). Data engine:
`data/load_families.py`.
