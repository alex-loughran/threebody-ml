# threebody-ml

Classical-ML layer on top of the [three-body orbit hunter](../PythonProject1)
(`threebody-physics`). Separate git repo by design — the physics engine stays a
stable importable library; ML experiments stay out of its history.

**Goal (the deliverable):** a *triage-grade* stability surrogate driving an
active-learning loop, measured against a non-adaptive baseline on the same
budget — "stable-orbits-found per CPU-hour, ML-guided vs uniform." Classical
models (Gaussian Process + gradient boosting), not deep learning: inputs are
2–3D and labels are scarce/expensive. See `docs/ml_project_kickoff.md` in the
physics repo for the full rationale.

## Run it in the terminal, not in PyCharm

This is a 16 GB / 8-core machine and label sweeps launch the same machinery that
has OOM-killed it twice. Run heavy sessions in a plain terminal; if you must edit
in an IDE, close it before a big run. `memguard` (shipped with the physics
package) is the safety net — it is imported and installed automatically by every
entry point here, and the import is made to **fail loud** if the engine isn't
installed editable.

## Setup

```bash
# 1. Install the physics engine editable into THIS environment (gives you the
#    physics modules AND memguard):
pip install -e ../PythonProject1

# 2. Install this repo + optional extras:
pip install -e .                 # core: sklearn GP, scipy, pandas, matplotlib
pip install -e '.[trees]'        # + xgboost / lightgbm comparators
pip install -e '.[parquet]'      # + pyarrow (parquet label tables)
```

Set `THREEBODY_PHYSICS_REPO` if the physics repo isn't at the default
`~/PycharmProjects/PythonProject1`.

## Layout

```
threebody-ml/
  ml_config.py            # paths (env-overridable) + (a,c,L) bounds + threshold
  data/
    build_labels.py       # (a,c,L) -> newton_refine_bhh -> analyse_orbit -> table
    load_scans.py         # read physics-repo RPF scan .npz into feature arrays
  surrogate/
    gp.py  trees.py       # GP (uncertainty for acquisition) + GBM comparators
    active_loop.py        # train -> propose -> label -> retrain (the deliverable)
  baselines/
    uniform_grid.py       # the comparison arm (Sobol / regular grid)
  experiments/
    exp01_surrogate_vs_grid.py
  results/                # gitignored data; tracked reports/*.md
```

## Quickstart

Always run from the repo root so the flat-layout packages import cleanly.

```bash
# Rebuild the labelled pool from pre-traced continuation families (no physics):
python -m data.load_families

# Active-learning-vs-random benchmark (in-memory, fast):
python -m experiments.exp02_active_vs_random --budget 160 --repeats 6

# Cross-family extrapolation test:
python -m experiments.exp03_cross_family

# (Re)generate stability labels for the 75 catalogue seeds (does physics integ.):
python -m data.build_labels --seeds
```

## Status — read this first if resuming

**Track A (classical surrogate) is characterised; findings in
[`reports/track_a_findings.md`](reports/track_a_findings.md).** Headline: a cheap
surrogate predicts `log10(λ_max)` from `(a,c,L)` at **R²≈0.996 within known
families**, but two honest negatives bound it — active learning gives **no
advantage over random** on the diverse 10-family pool (the early single-curve 4×
was an artifact), and it **does not extrapolate to unseen families** (hand-crafted
physics features E/T help only ~7%).

**Data engine:** continuation families (`data/load_families.py` reads
`<physics-repo>/mini_results/continuation_family_*.json` — 10 families, ~2,400
on-manifold points already traced and on disk). Random `(a,c,L)` sampling was
abandoned (only ~6% refine to an orbit). `ml_config.BOUNDS` is set to the real
seed extent.

### Where to pick up
1. **Representation learning (the real fix for extrapolation).** Learn features
   from the *dynamics* (trajectory / shape-sphere), not raw `(a,c,L)`. This is the
   deep-learning / Neural-ODE thread; scalars have been shown insufficient
   (`reports/track_a_findings.md`, Result 4).
2. **Physics writeup input:** [`reports/bifurcation_census.md`](reports/bifurcation_census.md)
   — the stable L≠0 orbit is pinned at `(a,c,L)=(0.2468,-2.0335,0.8305)`; `seed40`
   is a 20-stability-change bifurcation hotspot.

**Deprecated:** `experiments/exp01_surrogate_vs_grid.py`, `baselines/uniform_grid.py`,
and the Sobol pool in `surrogate/active_loop.py` assume the abandoned
random-sampling data engine — kept for history, not the current path.

Superseding context and rationale live in the user's project memory
(`project_threebody_ml.md`).
