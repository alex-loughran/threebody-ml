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

```bash
# Label the 75 Jankovic catalogue seeds (one variational integration each):
python -m data.build_labels --seeds

# Sanity-check on-disk scan maps:
python -m data.load_scans

# Run the headline comparison (writes results/exp01_*.json + .png):
python -m experiments.exp01_surrogate_vs_grid --budget 40 --acq stable_lcb
```

Always run from the repo root so the flat-layout packages import cleanly.

## Status / open gates

Starter scaffold. Before trusting a discovery run, tighten `ml_config.BOUNDS`
against the real Jankovic/BHH parameter extent (`build_labels --seeds` prints
the seed extent). Track B (discovery selection function) and the deferred DL
track are not built yet — see the kickoff doc, Sections 6–7 and the decision log
(Section 11). Korn / CERN feedback is still pending and may redirect the aim;
don't over-invest in ML infrastructure before it lands.
