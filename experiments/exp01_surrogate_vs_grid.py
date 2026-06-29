"""Experiment 01: the headline comparison.

Stable-orbits-found vs evaluations, ML-guided active loop vs a non-adaptive
sampler, on the SAME budget and SAME seed set. This plot *is* the deliverable
(Section 1, Section 11 success metric).

Run:
    python -m experiments.exp01_surrogate_vs_grid --budget 60 --acq stable_lcb

Both arms share the seed catalogue as the labelled prior; only the *new* points
each arm chooses differ. Memory-safety: arms run strictly one after another
(never concurrently — Section 9).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt

import ml_config as cfg
from data.build_labels import build_seed_table
from surrogate.active_loop import run_active_loop
from baselines.uniform_grid import run_uniform


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=int, default=40,
                    help="True-label evaluations per arm.")
    ap.add_argument("--acq", default="stable_lcb",
                    choices=["uncertainty", "stable_lcb"])
    ap.add_argument("--baseline", default="sobol", choices=["sobol", "grid"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg.ensure_dirs()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print("=== Seeding both arms from the catalogue ===")
    seeds = build_seed_table()

    print(f"\n=== Arm A: active loop (acq={args.acq}) ===")
    active = run_active_loop(seeds, budget=args.budget,
                             acquisition=args.acq, seed=args.seed)

    print(f"\n=== Arm B: baseline ({args.baseline}) ===")
    base = run_uniform(budget=args.budget, mode=args.baseline, seed=args.seed)
    base_curve = base.attrs.get("stable_found_curve", []) if not base.empty else []

    # --- report ---
    summary = {
        "stamp": stamp,
        "budget": args.budget,
        "acquisition": args.acq,
        "baseline": args.baseline,
        "active_stable_found": active.stable_found_curve[-1] if active.stable_found_curve else 0,
        "baseline_stable_found": base_curve[-1] if base_curve else 0,
        "active_curve": active.stable_found_curve,
        "baseline_curve": base_curve,
    }
    out_json = cfg.RESULTS / f"exp01_{stamp}.json"
    out_json.write_text(json.dumps(summary, indent=2))

    # --- plot ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(1, len(active.stable_found_curve) + 1),
            active.stable_found_curve, "-o", ms=3, label=f"active ({args.acq})")
    if base_curve:
        ax.plot(range(1, len(base_curve) + 1),
                base_curve, "-s", ms=3, label=f"baseline ({args.baseline})")
    ax.set_xlabel("true-label evaluations (= variational integrations)")
    ax.set_ylabel("cumulative stable orbits found")
    ax.set_title("Exp01: ML-guided vs non-adaptive sampling")
    ax.legend()
    fig.tight_layout()
    out_png = cfg.RESULTS / f"exp01_{stamp}.png"
    fig.savefig(out_png, dpi=130)

    print("\n=== Result ===")
    print(f"  active   found {summary['active_stable_found']} stable in {args.budget} evals")
    print(f"  baseline found {summary['baseline_stable_found']} stable in {args.budget} evals")
    print(f"  wrote {out_json}")
    print(f"  wrote {out_png}")


if __name__ == "__main__":
    main()
