# Continuation-family bifurcation census

*Physics-side findings (feeds the physics writeup, not the ML deliverable).
Generated from the 10 continuation families traced into
`mini_results/continuation_family_*.json`.*

Each family is a periodic-orbit branch traced by pseudo-arclength continuation.
**Folds** are turning points where the branch reverses in `L`; **stability
changes** are points where `n_unstable` (the count of Floquet multipliers off the
unit circle) jumps — i.e. bifurcations.

| family | pts | folds | stability changes | L-range | log10 λ_max | n_unstable |
|---|---:|---:|---:|---|---|---|
| `jankovic2_b3` | 789 | 1 | 8 | 0.757–1.246 | 0.00–5.43 | 0–4 |
| `seed40` | 241 | 2 | **20** | 0.392–0.962 | 1.00–2.67 | 1–4 |
| `seed10` | 241 | 3 | 6 | 0.483–1.089 | 0.23–2.07 | 2–4 |
| `seed1` | 241 | 3 | 4 | 0.573–0.926 | 0.59–1.71 | 2–4 |
| `seed30` | 205 | 1 | 3 | 0.422–1.079 | 1.04–1.73 | 2–4 |
| `seed20` | 146 | 1 | 0 | 0.725–1.041 | 1.20–1.86 | 2 |
| `seed50` | 241 | 0 | 0 | 0.792–1.120 | 1.04–1.17 | 2 |
| `seed60` | 241 | 0 | 0 | 0.974–1.186 | 0.98–1.14 | 2 |
| `seed70` | 241 | 0 | 0 | 0.832–1.092 | 0.93–1.11 | 2 |

## Highlights worth a figure / paragraph

1. **The stable orbit.** The only fully stable point (`n_unstable = 0`,
   `λ_max = 1.000001`) across all ~2,400 sits on `jankovic2_b3` at
   **`(a, c, L) = (0.2468, −2.0335, 0.8305)`**. This is the "first stable
   L≠0 orbit," pinned precisely by continuation.

2. **A bifurcation hotspot.** `seed40` (word length k=13) undergoes **20 stability
   changes** over L∈[0.39, 0.96], with `n_unstable` swinging between 1 and 4 —
   the richest structure in the set; a natural candidate for a detailed
   stability-vs-L figure.

3. **Rigid vs structured families.** `seed50/60/70` (high word length, k=19–33) are
   stability-*rigid*: no folds, no stability changes, `n_unstable ≡ 2` across their
   whole L-range. Contrast with the low-k families (`seed1`, `seed10`) that fold
   and change stability repeatedly. Whether stability rigidity correlates with
   word length is a concrete question the data can answer.
