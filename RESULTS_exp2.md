# Experiment 2 (feasibility pivot) — case14, 200 samples, 10 epochs, CPU

Question: can we get the DEQ-GNN's predictions to actually satisfy the
power-balance equations, and what does it cost?

| Model | Output | MSE vs solver | Viol mean (pu) | Viol max (pu) | Cost gap |
|---|---|---|---|---|---|
| DEQ (MSE only) | raw | 6.73e-05 | 0.128 | 0.261 | +3.11% |
| DEQ (MSE only) | + Newton correction | 2.94e-04 | **0.0000** | **0.0000** | **+0.40%** |
| DEQ + physics loss λ=1 | raw | 7.68e-04 | 0.0436 | 0.0814 | +149.5% |
| DEQ + physics loss λ=1 | + Newton correction | 9.95e-04 | **0.0000** | **0.0000** | +145.2% |

Correction details (both models): Newton converged on 100% of test
instances, in a mean of 2.0 iterations, moving the state by at most
~0.015 pu in Vm and ~0.035 rad in Va on average.
Layer validation: applied to solver solutions, the correction takes
0 iterations and moves nothing; the derived-cost function reproduces
pandapower's reported cost exactly on solver states.

## What this shows

1. **The correction layer works, decisively.** Two Newton iterations from
   the network's predicted state reach a point that satisfies the
   power-balance equations to 1e-4 pu — feasibility by construction, at
   negligible inference cost. Feasibility is a solved problem in this
   pipeline; it was never the network's job alone.

2. **Feasibility is not optimality.** Correction *worsened* voltage MSE
   (6.7e-05 → 2.9e-04) while improving the cost gap (+3.11% → +0.40%).
   Newton finds a nearby feasible point, not the OPF optimum; the three
   metrics genuinely measure different things and all must be reported.

3. **The naive physics loss fails, instructively.** λ=1 cut raw violation
   ~3x (0.128 → 0.0436 pu) but destroyed optimality (cost gap +149%).
   Diagnosis: the loss constrains only PQ-bus balance, leaving the
   generator dispatch implied by the state unconstrained — the model
   learned states that balance at load buses with a terrible dispatch.
   A physics loss needs the inequality side too (voltage/gen limits) and
   a much smaller λ, or it trades away the thing OPF is for.

## Next iteration (queued)

- λ sweep (0.01 / 0.1) + limit penalties, OR drop the physics loss.
- The stronger move: make the correction layer differentiable (implicit
  function theorem at the Newton fixed point — it is itself an equilibrium
  problem) and train the network *through* it, DC3-style, so the network
  learns to predict states whose correction is the optimum, not just a
  feasible point. This unifies the DEQ framing with the feasibility layer:
  learned equilibrium proposing, physics equilibrium disposing.
- Report cost gap of raw predictions with care: the "cost" of an
  infeasible state is the cost of its implied dispatch, not a true
  operating cost; the corrected cost gap is the meaningful number.

Reproduce: `.venv/bin/python src/run_experiment2.py --case case14 --samples 200 --epochs 10 --lam 1.0`
Full log: `exp2.log`.
