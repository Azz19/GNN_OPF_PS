# Experiment 1 — Does iterating to equilibrium beat iterating a fixed number of steps?

**Research question.** At matched parameter count, does a weight-tied GNN processor
iterated to a fixed point (DEQ-GNN, IGNN-style) predict AC-OPF bus voltages more
accurately and more feasibly than a conventional fixed-depth GNN — and what does
the fixed-point iteration count look like across easy vs. hard instances?

This is gap **G1** (no DEQ applied to AC-OPF/PF) with **G4** (residual-driven
adaptive compute) and **G9** (iteration-count / tolerance reporting) attached.

## Design

- **Task.** Surrogate AC-OPF: given per-bus loads and grid data, predict
  (Vm, Va) at all buses. (First experiment is deliberately the simple surrogate
  task; feasibility-completion and warm-starting are Experiments 2 and 3.)
- **Models.** Identical encoder/decoder. Baseline: 8 weight-untied MP layers.
  DEQ-GNN: 1 weight-tied MP block iterated to tolerance 1e-3 (max 50 iters),
  trained by implicit differentiation. Report parameter counts for both —
  the DEQ model should be far smaller; also run a parameter-matched wider
  baseline as a control.
- **Data (scaffold).** `src/opf_data.py` generates solved instances with
  pandapower by scaling loads uniformly in [0.8, 1.2]. Cases: case14 (smoke),
  case57, case118 (main), case300 (scale check).
- **Data (publishable).** Migrate to OPFData (FullTop + N-1 splits) and/or
  PGLib-OPF generation, per the gap map's evaluation playbook, before drawing
  conclusions. Also add per-load (non-uniform) perturbations and N-1 topology
  splits for the generalisation tests.
- **Splits.** i.i.d. test; plus two stress splits: (a) load scale outside the
  training range (1.2–1.3), (b) N-1 line outages (zero-shot).

## Metrics (report all, mean + tail)

1. Optimality: MSE of (Vm, Va); derived generation-cost gap % vs solver cost.
2. Feasibility: power-balance violation at PQ buses (max / mean, pu) computed
   through the true Ybus equations; voltage-limit violation rate.
3. Compute: inference ms/instance; training time; peak memory if available.
4. DEQ-specific: fixed-point iteration count distribution (mean / p95 / max),
   tolerance-vs-accuracy curve, iteration count vs. instance difficulty
   (load scale) — the adaptive-compute evidence for G4.

## Controls and reviewer-proofing (from the gap map)

- Include an **FCNN/MLP control** (Falconer & Mones found limited merit in
  topology on their systems — show the GNN earns its keep on yours, or report
  honestly if it doesn't).
- Compare against **DeepOPF-style predict-and-reconstruct** as a second
  baseline when moving to the publishable dataset.
- If any warm-start claim is later made, use IPOPT's **true default
  (bound-midpoint) start** as the baseline, not a flat start (WARP critique).

---

# Experiment 2 — feasibility pivot (added Oct 22, 2026, after Exp 1 smoke test)

**Decision (Hay, Oct 22):** pivot from raw accuracy to feasibility — make the
fixed point enforce the power-balance equations.

Components (`src/feasibility.py`, `src/run_experiment2.py`):
1. **Physics loss**: differentiable PQ-bus power-balance mismatch (torch,
   complex Ybus) added to the loss with weight λ.
2. **Newton correction layer**: at inference, derive PV setpoints from the
   prediction, hold PV/slack Vm, and run polar Newton-Raphson on the
   remaining state until mismatch < 1e-4 pu. Feasibility by construction.

**Smoke-scale findings** (full write-up in `RESULTS_exp2.md`): the correction
layer is decisive — 100% convergence, mean 2.0 iterations, violation →
0.0000 pu, cost gap +3.11% → +0.40% for the MSE-trained model — while
worsening voltage MSE, proving feasibility and optimality must be reported
separately. The naive physics loss (λ=1) failed: violation improved ~3x but
cost gap exploded to +149% because PQ-only balance leaves the implied
generator dispatch unconstrained.

**Next iteration:** λ sweep with limit penalties, and the principled version —
differentiate through the Newton fixed point (implicit function theorem) and
train the network through the correction, DC3-style.

## Success / kill criteria

- Proceed if DEQ-GNN matches or beats the fixed-depth baseline on violation
  and MSE at fewer parameters, OR shows a clean adaptive-compute signature
  (iterations grow with instance difficulty while accuracy holds).
- Revisit the design if the fixed point fails to converge on >5% of instances
  at case118 scale, or training is unstable without Jacobian regularisation —
  in which case Experiment 1b adds monDEQ-style monotone parameterisation or
  Jacobian regularisation (Bai et al., ICML 2021) and reports the difference.
