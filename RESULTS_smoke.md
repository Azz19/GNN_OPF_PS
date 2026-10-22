# Smoke test — case14, 200 samples, 10 epochs, CPU (Oct 22, 2026)

Purpose: verify the pipeline end-to-end, NOT to produce publishable numbers.
Single seed, tiny dataset, both models barely trained.

| Metric | FixedDepthGNN (8 layers) | DEQGNN (weight-tied, tol 1e-3) |
|---|---|---|
| Parameters | 107,522 | **22,082** (~5x fewer) |
| Test MSE (Vm, Va) | 1.40e-04 | **6.32e-05** |
| PQ power-balance violation, mean (pu) | 0.165 | **0.115** |
| PQ power-balance violation, max (pu) | 0.465 | **0.240** |
| Inference (ms/instance, CPU) | 1.84 | 7.47 |
| Fixed-point iterations | — | 23 (mean = p95 = max) |

Physics sanity check: the solver's own solutions score ~5e-06 pu violation
through the same Ybus scoring path, so the metric is sound.

## Honest read

- Encouraging: with ~5x fewer parameters the DEQ model reached lower MSE and
  roughly half the max violation after just 10 epochs. Consistent with the
  hypothesis, nowhere near proof.
- Both models remain far from feasible (violation ~0.1 pu is large). Expected
  at this training scale for a voltage-only surrogate; this is precisely
  what Experiment 2 (feasibility completion/correction) is for.
- The fixed point took exactly 23 iterations on every test instance, so no
  adaptive-compute signal (G4) is visible yet: iteration count is currently
  set by the tolerance, not by instance difficulty. Worth investigating
  (residual definition is a global norm over all hidden states) before
  claiming anything about adaptive compute.
- DEQ inference is ~4x slower here: 23 iterations of one block vs 8 layers.
  The parameter/memory advantage is real; the latency advantage is not, at
  least not naively — relevant to the hardware line (G7–G9).

Full log: `smoke_test.log`. Reproduce:
`.venv/bin/python src/run_experiment.py --case case14 --samples 200 --epochs 10`
