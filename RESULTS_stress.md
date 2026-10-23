# Experiment 3 — risk-axis stress evaluation

Case case14; predictors trained 10 epochs on 160 nominal-band (0.8-1.2x load) samples, seed 1, device cpu. Stress splits are out-of-distribution by construction.

## Split A — heavy loads (1.05x to 1.30x)
Solver itself failed on 0 stressed samples — the risk frontier where even the oracle has no answer.

### FixedDepthGNN
- corrected viol_max: mean 0.0000, p99 0.0000, max 0.0000 pu
- Newton converged 100.0% (iters mean 2.0, max 2)
- corrected cost gap: mean +0.38%, max +0.49%

### DEQGNN
- corrected viol_max: mean 0.0000, p99 0.0000, max 0.0000 pu
- Newton converged 100.0% (iters mean 2.0, max 2)
- corrected cost gap: mean +0.36%, max +0.64%

## Split B — N-1 line outages (nominal load)
15 branches tested, 0 skipped (islanding / solver infeasible), 0 solver failures on tested branches.

### FixedDepthGNN
- corrected viol_max: mean 0.0000, p99 0.0000, max 0.0000 pu
- Newton converged 100.0%
- corrected cost gap: mean +18.58%, max +182.99%
- worst branches (by corrected cost-gap tail): line 1: cost gap +183.0%, conv 100%; line 2: cost gap +53.5%, conv 100%; line 3: cost gap +19.7%, conv 100%; line 4: cost gap +12.5%, conv 100%; line 5: cost gap +3.5%, conv 100%

### DEQGNN
- corrected viol_max: mean 0.0000, p99 0.0000, max 0.0000 pu
- Newton converged 100.0%
- corrected cost gap: mean +4.82%, max +35.71%
- worst branches (by corrected cost-gap tail): line 2: cost gap +35.7%, conv 100%; line 3: cost gap +12.9%, conv 100%; line 4: cost gap +6.2%, conv 100%; line 0: cost gap +5.5%, conv 100%; line 5: cost gap +4.4%, conv 100%

## Caveats
- case14, toy training (10 epochs): validates the harness and the tail-reporting methodology, not a publishable risk claim.
- Newton correction needs a converged power flow; on contingencies where it stalls, the raw prediction is what you have — count those.
- Stressed-load samples re-solve OPF per sample; solver failures at 1.25-1.30x are expected and are data, not bugs.
- Next: same harness on case118 with OPFData N-1 split, on the GPU.
