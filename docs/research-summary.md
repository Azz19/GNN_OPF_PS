# Research summary — GNN for OPF: where the open problems are

Synthesis of two sourced literature passes (Oct 22, 2026): a DEQ+GNN+AC-OPF
gap map (44 papers) and an open-problems map for GNN-for-OPF specifically.
Full reports live outside this repo (Muse workspace research notes);
key claims below carry their main sources.

## Established

- **Fixed topology: a tuned MLP ≈ GNN.** Falconer & Mones (IEEE TPWRS 2023)
  found limited merit in topological information for OPF on fixed grids;
  HH-MPNN's own large-scale tables (2025 preprint) confirm it. The GNN's
  demonstrable advantage is handling *variable* topology in one model.
- **Zero-shot N-1 transfer works on average** (<3% cost gap, HH-MPNN) **but
  fails on the costliest contingencies**, sometimes worse than a
  topology-blind baseline. Cross-grid zero-shot is poor. Pretrain +
  fine-tune is what actually works (fine-tuning on 3,000 labels beat
  training from scratch on the same labels 40x in HH-MPNN).
- **Guaranteed AC feasibility is still open for GNNs.** Best reported mean
  power-balance violations ~1e-4–1e-3 pu (full topology); guarantees in the
  literature attach to non-GNN models, DC-OPF, or repair steps.
- **Warm-start evaluation was reset in 2026.** Primal-only warm starts do
  not beat IPOPT's true midpoint default (WARP preprint; Taheri & Molzahn);
  the value is in predicting the full primal-dual-barrier state
  (oracle state: IPOPT iterations 23 → 3.3 on case118).
- **Nobody has trained/evaluated a GNN on OPFData's 13,659-bus grid.**
  Largest credible GNN-OPF evaluation: CANOS at 10,000 buses.

## Ranked dissertation-sized open problems

1. **Primal-dual GNN warm-starting** under the corrected protocol, beyond
   case118. Baselines: IPOPT midpoint default, DC warm start (presolve
   counted), primal-only GNN. Data: PGLearn duals, OPFData.
2. **GNN-native feasibility layer** (DC3-style completion + correction
   inside a heterogeneous GNN), repair cost counted end-to-end, tail
   (max / p99.9) violations reported. ← Experiments 1–2 in this repo are
   the seed: a classical Newton correction already gives exact feasibility
   in ~2 iterations on case14.
3. **"Falconer 2.0"**: controlled factorial study of when graph structure
   helps — vary locality, grid size, task; matched parameters and tuning
   budgets; modern hybrid GNNs vs tuned MLP/linear controls.
4. **Independent verification of cross-grid pretraining** (GridSFM / LUMINA
   claims are unreplicated and partly inconsistent) + fine-tuning
   sample-efficiency curves, incl. the unused 13,659-bus grid.
5. **Hardware-aware GNN-OPF inference**: end-to-end GPU latency / energy /
   throughput profiling (batch-1, incl. graph construction and repair)
   vs CPU and GPU IPOPT. The niche appears empty; fits an AI-and-hardware
   lab directly.

Untouched by GNNs: true coupled SC-AC-OPF, multi-period / storage OPF,
stochastic/robust OPF with guarantees, learned branching/screening.

## Benchmarking rules adopted for this repo

Same-machine solver timing; midpoint + DC warm-start baselines with
presolve cost counted; report max and tail violations, not means only;
never report a cost gap for an infeasible point without saying so; count
repair/correction cost end-to-end.
