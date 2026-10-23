# OPFData loader validation — case14 shard 0 (Oct 23, 2026)

`src/opfdata_loader.py` converts OPFData shards to the repo's tensor
contract. Schema decoded empirically from the data (see the loader
docstring): generator cost columns verified against `metadata.objective`
over 500 examples (max error 2e-13); qmax/qmin/pmax columns confirmed
against empirical solution ranges.

Validation on real OPFData labels (not our own generated data):

- OPFData's own solutions score **max 5.0e-06 pu** PQ power-balance
  violation through the loader's Ybus — the branch/tap/shunt conversion
  (including the [bs, gs] shunt order) is correct.
- Newton correction applied to OPFData labels: **0 iterations, zero
  movement** — the correction layer agrees with the benchmark's solver.
- End-to-end: FixedDepthGNN trained on 320 OPFData examples (5 epochs,
  CPU) → test MSE 2.9e-04, raw violation 0.34 pu, **corrected violation
  3.0e-06 pu, Newton converged 80/80**. The feasibility result from
  Experiment 2 reproduces on the published benchmark's data.

Next: case118 (FullTop train/test shards + N-1 test shard) five-axis run.
