# OPFData scale-up brief — the scalability axis, concretely

How the "scalability" axis of the five-axis evaluation gets real: stop
generating labels with pandapower on this CPU box and train on the
published benchmark. Facts below are from a web check on 2026-10-22/23.

## What OPFData is

- "OPFData: Large-scale datasets for AC optimal power flow with topological
  perturbations" (Lovett et al., arXiv 2406.07234).
- Public Google Cloud bucket: `gs://gridopt-dataset/` (LICENSE + README in
  the bucket). Based on PGLib-OPF cases, solved with Julia / PowerModels.jl
  / Ipopt / MUMPS.
- One JSON file per example: `grid` (OPF inputs), `solution`, `metadata`.
- Canonical split per grid: 270k train / 15k val / 15k test (0.9/0.05/0.05),
  i.i.d. and sequential.
- Per grid, 300k examples in each of two variants: **full-topology** (active
  and reactive loads varied independently ±20%) and **N-1 topology** (each
  sample additionally has one randomly disconnected branch or generator).
- 10 grids, up to **13,659 buses**. Nobody has trained or evaluated a GNN on
  the 13,659-bus grid (per the Oct 22 literature pass); largest credible
  GNN-OPF evaluation is CANOS at 10,000 buses. HH-MPNN used six grids:
  14, 30, 57, 118, 500, 2000.

## Why it matters for this project

1. **Replaces our label bottleneck.** Our pandapower `runopp` generation is
   the scaling wall (case57 already doesn't converge out of the box here).
   OPFData hands us solved labels to 13,659 buses.
2. **The N-1 variant is our Experiment 3 at scale.** Instead of hand-rolled
   N-1 outages on case14, the risk evaluation runs on 15k held-out N-1
   samples per grid — the tail metrics become publishable.
3. **Benchmark comparability.** Training on the same splits as CANOS /
   HH-MPNN makes our feasibility + risk numbers directly comparable, which
   the benchmarking checklist in `research-summary.md` demands.

## Suggested scale-up path (GPU workstation)

1. **case118 full-topology**: validate the pipeline — our GNN + Newton
   correction trained on OPFData's 270k train split, evaluated on its 15k
   test split. Sanity target: reproduce the qualitative Experiment 1–2
   picture (feasibility via correction, small cost gap).
2. **case118 N-1 split**: run the Experiment 3 harness on 15k N-1 test
   samples. This is the risk result no one reports with tails.
3. **500 → 2000-bus grids**, then the **13,659-bus grid** (untouched by any
   GNN paper). Memory/latency profiling of inference + correction at each
   step feeds the hardware axis.

## Open practical questions (unverified)

- Exact download size per grid/variant — check with a bucket listing
  (`gsutil du`) before pulling; JSON-per-example at 300k files will be
  heavy, so sync selectively (start: case118 only).
- A conversion script JSON → our tensor format (`x`, `y`, `edge_index`,
  `edge_feat`, Ybus) is needed — the natural next code task, and it should
  live in `src/opfdata_loader.py`.
- Alternatives if OPFData stalls: **PGLearn** (2025, up to 24,000 buses,
  10M samples, N-1 line perturbations) for raw scale; **PFΔ** (Oct 2025,
  859,800 solved AC *power flow* instances with N-1/N-2 and
  close-to-infeasible cases) for risk-methodology ideas — note it is PF,
  not OPF, so it doesn't replace OPFData for our task.

## Immediate next step

Write `src/opfdata_loader.py`: stream the case118 full-topology test split
from the bucket (or a local sync), convert to our tensors, and run the
Experiment 2 evaluation unchanged. That single script turns the
scalability axis from a plan into a pipeline.
