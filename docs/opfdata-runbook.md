# OPFData GPU scale-up runbook

How to take the pipeline from case14-on-CPU to case118 (and beyond) on
Hay's GPU workstation, using the published OPFData benchmark instead of
pandapower label generation. Facts below were verified 2026-10-23 against
the public GCS bucket API and the OPFData paper (Lovett et al., arXiv
2406.07234); open questions are marked **[probe]** and are resolved by
running `src/schema_probe.py` on the workstation.

## Verified facts

- Bucket: `gs://gridopt-dataset/`, public, no auth needed (plain HTTPS
  download works too).
- Two variants:
  - FullTop (fixed topology, loads ±20%): `dataset_release_1/`
  - N-1 (each sample drops one random branch or generator, never the
    reference-bus gen, never disconnecting): `dataset_release_1_nminusone/`
- Sharding: 20 tar.gz shards per grid per variant, ~318 MB each for
  case118 (total ~6.4 GB per variant — do NOT sync everything; see below).
- File naming inside each tar: `example_{N}.json`, N = 0..299999.
- Canonical split (paper, sequential): train `example_0..269999`
  (270k), val `example_270000..284999` (15k), test `example_285000..299999`
  (15k). **[probe]**: confirm shard k holds examples k*15000..(k+1)*15000-1
  (i.e. test split ≈ shards 19 only); the probe prints the actual ranges.
- JSON skeleton (paper App. A):
  `{"grid": {"nodes": {"bus","generator","load","shunt"},
             "edges": {"ac_line": {"senders","receivers","features"},
                       "transformer": {"senders","receivers","features"},
                       "generator_link","load_link","shunt_link"},
             "context": ...},
    "solution": ..., "metadata": ...}`
- case118 dimensions (paper Table 1): 118 buses, 54 generators, 99 loads,
  14 shunts, 175 AC lines, 11 transformers.
- Solved with PowerModels.jl ACPPowerModel + Ipopt/MUMPS; cost is quadratic
  in Pg. Infeasible perturbations were discarded (so every example has a
  valid solution).

## Step 1 — download one test shard (workstation)

Only the test split is needed to validate the pipeline; training comes
later. Assuming the sequential shard mapping holds, the test split lives
in shard 19:

```bash
# with gsutil (pip install gsutil / gcloud CLI)
gsutil cp gs://gridopt-dataset/dataset_release_1/pglib_opf_case118_ieee_19.tar.gz .

# or plain HTTPS, no auth
curl -sSL -o pglib_opf_case118_ieee_19.tar.gz \
  "https://www.googleapis.com/download/storage/v1/b/gridopt-dataset/o/dataset_release_1%2Fpglib_opf_case118_ieee_19.tar.gz?alt=media"
```

~318 MB. The N-1 variant's shard 19 is the analogous path under
`dataset_release_1_nminusone/`.

## Step 2 — probe the schema (workstation)

```bash
python src/schema_probe.py --tar pglib_opf_case118_ieee_19.tar.gz --n-examples 3
```

This prints: member filename ranges (confirms the split mapping), the
exact per-node and per-edge feature names and vector lengths, units hints,
and the solution/metadata structure. Paste its output back here — it is
the input the loader is written against. It needs only the stdlib
(tarfile/json/urllib); no torch, no gsutil.

## Step 3 — write the loader (next code task)

`src/opfdata_loader.py` converts OPFData examples into **exactly** the
tensor contract `src/opf_data.py` already uses, so `run_experiment.py`,
`run_experiment2.py` and `evaluate_stress.py` run unchanged:

- `x`: [n, n_bus, 12] = [Pd, Qd, is_gen, is_slack, Pmin, Pmax, Qmin, Qmax,
  Vmin, Vmax, c2, c1] per bus, pu. Loads must be aggregated per bus
  (99 loads over 118 buses); gen limits/costs aggregated per gen bus.
- `edge_index`: [2, n_edges], both directions; `edge_feat`: [n_edges, 4] =
  [r, x, b, rate] pu, from ac_line + transformer features.
- `y`: [n, n_bus, 2] = [Vm, Va(rad)] per bus from `solution`.
- `cost`: [n] total generation cost (metadata or recomputed from solution
  Pg and the cost coefficients — report which).
- `Ybus`: [n_bus, n_bus] complex, built from the branch parameters
  (needed by the Newton correction layer and `pq_violation`).
- Bus indexing: OPFData node arrays are 0-based positional; verify against
  the probe (PowerModels-origin data is often 1-based — the probe reports
  index ranges; the loader must map to 0..n-1 consistently).

Scaling note for later: a streaming `torch.utils.data.IterableDataset`
over the tar members avoids extracting 300k JSONs to disk; start with
extract-then-load for the 15k test split only.

## Step 4 — run the Experiment 2 evaluation unchanged (workstation, GPU)

```bash
python src/run_experiment2.py --case case118 --opfdata <dir> --device cuda
```

Sanity targets: Newton correction converges ~100% with violation → 0 pu
on OPFData's own labels (validates the loader against solver truth), then
the trained GNN evaluated exactly as in `RESULTS_exp2.md`.

## Step 5 — the actual scale-up (after the pipeline validates)

1. Train on the 270k FullTop train split (case118), evaluate on 15k test.
2. N-1 variant, 15k test split — the Experiment 3 harness at publishable
   scale; this is the tail-risk result the literature doesn't report.
3. 500 → 2000-bus grids, then the 13,659-bus grid (no GNN paper has
   touched it). Profile inference + correction latency/memory at each
   step for the hardware axis.

## Open questions (for the probe / first workstation run)

- Exact feature key names and units inside each node/edge dict.
- Structure of `solution` (per-bus vm/va? per-gen pg/qg?) and `metadata`
  (cost present?).
- Shard↔split mapping (assumed sequential; probe prints filename ranges).
- Whether dropped components in N-1 examples are removed from the arrays
  entirely (paper says yes) — the loader must handle per-example graph
  sizes if so.
