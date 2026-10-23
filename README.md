# GNN for AC Optimal Power Flow — research scaffold

PhD research scaffold (Hay, AI & Energy Systems / AI-and-hardware lab).
Research line: open problems in GNNs for AC optimal power flow. Note
(2026-10-22): the DEQ (deep equilibrium) requirement was **dropped** —
it is no longer a constraint on the work. The `DEQGNN` model in `src`
is kept as an experimental comparison point, but the carrying-over
result is the **model-agnostic Newton power-flow correction layer**
(`src/feasibility.py`), which applies to any GNN-for-OPF predictor.
Background and positioning: the "Literature and Gap Map" page
(Oct 22, 2026), the Experiment 1 smoke test, and the Experiment 2
feasibility write-up.

## What's here

| File | Purpose |
|---|---|
| `EXPERIMENT.md` | Experiment 1 protocol: question, design, metrics, controls, kill criteria |
| `src/opf_data.py` | AC-OPF dataset generation with pandapower (load scaling → `runopp`), Ybus construction, power-balance violation scoring |
| `src/models.py` | Shared encoder/decoder; `FixedDepthGNN` baseline (8 untied MP layers); `DEQGNN` (weight-tied block iterated to a fixed point, implicit-differentiation backward via adjoint fixed-point solve) |
| `src/feasibility.py` | **Model-agnostic** feasibility tools: differentiable physics loss and the Newton power-flow correction layer |
| `src/device.py` | Device selection (`--device auto|cpu|cuda`); same scripts run on CPU and GPU |
| `src/run_experiment.py` | Trains both models, reports MSE, PQ-bus power-balance violation (pu), inference time, parameter counts, and DEQ iteration statistics |
| `src/run_experiment2.py` | Feasibility pivot: physics loss on/off × Newton correction on/off, with cost-gap and correction-distance reporting |
| `src/evaluate_stress.py` | **Risk axis**: trains both predictors on the nominal load band, then evaluates the corrected pipeline on out-of-distribution stress splits — heavy loads (1.05–1.30×) and N-1 line outages — reporting tail metrics (p99/max violation, correction convergence rate, cost gap) |
| `src/schema_probe.py` | Stdlib-only probe: prints the exact JSON schema of an OPFData shard (feature names, shapes, split mapping). Run on the GPU workstation after downloading one shard |
| `docs/opfdata-runbook.md` | GPU scale-up runbook: verified bucket/shard/split facts, download commands, tensor contract, and the loader spec for `src/opfdata_loader.py` (next code task) |
| `docs/opfdata-scaleup.md` | OPFData benchmark brief and the scale-up path for the scalability axis |

Message passing is plain-PyTorch (no torch-geometric dependency).

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# smoke test (case14, a few minutes on CPU):
.venv/bin/python src/run_experiment.py --case case14 --samples 400 --epochs 30
# main experiment (on the GPU workstation: --device cuda):
.venv/bin/python src/run_experiment.py --case case118 --samples 5000 --epochs 100 --device cuda
# feasibility experiment:
.venv/bin/python src/run_experiment2.py --case case14 --samples 200 --epochs 10 --device auto
# risk-axis stress evaluation (tail metrics under heavy loads and N-1 outages):
.venv/bin/python src/evaluate_stress.py --case case14 --samples 200 --epochs 10 --device auto
```

## Roadmap

1. **Experiment 1** (done, smoke scale): fixed-depth vs equilibrium processor, surrogate task — see `RESULTS_smoke.md`.
2. **Experiment 2** (done, smoke scale): feasibility pivot — physics loss + Newton correction layer (`src/feasibility.py`, `src/run_experiment2.py`). Correction reaches exact feasibility in ~2 iterations; naive physics loss fails on optimality — see `RESULTS_exp2.md`. The correction layer is **model-agnostic**: it bolts onto any GNN predictor, not just DEQ-GNN.
3. **Experiment 3** (done, smoke scale): risk-axis stress evaluation — heavy-load and N-1 stress splits with tail-metric reporting (`src/evaluate_stress.py`, `RESULTS_stress.md`). Feasibility holds under stress (Newton 100%); the risk lives in optimality tails under topology shift.
4. **Open problems** (current, per the Oct 22 pivot): ranked dissertation problems — primal-dual GNN warm-starting under the corrected WARP protocol, a GNN-native feasibility layer, "Falconer 2.0" (controlled study of when graph structure helps), cross-grid pretraining verification, and hardware-aware inference profiling (the empty niche; tail-made for the lab's GPU/FPGA angle).

## Workflow

Code and docs are maintained in git. The assistant has push access via a
repo-scoped deploy key and commits code/doc updates; Hay pulls on the GPU
workstation and runs `--device cuda` there (this box runs CPU). See the
open-problems map page for the deploy-key setup.
