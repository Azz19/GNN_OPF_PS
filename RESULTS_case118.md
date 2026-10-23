# Case118 five-axis evaluation on OPFData (Oct 23, 2026)

Setup: OPFData `pglib_opf_case118_ieee`, FullTop train shard 0 (first 800
examples), FullTop test shard 19 (first 200), N-1 test shard 19 (first 200,
zero-shot). Both models trained MSE-only, 6 epochs, Adam lr 1e-3.
Run on Hay's workstation: RTX A4000 (CUDA), Newton correction on CPU
(Windows). Script: `src/run_case118.py --train-n 800 --test-n 200
--epochs 6 --device cuda`.

## Results

| Axis / metric | FixedDepthGNN | DEQGNN |
|---|---|---|
| Parameters (fp32 size) | 107,522 (420 KiB) | 22,082 (86 KiB) |
| Train time (800×6) | 53 s (11 ms/sample-epoch) | 167 s (35 ms/sample-epoch) |
| Inference / example | 3.4 ms | 35.8 ms |
| Newton correction / example | ~284–301 ms | ~310–318 ms |
| Test MSE (FullTop) | 3.12e-04 | 3.75e-04 |
| Test MSE (N-1 zero-shot) | 8.99e-04 | 9.56e-04 |
| Raw violation (FullTop / N-1) | 4.17 / 4.63 pu | 5.59 / 5.79 pu |
| Corrected violation, mean | 4.2e-05 / 3.4e-05 pu | 3.0e-05 / 3.1e-05 pu |
| Corrected violation, max | 9.9e-05 / 1.0e-04 pu | 9.2e-05 / 9.1e-05 pu |
| Newton convergence | 100% (2.7–2.9 iters) | 100% (3.0–3.1 iters) |
| Cost gap, FullTop mean / p99 / max | **−31.1%** / −25.8% / −25.7% | **−11.6%** / −0.1% / +0.6% |
| Cost gap, N-1 mean / p99 / max | −30.8% / −12.8% / +2.3% | −11.4% / +13.6% / +16.8% |

Data loading: 55 s for 1,200 JSON examples (~46 ms/example — parsing,
not GPU, is the data-pipeline bottleneck at this scale). Peak-process RSS
was not captured on Windows (psutil absent); parameter memory above is
exact, and activations at 118 buses are small next to it.

## Reading the results — honestly

**Feasibility (power balance): solved at this scale, again.** Both models,
both splits, 100% Newton convergence, corrected violations ~1e-4 pu worst
case — despite raw predictions being far off (4–6 pu) after only 6 epochs
on 800 samples. The correction layer is doing the heavy lifting and it is
robust to weak predictions and unseen topology changes.

**The negative cost gaps are the most important number in the table —
and they are a warning, not a win.** A corrected state cannot legitimately
cost 31% *less* than the solver's optimum. It can only do so because the
Newton layer enforces the equality constraints (power balance) but NOT
the inequality constraints: the dispatch implied by the corrected
voltages is free to exceed generator P/Q limits and voltage bounds that
the true OPF respects. Negative gap = the corrected state is cheating on
limits, by an amount that varies by model (fixed-depth −31% uniformly;
DEQ −12% on average with its FullTop tail near 0%). **Next measurement
this demands:** limit-violation rates (Pg/Qg/Vm out-of-bounds) for the
corrected states, reported next to every cost gap. Until then, cost-gap
comparisons between models — including the N-1 tails (DEQ max +16.8% vs
fixed-depth max +2.3%) — are contaminated and should not be quoted as
optimality results.

**Risk axis:** the case14 pattern (tails, not means, carry the risk) is
visible again in the DEQ N-1 positive tail (p99 +13.6%), but see the
caveat above — a limit-aware re-scoring is required before ranking the
models on risk at this scale.

**Speed axis:** end-to-end latency is dominated by the correction, not
the network: 3.4 ms (fixed) / 35.8 ms (DEQ) of inference vs ~300 ms of
Newton in unoptimized Python (O(n²) Jacobian loops). Vectorizing the
Jacobian, or moving the correction to torch on the GPU, is the obvious
speed lever — the model choice barely matters for pipeline latency until
that is fixed.

**Scalability axis:** training at 118 buses is 11–35 ms/sample-epoch on
one A4000; nothing in the pipeline strained. The practical bottleneck
found is JSON data loading (~46 ms/example), which argues for caching
parsed shards as .npz for the 500/2000-bus steps.

**DEQ stability note:** the wild fixed-point behaviour seen in the
1-epoch CPU smoke test does NOT appear with 6 epochs of training — DEQ
trains and corrects cleanly at 118 buses, at ~3× the training cost and
~10× the inference cost of the fixed-depth model, with ~5× fewer
parameters.

---

## Follow-up: limit-violation metrics added to the evaluator

The evaluator (`src/run_case118.py`) now reports inequality-limit
violations for every corrected state, next to the cost gap, via
`limit_violations()` in `src/feasibility.py`: max out-of-bounds magnitude
(pu) and out-of-bounds rate for Pg vs [pmin, pmax], Qg vs [qmin, qmax]
(gen buses, from the injection equations), and Vm vs [vmin, vmax] (all
buses). OOB threshold is 1e-4 pu: verified that the OPFData solver labels
themselves read as fully limit-feasible under it (fracs 0.0; max
residuals ≤ ~1e-5 pu at binding constraints, i.e. honest float32 noise
that does not count as OOB). The per-example generator mask comes from
the example's own features, so N-1 examples that dropped a generator are
handled correctly; static limits were verified identical across the
FullTop and N-1 shards.

Re-running the same command picks these up; the resulting table will
show, for each model and split, how much of the negative cost gap is
bought with limit violations — i.e. whether "beating the optimum" is
limit-cheating, and by how much per axis.
