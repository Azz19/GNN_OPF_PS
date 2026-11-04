# Compression scaffold — what was built

*Built Nov 4, 2026. Companion to
`feasibility-aware-compression-brief.md`. This document describes
plumbing and how to run it. The smoke-test numbers below are
plumbing validation on toy training (48 examples, 1 epoch) — they
are NOT results and support no claim about compression quality.*

## What was built

**`src/run_compression.py`** (new; no changes to existing files):

- Loads OPFData with the same loader and shard protocol as
  `run_case118.py` (`--case case118` default: ft0 train / ft19
  FullTop test / n1_19 zero-shot N-1; `--case case14` uses the
  single cached shard and skips N-1). `--train-file` / `--test-file`
  / `--n1-file` override individual shards, mirroring
  `run_case118.py`.
- **Teacher**: `FixedDepthGNN(hidden=64, depth=8)` by default, or
  `DEQGNN` with `--teacher deq`. Trained with `run_case118.train()`
  imported unchanged (Adam lr 1e-3, per-example label MSE), or
  loaded from `--teacher-ckpt` (architecture must match
  `--teacher`). Saved to `runs/<run-tag>_teacher_*.pt`.
- **Students**: `FixedDepthGNN` at each `hidden x depth` in
  `--students` (default the full brief scan:
  `16x2,16x4,16x8,32x2,32x4,32x8,64x2,64x4,64x8`), each in two modes
  unless `--no-distill`:
  - `plain` — label-MSE training, same imported procedure;
  - `distill` — loss = label MSE + `--lambda-distill` (default 1.0)
    × MSE(student outputs, teacher outputs on the same inputs).
    Output matching only: no gradient-level feasibility term (the
    brief's caution from Exp 2 applies).
  Checkpoints: `runs/<run-tag>_student_<H>x<D>_<mode>.pt`
  (`runs/` stays gitignored). `--run-tag` defaults to `comp<case>`.
- **Scoring**: every model (teacher + each student/mode) prints one
  compression-table row — parameter count, on-disk size in MB of
  the serialised `state_dict` (the same bytes written to `runs/`),
  mean inference-only latency ms/example on the eval device (one
  warmup forward, CUDA-synced when on GPU) — followed by the
  **unmodified** Paper 1 audit imported from `run_case118`
  (`evaluate_audit` / `print_audit`): raw / Newton / clip /
  distributed stages with fully-feasible rate, Pg/Qg/Vm OOB, slack
  Pg, balance mismatch, feasible-only and all-points cost gaps, PF
  convergence, and the B-vs-A success bar, on FullTop test and
  (when present) N-1. A summary table at the end lists every model's
  params / size / latency side by side.

Not yet built (later levers from the brief): magnitude pruning,
quantization, feasibility-based checkpoint selection.

## Smoke-test evidence (plumbing only — NOT results)

Command (CPU, cached case14 shard):

```
.venv/bin/python src/run_compression.py --case case14 --train-n 48 \
    --test-n 16 --epochs 1 --students 16x2,32x4 --device cpu
```

Completed in ~27 s wall. Teacher trained and saved; both students
trained in both modes; all checkpoints written under `runs/`; the
audit ran for every model. Final summary table exactly as printed:

```
== Compression summary (feasibility per model: see audit blocks above) ==
  model                                      params   size MB    ms/ex
  teacher_FixedDepthGNN64x8                 107,522     0.448     3.71
  student_16x2_plain                          2,306     0.015     0.38
  student_16x2_distill                        2,306     0.015     4.05
  student_32x4_plain                         14,850     0.069     4.87
  student_32x4_distill                       14,850     0.069     0.63
```

Reading notes: at 1 epoch on 48 examples the models are essentially
untrained — audit feasible rates of 0% at the Newton stage and the
clip stage's 100%-feasible-with-large-balance-mismatch pattern are
the expected toy-training behaviour, already documented for Paper 1.
The latency column is noisy at this scale on a shared CPU box
(identical architectures print 0.38–4.87 ms across modes); treat it
as proof the timing path runs, not as a measurement. Real numbers
come from the GPU run below.

## The GPU run (Hay)

From `C:\Users\dagboola\GNN_OPF_PS`, after `git pull`:

```
python src/run_compression.py --train-n 800 --test-n 200 --epochs 6 --device cuda
```

Defaults: case118 shards, fixed 64x8 teacher, all nine student
configs × plain + distill (19 models, each fully audited on FullTop
+ N-1). Narrow it with `--students 16x4,32x4` or `--no-distill` for
a quicker first pass; reuse an existing teacher with e.g.
`--teacher-ckpt runs\compcase118_teacher_FixedDepthGNN64x8.pt`.

How to read the output: the per-model audit blocks are the score —
compare students against the teacher on **post-correction**
fully-feasible rate and Pg/Qg/Vm violations (Newton and distributed
stages), then divide by the size column. A student that matches the
teacher's raw MSE but loses feasibility after correction is a
failure of MSE-based selection (hypothesis H1 in the brief); a
student that keeps the teacher's post-correction feasibility at
~1/50th the parameters (16x2 is 2,306 vs 107,522) is the headline
the study is looking for.
