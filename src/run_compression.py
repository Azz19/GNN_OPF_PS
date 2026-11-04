"""
Feasibility-aware compression of OPF surrogates
(docs/feasibility-aware-compression-brief.md).

Trains a teacher (FixedDepthGNN 64x8 by default, or DEQGNN), then a
scan of smaller FixedDepthGNN students — width/depth scaling, lever 1
of the brief — each in up to two modes:

  plain    label-MSE training, identical procedure to run_case118.py
  distill  loss = label MSE + lambda_distill * MSE(student, teacher)
           on the same inputs (lever 2, output-matching term only;
           no gradient-level feasibility term — see the brief on why)

Every model (teacher + students) gets one compression-table row —
config, mode, parameter count, on-disk size of the serialised
state_dict, mean inference latency — followed by the UNMODIFIED
Paper 1 audit (run_case118.evaluate_audit / print_audit) on the
FullTop test split and, when present, the zero-shot N-1 split. The
audit is the scoreboard: compression is judged on post-correction
feasibility per parameter / per byte, not on raw MSE.

Checkpoints are saved to runs/<run-tag>_*.pt (gitignored), mirroring
run_case118.py. A previously trained teacher can be reused with
--teacher-ckpt (architecture must match --teacher).

Usage:
  python src/run_compression.py --train-n 800 --test-n 200 --epochs 6 --device cuda
  python src/run_compression.py --case case14 --train-n 48 --test-n 16 \
      --epochs 1 --students 16x2,32x4 --device cpu   # smoke test
"""
import argparse, io, os, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from opfdata_loader import load_opfdata
from models import FixedDepthGNN, DEQGNN
from device import resolve_device
from run_case118 import train as train_plain, evaluate_audit, print_audit

CACHE = os.path.join(os.path.dirname(__file__), "..", "data_cache")
NODE_DIM, EDGE_DIM = 12, 4  # OPFData feature widths, as in run_case118.py

CASE_FILES = {
    "case118": ("case118_ft0.tar.gz", "case118_ft19.tar.gz",
                "case118_n1_19.tar.gz"),
    "case14": ("case14_shard0.tar.gz", "case14_shard0.tar.gz", ""),
}


def model_size_mb(model):
    """On-disk size of the serialised state_dict, in MB — the same
    bytes torch.save writes to runs/*.pt."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1e6


def measure_latency_ms(model, d, idx, dev, per_example_topo=False):
    """Mean inference-only latency per example on the eval device
    (one warmup forward, then timed forwards, CUDA-synced)."""
    model.eval()

    def tensors(i):
        if per_example_topo:
            ei = torch.tensor(d["edge_index"][i], device=dev)
            ef = torch.tensor(d["edge_feat"][i], device=dev)
        else:
            ei = torch.tensor(d["edge_index"], device=dev)
            ef = torch.tensor(d["edge_feat"], device=dev)
        return torch.tensor(d["x"][i], device=dev), ei, ef

    def one(i):
        x, ei, ef = tensors(i)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x, ei, ef)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000

    idx = list(idx)
    one(idx[0])  # warmup
    return float(np.mean([one(i) for i in idx]))


def train_distill(student, teacher, d, n, epochs, dev, lam):
    """Same optimiser / per-example loop as run_case118.train(), with
    loss = MSE(student, label) + lam * MSE(student, teacher)."""
    ei = torch.tensor(d["edge_index"], device=dev)
    ef = torch.tensor(d["edge_feat"], device=dev)
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    lf = nn.MSELoss()
    teacher.eval()
    t0 = time.perf_counter()
    for ep in range(epochs):
        student.train()
        for i in range(n):
            x = torch.tensor(d["x"][i], device=dev)
            y = torch.tensor(d["y"][i], device=dev)
            with torch.no_grad():
                t_out = teacher(x, ei, ef)
            opt.zero_grad()
            s_out = student(x, ei, ef)
            loss = lf(s_out, y) + lam * lf(s_out, t_out)
            loss.backward(); opt.step()
    return time.perf_counter() - t0


def report_model(tag, model, dev, dte, dn1, test_n):
    """One compression-table row + the Paper 1 audit per split."""
    npar = sum(p.numel() for p in model.parameters())
    size = model_size_mb(model)
    lat = measure_latency_ms(model, dte, range(test_n), dev,
                             per_example_topo=dte["per_example_topology"])
    print(f"\n== {tag} | params {npar:,} | size {size:.3f} MB | "
          f"inference {lat:.2f} ms/example ==")
    for split, dd in (("FullTop test", dte), ("N-1 zero-shot", dn1)):
        if dd is None:
            continue
        audit = evaluate_audit(model, dd, range(test_n), dev,
                               per_example_topo=dd["per_example_topology"])
        print_audit(f"{tag} | {split}", audit)
    return {"tag": tag, "params": npar, "size_mb": size, "lat_ms": lat}


def parse_students(spec):
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        h, dp = tok.lower().split("x")
        out.append((int(h), int(dp)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="case118", choices=list(CASE_FILES),
                    help="selects the default OPFData shards in data_cache/")
    ap.add_argument("--train-n", type=int, default=800)
    ap.add_argument("--test-n", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--device", default="auto",
                    help="auto|cpu|cuda (auto picks cuda when available)")
    ap.add_argument("--train-file", default=None,
                    help="override the case default training shard")
    ap.add_argument("--test-file", default=None,
                    help="override the case default FullTop test shard")
    ap.add_argument("--n1-file", default=None,
                    help="override the case default N-1 shard; '' skips N-1")
    ap.add_argument("--teacher", default="fixed", choices=["fixed", "deq"],
                    help="teacher architecture: FixedDepthGNN 64x8 or DEQGNN")
    ap.add_argument("--teacher-ckpt", default=None,
                    help="load teacher weights from this runs/*.pt instead "
                         "of training (architecture must match --teacher)")
    ap.add_argument("--students", default="16x2,16x4,16x8,32x2,32x4,32x8,"
                                          "64x2,64x4,64x8",
                    help="comma list of hidden x depth student configs")
    ap.add_argument("--distill", dest="distill", action="store_true",
                    default=True, help="also train distilled students "
                                       "(default)")
    ap.add_argument("--no-distill", dest="distill", action="store_false",
                    help="plain students only")
    ap.add_argument("--lambda-distill", type=float, default=1.0)
    ap.add_argument("--run-tag", default=None,
                    help="checkpoint prefix; default comp<case>")
    args = ap.parse_args()
    dev = resolve_device(args.device)
    print(f"Device: {dev}")

    dft_train, dft_test, dft_n1 = CASE_FILES[args.case]
    train_file = args.train_file if args.train_file is not None else dft_train
    test_file = args.test_file if args.test_file is not None else dft_test
    n1_file = args.n1_file if args.n1_file is not None else dft_n1
    run_tag = args.run_tag or f"comp{args.case}"

    t0 = time.perf_counter()
    dtr = load_opfdata(os.path.join(CACHE, train_file), limit=args.train_n)
    dte = load_opfdata(os.path.join(CACHE, test_file), limit=args.test_n)
    dn1 = load_opfdata(os.path.join(CACHE, n1_file),
                       limit=args.test_n) if n1_file else None
    print(f"Data load: {time.perf_counter()-t0:.0f}s | train {dtr['x'].shape}, "
          f"test {dte['x'].shape}" +
          ("" if dn1 is not None else ", N-1 split skipped"))

    runs_dir = os.path.join(os.path.dirname(__file__), "..", "runs")
    os.makedirs(runs_dir, exist_ok=True)

    def save(model, name):
        path = os.path.join(runs_dir, f"{run_tag}_{name}.pt")
        torch.save(model.state_dict(), path)
        print(f"  checkpoint saved: {path}")
        return path

    # ---- teacher ----
    if args.teacher == "deq":
        teacher = DEQGNN(NODE_DIM, EDGE_DIM, 64)
        teacher_name = "teacher_DEQGNN"
    else:
        teacher = FixedDepthGNN(NODE_DIM, EDGE_DIM, 64, 8)
        teacher_name = "teacher_FixedDepthGNN64x8"
    teacher = teacher.to(dev)
    if args.teacher_ckpt:
        teacher.load_state_dict(torch.load(args.teacher_ckpt,
                                           map_location=dev))
        print(f"\nTeacher loaded from checkpoint: {args.teacher_ckpt}")
    else:
        tt = train_plain(teacher, dtr, args.train_n, args.epochs, dev)
        print(f"\nTeacher trained: {tt:.0f}s for "
              f"{args.train_n}x{args.epochs}")
        save(teacher, teacher_name)
    teacher.eval()
    summary = [report_model(teacher_name, teacher, dev, dte, dn1,
                            args.test_n)]

    # ---- student scan ----
    for hidden, depth in parse_students(args.students):
        for mode in (["plain", "distill"] if args.distill else ["plain"]):
            student = FixedDepthGNN(NODE_DIM, EDGE_DIM, hidden,
                                    depth).to(dev)
            name = f"student_{hidden}x{depth}_{mode}"
            if mode == "plain":
                tt = train_plain(student, dtr, args.train_n, args.epochs,
                                 dev)
            else:
                tt = train_distill(student, teacher, dtr, args.train_n,
                                   args.epochs, dev, args.lambda_distill)
            print(f"\nStudent {hidden}x{depth} [{mode}] trained: {tt:.0f}s")
            save(student, name)
            summary.append(report_model(name, student, dev, dte, dn1,
                                        args.test_n))

    # ---- compression table (the audit blocks above are the score) ----
    print("\n== Compression summary (feasibility per model: see audit "
          "blocks above) ==")
    print(f"  {'model':<38} {'params':>10} {'size MB':>9} {'ms/ex':>8}")
    for r in summary:
        print(f"  {r['tag']:<38} {r['params']:>10,} {r['size_mb']:>9.3f} "
              f"{r['lat_ms']:>8.2f}")


if __name__ == "__main__":
    main()
