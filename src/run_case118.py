"""
Five-axis evaluation at case118 scale on OPFData (see RESULTS_case118.md).

Axes: feasibility (raw + Newton-corrected violation), risk (zero-shot N-1
test split: cost-gap tails, correction convergence), scalability
(load/train/inference times at 118 buses vs the case14 numbers),
speed (inference + correction ms), memory (parameters + peak RSS).

Trains FixedDepthGNN and DEQGNN on a subset of the FullTop train shard,
evaluates on FullTop test and (zero-shot) N-1 test examples. CPU-feasible
subset sizes by default; scale up on the GPU workstation.

Usage:
  python src/run_case118.py --train-n 800 --test-n 200 --epochs 6
"""
import argparse, sys, os, time
try:
    import resource  # Unix only; absent on Windows
except ImportError:
    resource = None
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from opfdata_loader import load_opfdata, state_cost
from opf_data import pq_violation
from models import FixedDepthGNN, DEQGNN
from feasibility import newton_correct, limit_violations
from device import resolve_device

CACHE = os.path.join(os.path.dirname(__file__), "..", "data_cache")


def rss_mb():
    if resource is not None:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        return float("nan")


def train(model, d, n, epochs, dev):
    ei = torch.tensor(d["edge_index"], device=dev)
    ef = torch.tensor(d["edge_feat"], device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lf = nn.MSELoss()
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        for i in range(n):
            x = torch.tensor(d["x"][i], device=dev)
            y = torch.tensor(d["y"][i], device=dev)
            opt.zero_grad()
            loss = lf(model(x, ei, ef), y)
            loss.backward(); opt.step()
    return time.perf_counter() - t0


def evaluate(model, d, idx, dev, per_example_topo=False):
    st = d["static"]
    sb = st["static_bus"]  # [is_gen, is_slack, pmin, pmax, qmin, qmax,
                           #  vmin, vmax, c1, c2]
    rows = []
    model.eval()
    for i in idx:
        # is_gen / slack come from the example's own features (correct even
        # when an N-1 example dropped a generator)
        is_g = d["x"][i][:, 2] > 0
        sl = int(np.where(d["x"][i][:, 3] > 0)[0][0])
        gen_bus = d["gen_bus_list"][i]; gen_cost = d["gen_cost_list"][i]
        sol_pg = d["sol_pg"][i] if isinstance(d["sol_pg"], list) \
            else d["sol_pg"][i]
        if per_example_topo:
            ei = torch.tensor(d["edge_index"][i], device=dev)
            ef = torch.tensor(d["edge_feat"][i], device=dev)
            Ybus = d["Ybus"][i]
        else:
            ei = torch.tensor(d["edge_index"], device=dev)
            ef = torch.tensor(d["edge_feat"], device=dev)
            Ybus = d["Ybus"]
        x = torch.tensor(d["x"][i], device=dev)
        t0 = time.perf_counter()
        with torch.no_grad():
            p = model(x, ei, ef).cpu().numpy()
        t_inf = time.perf_counter() - t0
        mse = float(np.mean((p - d["y"][i]) ** 2))
        mx, _ = pq_violation(Ybus, p[:, 0], p[:, 1], -d["pd"][i], -d["qd"][i], ~is_g)
        t0 = time.perf_counter()
        cvm, cva, info = newton_correct(Ybus, p[:, 0], p[:, 1], d["pd"][i],
                                        d["qd"][i], is_g, slack_bus=sl)
        t_cor = time.perf_counter() - t0
        cmx, _ = pq_violation(Ybus, cvm, cva, -d["pd"][i], -d["qd"][i], ~is_g)
        ccost = state_cost(Ybus, cvm, cva, d["pd"][i], gen_bus,
                           gen_cost, sol_pg)
        # Inequality bounds the Newton layer does NOT enforce: how much
        # the corrected state cheats to sit under the solver optimum.
        clim = limit_violations(Ybus, cvm, cva, d["pd"][i], d["qd"][i],
                                is_g, sb[:, 2], sb[:, 3], sb[:, 4], sb[:, 5],
                                sb[:, 6], sb[:, 7])
        rows.append({"mse": mse, "viol_raw": mx, "viol_corr": cmx,
                     "cost_gap_pct": (ccost - d["cost"][i]) / d["cost"][i] * 100,
                     "conv": info["converged"], "iters": info["iters"],
                     "t_inf_ms": t_inf * 1000, "t_cor_ms": t_cor * 1000,
                     **clim})
    agg = {k: float(np.mean([r[k] for r in rows])) for k in
           ("mse", "viol_raw", "viol_corr", "cost_gap_pct", "iters",
            "t_inf_ms", "t_cor_ms")}
    agg["viol_corr_max"] = float(np.max([r["viol_corr"] for r in rows]))
    agg["cost_gap_p99"] = float(np.percentile([r["cost_gap_pct"] for r in rows], 99))
    agg["cost_gap_max"] = float(np.max([r["cost_gap_pct"] for r in rows]))
    agg["conv_rate"] = float(np.mean([r["conv"] for r in rows]))
    for k in ("lim_pg_max", "lim_pg_frac", "lim_qg_max", "lim_qg_frac",
              "lim_vm_max", "lim_vm_frac"):
        vals = [r[k] for r in rows]
        agg[k] = float(np.mean(vals))
        agg[k + "_max"] = float(np.max(vals))
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=800)
    ap.add_argument("--test-n", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--device", default="auto",
                    help="auto|cpu|cuda (auto picks cuda when available)")
    args = ap.parse_args()
    dev = resolve_device(args.device)
    print(f"Device: {dev}")

    t0 = time.perf_counter()
    dtr = load_opfdata(os.path.join(CACHE, "case118_ft0.tar.gz"),
                       limit=args.train_n)
    dte = load_opfdata(os.path.join(CACHE, "case118_ft19.tar.gz"),
                       limit=args.test_n)
    dn1 = load_opfdata(os.path.join(CACHE, "case118_n1_19.tar.gz"),
                       limit=args.test_n)
    print(f"Data load: {time.perf_counter()-t0:.0f}s | train {dtr['x'].shape}, "
          f"test {dte['x'].shape}, N-1 per-example topo: {dn1['per_example_topology']}")
    print(f"Peak RSS after load: {rss_mb():.0f} MB")

    for name, model in [("FixedDepthGNN", FixedDepthGNN(12, 4, 64, 8)),
                        ("DEQGNN", DEQGNN(12, 4, 64))]:
        npar = sum(p.numel() for p in model.parameters())
        model = model.to(dev)
        tt = train(model, dtr, args.train_n, args.epochs, dev)
        r_ft = evaluate(model, dte, range(args.test_n), dev)
        r_n1 = evaluate(model, dn1, range(args.test_n), dev,
                        per_example_topo=True)
        print(f"\n== {name} ({npar:,} params, {npar*4/1024:.0f} KiB fp32) ==")
        print(f"  train: {tt:.0f}s for {args.train_n}x{args.epochs} "
              f"({tt/(args.train_n*args.epochs)*1000:.0f} ms/sample-epoch)")
        for tag, r in [("FullTop test", r_ft), ("N-1 zero-shot", r_n1)]:
            print(f"  [{tag}] MSE {r['mse']:.2e} | viol raw {r['viol_raw']:.4f} -> "
                  f"corr {r['viol_corr']:.2e} (max {r['viol_corr_max']:.2e}) pu | "
                  f"cost gap mean {r['cost_gap_pct']:+.2f}% p99 {r['cost_gap_p99']:+.2f}% "
                  f"max {r['cost_gap_max']:+.2f}% | Newton conv {r['conv_rate']*100:.0f}% "
                  f"iters {r['iters']:.1f} | infer {r['t_inf_ms']:.1f} ms + corr "
                  f"{r['t_cor_ms']:.1f} ms")
            print(f"    limit violations (corrected state): "
                  f"Pg max {r['lim_pg_max']:.2e} pu "
                  f"({r['lim_pg_frac']*100:.1f}% of gen buses OOB) | "
                  f"Qg max {r['lim_qg_max']:.2e} pu "
                  f"({r['lim_qg_frac']*100:.1f}% OOB) | "
                  f"Vm max {r['lim_vm_max']:.2e} pu "
                  f"({r['lim_vm_frac']*100:.1f}% of buses OOB)")
        print(f"  peak RSS: {rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
