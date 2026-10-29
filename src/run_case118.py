"""
Five-axis evaluation at case118 scale on OPFData (see RESULTS_case118.md).

Axes: feasibility (raw + Newton-corrected violation), risk (zero-shot N-1
test split: cost-gap tails, correction convergence), scalability
(load/train/inference times at 118 buses vs the case14 numbers),
speed (inference + correction ms), memory (parameters + peak RSS).

Trains FixedDepthGNN and DEQGNN on a subset of the FullTop train shard,
evaluates on FullTop test and (zero-shot) N-1 test examples. CPU-feasible
subset sizes by default; scale up on the GPU workstation.

Paper 1 audit (docs/paper1-audit-protocol.md): the default evaluation is
now evaluate_audit(), which reports four stages side by side per model
and split — raw prediction, + plain Newton (variant A), + Newton then
clip to limits with no re-solve, + distributed-slack Newton (variant B)
— with per-class Pg/Qg/Vm OOB rates and magnitudes (slack Pg reported
separately), fully-feasible rate at tau = 1e-4 pu, cost gaps on feasible
points only and on all points (flagged), PF convergence, power-balance
mismatch per stage (so the clip stage cannot hide its imbalance), and
per-stage latency. Scope is Pg/Qg/Vm only: no branch/angle metrics.
The limit-aware repair stage from branch exp/limit-aware-repair is NOT
included (branch not merged); if it is merged later, add it as a fifth
stage in evaluate_audit() next to "distributed".

Trained weights are saved to runs/<run-tag>_<Model>.pt (gitignored) so
the audit can be re-scored without retraining.

Usage:
  python src/run_case118.py --train-n 800 --test-n 200 --epochs 6 --device cuda
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
from feasibility import (newton_correct, limit_violations,
                         newton_correct_distributed, bus_dispatch,
                         limit_violation_detail, dispatch_violation_detail,
                         clip_dispatch, balance_mismatch)
from device import resolve_device

CACHE = os.path.join(os.path.dirname(__file__), "..", "data_cache")
TAU = 1e-4  # the one OOB tolerance everywhere (protocol section 3)


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
    """Legacy corrected-state-only evaluation (kept for continuity with
    RESULTS_case118.md; main() now runs evaluate_audit())."""
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


# ------------------------- Paper 1 audit -------------------------

def _cost_from_bus_pg(pg_bus, gen_bus, gen_cost, sol_pg):
    """state_cost() with bus-level Pg given instead of derived from a
    state (the clip stage's dispatch is not a solved state). Same
    split-by-solution-proportions accounting as
    opfdata_loader.state_cost."""
    total = 0.0
    for gb in np.unique(gen_bus):
        idx = np.where(gen_bus == gb)[0]
        share = sol_pg[idx]
        share = share / share.sum() if share.sum() > 1e-12 \
            else np.full(len(idx), 1.0 / len(idx))
        for j, gi in enumerate(idx):
            pg = pg_bus[gb] * share[j]
            c2, c1, c0 = gen_cost[gi]
            total += c2 * pg ** 2 + c1 * pg + c0
    return float(total)


def _mpm(vals):
    """(mean, p99, max) of a per-example series."""
    v = np.asarray(vals, dtype=float)
    return (float(np.mean(v)), float(np.percentile(v, 99)),
            float(np.max(v)))


def _agg_stage(rows):
    feas = [r for r in rows if r["feasible"]]
    agg = {
        "n": len(rows),
        "feas_rate": float(np.mean([r["feasible"] for r in rows])),
        "pg_rate": float(np.mean([r["pg_oob"] for r in rows])),
        "pg": _mpm([r["pg_max"] for r in rows]),
        "qg_rate": float(np.mean([r["qg_oob"] for r in rows])),
        "qg": _mpm([r["qg_max"] for r in rows]),
        "vm_rate": float(np.mean([r["vm_oob"] for r in rows])),
        "vm": _mpm([r["vm_max"] for r in rows]),
        "slack_rate": float(np.mean([r["slack_oob"] for r in rows])),
        "slack": _mpm([r["slack_viol"] for r in rows]),
        "tot": _mpm([r["tot_max"] for r in rows]),
        "bal": _mpm([r["bal"] for r in rows]),
        "gap_all": _mpm([r["gap"] for r in rows]),
        "gap_feas": _mpm([r["gap"] for r in feas]) if feas else None,
        "n_feas": len(feas),
        "conv_rate": float(np.mean([r["conv"] for r in rows]))
        if rows[0]["conv"] is not None else None,
        "lat": _mpm([r["lat_ms"] for r in rows]),
        "corr": _mpm([r["corr_ms"] for r in rows]),
    }
    if "dist_ok" in rows[0]:
        agg["dist_ok_rate"] = float(np.mean([r["dist_ok"] for r in rows]))
        agg["rounds_mean"] = float(np.mean([r["rounds"] for r in rows]))
    return agg


def evaluate_audit(model, d, idx, dev, per_example_topo=False):
    """Paper 1 audit table (protocol section 3): four stages per example —
    raw prediction, + plain Newton (A), + Newton-then-clip (no re-solve),
    + distributed-slack Newton (B) — scored identically: per-class Pg/Qg/Vm
    OOB (rate = fraction of examples with any OOB; magnitudes mean/p99/max),
    slack Pg OOB separately, fully-feasible rate at tau, power-balance
    mismatch, cost gap on feasible points only and on all points, PF
    convergence, and per-stage latency (end-to-end incl. inference, and
    correction-only). Limits come from each example's own features, so
    N-1 examples with a dropped generator are scored against their own
    bounds (the legacy evaluate() used the first example's statics)."""
    rows = {s: [] for s in ("raw", "newton", "clip", "distributed")}
    model.eval()
    for i in idx:
        xi = d["x"][i]
        is_g = xi[:, 2] > 0
        sl = int(np.where(xi[:, 3] > 0)[0][0])
        # x cols: [pd, qd, is_gen, is_slack, pmin, pmax, qmin, qmax,
        #          vmin, vmax, c1, c2] — per-example limits
        pmin, pmax = (xi[:, 4].astype(np.float64),
                      xi[:, 5].astype(np.float64))
        qmin, qmax = (xi[:, 6].astype(np.float64),
                      xi[:, 7].astype(np.float64))
        vmin, vmax = (xi[:, 8].astype(np.float64),
                      xi[:, 9].astype(np.float64))
        lims = (pmin, pmax, qmin, qmax, vmin, vmax)
        gen_bus = d["gen_bus_list"][i]; gen_cost = d["gen_cost_list"][i]
        sol_pg = d["sol_pg"][i] if isinstance(d["sol_pg"], list) \
            else d["sol_pg"][i]
        pd, qd = d["pd"][i], d["qd"][i]
        label_cost = d["cost"][i]
        if per_example_topo:
            ei = torch.tensor(d["edge_index"][i], device=dev)
            ef = torch.tensor(d["edge_feat"][i], device=dev)
            Ybus = d["Ybus"][i]
        else:
            ei = torch.tensor(d["edge_index"], device=dev)
            ef = torch.tensor(d["edge_feat"], device=dev)
            Ybus = d["Ybus"]
        x = torch.tensor(xi, device=dev)
        t0 = time.perf_counter()
        with torch.no_grad():
            p = model(x, ei, ef).cpu().numpy()
        t_inf = (time.perf_counter() - t0) * 1000
        pvm, pva = p[:, 0], p[:, 1]

        def row(det, cost, bal, conv, lat_ms, corr_ms, **extra):
            r = {"feasible": det["feasible"],
                 "pg_oob": det["pg_oob"], "pg_max": det["lim_pg_max"],
                 "qg_oob": det["qg_oob"], "qg_max": det["lim_qg_max"],
                 "vm_oob": det["vm_oob"], "vm_max": det["lim_vm_max"],
                 "slack_oob": det["slack_pg_oob"],
                 "slack_viol": det["slack_pg_viol"],
                 "tot_max": det["lim_pg_max"] + det["lim_qg_max"]
                 + det["lim_vm_max"],
                 "bal": bal, "conv": conv,
                 "gap": (cost - label_cost) / label_cost * 100,
                 "lat_ms": lat_ms, "corr_ms": corr_ms}
            r.update(extra)
            return r

        # (a) raw GNN prediction
        det = limit_violation_detail(Ybus, pvm, pva, pd, qd, is_g, *lims,
                                     slack_bus=sl, tol=TAU)
        pg_raw, _ = bus_dispatch(Ybus, pvm, pva, pd, qd)
        bal = balance_mismatch(Ybus, pvm, pva, pd, qd, is_g, pg_raw,
                               slack_bus=sl)
        cost = state_cost(Ybus, pvm, pva, pd, gen_bus, gen_cost, sol_pg)
        rows["raw"].append(row(det, cost, bal, None, t_inf, 0.0))

        # (b) + plain Newton correction (variant A: single slack)
        t0 = time.perf_counter()
        cvm, cva, info = newton_correct(Ybus, pvm, pva, pd, qd, is_g,
                                        slack_bus=sl)
        t_new = (time.perf_counter() - t0) * 1000
        det = limit_violation_detail(Ybus, cvm, cva, pd, qd, is_g, *lims,
                                     slack_bus=sl, tol=TAU)
        cost = state_cost(Ybus, cvm, cva, pd, gen_bus, gen_cost, sol_pg)
        rows["newton"].append(row(det, cost, info["final_mismatch_pu"],
                                  info["converged"], t_inf + t_new, t_new))

        # (c) + Newton, then clip Pg/Qg/Vm to limits, NO re-solve
        t0 = time.perf_counter()
        vm_c, pg_c, qg_c = clip_dispatch(Ybus, cvm, cva, pd, qd, is_g,
                                         *lims)
        t_clip = (time.perf_counter() - t0) * 1000
        det = dispatch_violation_detail(pg_c, qg_c, vm_c, is_g, *lims,
                                        slack_bus=sl, tol=TAU)
        cost = _cost_from_bus_pg(pg_c, gen_bus, gen_cost, sol_pg)
        bal = balance_mismatch(Ybus, vm_c, cva, pd, qd, is_g, pg_c,
                               slack_bus=sl)
        rows["clip"].append(row(det, cost, bal, info["converged"],
                                t_inf + t_new + t_clip, t_new + t_clip))

        # (d) + distributed-slack Newton (variant B)
        t0 = time.perf_counter()
        dvm, dva, dinfo = newton_correct_distributed(
            Ybus, pvm, pva, pd, qd, is_g, pmin, pmax, slack_bus=sl)
        t_dis = (time.perf_counter() - t0) * 1000
        det = limit_violation_detail(Ybus, dvm, dva, pd, qd, is_g, *lims,
                                     slack_bus=sl, tol=TAU)
        cost = state_cost(Ybus, dvm, dva, pd, gen_bus, gen_cost, sol_pg)
        rows["distributed"].append(row(
            det, cost, dinfo["final_mismatch_pu"], dinfo["converged"],
            t_inf + t_dis, t_dis,
            dist_ok=dinfo["distributed_ok"], rounds=dinfo["rounds"]))
    return {s: _agg_stage(rows[s]) for s in rows}


def _fmt_gap(g):
    if g is None:
        return "n/a (no feasible points)"
    return f"mean {g[0]:+.2f}% p99 {g[1]:+.2f}% max {g[2]:+.2f}%"


def print_audit(tag, audit):
    print(f"  [{tag}] Paper 1 audit — scope Pg/Qg/Vm only, tau={TAU:g} pu; "
          f"OOB rate = % of examples with any OOB; magnitudes mean/p99/max pu")
    for stage in ("raw", "newton", "clip", "distributed"):
        a = audit[stage]
        conv = f"{a['conv_rate']*100:.0f}%" if a["conv_rate"] is not None \
            else "—"
        extra = ""
        if "dist_ok_rate" in a:
            extra = (f" | dist-ok {a['dist_ok_rate']*100:.0f}% "
                     f"rounds {a['rounds_mean']:.1f}")
        print(f"    {stage:<11} feas {a['feas_rate']*100:5.1f}% | "
              f"Pg OOB {a['pg_rate']*100:5.1f}% max {a['pg'][2]:.3f} | "
              f"Qg OOB {a['qg_rate']*100:5.1f}% max {a['qg'][2]:.3f} | "
              f"Vm OOB {a['vm_rate']*100:5.1f}% max {a['vm'][2]:.3f} | "
              f"slack Pg OOB {a['slack_rate']*100:5.1f}% "
              f"mean {a['slack'][0]:.3f} max {a['slack'][2]:.3f}")
        print(f"    {'':<11} bal max {a['bal'][2]:.2e} | "
              f"gap(feas, n={a['n_feas']}) {_fmt_gap(a['gap_feas'])} | "
              f"gap(all, flagged) {_fmt_gap(a['gap_all'])} | "
              f"PF conv {conv} | lat {a['lat'][0]:.1f} ms "
              f"(corr {a['corr'][0]:.1f} ms){extra}")
    # Protocol section 4 success bar for variant B vs variant A, judged on
    # this split's numbers — every component printed, pass or fail.
    A, B = audit["newton"], audit["distributed"]

    def mark(ok):
        return "PASS" if ok else "FAIL"

    def gap_mean(g):
        return g[0] if g is not None else None

    gf_ok = (gap_mean(B["gap_feas"]) is not None
             and gap_mean(A["gap_feas"]) is not None
             and gap_mean(B["gap_feas"]) >= gap_mean(A["gap_feas"]))
    lat_ratio = B["lat"][0] / A["lat"][0] if A["lat"][0] > 0 else float("inf")
    corr_ratio = B["corr"][0] / A["corr"][0] if A["corr"][0] > 0 \
        else float("inf")
    print(f"    success bar B vs A [{tag}]: "
          f"slack OOB rate {A['slack_rate']*100:.1f}% -> "
          f"{B['slack_rate']*100:.1f}% [{mark(B['slack_rate'] < A['slack_rate'])}] | "
          f"slack max {A['slack'][2]:.3f} -> {B['slack'][2]:.3f} "
          f"[{mark(B['slack'][2] < A['slack'][2])}] | "
          f"total OOB mean {A['tot'][0]:.3f} -> {B['tot'][0]:.3f} "
          f"[{mark(B['tot'][0] <= A['tot'][0])}] | "
          f"feas-gap not more negative [{mark(gf_ok)}] | "
          f"latency {lat_ratio:.2f}x end-to-end / {corr_ratio:.2f}x "
          f"correction-only (bar <=2x) [{mark(lat_ratio <= 2.0)}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=800)
    ap.add_argument("--test-n", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--device", default="auto",
                    help="auto|cpu|cuda (auto picks cuda when available)")
    ap.add_argument("--train-file", default="case118_ft0.tar.gz",
                    help="OPFData shard in data_cache/ for training")
    ap.add_argument("--test-file", default="case118_ft19.tar.gz",
                    help="OPFData shard in data_cache/ for FullTop test")
    ap.add_argument("--n1-file", default="case118_n1_19.tar.gz",
                    help="OPFData shard in data_cache/ for the zero-shot "
                         "N-1 test; pass '' to skip the N-1 split")
    ap.add_argument("--run-tag", default="case118",
                    help="checkpoint files are runs/<run-tag>_<Model>.pt")
    args = ap.parse_args()
    dev = resolve_device(args.device)
    print(f"Device: {dev}")

    t0 = time.perf_counter()
    dtr = load_opfdata(os.path.join(CACHE, args.train_file),
                       limit=args.train_n)
    dte = load_opfdata(os.path.join(CACHE, args.test_file),
                       limit=args.test_n)
    dn1 = load_opfdata(os.path.join(CACHE, args.n1_file),
                       limit=args.test_n) if args.n1_file else None
    print(f"Data load: {time.perf_counter()-t0:.0f}s | train {dtr['x'].shape}, "
          f"test {dte['x'].shape}" +
          (f", N-1 per-example topo: {dn1['per_example_topology']}"
           if dn1 is not None else ", N-1 split skipped"))
    print(f"Peak RSS after load: {rss_mb():.0f} MB")

    runs_dir = os.path.join(os.path.dirname(__file__), "..", "runs")
    os.makedirs(runs_dir, exist_ok=True)

    for name, model in [("FixedDepthGNN", FixedDepthGNN(12, 4, 64, 8)),
                        ("DEQGNN", DEQGNN(12, 4, 64))]:
        npar = sum(p.numel() for p in model.parameters())
        model = model.to(dev)
        tt = train(model, dtr, args.train_n, args.epochs, dev)
        ckpt = os.path.join(runs_dir, f"{args.run_tag}_{name}.pt")
        torch.save(model.state_dict(), ckpt)
        print(f"\n== {name} ({npar:,} params, {npar*4/1024:.0f} KiB fp32) ==")
        print(f"  train: {tt:.0f}s for {args.train_n}x{args.epochs} "
              f"({tt/(args.train_n*args.epochs)*1000:.0f} ms/sample-epoch)")
        print(f"  checkpoint saved: {ckpt}")
        r_ft = evaluate_audit(model, dte, range(args.test_n), dev,
                              per_example_topo=dte["per_example_topology"])
        print_audit("FullTop test", r_ft)
        if dn1 is not None:
            r_n1 = evaluate_audit(model, dn1, range(args.test_n), dev,
                                  per_example_topo=dn1["per_example_topology"])
            print_audit("N-1 zero-shot", r_n1)
        print(f"  peak RSS: {rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
