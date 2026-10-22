"""
Experiment 2 — feasibility pivot (see EXPERIMENT.md).

2x2 comparison on the DEQ-GNN:
  physics loss on/off  x  Newton correction layer on/off

Reports per cell: test MSE vs solver voltages, PQ + full-spec power-balance
violation (pu), derived generation-cost gap % (same derived-cost function
for solver and predicted states; matches pandapower res_cost on solver
states for case14), Newton convergence rate and iteration counts, and how
far the correction moved the state.

Usage: python src/run_experiment2.py --case case14 --samples 200 --epochs 10 --lam 1.0
       python src/run_experiment2.py --case case118 --samples 2000 --epochs 30 --device cuda
"""
import argparse, time, sys, os
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from opf_data import generate, pq_violation
from models import DEQGNN
from feasibility import physics_loss_torch, newton_correct, derived_cost
from device import resolve_device, to_dev
from device import resolve_device, to_dev


def train_deq(data, tr_idx, epochs, lam, Ybus_t, pq_mask_t, dev, seed=0):
    torch.manual_seed(seed)
    node_dim = data["x"].shape[2]; edge_dim = data["edge_feat"].shape[1]
    model = DEQGNN(node_dim, edge_dim, hidden=64).to(dev)
    ei = to_dev(data["edge_index"], dev).long(); ef = to_dev(data["edge_feat"], dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    for ep in range(epochs):
        model.train(); tot = 0.0
        for i in tr_idx:
            x = to_dev(data["x"][i], dev); y = to_dev(data["y"][i], dev)
            opt.zero_grad()
            pred = model(x, ei, ef)
            loss = mse(pred, y)
            if lam > 0:
                loss = loss + lam * physics_loss_torch(
                    pred, Ybus_t, x[:, 0], x[:, 1], pq_mask_t)
            loss.backward(); opt.step()
            tot += loss.item()
        if (ep + 1) % max(1, epochs // 5) == 0:
            print(f"    epoch {ep+1}/{epochs} train loss {tot/len(tr_idx):.6f}")
    return model


def eval_model(model, data, te_idx, dev):
    st = data["static"]; Ybus = data["Ybus"]
    is_gen = st["static_bus"][:, 0] > 0
    slack = int(np.where(st["static_bus"][:, 1] > 0)[0][0])
    ei = to_dev(data["edge_index"], dev).long(); ef = to_dev(data["edge_feat"], dev)
    rows = {"raw": [], "corr": []}
    model.eval()
    with torch.no_grad():
        for i in te_idx:
            x = to_dev(data["x"][i], dev)
            pred, _, _ = model.solve_with_stats(x, ei, ef)
            p = pred.detach().cpu().numpy()
            s = data["scale"][i]
            pd = st["base_load_p"] * s; qd = st["base_load_q"] * s
            solver_cost = derived_cost(Ybus, data["y"][i][:, 0],
                                       data["y"][i][:, 1], pd,
                                       st["static_bus"])
            for tag, (vm, va) in [("raw", (p[:, 0], p[:, 1]))]:
                mx, mn = pq_violation(Ybus, vm, va, -pd, -qd, ~is_gen)
                rows[tag].append({
                    "mse": float(np.mean((np.stack([vm, va], 1) - data["y"][i]) ** 2)),
                    "viol_max": mx, "viol_mean": mn,
                    "cost_gap_pct": (derived_cost(Ybus, vm, va, pd, st["static_bus"])
                                     - solver_cost) / solver_cost * 100})
            cvm, cva, info = newton_correct(Ybus, p[:, 0], p[:, 1], pd, qd,
                                            is_gen, slack_bus=slack)
            mx, mn = pq_violation(Ybus, cvm, cva, -pd, -qd, ~is_gen)
            rows["corr"].append({
                "mse": float(np.mean((np.stack([cvm, cva], 1) - data["y"][i]) ** 2)),
                "viol_max": mx, "viol_mean": mn,
                "cost_gap_pct": (derived_cost(Ybus, cvm, cva, pd, st["static_bus"])
                                 - solver_cost) / solver_cost * 100,
                **info})
    def agg(rs):
        out = {k: float(np.mean([r[k] for r in rs]))
               for k in ("mse", "viol_max", "viol_mean", "cost_gap_pct")}
        return out
    out = {"raw": agg(rows["raw"]), "corr": agg(rows["corr"])}
    cr = rows["corr"]
    out["corr"]["newton_converged_rate"] = float(np.mean([r["converged"] for r in cr]))
    out["corr"]["newton_iters_mean"] = float(np.mean([r["iters"] for r in cr]))
    out["corr"]["newton_iters_p95"] = float(np.percentile([r["iters"] for r in cr], 95))
    out["corr"]["move_vm_max"] = float(np.mean([r["move_vm_max"] for r in cr]))
    out["corr"]["move_va_max"] = float(np.mean([r["move_va_max"] for r in cr]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="case14")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "cuda"],
                    help="auto: cuda if available else cpu")
    args = ap.parse_args()

    dev = resolve_device(args.device)
    print(f"using device: {dev}")
    print(f"Generating {args.samples} AC-OPF samples ({args.case}) ...")
    data = generate(args.case, args.samples, seed=args.seed)
    n = len(data["x"])
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    tr, te = perm[: int(0.8 * n)], perm[int(0.8 * n):]
    st = data["static"]
    Ybus_t = torch.tensor(data["Ybus"], dtype=torch.complex64, device=dev)
    pq_mask_t = torch.tensor(st["static_bus"][:, 0] == 0, device=dev)

    for lam, label in [(0.0, "DEQ (MSE only)"), (args.lam, f"DEQ + physics loss (lam={args.lam})")]:
        print(f"\n== {label} ==")
        t0 = time.perf_counter()
        model = train_deq(data, tr, args.epochs, lam, Ybus_t, pq_mask_t, dev, args.seed)
        res = eval_model(model, data, te, dev)
        print(f"  trained in {time.perf_counter()-t0:.0f}s")
        for tag in ("raw", "corr"):
            r = res[tag]
            print(f"  [{tag}] MSE {r['mse']:.3e} | viol mean {r['viol_mean']:.4f} pu, "
                  f"max {r['viol_max']:.4f} pu | cost gap {r['cost_gap_pct']:+.3f}%")
        c = res["corr"]
        print(f"  [corr] Newton converged {c['newton_converged_rate']*100:.0f}% | "
              f"iters mean {c['newton_iters_mean']:.1f} p95 {c['newton_iters_p95']:.0f} | "
              f"state move: dVm {c['move_vm_max']:.4f} pu, dVa {c['move_va_max']:.4f} rad")


if __name__ == "__main__":
    main()
