"""
Experiment 1 (see EXPERIMENT.md): fixed-depth GNN vs DEQ-GNN for AC-OPF
surrogate prediction of bus voltages (Vm, Va).

Reports, for both models:
  - parameter count
  - test MSE on (Vm, Va)
  - power-balance violation at load (PQ) buses, max and mean, in pu
    (computed from predicted Vm/Va through the true Ybus equations)
  - mean inference time per instance
  - DEQ only: fixed-point iteration count (mean / p95 / max) and tolerance

Usage:
  python src/run_experiment.py --case case14 --samples 400 --epochs 30
  python src/run_experiment.py --case case118 --samples 5000 --epochs 100 --device cuda
"""
import argparse, time, sys, os
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from opf_data import generate, pq_violation
from models import FixedDepthGNN, DEQGNN
from device import resolve_device, to_dev


def evaluate(model, data, idx, is_deq, dev):
    st = data["static"]
    Ybus = data["Ybus"]
    ei = to_dev(data["edge_index"], dev).long()
    ef = to_dev(data["edge_feat"], dev)
    mses, vmaxs, vmeans, times, iters = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for i in idx:
            x = to_dev(data["x"][i], dev)
            t0 = time.perf_counter()
            if is_deq:
                pred, it, _ = model.solve_with_stats(x, ei, ef)
                iters.append(it)
            else:
                pred = model(x, ei, ef)
            times.append(time.perf_counter() - t0)
            p = pred.detach().cpu().numpy()
            mses.append(float(np.mean((p - data["y"][i]) ** 2)))
            s = data["scale"][i]
            is_pq = (st["static_bus"][:, 0] == 0)  # non-generator buses
            p_spec = -st["base_load_p"] * s
            q_spec = -st["base_load_q"] * s
            mx, mn = pq_violation(Ybus, p[:, 0], p[:, 1],
                                  p_spec, q_spec, is_pq)
            vmaxs.append(mx)
            vmeans.append(mn)
    out = {"mse": float(np.mean(mses)),
           "viol_max_pu": float(np.mean(vmaxs)),
           "viol_mean_pu": float(np.mean(vmeans)),
           "infer_ms": float(np.mean(times) * 1000)}
    if is_deq and iters:
        out["deq_iters_mean"] = float(np.mean(iters))
        out["deq_iters_p95"] = float(np.percentile(iters, 95))
        out["deq_iters_max"] = int(np.max(iters))
    return out


def train_one(model, data, tr_idx, epochs, dev, lr=1e-3):
    model.to(dev)
    ei = to_dev(data["edge_index"], dev).long()
    ef = to_dev(data["edge_feat"], dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for i in tr_idx:
            x = to_dev(data["x"][i], dev); y = to_dev(data["y"][i], dev)
            opt.zero_grad()
            loss = lossf(model(x, ei, ef), y)
            loss.backward()
            opt.step()
            tot += loss.item()
        if (ep + 1) % max(1, epochs // 5) == 0:
            print(f"    epoch {ep+1}/{epochs}  train MSE {tot/len(tr_idx):.6f}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="case14")
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "cuda"],
                    help="auto: cuda if available else cpu")
    args = ap.parse_args()

    dev = resolve_device(args.device)
    print(f"using device: {dev}")
    torch.manual_seed(args.seed)
    print(f"Generating {args.samples} AC-OPF samples ({args.case}) ...")
    data = generate(args.case, args.samples, seed=args.seed)
    n = len(data["x"])
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    tr, te = perm[: int(0.8 * n)], perm[int(0.8 * n):]
    node_dim = data["x"].shape[2]; edge_dim = data["edge_feat"].shape[1]

    results = {}
    for name, model, is_deq in [
        ("FixedDepthGNN", FixedDepthGNN(node_dim, edge_dim, args.hidden, args.depth), False),
        ("DEQGNN", DEQGNN(node_dim, edge_dim, args.hidden), True),
    ]:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n== {name} ({n_params:,} params) ==")
        t0 = time.perf_counter()
        train_one(model, data, tr, args.epochs, dev)
        res = evaluate(model, data, te, is_deq, dev)
        res["params"] = n_params
        res["train_s"] = time.perf_counter() - t0
        results[name] = res
        for k, v in res.items():
            print(f"  {k}: {v}")

    print("\n=== SUMMARY ===")
    for name, r in results.items():
        print(name, r)


if __name__ == "__main__":
    main()
