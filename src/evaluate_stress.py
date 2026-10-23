"""
Experiment 3 — risk-axis stress evaluation (see EXPERIMENT.md).

Trains both predictors (FixedDepthGNN, DEQGNN, MSE-only) on the nominal
0.8-1.2 load band of a case, then evaluates the feasibility-corrected
pipeline on two stress splits the training never saw:

  A. stressed loads : scales {1.05, 1.15, 1.25, 1.30} (out-of-distribution)
  B. N-1 topology   : one line out of service per test grid, nominal load

Reports tail metrics — max / p99 violation, Newton-correction convergence
rate, cost gap — for raw predictions and corrected states. Tails, not means,
are the risk story: the literature reports means, and failures live in tails.
Honestly counted alongside: solver failures (cases even the oracle cannot
solve) and branches skipped because the outage islands the grid.

Usage:
  python src/evaluate_stress.py --case case14 --samples 200 --epochs 10 --device cpu
  python src/evaluate_stress.py --case case118 --samples 2000 --epochs 30 --device cuda
"""
import argparse, time, sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from opf_data import generate, build_static, pq_violation, CASES
from models import FixedDepthGNN, DEQGNN
from feasibility import newton_correct, derived_cost
from device import resolve_device, to_dev
from run_experiment import train_one
from run_experiment2 import train_deq


def connected(edge_index, n_bus):
    """BFS connectivity check on the undirected graph."""
    adj = [[] for _ in range(n_bus)]
    for f, t in edge_index.T:
        adj[f].append(t)
    seen = {0}
    stack = [0]
    while stack:
        u = stack.pop()
        for w in adj[u]:
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return len(seen) == n_bus


def gen_split(case_name, net_mod, label, n_samples, load_lo, load_hi, seed):
    """One stress split; returns flat sample list + solver failure count."""
    data = generate(case_name, n_samples, load_lo=load_lo, load_hi=load_hi,
                    seed=seed, verbose=False, net=net_mod)
    st = data["static"]
    samples = []
    for i in range(len(data["x"])):
        samples.append({
            "x": data["x"][i], "y": data["y"][i],
            "scale": float(data["scale"][i]),
            "edge_index": data["edge_index"], "edge_feat": data["edge_feat"],
            "Ybus": data["Ybus"], "static": st, "split": label,
        })
    return samples, data["n_fail"]


def eval_sample(model, is_deq, s, dev):
    """Raw prediction + Newton-corrected metrics for one sample."""
    st = s["static"]
    Ybus = s["Ybus"]
    is_gen = st["static_bus"][:, 0] > 0
    slack = int(np.where(st["static_bus"][:, 1] > 0)[0][0])
    ei = to_dev(s["edge_index"], dev).long()
    ef = to_dev(s["edge_feat"], dev)
    x = to_dev(s["x"], dev)
    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        if is_deq:
            pred, iters, _ = model.solve_with_stats(x, ei, ef)
        else:
            pred = model(x, ei, ef)
            iters = None
        infer_ms = (time.perf_counter() - t0) * 1000.0
        p = pred.detach().cpu().numpy()
    pd = st["base_load_p"] * s["scale"]
    qd = st["base_load_q"] * s["scale"]
    solver_cost = derived_cost(Ybus, s["y"][:, 0], s["y"][:, 1], pd,
                               st["static_bus"])
    raw_mx, raw_mn = pq_violation(Ybus, p[:, 0], p[:, 1], -pd, -qd, ~is_gen)
    raw_cost = derived_cost(Ybus, p[:, 0], p[:, 1], pd, st["static_bus"])
    cvm, cva, info = newton_correct(Ybus, p[:, 0], p[:, 1], pd, qd,
                                    is_gen, slack_bus=slack)
    c_mx, c_mn = pq_violation(Ybus, cvm, cva, -pd, -qd, ~is_gen)
    c_cost = derived_cost(Ybus, cvm, cva, pd, st["static_bus"])
    return {
        "split": s["split"],
        "mse": float(np.mean((np.stack([p[:, 0], p[:, 1]], 1) - s["y"]) ** 2)),
        "raw_viol_max": raw_mx, "raw_viol_mean": raw_mn,
        "raw_cost_gap_pct": (raw_cost - solver_cost) / solver_cost * 100,
        "corr_viol_max": c_mx, "corr_viol_mean": c_mn,
        "corr_cost_gap_pct": (c_cost - solver_cost) / solver_cost * 100,
        "converged": info["converged"], "newton_iters": info["iters"],
        "infer_ms": infer_ms, "deq_iters": iters,
    }


def summarize(rows):
    r = np.asarray
    out = {"n": len(rows)}
    for k in ("mse", "raw_viol_max", "corr_viol_max", "corr_cost_gap_pct"):
        v = r([x[k] for x in rows])
        out[k] = {"mean": float(v.mean()), "p99": float(np.percentile(v, 99)),
                  "max": float(v.max())}
    out["newton_converged_rate"] = float(np.mean([x["converged"] for x in rows]))
    it = r([x["newton_iters"] for x in rows])
    out["newton_iters"] = {"mean": float(it.mean()),
                           "p95": float(np.percentile(it, 95)),
                           "max": int(it.max())}
    out["infer_ms_mean"] = float(np.mean([x["infer_ms"] for x in rows]))
    dit = [x["deq_iters"] for x in rows if x["deq_iters"] is not None]
    if dit:
        out["deq_iters_mean"] = float(np.mean(dit))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="case14")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--stress-per-scale", type=int, default=25)
    ap.add_argument("--n1-per-branch", type=int, default=8)
    ap.add_argument("--max-branches", type=int, default=0,
                    help="0 = all lines")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    dev = resolve_device(args.device)
    print(f"using device: {dev}")
    torch.manual_seed(args.seed)

    # ---- train both predictors on the nominal band (in-distribution) ----
    print(f"Generating {args.samples} nominal AC-OPF samples ({args.case}) ...")
    data = generate(args.case, args.samples, seed=args.seed)
    n = len(data["x"])
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    tr = perm[: int(0.8 * n)]
    node_dim = data["x"].shape[2]
    edge_dim = data["edge_feat"].shape[1]
    Ybus_t = torch.tensor(data["Ybus"], dtype=torch.complex64, device=dev)
    pq_mask_t = torch.tensor(data["static"]["static_bus"][:, 0] == 0, device=dev)

    models = []
    for name, mk, is_deq in [
        ("FixedDepthGNN", lambda: FixedDepthGNN(node_dim, edge_dim, 64, 8), False),
        ("DEQGNN", lambda: DEQGNN(node_dim, edge_dim, 64), True),
    ]:
        m = mk()
        print(f"\nTraining {name} (MSE-only, nominal band) ...")
        t0 = time.perf_counter()
        if is_deq:
            m = train_deq(data, tr, args.epochs, 0.0, Ybus_t, pq_mask_t, dev,
                          args.seed)
        else:
            m = train_one(m, data, tr, args.epochs, dev)
        print(f"  trained in {time.perf_counter() - t0:.0f}s")
        models.append((name, m, is_deq))

    # ---- stress split A: out-of-distribution loads ----
    print("\n== Stress split A: heavy loads ==")
    splitA, failA = [], 0
    for sc in (1.05, 1.15, 1.25, 1.30):
        s, f = gen_split(args.case, None, f"load_{sc:.2f}",
                         args.stress_per_scale, sc, sc, seed=args.seed + 11)
        splitA += s
        failA += f
        print(f"  scale {sc:.2f}: {len(s)} solved, {f} solver failures")

    # ---- stress split B: N-1 line outages ----
    print("\n== Stress split B: N-1 line outages ==")
    base_net = CASES[args.case]()
    n_lines = len(base_net.line)
    branch_ids = list(range(n_lines))
    if args.max_branches:
        branch_ids = branch_ids[: args.max_branches]
    splitB, failB, skipped = [], 0, []
    per_branch = {}
    for b in branch_ids:
        net2 = CASES[args.case]()
        net2.line.at[b, "in_service"] = False
        st2 = build_static(args.case, net=net2)
        if not connected(st2["edge_index"], st2["n_bus"]):
            skipped.append((b, "islands the grid"))
            continue
        s, f = gen_split(args.case, net2, f"n1_line{b}",
                         args.n1_per_branch, 1.0, 1.0, seed=args.seed + 101 + b)
        if not s:
            skipped.append((b, "solver failed on all samples"))
            continue
        splitB += s
        failB += f
        per_branch[b] = s
    print(f"  {len(per_branch)} branches tested, {len(skipped)} skipped "
          f"({len(splitB)} solved samples, {failB} solver failures)")
    for b, why in skipped:
        print(f"    line {b}: skipped — {why}")

    # ---- evaluate ----
    results = {}
    for name, m, is_deq in models:
        print(f"\n== Evaluating {name} on stress splits ==")
        rowsA = [eval_sample(m, is_deq, s, dev) for s in splitA]
        rowsB = [eval_sample(m, is_deq, s, dev) for s in splitB]
        results[name] = {
            "A_heavy_load": summarize(rowsA), "A_solver_fail": failA,
            "B_n1": summarize(rowsB), "B_solver_fail": failB,
            "B_branches_tested": len(per_branch),
            "B_branches_skipped": len(skipped),
        }
        # worst N-1 branches by corrected cost-gap tail (feasibility holds
        # everywhere here, so optimality is where the risk lives)
        worst = []
        for b, ss in per_branch.items():
            rs = [r for r in rowsB if r["split"] == f"n1_line{b}"]
            worst.append((b, max(x["corr_cost_gap_pct"] for x in rs),
                          float(np.mean([x["converged"] for x in rs]))))
        worst.sort(key=lambda t: -t[1])
        results[name]["B_worst_branches"] = [
            {"line": b, "worst_corr_cost_gap_pct": v, "converged_rate": c}
            for b, v, c in worst[:5]]
        for split, key in (("A_heavy_load", "A"), ("B_n1", "B")):
            sm = results[name][split]
            print(f"  [{key}] n={sm['n']} | corr viol_max mean {sm['corr_viol_max']['mean']:.4f} "
                  f"p99 {sm['corr_viol_max']['p99']:.4f} max {sm['corr_viol_max']['max']:.4f} pu | "
                  f"Newton converged {sm['newton_converged_rate']*100:.1f}% | "
                  f"cost gap mean {sm['corr_cost_gap_pct']['mean']:+.2f}% "
                  f"max {sm['corr_cost_gap_pct']['max']:+.2f}%")
        print(f"  [B] worst branches: " +
              ", ".join(f"line{b['line']} (cost gap {b['worst_corr_cost_gap_pct']:+.1f}%, "
                        f"conv {b['converged_rate']*100:.0f}%)"
                        for b in results[name]["B_worst_branches"]))

    # ---- write-up ----
    lines = ["# Experiment 3 — risk-axis stress evaluation", "",
             f"Case {args.case}; predictors trained {args.epochs} epochs on "
             f"{len(tr)} nominal-band (0.8-1.2x load) samples, seed {args.seed}, "
             f"device {dev}. Stress splits are out-of-distribution by construction.",
             "",
             "## Split A — heavy loads (1.05x to 1.30x)",
             f"Solver itself failed on {failA} stressed samples — the risk frontier "
             "where even the oracle has no answer.", ""]
    for name in results:
        sm = results[name]["A_heavy_load"]
        lines += [f"### {name}",
                  f"- corrected viol_max: mean {sm['corr_viol_max']['mean']:.4f}, "
                  f"p99 {sm['corr_viol_max']['p99']:.4f}, max {sm['corr_viol_max']['max']:.4f} pu",
                  f"- Newton converged {sm['newton_converged_rate']*100:.1f}% "
                  f"(iters mean {sm['newton_iters']['mean']:.1f}, max {sm['newton_iters']['max']})",
                  f"- corrected cost gap: mean {sm['corr_cost_gap_pct']['mean']:+.2f}%, "
                  f"max {sm['corr_cost_gap_pct']['max']:+.2f}%", ""]
    lines += ["## Split B — N-1 line outages (nominal load)",
              f"{results['DEQGNN']['B_branches_tested']} branches tested, "
              f"{results['DEQGNN']['B_branches_skipped']} skipped (islanding / solver infeasible), "
              f"{failB} solver failures on tested branches.", ""]
    for name in results:
        sm = results[name]["B_n1"]
        lines += [f"### {name}",
                  f"- corrected viol_max: mean {sm['corr_viol_max']['mean']:.4f}, "
                  f"p99 {sm['corr_viol_max']['p99']:.4f}, max {sm['corr_viol_max']['max']:.4f} pu",
                  f"- Newton converged {sm['newton_converged_rate']*100:.1f}%",
                  f"- corrected cost gap: mean {sm['corr_cost_gap_pct']['mean']:+.2f}%, "
                  f"max {sm['corr_cost_gap_pct']['max']:+.2f}%",
                  "- worst branches (by corrected cost-gap tail): " + "; ".join(
                      f"line {b['line']}: cost gap {b['worst_corr_cost_gap_pct']:+.1f}%, "
                      f"conv {b['converged_rate']*100:.0f}%"
                      for b in results[name]["B_worst_branches"]), ""]
    lines += ["## Caveats",
              "- case14, toy training (10 epochs): validates the harness and the "
              "tail-reporting methodology, not a publishable risk claim.",
              "- Newton correction needs a converged power flow; on contingencies "
              "where it stalls, the raw prediction is what you have — count those.",
              "- Stressed-load samples re-solve OPF per sample; solver failures at "
              "1.25-1.30x are expected and are data, not bugs.",
              "- Next: same harness on case118 with OPFData N-1 split, on the GPU."]
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(repo, "RESULTS_stress.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nWrote RESULTS_stress.md")


if __name__ == "__main__":
    main()
