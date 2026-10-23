"""
OPFData loader — converts OPFData (Lovett et al. 2024) tar shards into the
tensor contract used by opf_data.generate(), so run_experiment.py,
run_experiment2.py and evaluate_stress.py work on the published benchmark
unchanged in structure.

SCHEMA — decoded empirically from pglib_opf_case14_ieee shard 0 on
2026-10-23 (see schema_probe.py output; do not re-derive from memory):

grid.nodes.bus        [n_bus, 4]   [const 1.0, bus_type (3=ref, 2=PV, 1=PQ),
                                    vmin, vmax]
grid.nodes.generator  [n_gen, 11]  [mbase, ?, pmin, pmax, ?, qmin, qmax,
                                    const 1.0, cost_c2, cost_c1, cost_c0]
    Cost decode VERIFIED against metadata.objective over 500 examples:
        objective = sum_g (c2 * Pg^2 + c1 * Pg) + c0     (Pg in pu)
        max abs error 2e-13.  Empirical ranges also confirm qmax=col 6,
        qmin=col 5, pmax=col 3 (Pg never exceeds it; col 1 is NOT pmax).
grid.nodes.load       [n_load, 2]  [pd, qd] pu (per-example values)
grid.nodes.shunt      [n_shunt, 2] [bs, gs] pu — IEEE14's only shunt is the
                                    0.19 pu capacitor at bus 9, stored as
                                    [0.19, 0.0]; the physics validation in
                                    validate_opfdata.py arbitrates the order.
grid.edges.ac_line    features [9]:  [angmin, angmax, b_fr, b_to, r, x,
                                      rate_a, rate_b, rate_c]
grid.edges.transformer features [11]: [angmin, angmax, r, x, rate_a, rate_b,
                                      rate_c, tap, shift, 0, 0]  (no b)
solution.nodes.bus    [n_bus, 2]   [va (rad), vm]     <-- note the order
solution.nodes.generator [n_gen, 2] [pg, qg] pu
metadata.objective    float, total generation cost

Per-example loads vary (unlike the pandapower generator), so the returned
dict carries per-example 'pd'/'qd' arrays; evaluators should prefer them
over static base_load * scale when present. For the N-1 variant the
topology varies per example: 'Ybus' is then a list and 'edge_index' /
'edge_feat' are per-example lists as well (static-Ybus consumers should
use FullTop).
"""
import io
import json
import tarfile
import numpy as np

BASE_MVA = 100.0


def _iter_examples(tar_path, limit=None):
    with tarfile.open(tar_path) as tf:
        members = sorted([m for m in tf.getmembers()
                          if m.name.endswith(".json")],
                         key=lambda m: int(m.name.rsplit("_", 1)[1]
                                           .split(".")[0]))
        for k, m in enumerate(members):
            if limit is not None and k >= limit:
                break
            yield json.load(tf.extractfile(m))


def _build_ybus(n_bus, branches, shunts):
    """branches: (f, t, r, x, b_total, tap). shunts: (bus, gs, bs)."""
    Y = np.zeros((n_bus, n_bus), dtype=complex)
    for f, t, r, x, b, tap in branches:
        y = 1.0 / (r + 1j * x)
        if tap and tap > 0:
            Y[f, f] += y / tap ** 2
            Y[t, t] += y
            Y[f, t] -= y / tap
            Y[t, f] -= y / tap
        else:
            Y[f, f] += y + 1j * b / 2
            Y[t, t] += y + 1j * b / 2
            Y[f, t] -= y
            Y[t, f] -= y
    for bus, gs, bs in shunts:
        Y[bus, bus] += gs + 1j * bs
    return Y


def _convert_example(ex):
    g, sol = ex["grid"], ex["solution"]
    bus = np.array(g["nodes"]["bus"], dtype=np.float64)
    gen = np.array(g["nodes"]["generator"], dtype=np.float64)
    load = np.array(g["nodes"]["load"], dtype=np.float64)
    shunt = np.array(g["nodes"]["shunt"], dtype=np.float64) \
        if len(g["nodes"]["shunt"]) else np.zeros((0, 2))
    n_bus = bus.shape[0]

    gen_bus = np.array(g["edges"]["generator_link"]["receivers"], dtype=int)
    load_bus = np.array(g["edges"]["load_link"]["receivers"], dtype=int)
    shunt_bus = np.array(g["edges"]["shunt_link"]["receivers"], dtype=int) \
        if len(shunt) else np.zeros(0, dtype=int)

    pd = np.zeros(n_bus); qd = np.zeros(n_bus)
    for (p, q), b in zip(load, load_bus):
        pd[b] += p; qd[b] += q

    is_gen = np.zeros(n_bus); is_slack = np.zeros(n_bus)
    pmin = np.zeros(n_bus); pmax = np.zeros(n_bus)
    qmin = np.zeros(n_bus); qmax = np.zeros(n_bus)
    c1 = np.zeros(n_bus); c2 = np.zeros(n_bus)
    for row, b in zip(gen, gen_bus):
        is_gen[b] = 1.0
        pmin[b] += row[2]; pmax[b] += row[3]
        qmin[b] += row[5]; qmax[b] += row[6]
        c1[b] += row[9]; c2[b] += row[8]
    is_slack = (bus[:, 1] == 3).astype(np.float64)
    vmin, vmax = bus[:, 2], bus[:, 3]

    # feature scaling to roughly match the pandapower contract magnitudes
    static_bus = np.stack([is_gen, is_slack, pmin, pmax, qmin, qmax,
                           vmin, vmax, c1 / 1000.0, c2 / 1000.0], axis=1)
    x = np.concatenate([pd[:, None], qd[:, None], static_bus], axis=1)

    branches = []        # (f, t, r, x, b_total, tap)
    edge_index, edge_feat = [], []
    al = g["edges"]["ac_line"]
    for s, r_, f in zip(al["senders"], al["receivers"], al["features"]):
        rr, xx, b = f[4], f[5], f[2] + f[3]
        branches.append((s, r_, rr, xx, b, 0.0))
        edge_index += [[s, r_], [r_, s]]
        edge_feat += [[rr, xx, b, f[6]]] * 2
    tr = g["edges"]["transformer"]
    for s, r_, f in zip(tr["senders"], tr["receivers"], tr["features"]):
        rr, xx, tap = f[2], f[3], f[7]
        branches.append((s, r_, rr, xx, 0.0, tap))
        edge_index += [[s, r_], [r_, s]]
        edge_feat += [[rr, xx, 0.0, f[4]]] * 2

    shunts = [(int(b), float(gs), float(bs))
              for b, (bs, gs) in zip(shunt_bus, shunt)]  # stored [bs, gs]
    Ybus = _build_ybus(n_bus, branches, shunts)

    sb = np.array(sol["nodes"]["bus"], dtype=np.float64)  # [va, vm]
    y = np.stack([sb[:, 1], sb[:, 0]], axis=1)             # -> [vm, va]
    return {
        "x": x.astype(np.float32), "y": y.astype(np.float32),
        "cost": float(ex["metadata"]["objective"]),
        "pd": pd, "qd": qd, "Ybus": Ybus,
        "edge_index": np.array(edge_index).T,
        "edge_feat": np.array(edge_feat, dtype=np.float32),
        "static_bus": static_bus, "gen_bus": gen_bus,
        "gen_cost": gen[:, 8:11],  # per-gen [c2, c1, c0]
        "sol_pg": np.array(sol["nodes"]["generator"])[:, 0],
    }


def load_opfdata(tar_path, limit=None, verbose=True):
    """Load up to `limit` examples from one OPFData shard tar.

    Returns the generate()-style dict. For FullTop shards the topology is
    fixed, so 'edge_index'/'edge_feat'/'Ybus'/'static' are shared; if the
    topology varies across examples (N-1), they are returned per-example
    in lists under the same keys plus 'per_example_topology': True.
    """
    conv = [_convert_example(ex) for ex in _iter_examples(tar_path, limit)]
    if verbose:
        print(f"Loaded {len(conv)} OPFData examples from {tar_path}")
    first = conv[0]
    same_topo = all(np.array_equal(c["edge_index"], first["edge_index"])
                    for c in conv)
    out = {
        "x": np.stack([c["x"] for c in conv]),
        "y": np.stack([c["y"] for c in conv]),
        "cost": np.array([c["cost"] for c in conv]),
        "pd": np.stack([c["pd"] for c in conv]),
        "qd": np.stack([c["qd"] for c in conv]),
        "sol_pg": np.stack([c["sol_pg"] for c in conv]),
        "scale": np.ones(len(conv)),
        "static": {"static_bus": first["static_bus"],
                   "base_load_p": conv[0]["pd"], "base_load_q": conv[0]["qd"],
                   "gen_bus": first["gen_bus"], "gen_cost": first["gen_cost"],
                   "vmin": first["static_bus"][:, 6],
                   "vmax": first["static_bus"][:, 7]},
        "source": "opfdata",
    }
    if same_topo:
        out["edge_index"] = first["edge_index"]
        out["edge_feat"] = first["edge_feat"]
        out["Ybus"] = first["Ybus"]
        out["per_example_topology"] = False
    else:
        out["edge_index"] = [c["edge_index"] for c in conv]
        out["edge_feat"] = [c["edge_feat"] for c in conv]
        out["Ybus"] = [c["Ybus"] for c in conv]
        out["per_example_topology"] = True
    return out


def state_cost(Ybus, vm, va, pd, gen_bus, gen_cost, sol_pg):
    """Generation cost of an arbitrary state, OPFData accounting.

    Bus-level Pg comes from the injection equations; where several gens
    share a bus, the bus Pg is split in the solution's proportions (the
    only consistent choice without per-gen predictions). Exact when each
    gen bus hosts one generator, as in case14."""
    v = vm * np.exp(1j * va)
    s = v * np.conj(Ybus @ v)
    pg_bus = s.real + pd
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
