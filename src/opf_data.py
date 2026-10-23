"""
AC-OPF dataset generation with pandapower.

For each sample: scale all loads by a random factor in [load_lo, load_hi],
run AC-OPF (`runopp`), and record

  node features  x  : [Pd, Qd, is_gen, is_slack, Pmin, Pmax, Qmin, Qmax,
                       Vmin, Vmax, c2, c1]            (per bus, pu / scaled)
  edge features  e  : [r_pu, x_pu, b_pu, rate_pu]     (per branch, both dirs)
  targets        y  : [Vm, Va] per bus from the OPF solution
  meta              : solver objective (total generation cost), load scale

Branch data and Ybus come from pandapower's internal MATPOWER-format PPC
(`_pd2ppc` + `makeYbus`) so transformer taps and exact per-unit values are
handled correctly -- do NOT reconstruct branch parameters from the physical
line table (that ignores taps and mis-scales case14-style networks).

Physics scoring: with predicted (Vm, Va), bus injections follow
    S = V .* conj(Ybus @ V),
and the power-balance violation at PQ (load-only) buses is |S_calc - S_spec|
with S_spec = -(Pd + j Qd). For a solver solution this is ~1e-6 pu, which
`tests` below assert on a small generated set.

NOTE (research hygiene): this generator is a scaffold for pipeline
development and smoke tests. For publishable experiments, generate from
PGLib-OPF cases or use the OPFData benchmark (see EXPERIMENT.md) so results
are comparable with CANOS / HH-MPNN.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandapower as pp
import pandapower.networks as pn
from pandapower.pd2ppc import _pd2ppc
from pandapower.pypower.makeYbus import makeYbus

CASES = {"case14": pn.case14, "case57": pn.case57, "case118": pn.case118,
         "case300": pn.case300}


def build_static(case_name, net=None):
    """Static grid data: graph, features, Ybus, gen mapping (all pu).

    Pass a pre-built (possibly modified, e.g. an N-1 outage) pandapower net
    to build the static data for that topology; defaults to the intact case.
    """
    if net is None:
        net = CASES[case_name]()
    pp.runpp(net, numba=False)
    _, ppci = _pd2ppc(net)
    base = float(ppci["baseMVA"])
    bus = ppci["bus"]; branch = ppci["branch"]; gen = ppci["gen"]
    n_bus = bus.shape[0]

    Ybus, _, _ = makeYbus(base, bus, branch)
    Ybus = np.asarray(Ybus.todense()) if hasattr(Ybus, "todense") else np.asarray(Ybus)

    # branches in service
    br = branch[branch[:, 10] > 0] if branch.shape[1] > 10 else branch
    branches = np.stack([br[:, 0], br[:, 1], br[:, 2], br[:, 3], br[:, 4],
                         br[:, 5] / base], axis=1)

    # per-bus load at scale 1.0 (MATPOWER bus PD/QD are MW/Mvar)
    base_load_p = bus[:, 2] / base
    base_load_q = bus[:, 3] / base

    is_gen = np.zeros(n_bus); is_slack = np.zeros(n_bus)
    pmin = np.zeros(n_bus); pmax = np.zeros(n_bus)
    qmin = np.zeros(n_bus); qmax = np.zeros(n_bus)
    slack_buses = set(bus[bus[:, 1] == 3, 0].astype(int))  # bus type 3 = slack
    gen_buses = []
    for g in gen:
        if g[7] <= 0:  # gen status off
            continue
        b = int(g[0]); gen_buses.append(b)
        is_gen[b] = 1.0
        if b in slack_buses:
            is_slack[b] = 1.0
        pmax[b] += g[8] / base; pmin[b] += g[9] / base
        qmax[b] += g[3] / base; qmin[b] += g[4] / base
    # clip absurd PPC placeholder limits (e.g. +-1e9) to a sane range
    for arr in (pmin, qmin):
        arr[:] = np.clip(arr, -20.0, 0.0) if arr is qmin else np.clip(arr, -20.0, 20.0)
    pmax[:] = np.clip(pmax, 0.0, 20.0); qmax[:] = np.clip(qmax, 0.0, 20.0)

    c1 = np.zeros(n_bus); c2 = np.zeros(n_bus)
    if len(net.poly_cost):
        for _, pc in net.poly_cost.iterrows():
            el = int(pc.element)
            if pc.et == "gen":
                b = int(net.gen.at[el, "bus"])
            elif pc.et == "ext_grid":
                b = int(net.ext_grid.at[el, "bus"])
            else:
                continue
            c1[b] = pc.cp1_eur_per_mw / 100.0
            c2[b] = pc.cp2_eur_per_mw2 * base / 100.0
    vmin = net.bus.min_vm_pu.fillna(0.9).to_numpy(dtype=np.float64)
    vmax = net.bus.max_vm_pu.fillna(1.1).to_numpy(dtype=np.float64)
    # bus index in pandapower cases is 0..n-1 but res/ordering follows the
    # bus table order; keep an explicit order array for target extraction
    bus_order = list(net.bus.index)

    edge_index, edge_feat = [], []
    for f, t, r, x, b, rate in branches:
        edge_index += [[int(f), int(t)], [int(t), int(f)]]
        edge_feat += [[r, x, b, rate], [r, x, b, rate]]

    return {
        "net_name": case_name, "n_bus": n_bus, "branches": branches,
        "Ybus": Ybus,
        "static_bus": np.stack([is_gen, is_slack, pmin, pmax, qmin, qmax,
                                vmin, vmax, c1, c2], axis=1),
        "base_load_p": base_load_p, "base_load_q": base_load_q,
        "gen_buses": np.array(gen_buses), "vmin": vmin, "vmax": vmax,
        "edge_index": np.array(edge_index).T,
        "edge_feat": np.array(edge_feat), "bus_order": bus_order,
        "_net": net,
    }


def generate(case_name="case14", n_samples=500, load_lo=0.8, load_hi=1.2,
             seed=0, verbose=True, net=None):
    """Generate solved AC-OPF samples. Returns a dict of numpy arrays.

    Pass a modified pandapower `net` (e.g. with a line out of service) to
    generate samples for a perturbed topology; the returned dict carries
    that topology's edge_index / edge_feat / Ybus / static."""
    rng = np.random.default_rng(seed)
    st = build_static(case_name, net=net)
    net = st["_net"]
    base_p = net.load.p_mw.to_numpy().copy()
    base_q = net.load.q_mvar.to_numpy().copy()

    feats, targets, costs, scales = [], [], [], []
    n_fail = 0
    for i in range(n_samples):
        s = rng.uniform(load_lo, load_hi)
        net.load.p_mw = base_p * s
        net.load.q_mvar = base_q * s
        try:
            pp.runopp(net, numba=False, verbose=False)
            ok = bool(net.OPF_converged)
        except Exception:
            ok = False
        if not ok:
            n_fail += 1
            continue
        vm = net.res_bus.vm_pu.to_numpy()
        va = np.deg2rad(net.res_bus.va_degree.to_numpy())
        x = np.concatenate([
            (st["base_load_p"] * s)[:, None], (st["base_load_q"] * s)[:, None],
            st["static_bus"]], axis=1).astype(np.float32)
        feats.append(x)
        targets.append(np.stack([vm, va], axis=1).astype(np.float32))
        costs.append(float(net.res_cost))
        scales.append(s)
        if verbose and (i + 1) % 200 == 0:
            print(f"  {i+1}/{n_samples} generated ({n_fail} failed)")
    if verbose:
        print(f"Generated {len(feats)} solved samples, {n_fail} solver failures.")
    return {
        "x": np.stack(feats), "y": np.stack(targets),
        "cost": np.array(costs), "scale": np.array(scales),
        "edge_index": st["edge_index"],
        "edge_feat": st["edge_feat"].astype(np.float32),
        "Ybus": st["Ybus"], "static": st, "n_fail": n_fail,
    }


def pq_violation(Ybus, vm, va, p_spec, q_spec, pq_mask):
    """Power-balance violation (max, mean) in pu at PQ buses only.

    p_spec/q_spec are the specified NET injections at every bus; the mask
    restricts scoring to buses whose injection is actually specified a
    priori (load-only buses), which is the well-posed version of this metric.
    """
    v = vm * np.exp(1j * va)
    s_calc = v * np.conj(Ybus @ v)
    mis = np.abs(s_calc - (p_spec + 1j * q_spec))
    return float(mis[pq_mask].max()), float(mis[pq_mask].mean())
