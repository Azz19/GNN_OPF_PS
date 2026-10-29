"""
Experiment 2 — feasibility pivot.

Two complementary mechanisms, so their contributions can be separated:

1. PHYSICS LOSS (training time, differentiable).
   The power-balance mismatch at PQ buses, computed in torch with complex
   arithmetic through the true Ybus, added to the regression loss:
       L = MSE(V, V*) + lambda_phys * mean_PQ |S_calc(V) - S_spec|^2
   S_spec at PQ buses is known a priori: -(Pd + j Qd).

2. NEWTON CORRECTION LAYER (inference time, classical, not learned).
   Starting from the network's predicted state (Vm, Va):
     - derive each PV (generator) bus's active-power setpoint from the
       prediction itself: Pg_i = P_calc,i(V_pred) + Pd_i;
     - hold Vm at PV and slack buses at their predicted values and the
       slack angle at its predicted value;
     - run polar Newton-Raphson on the remaining unknowns
       (Va at non-slack buses, Vm at PQ buses) until the power-balance
       mismatch is below tol.
   The corrected point satisfies the power-balance equations to solver
   tolerance BY CONSTRUCTION. What must then be reported honestly is the
   price: how far the correction moved the state (distance from the OPF
   solution / cost drift) and whether Newton converged at all.

   This is the DC3 completion/correction idea with the correction done by
   Newton instead of gradient steps; in the feasibility-pivot framing, the
   fixed point being solved IS the power-flow equilibrium.
"""
import numpy as np
import torch


# ------------------------- differentiable physics -------------------------

def physics_loss_torch(pred, Ybus_t, pd, qd, pq_mask):
    """pred: (n_bus, 2) [Vm, Va]; Ybus_t: complex tensor; pd/qd: (n_bus,) pu
    (scaled loads for this sample); pq_mask: bool tensor over buses.
    Returns mean squared power-balance mismatch at PQ buses."""
    vm, va = pred[:, 0], pred[:, 1]
    v = torch.complex(vm * torch.cos(va), vm * torch.sin(va))
    s_calc = v * torch.conj(Ybus_t @ v)
    mis = torch.abs(s_calc - torch.complex(-pd, -qd)) ** 2
    return mis[pq_mask].mean()


# --------------------------- Newton correction ----------------------------

def _mismatch_and_jacobian(Ybus, vm, va, p_spec, q_spec, va_idx, vm_idx):
    """Polar power-flow mismatch and Jacobian.
    Unknowns x = [va at va_idx, vm at vm_idx]; equations are P at va_idx
    buses and Q at vm_idx (PQ) buses. Standard formulas (e.g. Grainger &
    Stevenson / MATPOWER dSbus_dV)."""
    G, B = Ybus.real, Ybus.imag
    v = vm * np.exp(1j * va)
    s_calc = v * np.conj(Ybus @ v)
    mis_p = p_spec[va_idx] - s_calc.real[va_idx]
    mis_q = q_spec[vm_idx] - s_calc.imag[vm_idx]
    mis = np.concatenate([mis_p, mis_q])

    n = len(vm)
    th = va
    P = s_calc.real; Q = s_calc.imag
    # off-diagonal / diagonal blocks
    H = np.zeros((n, n)); N = np.zeros((n, n)); M = np.zeros((n, n)); L = np.zeros((n, n))
    for i in range(n):
        for k in range(n):
            if i == k:
                H[i, i] = -Q[i] - B[i, i] * vm[i] ** 2
                N[i, i] = P[i] / vm[i] + G[i, i] * vm[i]
                M[i, i] = P[i] - G[i, i] * vm[i] ** 2
                L[i, i] = Q[i] / vm[i] - B[i, i] * vm[i]
            else:
                dth = th[i] - th[k]
                H[i, k] = vm[i] * vm[k] * (G[i, k] * np.sin(dth) - B[i, k] * np.cos(dth))
                N[i, k] = vm[i] * (G[i, k] * np.cos(dth) + B[i, k] * np.sin(dth))
                M[i, k] = -vm[i] * vm[k] * (G[i, k] * np.cos(dth) + B[i, k] * np.sin(dth))
                L[i, k] = vm[i] * (G[i, k] * np.sin(dth) - B[i, k] * np.cos(dth))
    J = np.block([[H[np.ix_(va_idx, va_idx)], N[np.ix_(va_idx, vm_idx)]],
                  [M[np.ix_(vm_idx, va_idx)], L[np.ix_(vm_idx, vm_idx)]]])
    return mis, J


def _pf_solve(Ybus, vm0, va0, p_spec, q_spec, va_idx, vm_idx,
              tol=1e-4, max_iter=50):
    """Polar Newton-Raphson core: solve P at the va_idx buses and Q at the
    vm_idx (PQ) buses to the given setpoints, starting from (vm0, va0).
    Returns (vm, va, info) with converged / iters / final max mismatch
    (pu, over the specified equations)."""
    vm, va = vm0.copy(), va0.copy()

    def full_mismatch(vm_, va_):
        v_ = vm_ * np.exp(1j * va_)
        s_ = v_ * np.conj(Ybus @ v_)
        mp = np.abs(p_spec[va_idx] - s_.real[va_idx]).max(initial=0.0)
        mq = np.abs(q_spec[vm_idx] - s_.imag[vm_idx]).max(initial=0.0)
        return max(mp, mq)

    iters = 0
    final = full_mismatch(vm, va)
    converged = final < tol
    while not converged and iters < max_iter:
        mis, J = _mismatch_and_jacobian(Ybus, vm, va, p_spec, q_spec,
                                        va_idx, vm_idx)
        try:
            dx = np.linalg.solve(J, mis)
        except np.linalg.LinAlgError:
            break
        va[va_idx] += dx[: len(va_idx)]
        vm[vm_idx] += dx[len(va_idx):]
        iters += 1
        final = full_mismatch(vm, va)
        converged = final < tol or np.max(np.abs(dx)) < 1e-10
    info = {"converged": bool(converged), "iters": iters,
            "final_mismatch_pu": float(final)}
    return vm, va, info


def bus_dispatch(Ybus, vm, va, pd, qd):
    """Bus-level generation implied by a state: Pg = P_inj + Pd and
    Qg = Q_inj + Qd at every bus (meaningful at generator buses)."""
    v = vm * np.exp(1j * va)
    s = v * np.conj(Ybus @ v)
    return s.real + pd, s.imag + qd


def newton_correct(Ybus, vm0, va0, pd, qd, is_gen, slack_bus=0,
                   tol=1e-4, max_iter=50):
    # tol 1e-4 pu: targets are stored float32, which puts a ~1e-6 floor under
    # the achievable mismatch; 1e-4 is still ~1000x tighter than any raw
    # network prediction in these experiments. A stall break (tiny step)
    # also counts as convergence.
    """Correct a predicted state to power-balance feasibility.

    Returns (vm, va, info) with info: converged, iters, final max mismatch
    (pu, over ALL specified equations), and state movement (max |dVm|,
    max |dVa| in rad)."""
    n = len(vm0)
    is_gen_b = np.asarray(is_gen) > 0
    pq_mask = ~is_gen_b
    # derived PV active-power setpoints from the predicted state
    v0 = vm0 * np.exp(1j * va0)
    s0 = v0 * np.conj(Ybus @ v0)
    p_spec = -pd.copy()
    gen_buses = np.where(is_gen_b)[0]
    p_spec[gen_buses] = s0.real[gen_buses]  # net injection implied by prediction
    q_spec = -qd.copy()

    va_idx = np.array([i for i in range(n) if i != slack_bus])
    vm_idx = np.where(pq_mask)[0]
    vm, va, info = _pf_solve(Ybus, vm0, va0, p_spec, q_spec, va_idx, vm_idx,
                             tol=tol, max_iter=max_iter)
    info["move_vm_max"] = float(np.max(np.abs(vm - vm0)))
    info["move_va_max"] = float(np.max(np.abs(va - va0)))
    return vm, va, info


def balance_mismatch(Ybus, vm, va, pd, qd, is_gen, pg_setpoints,
                     slack_bus=0):
    """Max power-balance mismatch (pu) of a state against a dispatch:
    P at every non-slack bus (generator buses specified by pg_setpoints,
    load buses by -pd) and Q at PQ buses (-qd) — the same convention as
    the Newton solver's own convergence check. A stage that clips its
    dispatch without re-solving (audit stage 3) shows up here instead of
    hiding inside the limit metrics."""
    is_gen_b = np.asarray(is_gen) > 0
    v = vm * np.exp(1j * va)
    s = v * np.conj(Ybus @ v)
    p_spec = -pd.copy()
    g = np.where(is_gen_b)[0]
    p_spec[g] = pg_setpoints[g] - pd[g]
    non_slack = np.array([i for i in range(len(vm)) if i != slack_bus])
    pq = np.where(~is_gen_b)[0]
    mp = np.abs(p_spec[non_slack] - s.real[non_slack]).max(initial=0.0)
    mq = np.abs(-qd[pq] - s.imag[pq]).max(initial=0.0)
    return float(max(mp, mq))


def clip_dispatch(Ybus, vm, va, pd, qd, is_gen, pmin, pmax, qmin, qmax,
                  vmin, vmax):
    """Naive clip baseline (Paper 1 audit stage 3): clip the corrected
    state's implied Pg/Qg and its Vm to their limits, with NO re-solve.
    The clipped dispatch satisfies the limits by construction but
    generally breaks power balance — score it with balance_mismatch()
    against the clipped setpoints to expose that. Returns
    (vm_clipped, pg_clipped, qg_clipped) as bus-level arrays."""
    pg, qg = bus_dispatch(Ybus, vm, va, pd, qd)
    pg_c, qg_c = pg.copy(), qg.copy()
    g = np.where(np.asarray(is_gen) > 0)[0]
    pg_c[g] = np.clip(pg[g], pmin[g], pmax[g])
    qg_c[g] = np.clip(qg[g], qmin[g], qmax[g])
    vm_c = np.clip(vm, vmin, vmax)
    return vm_c, pg_c, qg_c


def newton_correct_distributed(Ybus, vm0, va0, pd, qd, is_gen, pmin, pmax,
                               slack_bus=0, tol=1e-4, oob_tol=1e-4,
                               max_iter=50, max_rounds=25):
    """Paper 1, slack-decoding variant B — headroom-proportional
    distributed slack.

    Runs the plain correction first (variant A behaviour: the slack bus
    absorbs the entire balance residual). If the slack generator's Pg
    then sits outside [pmin, pmax], the excess is moved to the other
    generators in proportion to their remaining headroom (pmax - Pg)
    when the slack is over pmax, or in proportion to (Pg - pmin) when it
    is under pmin; the PF solve is repeated with the updated Pg
    setpoints, and the loop repeats until the slack is inside its limits
    or no headroom remains. Failure is reported honestly via
    info["distributed_ok"] = False — the slack is never silently
    clipped, and no setpoint is ever pushed past its own limit (each
    shift is capped by that generator's room).

    Returns (vm, va, info): converged (final PF solve), iters (total
    Newton iterations across all solves), rounds, distributed_ok,
    slack_pg_viol_before / _after (pu beyond-limit magnitude), and
    final_mismatch_pu of the last solve."""
    is_gen_b = np.asarray(is_gen) > 0
    gen_buses = np.where(is_gen_b)[0]
    n = len(vm0)
    va_idx = np.array([i for i in range(n) if i != slack_bus])
    vm_idx = np.where(~is_gen_b)[0]
    q_spec = -qd.copy()

    vm, va, info = newton_correct(Ybus, vm0, va0, pd, qd, is_gen_b,
                                  slack_bus=slack_bus, tol=tol,
                                  max_iter=max_iter)
    total_iters = info["iters"]
    converged = info["converged"]
    final_mismatch = info["final_mismatch_pu"]

    def slack_viol(pg_):
        return max(pg_[slack_bus] - pmax[slack_bus],
                   pmin[slack_bus] - pg_[slack_bus], 0.0)

    pg, _ = bus_dispatch(Ybus, vm, va, pd, qd)
    viol_before = float(slack_viol(pg))
    rounds = 0
    ok = viol_before <= oob_tol
    while not ok and rounds < max_rounds:
        over = pg[slack_bus] > pmax[slack_bus]
        if over:
            need = pg[slack_bus] - pmax[slack_bus]
            room = np.maximum(pmax - pg, 0.0)
        else:
            need = pmin[slack_bus] - pg[slack_bus]
            room = np.maximum(pg - pmin, 0.0)
        room[~is_gen_b] = 0.0
        room[slack_bus] = 0.0
        total_room = float(room.sum())
        if total_room <= 1e-12:
            break  # no headroom anywhere: honest failure, reported below
        shift = room / total_room * min(need, total_room)
        pg_set = pg + shift if over else pg - shift
        p_spec = -pd.copy()
        p_spec[gen_buses] = pg_set[gen_buses] - pd[gen_buses]
        vm, va, sinfo = _pf_solve(Ybus, vm, va, p_spec, q_spec,
                                  va_idx, vm_idx, tol=tol,
                                  max_iter=max_iter)
        total_iters += sinfo["iters"]
        converged = sinfo["converged"]
        final_mismatch = sinfo["final_mismatch_pu"]
        rounds += 1
        pg, _ = bus_dispatch(Ybus, vm, va, pd, qd)
        ok = float(slack_viol(pg)) <= oob_tol
        if not converged:
            break
    info = {"converged": bool(converged), "iters": total_iters,
            "rounds": rounds, "distributed_ok": bool(ok),
            "slack_pg_viol_before": viol_before,
            "slack_pg_viol_after": float(slack_viol(pg)),
            "final_mismatch_pu": float(final_mismatch),
            "move_vm_max": float(np.max(np.abs(vm - vm0))),
            "move_va_max": float(np.max(np.abs(va - va0)))}
    return vm, va, info


# -------------------- inequality-limit violations --------------------

def _oob_summary(pg, qg, vm, is_gen, pmin, pmax, qmin, qmax, vmin, vmax,
                 slack_bus, tol):
    """Shared out-of-bounds accounting over bus-level Pg/Qg/Vm arrays.

    Carries the original limit_violations() metrics (max magnitude and
    fraction of gen buses / buses OOB) plus the Paper 1 audit extras:
    per-class any-OOB flags for THIS example, overall feasibility, and
    the slack generator's Pg violation reported separately (0.0 when the
    slack bus hosts no generator)."""
    g = np.where(np.asarray(is_gen) > 0)[0]
    pg_viol = np.maximum(np.maximum(pg[g] - pmax[g], pmin[g] - pg[g]), 0.0)
    qg_viol = np.maximum(np.maximum(qg[g] - qmax[g], qmin[g] - qg[g]), 0.0)
    vm_viol = np.maximum(np.maximum(vm - vmax, vmin - vm), 0.0)
    out = {
        "lim_pg_max": float(np.max(pg_viol, initial=0.0)),
        "lim_pg_frac": float(np.mean(pg_viol > tol)) if len(g) else 0.0,
        "lim_qg_max": float(np.max(qg_viol, initial=0.0)),
        "lim_qg_frac": float(np.mean(qg_viol > tol)) if len(g) else 0.0,
        "lim_vm_max": float(np.max(vm_viol, initial=0.0)),
        "lim_vm_frac": float(np.mean(vm_viol > tol)),
        "pg_oob": bool(np.max(pg_viol, initial=0.0) > tol),
        "qg_oob": bool(np.max(qg_viol, initial=0.0) > tol),
        "vm_oob": bool(np.max(vm_viol, initial=0.0) > tol),
    }
    out["feasible"] = not (out["pg_oob"] or out["qg_oob"] or out["vm_oob"])
    pos = np.where(g == slack_bus)[0]
    out["slack_pg_viol"] = float(pg_viol[pos[0]]) if len(pos) else 0.0
    out["slack_pg_oob"] = bool(out["slack_pg_viol"] > tol)
    return out


def limit_violations(Ybus, vm, va, pd, qd, is_gen, pmin, pmax, qmin, qmax,
                     vmin, vmax, tol=1e-4):
    """How much a state cheats the inequality bounds the AC OPF enforces.

    The Newton correction layer enforces only the power-balance EQUALITIES,
    so a corrected state can sit "under" the solver's optimum by violating
    generator P/Q limits or voltage bounds. Pg/Qg come from the injection
    equations (Pg = P_inj + Pd, Qg = Q_inj + Qd) at generator buses; bus
    limits are the sums over colocated generators (each gen occupies its
    own bus in these grids). Vm is checked at every bus.

    Returns per-metric: max out-of-bounds magnitude (pu) and the fraction
    of gen buses / buses with any out-of-bounds value. `tol` guards the
    OOB fractions against float32 residue at binding constraints."""
    pg, qg = bus_dispatch(Ybus, vm, va, pd, qd)
    s = _oob_summary(pg, qg, vm, is_gen, pmin, pmax, qmin, qmax, vmin, vmax,
                     slack_bus=-1, tol=tol)
    return {k: s[k] for k in ("lim_pg_max", "lim_pg_frac", "lim_qg_max",
                              "lim_qg_frac", "lim_vm_max", "lim_vm_frac")}


def limit_violation_detail(Ybus, vm, va, pd, qd, is_gen, pmin, pmax, qmin,
                           qmax, vmin, vmax, slack_bus=0, tol=1e-4):
    """limit_violations() plus the Paper 1 audit extras for one example:
    per-class any-OOB flags, overall `feasible` (zero OOB on Pg, Qg and
    Vm at `tol`), and the slack generator's Pg violation on its own."""
    pg, qg = bus_dispatch(Ybus, vm, va, pd, qd)
    return _oob_summary(pg, qg, vm, is_gen, pmin, pmax, qmin, qmax, vmin,
                        vmax, slack_bus=slack_bus, tol=tol)


def dispatch_violation_detail(pg, qg, vm, is_gen, pmin, pmax, qmin, qmax,
                              vmin, vmax, slack_bus=0, tol=1e-4):
    """Same summary as limit_violation_detail() but for a dispatch given
    directly as bus-level Pg/Qg/Vm arrays — the clip stage, whose
    dispatch is not a solved state."""
    return _oob_summary(pg, qg, vm, is_gen, pmin, pmax, qmin, qmax, vmin,
                        vmax, slack_bus=slack_bus, tol=tol)


def derived_cost(Ybus, vm, va, pd, static_bus, base_mva=100.0):
    """Generation cost implied by a state: Pg at gen buses from the
    injection equations, priced with the quadratic cost features stored in
    static_bus (c1 = cp1/100, c2 = cp2 * base/100, from opf_data).
    Self-consistent: use the SAME function for solver and predicted states
    when computing a cost gap."""
    v = vm * np.exp(1j * va)
    s = v * np.conj(Ybus @ v)
    is_gen = static_bus[:, 0] > 0
    pg_pu = s.real[is_gen] + pd[is_gen]      # Pg = P_inj + Pd
    pg_mw = pg_pu * base_mva
    # static_bus columns: [..., vmin(6), vmax(7), c1(8), c2(9)]
    c1 = static_bus[is_gen, 8] * 100.0       # back to cp1 (per MW)
    c2 = static_bus[is_gen, 9] * 100.0 / base_mva  # back to cp2 (per MW^2)
    return float(np.sum(c2 * pg_mw ** 2 + c1 * pg_mw))
