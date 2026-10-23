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
    vm, va = vm0.copy(), va0.copy()
    n = len(vm)
    pq_mask = ~is_gen
    # derived PV active-power setpoints from the predicted state
    v0 = vm0 * np.exp(1j * va0)
    s0 = v0 * np.conj(Ybus @ v0)
    p_spec = -pd.copy()
    gen_buses = np.where(is_gen)[0]
    p_spec[gen_buses] = s0.real[gen_buses]  # net injection implied by prediction
    q_spec = -qd.copy()

    va_idx = np.array([i for i in range(n) if i != slack_bus])
    vm_idx = np.where(pq_mask)[0]

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
            "final_mismatch_pu": float(final),
            "move_vm_max": float(np.max(np.abs(vm - vm0))),
            "move_va_max": float(np.max(np.abs(va - va0)))}
    return vm, va, info


# -------------------- inequality-limit violations --------------------

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
    v = vm * np.exp(1j * va)
    s = v * np.conj(Ybus @ v)
    g = np.where(np.asarray(is_gen) > 0)[0]
    pg = s.real[g] + pd[g]
    qg = s.imag[g] + qd[g]
    pg_viol = np.maximum(np.maximum(pg - pmax[g], pmin[g] - pg), 0.0)
    qg_viol = np.maximum(np.maximum(qg - qmax[g], qmin[g] - qg), 0.0)
    vm_viol = np.maximum(np.maximum(vm - vmax, vmin - vm), 0.0)
    return {
        "lim_pg_max": float(np.max(pg_viol, initial=0.0)),
        "lim_pg_frac": float(np.mean(pg_viol > tol)) if len(g) else 0.0,
        "lim_qg_max": float(np.max(qg_viol, initial=0.0)),
        "lim_qg_frac": float(np.mean(qg_viol > tol)) if len(g) else 0.0,
        "lim_vm_max": float(np.max(vm_viol, initial=0.0)),
        "lim_vm_frac": float(np.mean(vm_viol > tol)),
    }


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
