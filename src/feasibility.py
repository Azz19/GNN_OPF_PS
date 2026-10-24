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


def _newton_solve(Ybus, vm, va, p_spec, q_spec, va_idx, vm_idx,
                tol=1e-4, max_iter=50):
    """Polar Newton-Raphson power-flow solve, in place on copies.

    Unknowns x = [va at va_idx, vm at vm_idx]; equations are P at va_idx
    buses and Q at vm_idx buses. Returns (vm, va, converged, iters,
    final max mismatch over all specified equations). A stall break
    (tiny step) also counts as convergence."""
    vm, va = vm.copy(), va.copy()
    n = len(vm)

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
    return vm, va, bool(converged), iters, float(final)
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
    pq_mask = ~np.asarray(is_gen, dtype=bool)
    # derived PV active-power setpoints from the predicted state
    v0 = vm0 * np.exp(1j * va0)
    s0 = v0 * np.conj(Ybus @ v0)
    p_spec = -pd.copy()
    gen_buses = np.where(np.asarray(is_gen) > 0)[0]
    p_spec[gen_buses] = s0.real[gen_buses]  # net injection implied by prediction
    q_spec = -qd.copy()

    va_idx = np.array([i for i in range(n) if i != slack_bus])
    vm_idx = np.where(pq_mask)[0]
    vm = vm0.copy()
    vm[~pq_mask] = vm0[~pq_mask]  # held at PV/slack buses (explicit)
    va = va0.copy()
    vm, va, converged, iters, final = _newton_solve(
        Ybus, vm, va, p_spec, q_spec, va_idx, vm_idx, tol, max_iter)
    info = {"converged": converged, "iters": iters,
            "final_mismatch_pu": final,
            "move_vm_max": float(np.max(np.abs(vm - vm0))),
            "move_va_max": float(np.max(np.abs(va - va0)))}
    return vm, va, info


# -------------------- limit-aware Newton repair --------------------

def newton_correct_limited(Ybus, vm0, va0, pd, qd, is_gen, pmin, pmax, qmin,
                           qmax, vmin, vmax, slack_bus=0, tol=1e-4,
                           lim_tol=1e-4, max_iter=50, max_outer=25):
    """Limit-aware repair: power-flow correction that also respects the
    AC-OPF inequality bounds, not just the power-balance equalities.

    The plain Newton layer can "beat the optimum" by violating generator
    P/Q limits or voltage bounds (the negative cost-gap finding). This
    wraps the same Newton solve in an outer loop with MATPOWER-style
    limit handling, applying ONE repair per outer iteration, worst
    violation first:

      0. Init (before any solve): held Vm at PV/slack buses clamped to
         [vmin, vmax]; GNN-derived Pg setpoints clamped to [pmin, pmax].
         Clamping held quantities cannot disturb power-balance
         convergence, so this is free.
      1. Newton power-flow solve with the current bus typing.
      2. Inspect the solved state and repair the worst violation:
         - Qg outside [qmin, qmax] at a PV bus -> switch it to PQ with Q
           fixed at the violated limit (standard PV->PQ switching);
         - Vm outside [vmin, vmax] at a PQ (load) bus -> hold Vm at the
           violated bound (bus becomes PV-type with P = -Pd, Q free);
         - Pg outside [pmin, pmax] at the slack -> redispatch: shift the
           excess/deficit across the other generators' P setpoints in
           proportion to their remaining headroom, applying as much as
           the headroom allows.
      3. Re-solve warm-started; stop when every bound holds within
         lim_tol, when the worst violation has no applicable repair
         (a twice-switched bus), when a post-repair solve fails (state
         reverts to the last converged solve), or at max_outer.
      4. Phase B (slack Qg only): the slack's Qg can survive Phase A since
         no bus-type switch applies to the slack. Greedy, sensitivity-
         guided PV voltage coordination (bounded, trust-region steps,
         accepted only if the worst residual over ALL bounds decreases)
         then tries to relieve it by moving held Vm setpoints at PV
         buses within [vmin, vmax], re-solving Newton exactly each time.

    Returns (vm, va, info). info carries converged / iters (total Newton
    iterations) / final_mismatch_pu / move metrics like newton_correct,
    plus outer_iters, repairs (chronological list of what was done),
    limited_ok (balance converged AND every bound within lim_tol), and
    residual = limit_violations() of the final state. Honest by
    construction: a repair that cannot be made is reported in residual,
    never silently dropped. Two robustness guards: a spurious-branch
    fallback (a converged solve with a huge Vm violation at a PQ bus is
    retried from a flat start, keeping the less-violating solution), and
    best-of-trajectory (if a later repair lands somewhere worse, the
    best converged state seen is restored and flagged).
    """
    is_gen = np.asarray(is_gen, dtype=bool)
    n = len(vm0)
    gen = np.where(is_gen)[0]
    gens_noslack = np.array([b for b in gen if b != slack_bus])

    pmin = np.asarray(pmin, dtype=float); pmax = np.asarray(pmax, dtype=float)
    qmin = np.asarray(qmin, dtype=float); qmax = np.asarray(qmax, dtype=float)
    vmin = np.asarray(vmin, dtype=float); vmax = np.asarray(vmax, dtype=float)

    held_vm = np.zeros(n, dtype=bool)   # buses with Vm fixed (PV-type)
    held_vm[gen] = True                # PV buses + slack
    switched = np.zeros(n, dtype=bool)  # PV->PQ switched (Q pinned at limit)

    vm = vm0.copy(); va = va0.copy()
    repairs = []

    def _injections(vm_, va_):
        v_ = vm_ * np.exp(1j * va_)
        s_ = v_ * np.conj(Ybus @ v_)
        return s_.real + pd, s_.imag + qd  # Pg, Qg per bus

    # ---- init: clamp held quantities to bounds (free w.r.t. balance) ----
    # The clamps always apply (harmless at any scale), but only
    # violations above lim_tol are recorded as repairs, so float32
    # residue on binding constraints doesn't inflate the repair log.
    for b in gen:
        lo, hi = vmin[b], vmax[b]
        if vm[b] < lo or vm[b] > hi:
            if max(lo - vm[b], vm[b] - hi) > lim_tol:
                repairs.append({"bus": int(b), "kind": "vm_clamp_init",
                                "from": float(vm[b]),
                                "to": float(min(max(vm[b], lo), hi))})
            vm[b] = min(max(vm[b], lo), hi)
    # derived Pg setpoints from the predicted state, clamped to [pmin, pmax]
    v0 = vm0 * np.exp(1j * va0)
    s0 = v0 * np.conj(Ybus @ v0)
    p_spec = -pd.copy().astype(float)
    p_spec[gen] = s0.real[gen]
    for b in gen:
        pg = p_spec[b] + pd[b]
        pg_c = min(max(pg, pmin[b]), pmax[b])
        if abs(pg_c - pg) > lim_tol:
            repairs.append({"bus": int(b), "kind": "pg_clamp_init",
                            "from": float(pg), "to": float(pg_c)})
        p_spec[b] = pg_c - pd[b]
    q_spec = -qd.copy().astype(float)

    va_idx = np.array([i for i in range(n) if i != slack_bus])
    total_iters = 0
    outer = 0
    converged = False
    final = np.inf
    reverted = False

    def _solve_current(vm_, va_):
        vm_idx_ = np.where(~held_vm)[0]
        vm_idx_ = vm_idx_[vm_idx_ != slack_bus]
        return _newton_solve(Ybus, vm_, va_, p_spec, q_spec,
                             va_idx, vm_idx_, tol, max_iter)

    vm, va, converged, it, final = _solve_current(vm, va)
    total_iters += it

    def _worst_resid(vm_, va_):
        r_ = limit_violations(Ybus, vm_, va_, pd, qd, is_gen, pmin, pmax,
                              qmin, qmax, vmin, vmax, tol=lim_tol)
        return max(r_["lim_pg_max"], r_["lim_qg_max"], r_["lim_vm_max"])

    best = None  # (worst_resid, vm, va, final_mismatch): best converged
                 # state seen on this trajectory; restored at the end if
                 # a later repair made things worse.

    def _track(vm_, va_, converged_, final_):
        nonlocal best
        if not converged_:
            return
        w_ = _worst_resid(vm_, va_)
        if best is None or w_ < best[0] - 1e-12:
            best = (w_, vm_.copy(), va_.copy(), final_)

    _track(vm, va, converged, final)

    if converged:
        # Spurious-branch guard: a converged solve with a huge Vm
        # violation at a PQ bus is almost surely a non-physical Newton
        # branch (bad basin from a wild prediction). Retry from a flat
        # start and continue from whichever converged solution violates
        # the bounds less.
        pq_all = [i for i in range(n) if i != slack_bus and not held_vm[i]]
        vv = max([max(vmin[b] - vm[b], vm[b] - vmax[b])
                  for b in pq_all] + [0.0])
        if vv > 0.25:
            w_before = best[0]
            vm_f = np.ones(n); vm_f[held_vm] = vm[held_vm]
            va_f = np.zeros(n); va_f[slack_bus] = va0[slack_bus]
            vm2, va2, ok2, it2, f2 = _solve_current(vm_f, va_f)
            total_iters += it2
            _track(vm2, va2, ok2, f2)
            if best[0] < w_before - 1e-12:
                _, vm, va, final = best
                repairs.append({"bus": int(slack_bus),
                                "kind": "flatstart_fallback",
                                "worst_from": float(w_before),
                                "worst_to": float(best[0])})

    while converged and outer < max_outer:
        pg, qg = _injections(vm, va)
        # gather repairable violations, worst first
        cands = []  # (magnitude, kind, bus, detail)
        for b in gens_noslack:
            if held_vm[b] and not switched[b]:
                qv = max(qmin[b] - qg[b], qg[b] - qmax[b])
                if qv > lim_tol:
                    cands.append((qv, "pv2pq", b,
                                  "qg_at_limit"))
        pq_buses = [i for i in range(n)
                    if i != slack_bus and not held_vm[i] and not switched[i]]
        for b in pq_buses:
            vv = max(vmin[b] - vm[b], vm[b] - vmax[b])
            if vv > lim_tol:
                cands.append((vv, "vm_hold", b, "vm_at_bound"))
        sv = max(pmin[slack_bus] - pg[slack_bus],
                 pg[slack_bus] - pmax[slack_bus])
        if sv > lim_tol:
            cands.append((sv, "redispatch", slack_bus, "slack_pg"))
        if not cands:
            break
        cands.sort(key=lambda c: -c[0])
        mag, kind, b, _ = cands[0]

        # snapshot everything: on a post-repair solve failure we revert
        # the repair entirely (typing, specs and state) and keep the last
        # converged solve, flagged honestly.
        snap = (vm.copy(), va.copy(), p_spec.copy(), q_spec.copy(),
                held_vm.copy(), switched.copy())
        if kind == "pv2pq":
            held_vm[b] = False
            switched[b] = True
            q_spec[b] = qmin[b] if qg[b] < qmin[b] else qmax[b]
            repairs.append({"bus": int(b), "kind": "pv2pq",
                            "q_pinned_pu": float(q_spec[b]),
                            "viol_pu": float(mag)})
        elif kind == "vm_hold":
            bound = vmin[b] if vm[b] < vmin[b] else vmax[b]
            held_vm[b] = True
            vm[b] = bound
            repairs.append({"bus": int(b), "kind": "vm_hold",
                            "vm_pinned_pu": float(bound),
                            "viol_pu": float(mag)})
        else:  # redispatch around the slack
            target = min(max(pg[slack_bus], pmin[slack_bus]),
                         pmax[slack_bus])
            # need > 0: slack Pg over pmax -> other gens must take MORE so
            # the slack can back down; need < 0: slack under pmin ->
            # others take LESS so the slack can pick up.
            need = pg[slack_bus] - target
            head = np.zeros(len(gens_noslack))
            for j, gb in enumerate(gens_noslack):
                pg_gb = p_spec[gb] + pd[gb]
                head[j] = (pmax[gb] - pg_gb) if need > 0 else (pg_gb - pmin[gb])
            head = np.maximum(head, 0.0)
            tot = head.sum()
            applied = 0.0
            if tot > 1e-12:
                for j, gb in enumerate(gens_noslack):
                    step = need * head[j] / tot
                    step = np.sign(need) * min(abs(step), head[j])
                    p_spec[gb] += step
                    applied += step
            repairs.append({"bus": int(slack_bus), "kind": "redispatch",
                            "need_pu": float(need),
                            "applied_pu": float(applied),
                            "viol_pu": float(mag)})
            if abs(applied) < 1e-12 and abs(need) > lim_tol:
                # no headroom anywhere: the slack violation is unfixable
                # by redispatch; stop rather than spin the outer loop.
                outer += 1
                break
        outer += 1
        vm, va, ok, it, final = _solve_current(vm, va)
        total_iters += it
        if not ok:
            vm, va, p_spec, q_spec, held_vm, switched = snap
            converged = True
            reverted = True
            repairs[-1]["reverted"] = True
            vm_idx_ = np.where(~held_vm)[0]
            vm_idx_ = vm_idx_[vm_idx_ != slack_bus]
            v_ = vm * np.exp(1j * va)
            s_ = v_ * np.conj(Ybus @ v_)
            mp = np.abs(p_spec[va_idx] - s_.real[va_idx]).max(initial=0.0)
            mq = np.abs(q_spec[vm_idx_] - s_.imag[vm_idx_]).max(initial=0.0)
            final = float(max(mp, mq))
            break
        converged = ok
        _track(vm, va, converged, final)

    # ---- Phase B: slack-Q relief via PV voltage coordination ----
    # The slack's Qg can survive Phase A: no bus-type switch applies to the
    # slack. The remaining power-flow lever is the held Vm setpoints at PV
    # buses (raising a PV bus's Vm makes it produce more Q, so the slack
    # absorbs less). Greedy and bounded: pick the most sensitive PV bus
    # with headroom, step its Vm toward relieving the slack Qg, re-solve
    # Newton exactly, and accept only if the worst residual over ALL
    # bounds decreases. Every accepted move is recorded.
    max_coord = 8
    coord = 0
    while converged and coord < max_coord:
        pg, qg = _injections(vm, va)
        qv = max(qmin[slack_bus] - qg[slack_bus],
                 qg[slack_bus] - qmax[slack_bus])
        if qv <= lim_tol:
            break
        resid = limit_violations(Ybus, vm, va, pd, qd, is_gen, pmin, pmax,
                                 qmin, qmax, vmin, vmax, tol=lim_tol)
        worst = max(resid["lim_pg_max"], resid["lim_qg_max"],
                    resid["lim_vm_max"])
        v_base = vm * np.exp(1j * va)
        yrow = Ybus[slack_bus]
        h = 1e-5
        scored = []
        for b in gens_noslack:
            if not (held_vm[b] and not switched[b]):
                continue
            dv = h * np.exp(1j * va[b])
            vp = v_base.copy(); vp[b] += dv
            vmn = v_base.copy(); vmn[b] -= dv
            qp = float((v_base[slack_bus] * np.conj(yrow @ vp)).imag)
            qm = float((v_base[slack_bus] * np.conj(yrow @ vmn)).imag)
            scored.append((abs((qp - qm) / (2 * h)), b, (qp - qm) / (2 * h)))
        scored.sort(key=lambda t: -t[0])
        target = (qmin[slack_bus] if qg[slack_bus] < qmin[slack_bus]
                  else qmax[slack_bus])
        need = target - qg[slack_bus]
        improved = False
        for _, b, s_b in scored:
            if abs(s_b) < 1e-9:
                continue
            step = need / s_b
            step = float(np.sign(step) * min(abs(step), 0.02))  # trust region
            old_v = float(vm[b])
            new_v = min(max(vm[b] + step, vmin[b]), vmax[b])
            if abs(new_v - old_v) < 1e-9:
                continue
            vm_try = vm.copy(); vm_try[b] = new_v
            vm2, va2, ok2, it2, f2 = _solve_current(vm_try, va)
            total_iters += it2
            if not ok2:
                continue
            r2 = limit_violations(Ybus, vm2, va2, pd, qd, is_gen, pmin,
                                  pmax, qmin, qmax, vmin, vmax, tol=lim_tol)
            worst2 = max(r2["lim_pg_max"], r2["lim_qg_max"], r2["lim_vm_max"])
            if worst2 < worst - 1e-9:
                vm, va, converged, final = vm2, va2, True, f2
                repairs.append({"bus": int(b), "kind": "vm_coord",
                                "vm_from": old_v, "vm_to": float(new_v),
                                "slack_qg_from": float(qg[slack_bus]),
                                "worst_from": float(worst),
                                "worst_to": float(worst2)})
                _track(vm, va, True, final)
                improved = True
                break
        coord += 1
        if not improved:
            break

    restored_best = False
    if best is not None:
        w_final = _worst_resid(vm, va)
        if best[0] < w_final - 1e-12:
            _, vm, va, final = best
            restored_best = True

    residual = limit_violations(Ybus, vm, va, pd, qd, is_gen, pmin, pmax,
                                qmin, qmax, vmin, vmax, tol=lim_tol)
    worst_resid = max(residual["lim_pg_max"], residual["lim_qg_max"],
                      residual["lim_vm_max"])
    info = {"converged": bool(converged), "iters": total_iters,
            "outer_iters": outer, "final_mismatch_pu": float(final),
            "move_vm_max": float(np.max(np.abs(vm - vm0))),
            "move_va_max": float(np.max(np.abs(va - va0))),
            "repairs": repairs, "reverted": reverted,
            "restored_best": restored_best,
            "limited_ok": bool(converged and worst_resid <= lim_tol),
            "residual": residual,
            "n_pv2pq": sum(1 for r in repairs if r["kind"] == "pv2pq"),
            "n_vm_hold": sum(1 for r in repairs if r["kind"] == "vm_hold"),
            "n_vm_coord": sum(1 for r in repairs if r["kind"] == "vm_coord"),
            "n_redispatch": sum(1 for r in repairs
                                if r["kind"] == "redispatch")}
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
