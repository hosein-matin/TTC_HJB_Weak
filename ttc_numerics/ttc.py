"""Numerical first-contact TTC for the prescribed OVM prediction.

Coordinates are y=(Xi,Lambda,zeta_L), Lambda=v_L-v_F.  The solver preserves
actual initial times, uses classical RK4 with the constant forcing on each
subinterval, and splits every step at forcing jumps, requested snapshot times,
and the terminal time.  Collision roots are bisected on the RK4 one-step map.
Closing-to-opening minima are checked as well as endpoint sign changes.

This is a floating-point numerical solver, not a validated event enclosure.
The discontinuous first-contact classification at grazing cannot be certified
by a step-refinement check.  The reported accuracy checks are empirical and
are separated from the abstract mathematical estimates in the manuscript.
"""
import numpy as np

ALPHA = 2.0
TH = 1.0
DT = 1e-3
AL_VALUES = np.array([0.5, -1.2, 0.8, -0.4, 0.0])
AL_BREAKS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
# Preserve the supplied training distribution and checkpoint normalization.
VMAX = 2.0 * np.tanh(2.0)  # sampling speed cap, NOT the global supremum of Vopt
VOPT_SUP = 1.0 + np.tanh(2.0)
BOX = {"s": (0.0, 1.0), "Xi": (0.0, 1.0), "vF": (0.0, VMAX),
       "zeta": (0.25, 1.75), "Lam": (0.25 - VMAX, 1.75)}
# These inequalities select a test subset; they do not by themselves certify
# compact containment of a continuum neighborhood in the collision domain.
VMIN, GAMMA, GAMMA_T = 0.1, 0.5, 0.2


def Vopt(r):
    return np.tanh(r - 2.0) + np.tanh(2.0)


def aL(t):
    """Right-continuous representative; its endpoint values do not affect flow."""
    t = np.asarray(t)
    k = np.clip(np.searchsorted(AL_BREAKS[1:], t, side="right"), 0, 4)
    return AL_VALUES[k]


def primitive(t):
    """Integral of the prescribed leader acceleration from zero to t."""
    t = np.asarray(t, dtype=float)
    if np.any(t < -1e-13) or np.any(t > TH + 1e-13):
        raise ValueError("Forcing primitive is defined on [0,TH]")
    t = np.clip(t, 0.0, TH)
    lengths = np.maximum(0.0, np.minimum(t[..., None], AL_BREAKS[1:]) - AL_BREAKS[:-1])
    return np.sum(lengths * AL_VALUES, axis=-1)


def field_const(a, y):
    Xi, Lam, ze = y[..., 0], y[..., 1], y[..., 2]
    dy = np.empty_like(y)
    dy[..., 0] = Lam
    dy[..., 1] = a - ALPHA * (Vopt(Xi) - ze + Lam)
    dy[..., 2] = a
    return dy


def field(t, y):
    return field_const(aL(t), y)


def rk4_step(t, y, h):
    """One RK4 step with the forcing fixed to its value at the left endpoint.

    The caller must split steps at forcing breakpoints.  Keeping the same
    forcing in all four stages also avoids assigning the right-hand forcing
    to the endpoint stage of a step ending at a jump.
    """
    y = np.asarray(y, dtype=float)
    hh = h if np.isscalar(h) else np.asarray(h)[:, None]
    a = aL(t)
    k1 = field_const(a, y)
    k2 = field_const(a, y + 0.5 * hh * k1)
    k3 = field_const(a, y + 0.5 * hh * k2)
    k4 = field_const(a, y + hh * k3)
    return y + hh / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def _first_crossing(t, y, h, ynew, nbis):
    """Return (cross, upper bracket). Also check a within-step local minimum."""
    cross = ynew[:, 0] <= 0.0
    upper = h.copy()
    turn = (y[:, 1] < 0.0) & (ynew[:, 1] >= 0.0) & ~cross
    if np.any(turn):
        ii = np.flatnonzero(turn)
        lo, hi = np.zeros(len(ii)), h[ii].copy()
        for _ in range(nbis):
            mid = (lo + hi) * 0.5
            ym = rk4_step(t[ii], y[ii], mid)
            closing = ym[:, 1] < 0.0
            lo = np.where(closing, mid, lo)
            hi = np.where(closing, hi, mid)
        hm = (lo + hi) * 0.5
        ym = rk4_step(t[ii], y[ii], hm)
        entered = ym[:, 0] <= 0.0
        cross[ii[entered]] = True
        upper[ii[entered]] = hm[entered]
    return cross, upper


def ttc(s, y0, snap_times=(), dt=DT, nbis=40, _stop_after=None):
    """Compute capped first-contact time, preserving the supplied initial s.

    s has shape (N,), y0 shape (N,3).  All inputs must be finite and
    0<=s<=TH.  Initial target points Xi<=0 and terminal states stop immediately.
    Output keys used by the original project are preserved.  A snapshot k is
    the state at min(k,V), including the terminal state for a no-hit prediction.
    `hit` includes a contact at the horizon; `in_DD` restricts to strict
    pre-horizon transversal contact from a positive initial gap.
    """
    s = np.asarray(s, dtype=float).copy()
    y = np.asarray(y0, dtype=float).copy()
    if s.ndim != 1 or y.shape != (len(s), 3):
        raise ValueError("Expected s:(N,) and y0:(N,3)")
    if not np.all(np.isfinite(s)) or not np.all(np.isfinite(y)):
        raise ValueError("Initial data must be finite")
    if np.any(s < -1e-13) or np.any(s > TH + 1e-13):
        raise ValueError("Initial times must lie in [0,TH]")
    if dt <= 0 or nbis < 1:
        raise ValueError("dt must be positive and nbis at least one")
    s = np.clip(s, 0.0, TH)
    snaps_order = np.array(sorted(set(float(k) for k in snap_times)))
    if np.any(snaps_order < 0) or not np.all(np.isfinite(snaps_order)):
        raise ValueError("Snapshot durations must be finite and nonnegative")
    N, ns = len(s), len(snaps_order)
    end_time = np.full(N, TH)
    if _stop_after is not None:
        duration = np.broadcast_to(np.asarray(_stop_after, dtype=float), (N,))
        if np.any(duration < 0) or not np.all(np.isfinite(duration)):
            raise ValueError("Stopping durations must be finite and nonnegative")
        end_time = np.minimum(TH, s + duration)
    initial_gap = y[:, 0].copy()
    V = end_time - s
    hit = initial_gap <= 0.0
    V[hit] = 0.0
    lam_imp = np.full(N, np.nan)
    y_imp = np.full((N, 3), np.nan)
    lam_imp[hit] = y[hit, 1]
    y_imp[hit] = y[hit]
    snaps = {float(k): np.full((N, 3), np.nan) for k in snaps_order}
    active = ~hit & (s < end_time)
    t = s.copy()
    next_idx = np.zeros(N, dtype=int)
    if ns and snaps_order[0] == 0.0:
        snaps[0.0][:] = y
        next_idx[:] = 1
    for k in snaps_order:
        snaps[float(k)][~active] = y[~active]
    # A split at each requested time/forcing jump costs at most one extra step.
    max_steps = int(np.ceil(TH / dt)) + ns + len(AL_BREAKS) + 16
    time_tol = 4e-14
    for _ in range(max_steps):
        idx = np.flatnonzero(active)
        if not len(idx):
            break
        tj, yj = t[idx], y[idx]
        seg = np.clip(np.searchsorted(AL_BREAKS[1:], tj, side="right"), 0, 4)
        next_break = AL_BREAKS[seg + 1]
        target = np.minimum(next_break, end_time[idx])
        if ns:
            ni = next_idx[idx]
            next_snap = np.full(len(idx), np.inf)
            pending = ni < ns
            next_snap[pending] = s[idx[pending]] + snaps_order[ni[pending]]
            target = np.minimum(target, next_snap)
        h = np.minimum(dt, target - tj)
        if np.any(h <= 0):
            raise RuntimeError("Nonpositive integration step; snapshot/time bookkeeping failed")
        yn = rk4_step(tj, yj, h)
        cross, upper = _first_crossing(tj, yj, h, yn, nbis)
        if np.any(cross):
            ci = np.flatnonzero(cross)
            c = idx[ci]
            lo, hi = np.zeros(len(c)), upper[ci].copy()
            for _b in range(nbis):
                mid = (lo + hi) * 0.5
                ym = rk4_step(tj[ci], yj[ci], mid)
                neg = ym[:, 0] <= 0.0
                hi = np.where(neg, mid, hi)
                lo = np.where(neg, lo, mid)
            hc = (lo + hi) * 0.5
            yc = rk4_step(tj[ci], yj[ci], hc)
            yc[:, 0] = 0.0
            V[c] = tj[ci] + hc - s[c]
            hit[c] = True
            lam_imp[c] = yc[:, 1]
            y_imp[c] = yc
            y[c] = yc
            t[c] = s[c] + V[c]
            active[c] = False
            for j, k in enumerate(snaps_order):
                pending = next_idx[c] <= j
                snaps[float(k)][c[pending]] = yc[pending]
            next_idx[c] = ns
        keep = idx[~cross]
        if len(keep):
            y[keep] = yn[~cross]
            tn = tj[~cross] + h[~cross]
            targ = target[~cross]
            close = np.abs(tn - targ) <= time_tol
            tn[close] = targ[close]
            t[keep] = tn
            if ns:
                ni = next_idx[keep]
                pending = ni < ns
                due = pending.copy()
                due[pending] = (t[keep[pending]] >= s[keep[pending]] + snaps_order[ni[pending]] - time_tol)
                for j in np.unique(ni[due]):
                    sel = keep[due & (ni == j)]
                    snaps[float(snaps_order[j])][sel] = y[sel]
                    next_idx[sel] += 1
            done = keep[t[keep] >= end_time[keep] - time_tol]
            if len(done):
                t[done] = end_time[done]
                active[done] = False
                for j, k in enumerate(snaps_order):
                    pending = next_idx[done] <= j
                    snaps[float(k)][done[pending]] = y[done[pending]]
                next_idx[done] = ns
    if np.any(active):
        raise RuntimeError("Integration did not reach a stop within its step budget")
    if any(np.isnan(v).any() for v in snaps.values()):
        raise RuntimeError("An output snapshot was not assigned")
    return dict(s=s, V=V, hit=hit, lam_imp=lam_imp, y_imp=y_imp,
                y_final=y, initial_gap=initial_gap, stop_time=t, snaps=snaps)


def in_D(res, vmin=VMIN, gamma=GAMMA, gamma_t=GAMMA_T):
    """Numerically selected transversal test subset; not a coverage certificate."""
    return (in_DD(res) & (res["V"] >= vmin) & (res["lam_imp"] <= -gamma)
            & (res["V"] <= TH - res["s"] - gamma_t))


def in_DD(res):
    positive_start = res.get("initial_gap", np.ones(len(res["V"]))) > 0.0
    return (res["hit"] & positive_start & (res["lam_imp"] < 0.0)
            & (res["V"] > 0.0) & (res["V"] < TH - res["s"] - 1e-13))


def sample_box(n, rng):
    s = rng.uniform(*BOX["s"], n)
    Xi = rng.uniform(*BOX["Xi"], n)
    vF = rng.uniform(*BOX["vF"], n)
    ze = rng.uniform(*BOX["zeta"], n)
    return s, np.stack([Xi, ze - vF, ze], axis=1)


def physical(y, tol=1e-9):
    vF = y[..., 2] - y[..., 1]
    return ((vF >= -tol) & (vF <= VOPT_SUP + tol)
            & (y[..., 2] >= -tol) & (y[..., 2] <= VMAX + tol))


def advance(s, y0, duration, dt=DT, nbis=40):
    """Integrate for duration or until first contact/TH, whichever comes first.

    This computes one stopped-step landing without integrating to the full
    horizon. Return keys: ell (elapsed time), y (landing state), hit, time.
    """
    r = ttc(s, y0, dt=dt, nbis=nbis, _stop_after=duration)
    return dict(ell=r["V"], y=r["y_final"], hit=r["hit"],
                time=r["s"] + r["V"], lam_imp=r["lam_imp"])
