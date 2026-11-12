#!/usr/bin/env python
"""APPDF (adaptive Gaussian-mixture filter with pairwise-distance merging) for the 6-D coupled
Van der Pol system.

Model (three ring-coupled Van der Pol oscillators, state s = (x1, y1, x2, y2, x3, y3)):

    dx_i = y_i dt
    dy_i = [mu (1 - x_i^2) y_i - x_i + k (x_{i-1} - x_i) + k (x_{i+1} - x_i)] dt + dW_i ,
    E[dW dW^T] = Q dt   (additive noise on all six states, Q = 6e-3 I6 in the paper case)

Measurement (per oscillator, polar):

    h(s) = (r_1, theta_1, r_2, theta_2, r_3, theta_3),  r_i = sqrt(x_i^2 + y_i^2),  theta_i = atan2(y_i, x_i)
    z = h(s) + v,  v ~ N(0, R),  R = diag(0.2^2, 0.2^2) x 3

Algorithm: every Gaussian component (weight w, mean mu, square-root factor M) is propagated with the
LSKF mean / square-root ODE. A component is split into three along the principal axis of its
covariance (SVD) with the largest linearisation error whenever that error exceeds `tol`
(rank-1 update of M). After each step, components whose means are closer than `combine_radius`
(within the same grid block of size `blocksize`) are merged by moment matching, and components with
weight <= `weight_thresh` are discarded. Measurements are assimilated with a square-root cubature
update per component plus a weight update. Components are propagated in a multiprocessing pool.

Paper case: mu = 1.5, k = 0.2, x0 = (2, 0) x 3, P0 = 0.01 I6, dt = 0.01, t in [0, 10], one measurement
at t = 5; tol = 0.05, min_dt = 0.0005, weight_thresh = 1e-4, blocksize = 1.0, combine_radius = 0.005.

Usage:
    python coupled_vdp6d/appdf.py                      # paper case
    python coupled_vdp6d/appdf.py --t-end 2 --no-measurements --workers 4 --out results/test
    python coupled_vdp6d/appdf.py --mu 3.0 --k-coupling 0.2 --measurement "5.0,1.9,3.0"
"""
import argparse
import math
import multiprocessing as mp
import os
import time
import warnings

# one BLAS thread per pool worker (must be set before numpy is imported)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.linalg import cholesky, qr, solve_triangular
from scipy.stats import multivariate_normal

NUM_OSCILLATORS = 3
SYSTEM_DIM = NUM_OSCILLATORS * 2
STATE_NAMES = ["x1", "y1", "x2", "y2", "x3", "y3"]

# Measurement (r, theta) at t = 5 for the nine (mu, k) cases of the paper; the same value is used
# for the three oscillators.
Z_TABLE = {
    (1.5, 0.1): [1.476, 2.65], (1.5, 0.2): [1.518, 2.62], (1.5, 0.3): [1.5, 2.64],
    (2.0, 0.1): [1.665, 2.86], (2.0, 0.2): [1.65, 2.87], (2.0, 0.3): [1.66, 2.87],
    (2.5, 0.1): [1.782, 2.98], (2.5, 0.2): [1.787, 2.96], (2.5, 0.3): [1.795, 2.97],
}


# ==============================================================================
# 1. Dynamics: three ring-coupled Van der Pol oscillators
# ==============================================================================
def coupled_vdp_dynamics(t, state, mu_vdp=2.0, k_coupling=0.5):
    """Drift of the ring-coupled Van der Pol system. state shape: (..., SYSTEM_DIM)."""
    if state.ndim == 1:
        state = state[np.newaxis, :]
        is_1d = True
    else:
        is_1d = False

    # x indices: 0, 2, 4 / y indices: 1, 3, 5
    x = state[..., 0::2]
    y = state[..., 1::2]

    # ring topology: neighbours i-1 and i+1
    x_left = np.roll(x, 1, axis=-1)
    x_right = np.roll(x, -1, axis=-1)
    coupling = k_coupling * (x_left - x) + k_coupling * (x_right - x)

    dxdt = y
    dydt = mu_vdp * (1 - x**2) * y - x + coupling

    derivs = np.zeros_like(state)
    derivs[..., 0::2] = dxdt
    derivs[..., 1::2] = dydt

    if is_1d:
        return derivs.flatten()
    return derivs


# ==============================================================================
# 2. Gaussian component and splitting (SVD axes, rank-1 update)
# ==============================================================================
class Component:
    """One Gaussian component: weight w, mean mu, square-root factor M (P = M M^T)."""

    def __init__(self, w, mu, M):
        self.w, self.mu, self.M = float(w), mu.copy(), M.copy()

    def axis_error(self, t, mu_vdp, k_coupling):
        """Check the principal axes of the covariance (SVD) and return the scaled axis with the
        largest linearisation error, together with that error."""
        P = self.M @ self.M.T

        try:
            U, S, _ = np.linalg.svd(P)
        except np.linalg.LinAlgError:
            return None, 0.0

        best_err = -1.0
        best_axis = None

        for i in range(len(S)):
            d = U[:, i] * math.sqrt(S[i])
            err = self.lin_approx_rel_error_vdp(d, t, mu_vdp, k_coupling)

            if err > best_err:
                best_err = err
                best_axis = d

        return best_axis, best_err

    def lin_approx_rel_error_vdp(self, offset, t, mu_vdp, k_coupling):
        """Relative deviation from linearity of the drift along `offset`."""
        cder = coupled_vdp_dynamics(t, self.mu, mu_vdp, k_coupling)
        c_norm = np.linalg.norm(cder)
        if c_norm < 1e-12:
            return 0.0

        near = coupled_vdp_dynamics(t, self.mu + offset, mu_vdp, k_coupling) - cder
        far = 0.5 * (coupled_vdp_dynamics(t, self.mu + 2 * offset, mu_vdp, k_coupling) - cder)
        return np.linalg.norm(near - far) / c_norm


def split_component(p, d, w_=0.21921, a=1.03332):
    """Split a component into three along the direction d using a rank-1 update of M."""
    f = 1 - 1 / np.sqrt(2)

    if d.ndim == 1:
        d = d[:, np.newaxis]

    d_dot_d = np.dot(d.T, d)
    if d_dot_d == 0:
        return [p]

    try:
        term = np.outer(d, p.M.T @ d)
        N = p.M - f * term / d_dot_d
    except Exception:
        return [p]

    d_flat = d.flatten()

    w_c = p.w * (1 - 2 * w_)
    w_lr = p.w * w_

    c = Component(w_c, p.mu, N)
    l = Component(w_lr, p.mu - a * d_flat, N.copy())
    r = Component(w_lr, p.mu + a * d_flat, N.copy())

    return [c, l, r]


# ==============================================================================
# 3. Mixture reduction (pairwise distance within grid blocks)
# ==============================================================================
def combine_components_pairwise(plist, blocksize, combine_radius):
    """Merge (moment matching) components whose means are within combine_radius of each other.
    Only components that fall in the same grid block (edge length blocksize) are compared."""
    if not plist:
        return []

    dims = plist[0].mu.size

    # --- 1. assign the components to blocks ---
    min_mu = np.min([p.mu for p in plist], axis=0)
    max_mu = np.max([p.mu for p in plist], axis=0)

    if np.isscalar(blocksize):
        blocksize = np.full(dims, blocksize)

    blocksize_adjusted = np.where(blocksize == 0, 1e-9, blocksize)

    n_blocks = np.ceil((max_mu - min_mu) / blocksize_adjusted).astype(int) + 1
    n_blocks = np.maximum(1, n_blocks)

    def block_id(mu):
        idx = np.floor((mu - min_mu) / blocksize_adjusted).astype(int)
        idx = np.clip(idx, 0, n_blocks - 1)
        prod_values = [np.prod(n_blocks[:k]) if k > 0 else 1 for k in range(dims)]
        return sum(idx[k] * prod_values[k] for k in range(dims))

    blocks = {}
    for p in plist:
        blocks.setdefault(block_id(p.mu), []).append(p)

    new_list = []

    # --- 2. merge inside each block ---
    for group in blocks.values():
        n = len(group)
        merged_flag = [False] * n

        for i in range(n):
            if merged_flag[i]:
                continue

            merge_candidates = [group[i]]
            merged_flag[i] = True

            for j in range(i + 1, n):
                if merged_flag[j]:
                    continue

                if np.linalg.norm(group[i].mu - group[j].mu) <= combine_radius:
                    merge_candidates.append(group[j])
                    merged_flag[j] = True

            if len(merge_candidates) == 1:
                new_list.append(merge_candidates[0])
                continue

            w_sum = sum(p.w for p in merge_candidates)
            mu_bar = sum(p.w * p.mu for p in merge_candidates) / w_sum
            P_bar = sum(p.w / w_sum * (p.M @ p.M.T + np.outer(p.mu - mu_bar, p.mu - mu_bar))
                        for p in merge_candidates)

            try:
                L = np.linalg.cholesky(P_bar + 1e-12 * np.eye(dims))
                new_list.append(Component(w_sum, mu_bar, L))
            except np.linalg.LinAlgError:
                best_p = max(merge_candidates, key=lambda p: p.w)
                new_list.append(Component(w_sum, mu_bar, best_p.M.copy()))

    return new_list


# ==============================================================================
# 4. Component propagation with adaptive splitting
# ==============================================================================
def dxMdt(t, G, k_noise, mu_vdp, k_coupling):
    """ODE right-hand side for the packed [mu, M] of one component (k_noise = Q)."""
    dim = SYSTEM_DIM
    mu = G[:dim]
    M = G[dim:].reshape(dim, dim)

    dmu_sum = np.zeros_like(mu)
    for i in range(dim):
        dmu_sum += 0.5 * (coupled_vdp_dynamics(t, mu + M[:, i], mu_vdp, k_coupling) +
                          coupled_vdp_dynamics(t, mu - M[:, i], mu_vdp, k_coupling))
    dmu = dmu_sum / dim

    invMT = np.linalg.pinv(M.T + 1e-12 * np.eye(dim))
    dM = np.zeros_like(M)
    for i in range(dim):
        e = np.zeros(dim)
        e[i] = 1
        v_tip = coupled_vdp_dynamics(t, mu + M[:, i], mu_vdp, k_coupling)
        dM[:, i] = v_tip - dmu + 0.5 * (k_noise @ invMT @ e)

    return np.hstack((dmu, dM.ravel()))


def pack(p):
    return np.hstack((p.mu, p.M.ravel()))


def unpack(v, w):
    return Component(w, v[:SYSTEM_DIM], v[SYSTEM_DIM:].reshape(SYSTEM_DIM, SYSTEM_DIM))


def integrate_until_split(p, t0, dt, k_noise, mu_vdp, k_coupling, tol, min_dt):
    """Integrate one component from t0 for at most dt. If the linearisation error exceeds tol
    inside the interval, bisect for the time at which the error is in (0.8 tol, tol].

    Returns (component at t0 + time_used, time_used, needsplit).
    """
    G0 = pack(p)
    currentstep = dt
    adjustment = dt
    needsplit = True
    time_used = 0.0
    did_once = False
    max_attempts = 50
    attempts = 0

    while (not did_once) or (adjustment > min_dt and attempts < max_attempts):
        did_once = True
        attempts += 1
        currentstep = max(currentstep, 1e-9)

        sol = solve_ivp(lambda t, G: dxMdt(t, G, k_noise, mu_vdp, k_coupling),
                        (t0, t0 + currentstep), G0, method='RK45',
                        rtol=1e-8, atol=1e-10, t_eval=[t0 + currentstep])

        if not sol.success:
            adjustment *= 0.5
            currentstep -= adjustment
            needsplit = True
            continue

        p_mid = unpack(sol.y[:, 0], p.w)
        axis, err = p_mid.axis_error(t0 + currentstep, mu_vdp, k_coupling)

        if currentstep == dt and err < tol:
            needsplit = False
            time_used = currentstep
            break
        if err > tol:
            adjustment *= 0.5
            currentstep -= adjustment
        elif err > 0.8 * tol:
            needsplit = True
            time_used = currentstep
            break
        else:
            adjustment *= 0.5
            currentstep += adjustment
            currentstep = min(currentstep, dt)

    if time_used == 0:
        final_step = max(currentstep, min_dt)
        sol = solve_ivp(lambda t, G: dxMdt(t, G, k_noise, mu_vdp, k_coupling),
                        (t0, t0 + final_step), G0, method='RK45',
                        rtol=1e-6, atol=1e-8, t_eval=[t0 + final_step])
        if sol.success:
            p_mid = unpack(sol.y[:, 0], p.w)
            time_used = final_step
        else:
            p_mid = p
            time_used = 0.0
        needsplit = True

    return p_mid, time_used, needsplit


def propagate_adaptive(p, t0, dt, k_noise, mu_vdp, k_coupling, tol, min_dt, weight_thresh):
    """Propagate one component over dt, splitting it when required.

    When a split occurs the children are divided by weight:
      - big   (w >  weight_thresh): propagated recursively over the remaining time (may split again)
      - small (w <= weight_thresh): integrated over the remaining time with a single call of
        integrate_until_split, without further splitting
    The case in which the integration makes no progress (t_used ~ 0) is handled the same way.
    """
    if dt <= 0.0:
        return [p]

    t_rem = dt
    current_p = p
    propagated = []

    while t_rem > 1e-9:
        p_mid, t_used, needsplit = integrate_until_split(
            current_p, t0, t_rem, k_noise, mu_vdp, k_coupling, tol, min_dt
        )

        # ---- (1) the integration made no progress -> forced split ----
        if t_used <= 1e-9:
            axis, _ = p_mid.axis_error(t0, mu_vdp, k_coupling)
            if axis is None:
                propagated.append(p_mid)
                break

            children = split_component(p_mid, axis)

            big = [q for q in children if q.w > weight_thresh]
            small = [q for q in children if q.w <= weight_thresh]

            # small: integrate over the remaining time (t_rem) without further splitting
            for q in small:
                p_end, _, _ = integrate_until_split(
                    q, t0, t_rem, k_noise, mu_vdp, k_coupling, tol, min_dt
                )
                propagated.append(p_end)

            # big: recursion allowed
            for q in big:
                propagated.extend(
                    propagate_adaptive(q, t0, t_rem, k_noise, mu_vdp, k_coupling, tol, min_dt, weight_thresh)
                )
            warnings.warn(f"APPDF: integration failed at t = {t0:.6f}; the component was split instead.")
            break

        # ---- (2) no split needed -> accept ----
        if not needsplit:
            propagated.append(p_mid)
            break

        # ---- (3) split needed ----
        axis, _ = p_mid.axis_error(t0 + t_used, mu_vdp, k_coupling)
        if axis is None:
            propagated.append(p_mid)
            break

        children = split_component(p_mid, axis)

        big = [q for q in children if q.w > weight_thresh]
        small = [q for q in children if q.w <= weight_thresh]

        # small: integrate over the remaining time (t_rem - t_used) without further splitting
        for q in small:
            p_end, _, _ = integrate_until_split(
                q, t0 + t_used, t_rem - t_used, k_noise, mu_vdp, k_coupling, tol, min_dt
            )
            propagated.append(p_end)

        # big: recursive splitting over the remaining time
        for q in big:
            propagated.extend(
                propagate_adaptive(
                    q, t0 + t_used, t_rem - t_used,
                    k_noise, mu_vdp, k_coupling, tol, min_dt, weight_thresh
                )
            )

        break

    return propagated


def _propagate_component_worker(args):
    p, t0, dt, k_noise, mu_vdp, k_coupling, tol, min_dt, weight_thresh = args
    return propagate_adaptive(p, t0, dt, k_noise, mu_vdp, k_coupling, tol, min_dt, weight_thresh)


# ==============================================================================
# 5. Measurement model and square-root cubature update
# ==============================================================================
def normalize_angle(angle):
    """Wrap an angle to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def h_fun_general(state):
    """Polar measurement of each oscillator: [x1, y1, ...] -> [r1, th1, ...]."""
    meas = []
    n_osc = len(state) // 2
    for i in range(n_osc):
        x, y = state[2 * i], state[2 * i + 1]
        meas.extend([np.sqrt(x**2 + y**2), np.arctan2(y, x)])
    return np.array(meas)


def srckf_measurement_update_general(p, z, h_fun, R):
    """Square-root cubature measurement update of one component (angles at the odd measurement
    indices). The returned component carries the un-normalised updated weight."""
    n = p.mu.size
    nz = z.shape[0]
    S_pred = p.M

    xi = np.sqrt(n) * np.hstack((np.eye(n), -np.eye(n)))
    X_cubature = p.mu[:, np.newaxis] + S_pred @ xi
    Z_cubature = np.array([h_fun(X_cubature[:, i]) for i in range(2 * n)]).T

    # predicted measurement (circular mean for the angles)
    z_pred = np.zeros(nz)
    for i in range(nz):
        if i % 2 == 1:
            mean_cos = np.mean(np.cos(Z_cubature[i, :]))
            mean_sin = np.mean(np.sin(Z_cubature[i, :]))
            z_pred[i] = np.arctan2(mean_sin, mean_cos)
        else:
            z_pred[i] = np.mean(Z_cubature[i, :])

    Z_centered = np.zeros_like(Z_cubature)
    innovation = z - z_pred
    for i in range(nz):
        diff = Z_cubature[i, :] - z_pred[i]
        innov_diff = innovation[i]
        if i % 2 == 1:
            diff = normalize_angle(diff)
            innov_diff = normalize_angle(innov_diff)
        Z_centered[i, :] = diff
        innovation[i] = innov_diff

    Z_centered /= np.sqrt(2 * n)
    X_centered = (X_cubature - p.mu[:, np.newaxis]) / np.sqrt(2 * n)

    S_R = cholesky(R, lower=True)
    A_top = np.hstack([Z_centered, S_R])
    A_bottom = np.hstack([X_centered, np.zeros((n, nz))])
    A = np.vstack([A_top, A_bottom])
    _, R_qr = qr(A.T, mode='economic')
    T = R_qr.T

    T11 = T[:nz, :nz]
    T21 = T[nz:, :nz]
    T22 = T[nz:, nz:]

    Wk_T = solve_triangular(T11, T21.T, lower=True, trans='T')
    W_k = Wk_T.T

    mu_up = p.mu + W_k @ innovation
    M_up = T22

    # component likelihood (mixture weight update): sum of N(z; h(X_j), R) over the cubature points
    component_likelihood = 0.0
    num_points = 2 * n
    for j in range(num_points):
        mean_j = Z_cubature[:, j]
        residual_j = z - mean_j
        for k_idx in range(1, nz, 2):
            residual_j[k_idx] = normalize_angle(residual_j[k_idx])
        likelihood_at_point = multivariate_normal.pdf(residual_j, mean=np.zeros(nz), cov=R)
        component_likelihood += likelihood_at_point

    w_unnormalized = p.w * component_likelihood
    return Component(w_unnormalized, mu_up, M_up)


def _measurement_update_worker(args):
    p, z_obs, h_fun, R = args
    return srckf_measurement_update_general(p, z_obs, h_fun, R)


# ==============================================================================
# 6. Filter loop
# ==============================================================================
def run_appdf(t_end=10.0, dt=0.01, mu_vdp=1.5, k_coupling=0.2, Q=None, measurements=None, R=None,
              x0=None, P0=None, tol=0.05, min_dt=0.0005, weight_thresh=1e-4,
              blocksize=1.0, combine_radius=0.005, components_per_worker_min=1,
              n_workers=None, return_components=False, verbose=True):
    """Run the APPDF on the 6-D coupled Van der Pol system.

    measurements: dict {float(round(t, 6)): z (6,)}; None or {} -> prediction only.
    Q, R, x0, P0: process-noise covariance, measurement-noise covariance, initial mean / covariance
    (None -> paper values).
    tol: split tolerance on the linearisation error; min_dt: smallest bisection interval of the split
    search; weight_thresh: components with weight <= weight_thresh are not split further and are
    discarded at the end of the step.
    blocksize (scalar or (6,)), combine_radius: grid block size and merge distance of the pairwise merge.
    n_workers: maximum pool size (None -> mp.cpu_count()).

    Returns a dict with (T = number of steps; no t = 0 entry)
        time_hist (T,), mu_hist (T, 6), P_hist (T, 6, 6) [moment-matched mixture mean / covariance],
        count_hist (T,), measurements,
        components_hist (only if return_components): list of (weights (n,), means (n, 6), M (n, 6, 6)).
    """
    steps = int(np.round(t_end / dt))

    if Q is None:
        Q = (6e-3) * np.eye(SYSTEM_DIM)
    if R is None:
        R = np.diag([0.2**2, 0.2**2] * NUM_OSCILLATORS)
    if x0 is None:
        x0 = np.zeros(SYSTEM_DIM)
        for i in range(NUM_OSCILLATORS):
            x0[2 * i] = 2.0
    if P0 is None:
        P0 = np.eye(SYSTEM_DIM) * 0.01
    x0 = np.asarray(x0, dtype=float)
    blocksize = np.ones(SYSTEM_DIM) * blocksize

    M0 = np.linalg.cholesky(P0)
    components = [Component(1.0, x0, M0)]

    mu_hist, P_hist, count_hist, time_hist = [], [], [], []
    components_hist = []

    t0 = 0.0

    max_cpu = max(1, mp.cpu_count() if n_workers is None else int(n_workers))
    pool = None
    curr_pool_size = 0

    if verbose:
        print(f"Running APPDF (6-D coupled Van der Pol, mu={mu_vdp}, k={k_coupling})...")

    for s in range(steps):
        if verbose and s % 100 == 0:
            print(f"Step {s}/{steps}, time = {t0:.2f}, components = {len(components)}")

        n_parts = len(components)
        needed_procs = min(max_cpu, max(1, math.ceil(n_parts / components_per_worker_min)))

        if pool is None or needed_procs != curr_pool_size:
            if pool:
                pool.close()
                pool.join()
            curr_pool_size = needed_procs
            pool = mp.Pool(processes=curr_pool_size)

        # 1. parallel propagation (with splitting)
        task_args = [(p, t0, dt, Q, mu_vdp, k_coupling, tol, min_dt, weight_thresh) for p in components]
        results = pool.map(_propagate_component_worker, task_args)

        new_parts = [p for sub in results for p in sub if p.w > weight_thresh]

        # 2. merge
        components = combine_components_pairwise(new_parts, blocksize=blocksize,
                                                 combine_radius=combine_radius)

        current_time_rounded = float(round(t0 + dt, 6))

        # 3. measurement update
        if measurements and current_time_rounded in measurements:
            if verbose:
                print(f"  -> measurement update at t = {current_time_rounded:.2f}")
            z_obs = measurements[current_time_rounded]

            if len(components) == 1:
                p_up = srckf_measurement_update_general(components[0], z_obs, h_fun_general, R)
                components = [p_up]
            else:
                update_args = [(p, z_obs, h_fun_general, R) for p in components]
                updated_list = pool.map(_measurement_update_worker, update_args)
                components = [p for p in updated_list if p.w > 1e-15]

            # merge again after the update
            components = combine_components_pairwise(components, blocksize=blocksize,
                                                     combine_radius=combine_radius)

        # 4. weight normalisation
        total_w = sum(p.w for p in components)
        if total_w > 0:
            for p in components:
                p.w /= total_w
        else:
            warnings.warn("APPDF: component weights sum to 0; resetting to uniform weights.")
            for p in components:
                p.w = 1.0 / len(components)

        # 5. mixture statistics (moment matching)
        if len(components) > 0:
            mu_mix = np.sum([p.w * p.mu for p in components], axis=0)
            P_mix = np.sum([p.w * (p.M @ p.M.T + np.outer(p.mu - mu_mix, p.mu - mu_mix))
                            for p in components], axis=0)
        else:
            mu_mix = mu_hist[-1] if mu_hist else x0
            P_mix = P_hist[-1] if P_hist else P0

        mu_hist.append(mu_mix)
        P_hist.append(P_mix)
        count_hist.append(len(components))
        time_hist.append(t0 + dt)
        if return_components:
            components_hist.append((np.array([p.w for p in components]),
                                    np.array([p.mu for p in components]),
                                    np.array([p.M for p in components])))
        t0 += dt

    if pool:
        pool.close()
        pool.join()

    out = {
        "mu_hist": np.array(mu_hist), "P_hist": np.array(P_hist), "count_hist": np.array(count_hist),
        "time_hist": np.array(time_hist), "measurements": measurements
    }
    if return_components:
        out["components_hist"] = components_hist
    return out


# ==============================================================================
# 7. Command-line interface
# ==============================================================================
def build_measurements(args, parser):
    """Measurement dict from the CLI: --no-measurements, explicit --measurement, or the paper table."""
    if args.no_measurements:
        return {}
    if args.measurement:
        measurements = {}
        for item in args.measurement:
            try:
                t_meas, r_meas, th_meas = (float(v) for v in item.split(","))
            except ValueError:
                parser.error(f'--measurement expects "t,r,theta", got "{item}"')
            measurements[float(round(t_meas, 6))] = np.array([r_meas, th_meas] * 3)
        return measurements
    key = (float(args.mu), float(args.k_coupling))
    if key not in Z_TABLE:
        parser.error(f"no tabulated measurement for (mu, k) = {key}; "
                     'use --no-measurements or --measurement "t,r,theta"')
    return {float(round(5.0, 6)): np.array(Z_TABLE[key] * 3)}


def save_components_csv(results, path):
    """One row per component and stored step: time, weight, mean, upper triangle of P = M M^T."""
    iu = np.triu_indices(SYSTEM_DIM)
    cov_names = [f"P{i + 1}{j + 1}" for i, j in zip(*iu)]
    rows = []
    for t, (w, mu, M) in zip(results["time_hist"], results["components_hist"]):
        P = M @ np.transpose(M, (0, 2, 1))
        rows.append(np.column_stack([np.full(len(w), t), w, mu, P[:, iu[0], iu[1]]]))
    columns = ["time", "weight"] + STATE_NAMES + cov_names
    pd.DataFrame(np.vstack(rows), columns=columns).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="APPDF for the 6-D coupled Van der Pol system (defaults: paper case).")
    parser.add_argument("--mu", type=float, default=1.5, help="Van der Pol parameter mu")
    parser.add_argument("--k-coupling", type=float, default=0.2, help="coupling strength k")
    parser.add_argument("--t-end", type=float, default=10.0, help="final time")
    parser.add_argument("--dt", type=float, default=0.01, help="filter time step")
    parser.add_argument("--no-measurements", action="store_true", help="prediction only")
    parser.add_argument("--measurement", action="append", metavar='"t,r,theta"',
                        help="measurement (same r, theta for the three oscillators) at time t; repeatable. "
                             "Default: the paper measurement at t = 5 for the given (mu, k)")
    parser.add_argument("--tol", type=float, default=0.05, help="split tolerance (linearisation error)")
    parser.add_argument("--min-dt", type=float, default=0.0005, help="smallest interval of the split search")
    parser.add_argument("--weight-thresh", type=float, default=1e-4,
                        help="components with weight <= this value are not split and are discarded")
    parser.add_argument("--blocksize", type=float, default=1.0, help="grid block size of the merge")
    parser.add_argument("--combine-radius", type=float, default=0.005, help="merge distance between means")
    parser.add_argument("--workers", type=int, default=None, help="maximum pool size (default: all CPUs)")
    parser.add_argument("--save-components", action="store_true", help="also write components.csv")
    parser.add_argument("--out", default=os.path.join("results", "coupled_vdp6d_appdf"),
                        help="output directory")
    parser.add_argument("--quiet", action="store_true", help="no progress output")
    args = parser.parse_args()

    measurements = build_measurements(args, parser)

    start = time.time()
    results = run_appdf(t_end=args.t_end, dt=args.dt, mu_vdp=args.mu, k_coupling=args.k_coupling,
                        measurements=measurements, tol=args.tol, min_dt=args.min_dt,
                        weight_thresh=args.weight_thresh, blocksize=args.blocksize,
                        combine_radius=args.combine_radius, n_workers=args.workers,
                        return_components=args.save_components, verbose=not args.quiet)
    wall = time.time() - start

    os.makedirs(args.out, exist_ok=True)
    table = pd.DataFrame(np.column_stack([results["time_hist"], results["mu_hist"]]),
                         columns=["time"] + STATE_NAMES)
    table["n_components"] = results["count_hist"]
    table.to_csv(os.path.join(args.out, "mean_trajectory.csv"), index=False)
    np.savez(os.path.join(args.out, "covariance.npz"), times=results["time_hist"],
             means=results["mu_hist"], covs=results["P_hist"])
    saved = "mean_trajectory.csv and covariance.npz"
    if args.save_components:
        save_components_csv(results, os.path.join(args.out, "components.csv"))
        saved += " and components.csv"

    print(f"Saved {saved} to '{args.out}'")
    print(f"Components: max = {int(results['count_hist'].max())}, final = {int(results['count_hist'][-1])}")
    print(f"Wall time: {wall:.2f} s")


if __name__ == "__main__":
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    main()
