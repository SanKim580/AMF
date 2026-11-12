#!/usr/bin/env python
"""AMF (adaptive Gaussian-mixture filter with blocked KL-bound merging) for the 6-D coupled
Van der Pol system.

Model (three ring-coupled Van der Pol oscillators, state s = (x1, y1, x2, y2, x3, y3)):

    dx_i = y_i dt
    dy_i = [mu (1 - x_i^2) y_i - x_i + k (x_{i-1} - x_i) + k (x_{i+1} - x_i)] dt + dW_i ,
    E[dW dW^T] = Q dt   (additive noise on all six states, Q = 6e-3 I6 in the paper case)

Measurement (per oscillator, polar):

    h(s) = (r_1, theta_1, r_2, theta_2, r_3, theta_3),  r_i = sqrt(x_i^2 + y_i^2),  theta_i = atan2(y_i, x_i)
    z = h(s) + v,  v ~ N(0, R),  R = diag(0.2^2, 0.2^2) x 3

Algorithm: every Gaussian component (weight w, mean mu, square-root factor M) is propagated with the
LSKF mean / square-root ODE. A component is split into three along the column of M with the largest
linearisation error whenever that error exceeds `tol`. After each step the mixture is reduced:
components are grouped into grid blocks of edge `blocksize`, and inside each block pairs are merged
greedily in the order of the upper bound of the KL discrimination until no pair is below `kl_lambda`
(and at most `kl_mu_per_block` components are left per block). Components with weight <=
`weight_thresh` are discarded. Measurements are assimilated with a square-root cubature update per
component plus a weight update. Components are propagated in a multiprocessing pool.

Paper case: mu = 1.5, k = 0.2, x0 = (2, 0) x 3, P0 = 0.01 I6, dt = 0.01, t in [0, 10], one measurement
at t = 5; tol = 0.05, min_dt = 0.0005, weight_thresh = 1e-4, blocksize = 1.0 (every dimension),
kl_ml_per_block = 1, kl_mu_per_block = 100, kl_lambda = 0.05. These are the defaults.

Alternative tuned setting used for the consistency study:
    --tol 0.02 --blocksize 0.5 --weight-thresh 1e-6

Usage:
    python coupled_vdp6d/amf.py                        # paper case
    python coupled_vdp6d/amf.py --tol 0.02 --blocksize 0.5 --weight-thresh 1e-6
    python coupled_vdp6d/amf.py --t-end 2 --no-measurements --workers 4 --out results/test
    python coupled_vdp6d/amf.py --mu 3.0 --k-coupling 0.2 --measurement "5.0,1.9,3.0"
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

SYSTEM_DIM = 6  # 3 oscillators x 2 states
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
def coupled_vdp_6d(t, state, mu_vdp=2.0, k_coupling=0.5):
    """Drift of the ring-coupled Van der Pol system. state shape: (..., 6) = [x1, y1, x2, y2, x3, y3]."""
    if state.ndim == 1:
        state = state[np.newaxis, :]
        is_1d = True
    else:
        is_1d = False

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
# 2. Gaussian component and splitting
# ==============================================================================
class Component:
    """One Gaussian component: weight w, mean mu, square-root factor M (P = M M^T)."""

    def __init__(self, w, mu, M):
        self.w, self.mu, self.M = float(w), mu.copy(), M.copy()

    def axis_error(self, t, mu_vdp, k_coupling):
        """Linearisation error along every column of M. Returns the column with the largest error
        and the list of all errors."""
        best_err = -1.0
        best_axis = None
        err_list = []

        for i in range(self.M.shape[1]):
            d = self.M[:, i]
            err = self.lin_approx_rel_error_vdp(d, t, mu_vdp, k_coupling)
            err_list.append(err)
            if err > best_err:
                best_err = err
                best_axis = d

        return best_axis, err_list

    def lin_approx_rel_error_vdp(self, offset, t, mu_vdp, k_coupling):
        """Relative deviation from linearity of the drift along `offset`."""
        cder = coupled_vdp_6d(t, self.mu, mu_vdp, k_coupling)
        c_norm = np.linalg.norm(cder)
        if c_norm < 1e-12:
            return 0.0

        near = coupled_vdp_6d(t, self.mu + offset, mu_vdp, k_coupling) - cder
        far = 0.5 * (coupled_vdp_6d(t, self.mu + 2 * offset, mu_vdp, k_coupling) - cder)
        return np.linalg.norm(near - far) / c_norm


def split_component(p, d):
    """Split a component into three along the direction d.

    The children share the covariance Sigma - 0.5 v v^T / (d^T Sigma d) with v = Sigma d (closed
    form, no transformation matrix needed) and are displaced by +/- 1.03332 v / sqrt(d^T Sigma d).
    """
    n = p.mu.size
    Sigma = p.M @ p.M.T

    d_norm_val = np.linalg.norm(d)
    if d_norm_val < 1e-12:
        return [p]
    d_vec = d / d_norm_val

    v = Sigma @ d_vec
    s_dd = d_vec.T @ v

    if s_dd < 1e-12:
        return [p]

    sigma_d = np.sqrt(s_dd)

    Sigma_final = Sigma - 0.5 * np.outer(v, v) / s_dd

    try:
        M_final = np.linalg.cholesky(Sigma_final + 1e-12 * np.eye(n))
    except np.linalg.LinAlgError:
        return [p]

    a_scale = 1.03332 / sigma_d
    disp_vector = a_scale * v

    w_ = 0.21921
    w_parent = p.w
    w_c = w_parent * (1 - 2 * w_)
    w_lr = w_parent * w_

    c = Component(w_c, p.mu, M_final)
    l = Component(w_lr, p.mu - disp_vector, M_final.copy())
    r = Component(w_lr, p.mu + disp_vector, M_final.copy())

    return [c, l, r]


# ==============================================================================
# 3. Mixture reduction (KL-bound merging inside grid blocks)
# ==============================================================================
def combine_components_kl_reduction(plist, Ml, Mu, lambda_thresh):
    """Greedy pairwise merging by the upper bound of the KL discrimination.

    Pairs are merged (moment matching) in increasing order of the bound while more than Mu components
    are left, or while more than Ml are left and the smallest bound is below lambda_thresh.
    """
    if not plist or len(plist) <= Ml:
        return plist

    def _merge_two_components(p_i, p_j):
        w_i, w_j = p_i.w, p_j.w
        w_ij = w_i + w_j
        if w_ij == 0:
            return None
        mu_i, mu_j = p_i.mu, p_j.mu
        P_i, P_j = p_i.M @ p_i.M.T, p_j.M @ p_j.M.T

        w_i_n, w_j_n = w_i / w_ij, w_j / w_ij
        mu_ij = w_i_n * mu_i + w_j_n * mu_j
        P_ij = w_i_n * P_i + w_j_n * P_j + w_i_n * w_j_n * np.outer(mu_i - mu_j, mu_i - mu_j)

        try:
            L_ij = np.linalg.cholesky(P_ij + 1e-12 * np.eye(mu_i.size))
            return Component(w_ij, mu_ij, L_ij)
        except np.linalg.LinAlgError:
            return None

    def _calculate_kl_bound(p_i, p_j, p_merged):
        if p_merged is None:
            return np.inf
        det_Pi = np.linalg.det(p_i.M @ p_i.M.T + 1e-12 * np.eye(p_i.mu.size))
        det_Pj = np.linalg.det(p_j.M @ p_j.M.T + 1e-12 * np.eye(p_j.mu.size))
        det_Pij = np.linalg.det(p_merged.M @ p_merged.M.T + 1e-12 * np.eye(p_merged.mu.size))
        if det_Pi <= 0 or det_Pj <= 0 or det_Pij <= 0:
            return np.inf
        return 0.5 * (p_merged.w * np.log(det_Pij) - p_i.w * np.log(det_Pi) - p_j.w * np.log(det_Pj))

    components_dict = {i: p for i, p in enumerate(plist)}
    B_matrix = np.full((len(plist), len(plist)), np.inf)
    merged_cache = {}
    indices = list(components_dict.keys())

    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            idx_i, idx_j = indices[i], indices[j]
            p_m = _merge_two_components(components_dict[idx_i], components_dict[idx_j])
            merged_cache[(idx_i, idx_j)] = p_m
            B_matrix[idx_i, idx_j] = _calculate_kl_bound(components_dict[idx_i], components_dict[idx_j], p_m)

    num_active = len(plist)

    while num_active > Mu or (num_active > Ml and np.nanmin(B_matrix) < lambda_thresh):
        if num_active <= 1:
            break
        min_val = np.nanmin(B_matrix)
        if min_val == np.inf:
            break
        i_star, j_star = np.unravel_index(np.nanargmin(B_matrix), B_matrix.shape)

        p_merged = merged_cache.get((min(i_star, j_star), max(i_star, j_star)))
        if p_merged is None:
            p_merged = _merge_two_components(components_dict[i_star], components_dict[j_star])
        if p_merged is None:
            B_matrix[i_star, j_star] = np.inf
            continue

        components_dict[i_star] = p_merged
        del components_dict[j_star]
        num_active -= 1

        B_matrix[j_star, :] = np.inf
        B_matrix[:, j_star] = np.inf

        # update the bounds between the merged component and all remaining ones
        p_new = components_dict[i_star]
        for idx_k in components_dict.keys():
            if idx_k == i_star:
                continue
            p_m_new = _merge_two_components(p_new, components_dict[idx_k])
            key = (min(i_star, idx_k), max(i_star, idx_k))
            merged_cache[key] = p_m_new
            cost = _calculate_kl_bound(p_new, components_dict[idx_k], p_m_new)
            B_matrix[i_star, idx_k] = cost
            B_matrix[idx_k, i_star] = cost

    return list(components_dict.values())


def combine_components_blocked_kl(plist, blocksize, kl_ml, kl_mu, kl_lambda):
    """Group the components into grid blocks (edge length blocksize per dimension) by their means
    and apply the KL-bound reduction inside every block."""
    if not plist:
        return []
    dims = plist[0].mu.size

    min_mu = np.min([p.mu for p in plist], axis=0)
    max_mu = np.max([p.mu for p in plist], axis=0)
    blocksize_adj = np.where(blocksize == 0, 1e-9, blocksize)
    n_blocks = np.maximum(1, np.ceil((max_mu - min_mu) / blocksize_adj).astype(int) + 1)

    def block_id(mu):
        idx = np.clip(np.floor((mu - min_mu) / blocksize_adj).astype(int), 0, n_blocks - 1)
        prod = [np.prod(n_blocks[:k]) if k > 0 else 1 for k in range(dims)]
        return sum(idx[k] * prod[k] for k in range(dims))

    blocks = {}
    for p in plist:
        blocks.setdefault(block_id(p.mu), []).append(p)

    new_list = []
    for _, group in blocks.items():
        if len(group) <= 1:
            new_list.extend(group)
        else:
            new_list.extend(combine_components_kl_reduction(group, kl_ml, kl_mu, kl_lambda))

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
        dmu_sum += 0.5 * (coupled_vdp_6d(t, mu + M[:, i], mu_vdp, k_coupling) +
                          coupled_vdp_6d(t, mu - M[:, i], mu_vdp, k_coupling))
    dmu = dmu_sum / dim

    invMT = np.linalg.pinv(M.T + 1e-12 * np.eye(dim))
    dM = np.zeros_like(M)
    for i in range(dim):
        e = np.zeros(dim)
        e[i] = 1
        v_tip = coupled_vdp_6d(t, mu + M[:, i], mu_vdp, k_coupling)
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

        sol = solve_ivp(
            lambda t, G: dxMdt(t, G, k_noise, mu_vdp, k_coupling),
            (t0, t0 + currentstep),
            G0,
            method='RK45',
            rtol=1e-7,
            atol=1e-9,
            t_eval=[t0 + currentstep]
        )

        if not sol.success:
            adjustment *= 0.5
            currentstep -= adjustment
            needsplit = True
            continue

        p_mid = unpack(sol.y[:, 0], p.w)

        axis, err_list = p_mid.axis_error(t0 + currentstep, mu_vdp, k_coupling)
        err = max(err_list) if err_list else 0.0

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

        sol = solve_ivp(
            lambda t, G: dxMdt(t, G, k_noise, mu_vdp, k_coupling),
            (t0, t0 + final_step),
            G0,
            method='RK45',
            rtol=1e-7,
            atol=1e-9,
            t_eval=[t0 + final_step]
        )
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

        # the integration failed / made no progress -> force a split
        if t_used <= 1e-9:
            axis, _ = p_mid.axis_error(t0, mu_vdp, k_coupling)
            children = split_component(p_mid, axis)

            big = [q for q in children if q.w > weight_thresh]
            small = [q for q in children if q.w <= weight_thresh]

            # small: propagate over the remaining time without further splitting
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
            warnings.warn(f"AMF: integration failed at t = {t0:.6f}; the component was split instead.")
            break

        # no split needed -> accept
        if not needsplit:
            propagated.append(p_mid)
            break

        # split needed
        axis, _ = p_mid.axis_error(t0 + t_used, mu_vdp, k_coupling)
        children = split_component(p_mid, axis)

        big = [q for q in children if q.w > weight_thresh]
        small = [q for q in children if q.w <= weight_thresh]

        # small: propagate over the remaining time without further splitting
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


def h_fun_6d(state):
    """Polar measurement of each oscillator: [x1, y1, x2, y2, x3, y3] -> [r1, th1, r2, th2, r3, th3]."""
    meas = []
    for i in range(3):
        x, y = state[2 * i], state[2 * i + 1]
        meas.extend([np.sqrt(x**2 + y**2), np.arctan2(y, x)])
    return np.array(meas)


def srckf_measurement_update(p, z, h_fun, R):
    """Square-root cubature measurement update of one component (angles at the odd measurement
    indices 1, 3, 5). The returned component carries the un-normalised updated weight."""
    n = p.mu.size
    nz = z.shape[0]
    S_pred = p.M

    xi = np.sqrt(n) * np.hstack((np.eye(n), -np.eye(n)))
    X_cubature = p.mu[:, np.newaxis] + S_pred @ xi
    Z_cubature = np.array([h_fun(X_cubature[:, i]) for i in range(2 * n)]).T  # (nz, 2n)

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
    return srckf_measurement_update(p, z_obs, h_fun, R)


# ==============================================================================
# 6. Filter loop
# ==============================================================================
def run_amf(t_end=10.0, dt=0.01, mu_vdp=1.5, k_coupling=0.2, Q=None, measurements=None, R=None,
            x0=None, P0=None, tol=0.05, min_dt=0.0005, weight_thresh=1e-4,
            blocksize=1.0, kl_ml_per_block=1, kl_mu_per_block=100, kl_lambda=0.05,
            components_per_worker_min=1, n_workers=None, return_components=False, verbose=True):
    """Run the AMF on the 6-D coupled Van der Pol system.

    measurements: dict {float(round(t, 6)): z (6,)}; None or {} -> prediction only.
    Q, R, x0, P0: process-noise covariance, measurement-noise covariance, initial mean / covariance
    (None -> paper values).
    tol: split tolerance on the linearisation error; min_dt: smallest bisection interval of the split
    search; weight_thresh: components with weight <= weight_thresh are not split further and are
    discarded at the end of the step.
    blocksize (scalar or (6,)): grid block edge of the merge; kl_ml_per_block / kl_mu_per_block:
    minimum / maximum number of components per block; kl_lambda: merge threshold on the KL bound.
    n_workers: maximum pool size (None -> mp.cpu_count()).

    Returns a dict with (T = number of steps; no t = 0 entry)
        time_hist (T,), mu_hist (T, 6), P_hist (T, 6, 6) [moment-matched mixture mean / covariance],
        count_hist (T,), measurements,
        added_by_prop_hist, reduced_by_thresh_hist, reduced_by_kl_hist, reduced_by_meas_hist (T,)
        [change of the number of components per step: splitting, weight threshold, merge after the
        propagation, measurement update + merge],
        components_hist (only if return_components): list of (weights (n,), means (n, 6), M (n, 6, 6)).
    """
    steps = int(np.round(t_end / dt))

    if Q is None:
        Q = (6e-3) * np.eye(SYSTEM_DIM)
    if R is None:
        R = np.diag([0.2**2, 0.2**2] * 3)
    if x0 is None:
        x0 = np.array([2.0, 0.0, 2.0, 0.0, 2.0, 0.0])
    if P0 is None:
        P0 = np.eye(SYSTEM_DIM) * 0.01
    x0 = np.asarray(x0, dtype=float)
    blocksize = np.ones(SYSTEM_DIM) * blocksize

    M0 = np.linalg.cholesky(P0)
    components = [Component(1.0, x0, M0)]

    mu_hist, P_hist, count_hist, time_hist = [], [], [], []
    components_hist = []

    added_by_prop_hist = []
    reduced_by_thresh_hist = []
    reduced_by_kl_hist = []
    reduced_by_meas_hist = []

    t0 = 0.0

    max_cpu = max(1, mp.cpu_count() if n_workers is None else int(n_workers))
    pool = None
    curr_pool_size = 0

    if verbose:
        print(f"Running AMF (6-D coupled Van der Pol, mu={mu_vdp}, k={k_coupling})...")

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

        N0 = len(components)

        # --- 1) propagation (with splitting) ---
        task_args = [(p, t0, dt, Q, mu_vdp, k_coupling, tol, min_dt, weight_thresh) for p in components]
        results = pool.map(_propagate_component_worker, task_args)

        new_parts_all = []
        for out_parts in results:
            new_parts_all.extend(out_parts)

        Nprop_all = len(new_parts_all)

        new_parts = [p for p in new_parts_all if p.w > weight_thresh]
        Nprop_kept = len(new_parts)

        # --- 2) reduction (blocked KL merge) ---
        components = combine_components_blocked_kl(new_parts, blocksize, kl_ml_per_block,
                                                   kl_mu_per_block, kl_lambda)
        Nkl = len(components)

        current_time_rounded = float(round(t0 + dt, 6))

        # --- 3) measurement update ---
        if measurements and current_time_rounded in measurements:
            if verbose:
                print(f"  -> measurement update at t = {current_time_rounded:.2f}")
            z_obs = measurements[current_time_rounded]

            if len(components) == 1:
                p_up = srckf_measurement_update(components[0], z_obs, h_fun_6d, R)
                components = [p_up]
            else:
                update_args = [(p, z_obs, h_fun_6d, R) for p in components]
                updated_list = pool.map(_measurement_update_worker, update_args)
                components = [p for p in updated_list if p.w > 1e-15]

            components = combine_components_blocked_kl(components, blocksize, kl_ml_per_block,
                                                       kl_mu_per_block, kl_lambda)
            Nmeas = len(components)
        else:
            Nmeas = Nkl

        added_by_prop_hist.append(Nprop_all - N0)
        reduced_by_thresh_hist.append(Nprop_all - Nprop_kept)
        reduced_by_kl_hist.append(Nprop_kept - Nkl)
        reduced_by_meas_hist.append(Nkl - Nmeas)  # reduction stored as a positive number

        # --- 4) weight normalisation ---
        total_w = sum(p.w for p in components)
        if total_w > 0:
            for p in components:
                p.w /= total_w
        else:
            if len(components) == 0:
                components = [Component(1.0, x0, M0)]
            for p in components:
                p.w = 1.0 / len(components)

        # --- 5) mixture statistics (moment matching) ---
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
        "mu_hist": np.array(mu_hist),
        "P_hist": np.array(P_hist),
        "count_hist": np.array(count_hist),
        "time_hist": np.array(time_hist),
        "measurements": measurements,
        "added_by_prop_hist": np.array(added_by_prop_hist),
        "reduced_by_thresh_hist": np.array(reduced_by_thresh_hist),
        "reduced_by_kl_hist": np.array(reduced_by_kl_hist),
        "reduced_by_meas_hist": np.array(reduced_by_meas_hist)
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
        description="AMF for the 6-D coupled Van der Pol system (defaults: paper case).")
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
    parser.add_argument("--blocksize", type=float, default=1.0,
                        help="grid block edge of the merge (same for every dimension)")
    parser.add_argument("--kl-lambda", type=float, default=0.05, help="merge threshold on the KL bound")
    parser.add_argument("--kl-mu", type=int, default=100, help="maximum number of components per block")
    parser.add_argument("--workers", type=int, default=None, help="maximum pool size (default: all CPUs)")
    parser.add_argument("--save-components", action="store_true", help="also write components.csv")
    parser.add_argument("--out", default=os.path.join("results", "coupled_vdp6d_amf"),
                        help="output directory")
    parser.add_argument("--quiet", action="store_true", help="no progress output")
    args = parser.parse_args()

    measurements = build_measurements(args, parser)

    start = time.time()
    results = run_amf(t_end=args.t_end, dt=args.dt, mu_vdp=args.mu, k_coupling=args.k_coupling,
                      measurements=measurements, tol=args.tol, min_dt=args.min_dt,
                      weight_thresh=args.weight_thresh, blocksize=args.blocksize,
                      kl_lambda=args.kl_lambda, kl_mu_per_block=args.kl_mu, n_workers=args.workers,
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
