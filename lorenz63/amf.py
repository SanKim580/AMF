"""AMF (adaptive Gaussian-mixture level-set filter with level-set-vector split and blocked
KL-bound merge) for the stochastic Lorenz-63 system.

Model (additive white-noise forcing):

    dx = sigma (y - x) dt        + sqrt(2 Gamma) dW_x
    dy = [x (rho - z) - y] dt    + sqrt(2 Gamma) dW_y
    dz = [x y - beta z] dt       + sqrt(2 Gamma) dW_z

i.e. dX = f(X) dt + sqrt(2 Gamma) dW with diffusion covariance Q = 2 Gamma I3.
Measurement: direct observation of the state, z = h(x) + v, h(x) = x, v ~ N(0, R).

Every mixture component (weight w, mean mu, covariance factor M, P = M M^T) is propagated with the
level-set equations.  The relative linearisation error of the drift is monitored along the columns
of M (the level-set vectors); as soon as it exceeds the tolerance `tol` the component is split into
three along the worst column.  After every step the mixture is reduced: the components are sorted
into grid blocks of size `blocksize` and, inside every block, pairs are merged in the order of the
smallest upper bound of the KL divergence while that bound is below `kl_lambda` (or while the block
holds more than `kl_mu_per_block` components).  At measurement times every component gets a
square-root cubature update and is re-weighted by its likelihood.

Usage:
    python lorenz63/amf.py                              # paper case
    python lorenz63/amf.py --t-end 1 --no-measurements --workers 4 --out results/amf_test
    python lorenz63/amf.py --tol 0.05 --blocksize 1.0 --kl-lambda 0.05 --kl-mu 100 --save-components
"""
import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import multiprocessing as mp
import time
import warnings

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.linalg import cholesky, qr, solve_triangular
from scipy.stats import multivariate_normal

SYSTEM_DIM = 3

# Paper case
PAPER_X0 = (1.0, 1.0, 1.0)
PAPER_P0_VAR = 0.1
PAPER_MEAS_SIGMA = 0.8
PAPER_MEASUREMENT = (10.0, -0.25, 0.02, 20.09)   # t, x, y, z

# Default method parameters (paper setting)
TOL = 0.05               # split tolerance on the relative linearisation error
MIN_DT = 0.0005          # smallest step of the split-time bisection
WEIGHT_THRESH = 0.00001  # components with a smaller weight are not split further / are discarded
BLOCKSIZE = 1.0          # edge length of the merge grid blocks (every dimension)
KL_ML_PER_BLOCK = 1      # never reduce a block below this number of components
KL_MU_PER_BLOCK = 100    # always reduce a block down to this number of components
KL_LAMBDA = 0.05         # merge pairs whose KL upper bound is below this threshold


# ==============================================================================
# 1. Lorenz-63 drift (the noise covariance 2 Gamma I enters the level-set equations)
# ==============================================================================
def lorenz_drift(t, state, sigma=3.0, rho=26.5, beta=1.0):
    state = np.asarray(state)

    if state.ndim == 1:
        state = state[np.newaxis, :]
        is_1d = True
    else:
        is_1d = False

    x = state[..., 0]
    y = state[..., 1]
    z = state[..., 2]

    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z

    derivs = np.stack([dxdt, dydt, dzdt], axis=-1)

    if is_1d:
        return derivs[0]
    return derivs


# ==============================================================================
# 2. Mixture component and splitting
# ==============================================================================
class Component:
    """Gaussian mixture component: weight w, mean mu, covariance factor M (P = M M^T)."""

    def __init__(self, w, mu, M):
        self.w = float(w)
        self.mu = mu.copy()
        self.M = M.copy()

    def axis_error(self, t, sigma, rho, beta):
        """Column of M with the largest linearisation error, and the errors of all columns."""
        best_err = -1.0
        best_axis = None
        err_list = []

        for i in range(self.M.shape[1]):
            d = self.M[:, i]
            err = self.lin_approx_rel_error(d, t, sigma, rho, beta)
            err_list.append(err)

            if err > best_err:
                best_err = err
                best_axis = d

        return best_axis, err_list

    def lin_approx_rel_error(self, offset, t, sigma, rho, beta):
        """Relative deviation of the drift from linearity along `offset` (compares mu+d and mu+2d)."""
        cder = lorenz_drift(t, self.mu, sigma, rho, beta)
        c_norm = np.linalg.norm(cder)

        if c_norm < 1e-12:
            return 0.0

        near = lorenz_drift(t, self.mu + offset, sigma, rho, beta) - cder
        far = 0.5 * (
            lorenz_drift(t, self.mu + 2 * offset, sigma, rho, beta) - cder
        )

        return np.linalg.norm(near - far) / c_norm


def split_component(p, d):
    """Split p into three components (centre, left, right) along the direction d.

    The children share the covariance Sigma - 0.5 v v^T / s_dd with v = Sigma d_hat and
    s_dd = d_hat^T Sigma d_hat; the outer means are displaced by +- 1.03332 v / sqrt(s_dd).
    """
    if d is None:
        return [p]

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
# 3. Mixture reduction: blocked KL-bound merge
# ==============================================================================
def combine_components_kl_reduction(plist, Ml, Mu, lambda_thresh):
    """Greedy pairwise reduction of one block.

    The pair with the smallest KL upper bound is merged (moment matching) while the block holds
    more than Mu components, or more than Ml components and the smallest bound is below
    lambda_thresh.
    """
    if not plist or len(plist) <= Ml:
        return plist

    def _merge_two_components(p_i, p_j):
        w_i, w_j = p_i.w, p_j.w
        w_ij = w_i + w_j

        if w_ij == 0:
            return None

        mu_i, mu_j = p_i.mu, p_j.mu
        P_i = p_i.M @ p_i.M.T
        P_j = p_j.M @ p_j.M.T

        w_i_n = w_i / w_ij
        w_j_n = w_j / w_ij

        mu_ij = w_i_n * mu_i + w_j_n * mu_j

        P_ij = (
            w_i_n * P_i
            + w_j_n * P_j
            + w_i_n * w_j_n * np.outer(mu_i - mu_j, mu_i - mu_j)
        )

        try:
            L_ij = np.linalg.cholesky(P_ij + 1e-12 * np.eye(mu_i.size))
            return Component(w_ij, mu_ij, L_ij)
        except np.linalg.LinAlgError:
            return None

    def _calculate_kl_bound(p_i, p_j, p_merged):
        if p_merged is None:
            return np.inf

        n = p_i.mu.size

        det_Pi = np.linalg.det(p_i.M @ p_i.M.T + 1e-12 * np.eye(n))
        det_Pj = np.linalg.det(p_j.M @ p_j.M.T + 1e-12 * np.eye(n))
        det_Pij = np.linalg.det(p_merged.M @ p_merged.M.T + 1e-12 * np.eye(n))

        if det_Pi <= 0 or det_Pj <= 0 or det_Pij <= 0:
            return np.inf

        return 0.5 * (
            p_merged.w * np.log(det_Pij)
            - p_i.w * np.log(det_Pi)
            - p_j.w * np.log(det_Pj)
        )

    components_dict = {i: p for i, p in enumerate(plist)}
    B_matrix = np.full((len(plist), len(plist)), np.inf)
    merged_cache = {}

    indices = list(components_dict.keys())

    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            idx_i, idx_j = indices[i], indices[j]
            p_m = _merge_two_components(components_dict[idx_i], components_dict[idx_j])
            merged_cache[(idx_i, idx_j)] = p_m
            B_matrix[idx_i, idx_j] = _calculate_kl_bound(
                components_dict[idx_i],
                components_dict[idx_j],
                p_m
            )

    num_active = len(plist)

    while num_active > Mu or (
        num_active > Ml and np.nanmin(B_matrix) < lambda_thresh
    ):
        if num_active <= 1:
            break

        min_val = np.nanmin(B_matrix)
        if min_val == np.inf:
            break

        i_star, j_star = np.unravel_index(
            np.nanargmin(B_matrix),
            B_matrix.shape
        )

        p_merged = merged_cache.get((min(i_star, j_star), max(i_star, j_star)))

        if p_merged is None:
            p_merged = _merge_two_components(
                components_dict[i_star],
                components_dict[j_star]
            )

        if p_merged is None:
            B_matrix[i_star, j_star] = np.inf
            continue

        components_dict[i_star] = p_merged
        del components_dict[j_star]
        num_active -= 1

        B_matrix[j_star, :] = np.inf
        B_matrix[:, j_star] = np.inf

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
    """Sort the components into grid blocks (by their means) and reduce every block separately."""
    if not plist:
        return []

    dims = plist[0].mu.size

    min_mu = np.min([p.mu for p in plist], axis=0)
    max_mu = np.max([p.mu for p in plist], axis=0)

    blocksize_adj = np.where(blocksize == 0, 1e-9, blocksize)
    n_blocks = np.maximum(
        1,
        np.ceil((max_mu - min_mu) / blocksize_adj).astype(int) + 1
    )

    def block_id(mu):
        idx = np.clip(
            np.floor((mu - min_mu) / blocksize_adj).astype(int),
            0,
            n_blocks - 1
        )

        prod = [
            np.prod(n_blocks[:k]) if k > 0 else 1
            for k in range(dims)
        ]

        return sum(idx[k] * prod[k] for k in range(dims))

    blocks = {}

    for p in plist:
        blocks.setdefault(block_id(p.mu), []).append(p)

    new_list = []

    for _, group in blocks.items():
        if len(group) <= 1:
            new_list.extend(group)
        else:
            new_list.extend(
                combine_components_kl_reduction(
                    group,
                    kl_ml,
                    kl_mu,
                    kl_lambda
                )
            )

    return new_list


# ==============================================================================
# 4. Component propagation (level-set equations) with adaptive splitting
# ==============================================================================
def dxMdt(t, G, k_noise, sigma, rho, beta):
    dim = SYSTEM_DIM
    mu = G[:dim]
    M = G[dim:].reshape(dim, dim)

    dmu_sum = np.zeros_like(mu)

    for i in range(dim):
        dmu_sum += 0.5 * (
            lorenz_drift(t, mu + M[:, i], sigma, rho, beta)
            + lorenz_drift(t, mu - M[:, i], sigma, rho, beta)
        )

    dmu = dmu_sum / dim

    invMT = np.linalg.pinv(M.T + 1e-12 * np.eye(dim))
    dM = np.zeros_like(M)

    for i in range(dim):
        e = np.zeros(dim)
        e[i] = 1.0

        v_tip = lorenz_drift(t, mu + M[:, i], sigma, rho, beta)
        dM[:, i] = v_tip - dmu + 0.5 * (k_noise @ invMT @ e)

    return np.hstack((dmu, dM.ravel()))


def pack(p):
    return np.hstack((p.mu, p.M.ravel()))


def unpack(v, w):
    return Component(
        w,
        v[:SYSTEM_DIM],
        v[SYSTEM_DIM:].reshape(SYSTEM_DIM, SYSTEM_DIM)
    )


def integrate_until_split(
    p,
    t0,
    dt,
    k_noise,
    sigma,
    rho,
    beta,
    tol=TOL,
    min_dt=MIN_DT
):
    """Integrate p over [t0, t0 + dt]; if the linearisation error exceeds tol inside the step,
    bisect for the time at which the error lies in (0.8 tol, tol].

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
            lambda t, G: dxMdt(t, G, k_noise, sigma, rho, beta),
            (t0, t0 + currentstep),
            G0,
            method="RK45",
            rtol=1e-7,
            atol=1e-9,
            t_eval=[t0 + currentstep],
        )

        if not sol.success:
            adjustment *= 0.5
            currentstep -= adjustment
            needsplit = True
            continue

        p_mid = unpack(sol.y[:, 0], p.w)

        axis, err_list = p_mid.axis_error(
            t0 + currentstep,
            sigma,
            rho,
            beta
        )

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
            lambda t, G: dxMdt(t, G, k_noise, sigma, rho, beta),
            (t0, t0 + final_step),
            G0,
            method="RK45",
            rtol=1e-7,
            atol=1e-9,
            t_eval=[t0 + final_step],
        )

        if sol.success:
            p_mid = unpack(sol.y[:, 0], p.w)
            time_used = final_step
        else:
            p_mid = p
            time_used = 0.0

        needsplit = True

    return p_mid, time_used, needsplit


def propagate_adaptive(
    p,
    t0,
    dt,
    k_noise,
    sigma,
    rho,
    beta,
    tol=TOL,
    min_dt=MIN_DT,
    weight_thresh=WEIGHT_THRESH
):
    """Propagate one component over dt, splitting it (recursively) whenever required.
    Children with weight <= weight_thresh are propagated without further splitting."""
    if dt <= 0.0:
        return [p]

    t_rem = dt
    current_p = p
    propagated = []

    while t_rem > 1e-9:
        p_mid, t_used, needsplit = integrate_until_split(
            current_p,
            t0,
            t_rem,
            k_noise,
            sigma,
            rho,
            beta,
            tol,
            min_dt
        )

        if t_used <= 1e-9:
            # The integration made no progress: split at t0 and continue with the children.
            axis, _ = p_mid.axis_error(t0, sigma, rho, beta)
            children = split_component(p_mid, axis)

            big = [q for q in children if q.w > weight_thresh]
            small = [q for q in children if q.w <= weight_thresh]

            for q in small:
                p_end, _, _ = integrate_until_split(
                    q,
                    t0,
                    t_rem,
                    k_noise,
                    sigma,
                    rho,
                    beta,
                    tol,
                    min_dt
                )
                propagated.append(p_end)

            for q in big:
                propagated.extend(
                    propagate_adaptive(
                        q,
                        t0,
                        t_rem,
                        k_noise,
                        sigma,
                        rho,
                        beta,
                        tol,
                        min_dt,
                        weight_thresh
                    )
                )

            warnings.warn(f"AMF: integration failed at t = {t0:.6f}; component split without propagation.")
            break

        if not needsplit:
            propagated.append(p_mid)
            break

        axis, _ = p_mid.axis_error(t0 + t_used, sigma, rho, beta)
        children = split_component(p_mid, axis)

        big = [q for q in children if q.w > weight_thresh]
        small = [q for q in children if q.w <= weight_thresh]

        for q in small:
            p_end, _, _ = integrate_until_split(
                q,
                t0 + t_used,
                t_rem - t_used,
                k_noise,
                sigma,
                rho,
                beta,
                tol,
                min_dt
            )
            propagated.append(p_end)

        for q in big:
            propagated.extend(
                propagate_adaptive(
                    q,
                    t0 + t_used,
                    t_rem - t_used,
                    k_noise,
                    sigma,
                    rho,
                    beta,
                    tol,
                    min_dt,
                    weight_thresh
                )
            )

        break

    return propagated


def _propagate_component(args):
    p, t0, dt, k_noise, sigma, rho, beta, tol, min_dt, weight_thresh = args
    return propagate_adaptive(p, t0, dt, k_noise, sigma, rho, beta, tol, min_dt, weight_thresh)


# ==============================================================================
# 5. Measurement model and square-root cubature update
# ==============================================================================
def h_fun_lorenz(state):
    return state.copy()


def srckf_measurement_update(p, z, h_fun, R):
    """Square-root cubature update of one component; the returned weight is the prior weight times
    the component likelihood (not normalised)."""
    n = p.mu.size
    nz = z.shape[0]
    S_pred = p.M

    xi = np.sqrt(n) * np.hstack((np.eye(n), -np.eye(n)))
    X_cubature = p.mu[:, np.newaxis] + S_pred @ xi

    Z_cubature = np.array(
        [h_fun(X_cubature[:, i]) for i in range(2 * n)]
    ).T

    z_pred = np.mean(Z_cubature, axis=1)

    innovation = z - z_pred
    Z_centered = (Z_cubature - z_pred[:, np.newaxis]) / np.sqrt(2 * n)
    X_centered = (X_cubature - p.mu[:, np.newaxis]) / np.sqrt(2 * n)

    S_R = cholesky(R, lower=True)

    A_top = np.hstack([Z_centered, S_R])
    A_bottom = np.hstack([X_centered, np.zeros((n, nz))])
    A = np.vstack([A_top, A_bottom])

    _, R_qr = qr(A.T, mode="economic")
    T = R_qr.T

    T11 = T[:nz, :nz]
    T21 = T[nz:, :nz]
    T22 = T[nz:, nz:]

    Wk_T = solve_triangular(T11, T21.T, lower=True, trans="T")
    W_k = Wk_T.T

    mu_up = p.mu + W_k @ innovation
    M_up = T22

    # Component likelihood: sum of N(z; h(X_j), R) over the cubature points
    component_likelihood = 0.0
    num_points = 2 * n

    for j in range(num_points):
        mean_j = Z_cubature[:, j]
        residual_j = z - mean_j

        likelihood_at_point = multivariate_normal.pdf(
            residual_j,
            mean=np.zeros(nz),
            cov=R
        )

        component_likelihood += likelihood_at_point

    w_unnormalized = p.w * component_likelihood

    return Component(w_unnormalized, mu_up, M_up)


def _measurement_update_worker(args):
    p, z_obs, h_fun, R = args
    return srckf_measurement_update(p, z_obs, h_fun, R)


# ==============================================================================
# 6. Main AMF loop
# ==============================================================================
def run_amf(
    t_end=20.0,
    dt=0.01,
    sigma=3.0,
    rho=26.5,
    beta=1.0,
    gamma=0.2,
    measurements=None,
    h_fun=h_fun_lorenz,
    R=None,
    x0=PAPER_X0,
    P0=None,
    tol=TOL,
    min_dt=MIN_DT,
    weight_thresh=WEIGHT_THRESH,
    blocksize=BLOCKSIZE,
    kl_ml_per_block=KL_ML_PER_BLOCK,
    kl_mu_per_block=KL_MU_PER_BLOCK,
    kl_lambda=KL_LAMBDA,
    components_per_worker_min=1,
    n_workers=None,
    verbose=False,
):
    """Run the AMF on the stochastic Lorenz-63 system.

    Parameters
    ----------
    t_end, dt : final time and filter step.
    sigma, rho, beta : Lorenz parameters.
    gamma : forcing strength Gamma; the diffusion covariance is Q = 2 Gamma I3.
    measurements : dict {round(t, 6): z (3,)} or None (prediction only).
    h_fun, R : measurement function and noise covariance (default 0.8^2 I3).
    x0, P0 : initial mean and covariance (default (1,1,1), 0.1 I3).
    tol : split tolerance on the relative linearisation error.
    min_dt : smallest step of the split-time bisection.
    weight_thresh : components below this weight are not split further and are discarded after
        the propagation step.
    blocksize : edge length of the merge grid blocks (scalar or one value per dimension).
    kl_ml_per_block, kl_mu_per_block : lower / upper number of components per block in the merge.
    kl_lambda : KL-bound threshold of the merge.
    components_per_worker_min : minimum number of components per pool worker.
    n_workers : maximum pool size (None -> mp.cpu_count()).

    Returns
    -------
    dict with time_hist (T,), mu_hist (T,3), P_hist (T,3,3), count_hist (T,), measurements, the
    per-step component bookkeeping (added_by_prop_hist, reduced_by_thresh_hist, reduced_by_kl_hist,
    reduced_by_meas_hist) and components (final mixture as a list of (w, mu, M));
    T = n_steps (no entry for t = 0).
    """
    steps = int(np.round(t_end / dt))

    x0 = np.array(x0, dtype=float)
    if P0 is None:
        P0 = np.eye(SYSTEM_DIM) * PAPER_P0_VAR
    if R is None:
        R = np.diag([PAPER_MEAS_SIGMA ** 2] * SYSTEM_DIM)
    if np.isscalar(blocksize):
        blocksize = np.ones(SYSTEM_DIM) * blocksize
    else:
        blocksize = np.asarray(blocksize, dtype=float)

    # <eta_i(t) eta_j(t')> = 2 Gamma delta_ij delta(t - t')  ->  diffusion covariance 2 Gamma I
    k_noise = (2.0 * gamma) * np.eye(SYSTEM_DIM)

    M0 = np.linalg.cholesky(P0)
    components = [Component(1.0, x0, M0)]

    mu_hist, P_hist, count_hist, time_hist = [], [], [], []

    added_by_prop_hist = []
    reduced_by_thresh_hist = []
    reduced_by_kl_hist = []
    reduced_by_meas_hist = []

    t0 = 0.0

    max_cpu = max(1, mp.cpu_count() if n_workers is None else int(n_workers))
    pool = None
    curr_pool_size = 0

    for s_idx in range(steps):
        if verbose and s_idx % 100 == 0:
            print(f"Step {s_idx}/{steps}, time: {t0:.3f}, components: {len(components)}")

        n_comp = len(components)
        needed_procs = min(
            max_cpu,
            max(1, math.ceil(n_comp / components_per_worker_min))
        )

        if pool is None or needed_procs != curr_pool_size:
            if pool:
                pool.close()
                pool.join()

            curr_pool_size = needed_procs
            pool = mp.Pool(processes=curr_pool_size)

        N0 = len(components)

        # --- 1) Propagation with adaptive splitting ---
        task_args = [
            (p, t0, dt, k_noise, sigma, rho, beta, tol, min_dt, weight_thresh)
            for p in components
        ]

        results = pool.map(_propagate_component, task_args)

        new_comps_all = []

        for out_comps in results:
            new_comps_all.extend(out_comps)

        Nprop_all = len(new_comps_all)

        new_comps = [p for p in new_comps_all if p.w > weight_thresh]
        Nprop_kept = len(new_comps)

        # --- 2) Reduction: blocked KL merge ---
        components = combine_components_blocked_kl(
            new_comps,
            blocksize,
            kl_ml_per_block,
            kl_mu_per_block,
            kl_lambda
        )

        Nkl = len(components)

        current_time_rounded = float(round(t0 + dt, 6))

        # --- 3) Measurement update ---
        if measurements and current_time_rounded in measurements:
            if verbose:
                print(f"  --> Measurement update at t={current_time_rounded:.2f}")
            z_obs = measurements[current_time_rounded]

            if len(components) == 1:
                p_up = srckf_measurement_update(
                    components[0],
                    z_obs,
                    h_fun,
                    R
                )
                components = [p_up]
            else:
                update_args = [
                    (p, z_obs, h_fun, R)
                    for p in components
                ]

                updated_list = pool.map(
                    _measurement_update_worker,
                    update_args
                )

                components = [
                    p for p in updated_list
                    if p.w > 1e-15
                ]

            components = combine_components_blocked_kl(
                components,
                blocksize,
                kl_ml_per_block,
                kl_mu_per_block,
                kl_lambda
            )

            Nmeas = len(components)
        else:
            Nmeas = Nkl

        added_by_prop_hist.append(Nprop_all - N0)
        reduced_by_thresh_hist.append(Nprop_all - Nprop_kept)
        reduced_by_kl_hist.append(Nprop_kept - Nkl)
        reduced_by_meas_hist.append(Nkl - Nmeas)

        # --- 4) Weight normalisation ---
        total_w = sum(p.w for p in components)

        if total_w > 0:
            for p in components:
                p.w /= total_w
        else:
            if len(components) == 0:
                components = [Component(1.0, x0, M0)]

            for p in components:
                p.w = 1.0 / len(components)

        # --- 5) Mixture mean and covariance (moment matching) ---
        if len(components) > 0:
            mu_mix = np.sum([p.w * p.mu for p in components], axis=0)

            P_mix = np.sum(
                [
                    p.w * (
                        p.M @ p.M.T
                        + np.outer(p.mu - mu_mix, p.mu - mu_mix)
                    )
                    for p in components
                ],
                axis=0
            )
        else:
            mu_mix = mu_hist[-1] if mu_hist else x0
            P_mix = P_hist[-1] if P_hist else P0

        mu_hist.append(mu_mix)
        P_hist.append(P_mix)
        count_hist.append(len(components))
        time_hist.append(t0 + dt)

        t0 += dt

    if pool:
        pool.close()
        pool.join()

    return {
        "mu_hist": np.array(mu_hist),
        "P_hist": np.array(P_hist),
        "count_hist": np.array(count_hist),
        "time_hist": np.array(time_hist),
        "measurements": measurements,
        "added_by_prop_hist": np.array(added_by_prop_hist),
        "reduced_by_thresh_hist": np.array(reduced_by_thresh_hist),
        "reduced_by_kl_hist": np.array(reduced_by_kl_hist),
        "reduced_by_meas_hist": np.array(reduced_by_meas_hist),
        "components": [(p.w, p.mu.copy(), p.M.copy()) for p in components],
    }


# ==============================================================================
# 7. Command line
# ==============================================================================
def parse_measurements(specs):
    """["t,x,y,z", ...] -> {round(t, 6): array([x, y, z])}."""
    measurements = {}
    for spec in specs:
        vals = [float(v) for v in spec.split(",")]
        if len(vals) != 1 + SYSTEM_DIM:
            raise ValueError(f"--measurement expects 't,x,y,z', got '{spec}'")
        measurements[float(round(vals[0], 6))] = np.array(vals[1:])
    return measurements


def save_results(results, out_dir, save_components=False):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(np.c_[results["time_hist"], results["mu_hist"]], columns=["time", "x", "y", "z"])
    df["n_components"] = results["count_hist"]
    df.to_csv(os.path.join(out_dir, "mean_trajectory.csv"), index=False)

    np.savez(
        os.path.join(out_dir, "covariance.npz"),
        times=results["time_hist"],
        means=results["mu_hist"],
        covs=results["P_hist"],
    )

    if save_components:
        # Final mixture: weight, mean, covariance factor M and covariance P = M M^T per component
        t_final = results["time_hist"][-1] if len(results["time_hist"]) else 0.0
        rows = []
        for j, (w, mu, M) in enumerate(results["components"]):
            P = M @ M.T
            row = {
                "time": t_final,
                "component_id": j,
                "weight": w,
                "mu_x": mu[0],
                "mu_y": mu[1],
                "mu_z": mu[2],
            }
            for a in range(SYSTEM_DIM):
                for b in range(SYSTEM_DIM):
                    row[f"M_{a}{b}"] = M[a, b]
            for a in range(SYSTEM_DIM):
                for b in range(SYSTEM_DIM):
                    row[f"P_{a}{b}"] = P[a, b]
            rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "components.csv"), index=False)


def main():
    ap = argparse.ArgumentParser(description="AMF for the stochastic Lorenz-63 system.")
    ap.add_argument("--t-end", type=float, default=20.0)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--rho", type=float, default=26.5)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.2, help="forcing strength Gamma (Q = 2 Gamma I3)")
    ap.add_argument("--measurement", action="append", metavar="t,x,y,z",
                    help="measurement time and value; may be repeated "
                         "(default: the paper measurement 10.0,-0.25,0.02,20.09)")
    ap.add_argument("--no-measurements", action="store_true", help="prediction only")
    ap.add_argument("--meas-sigma", type=float, default=PAPER_MEAS_SIGMA,
                    help="measurement noise standard deviation (R = meas_sigma^2 I3)")
    ap.add_argument("--tol", type=float, default=TOL, help="split tolerance")
    ap.add_argument("--min-dt", type=float, default=MIN_DT, help="smallest step of the split-time bisection")
    ap.add_argument("--weight-thresh", type=float, default=WEIGHT_THRESH,
                    help="weight below which components are not split / are discarded")
    ap.add_argument("--blocksize", type=float, default=BLOCKSIZE, help="merge grid block size")
    ap.add_argument("--kl-lambda", type=float, default=KL_LAMBDA, help="KL-bound merge threshold")
    ap.add_argument("--kl-mu", type=int, default=KL_MU_PER_BLOCK,
                    help="maximum number of components per block after the merge")
    ap.add_argument("--workers", type=int, default=None, help="maximum pool size (default: all CPUs)")
    ap.add_argument("--out", default=os.path.join("results", "lorenz63_amf"))
    ap.add_argument("--save-components", action="store_true",
                    help="also write the final mixture to components.csv")
    ap.add_argument("--quiet", action="store_true", help="no progress output")
    args = ap.parse_args()

    if args.no_measurements:
        measurements = {}
    elif args.measurement:
        measurements = parse_measurements(args.measurement)
    else:
        measurements = parse_measurements([",".join(str(v) for v in PAPER_MEASUREMENT)])

    start_time = time.time()
    results = run_amf(
        t_end=args.t_end,
        dt=args.dt,
        sigma=args.sigma,
        rho=args.rho,
        beta=args.beta,
        gamma=args.gamma,
        measurements=measurements,
        R=np.diag([args.meas_sigma ** 2] * SYSTEM_DIM),
        tol=args.tol,
        min_dt=args.min_dt,
        weight_thresh=args.weight_thresh,
        blocksize=args.blocksize,
        kl_lambda=args.kl_lambda,
        kl_mu_per_block=args.kl_mu,
        n_workers=args.workers,
        verbose=not args.quiet,
    )
    wall = time.time() - start_time

    save_results(results, args.out, save_components=args.save_components)
    print(f"AMF (Lorenz-63) finished: {len(results['time_hist'])} steps, "
          f"max components {int(results['count_hist'].max())}, wall time {wall:.2f} s")
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass

    main()
