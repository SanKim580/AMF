"""Adaptive Gaussian-mixture filter (AMF) for the 2-D stochastic Van der Pol oscillator.

Model
    dy = z dt + sqrt(q) dW_1
    dz = (mu (1 - y^2) z - y) dt + sqrt(q) dW_2          (Q = q I, paper: q = 6e-3)
    measurement  h(y, z) = (r, theta) = (sqrt(y^2 + z^2), atan2(z, y)),  R = diag(0.5^2, 0.8^2)

The density is a Gaussian mixture.  Every component N(mu, M M^T) is propagated with the
level-set ODE.  Whenever the relative linearisation error of the drift along a level-set
direction (a column of M) exceeds a tolerance, the component is split into three along that
direction with a moment-preserving split.  After every step the mixture is reduced: the state
space is divided into grid blocks and, inside every block, pairs of components are merged
greedily in the order of the smallest upper bound on the Kullback-Leibler discrimination until
the bound exceeds a threshold.  At measurement times every component is updated with a
square-root cubature update with a wrapped angle and re-weighted with its likelihood.
Components are propagated in parallel with a multiprocessing pool.

The script runs the paper case (x0 = (2, 0), P0 = 0.01 I, dt = 1/20, t in [0, 30], measurements
at t = 10 and t = 20; block size (0.04, 0.04), KL threshold 0.03, at most 100 components per
block, split tolerance 0.05) and writes the mixture mean trajectory and covariance history.

Usage
    python vdp2d/amf.py                        # paper case, mu = 1.5
    python vdp2d/amf.py --mu 2.0 --workers 8 --out results/amf_mu2 --save-components
    python vdp2d/amf.py --no-measurements --t-end 10
"""
import os

# one BLAS/OpenMP thread per process: the parallelism comes from the multiprocessing pool
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import math
import multiprocessing as mp
import time
import warnings

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import cholesky, qr, solve_triangular
from scipy.stats import multivariate_normal

# (r, theta) measured at t = 10 and t = 20 in the paper, for each value of mu
MEASUREMENT_TIMES = [10.0, 20.0]
R_THETA = {
    1.0: [[2, 3.2659], [2, 0.245]],
    1.25: [[2, 3.142], [10 ** 0.5, 1.249]],
    1.5: [[5 ** 0.5, -2.0344], [2 ** 0.5, 2.3562]],
    1.75: [[2 ** 0.5, -0.7854], [2, 2.8967]],
    2.0: [[2 ** 0.5, -0.7854], [2, 3.0172]],
}

# default split parameters
TOL, MIN_DT = 0.05, 0.0005


###############################################################################
# Van der Pol model
###############################################################################
def van_der_pol(t, x, mu_vdp=1.0):
    """Van der Pol oscillator equations: y' = z, z' = mu*(1 - y^2)*z - y."""
    y, z = x[..., 0], x[..., 1]
    dy_dt = z
    dz_dt = mu_vdp * (1 - y**2) * z - y
    return np.stack((dy_dt, dz_dt), axis=-1)


###############################################################################
# Mixture component, splitting and merging
###############################################################################
class Component:
    """Gaussian mixture component with weight w, mean mu and square-root factor M (P = M M^T)."""

    def __init__(self, w, mu, M):
        self.w, self.mu, self.M = float(w), mu.copy(), M.copy()

    def axis_error(self, t, mu_vdp):
        """Linearisation error along every column of M (the level-set directions).

        Returns the column with the largest error and the list of all errors.
        """
        best_err = -1.0
        best_axis = None
        err_list = []

        for i in range(self.M.shape[1]):
            d = self.M[:, i]
            err = self.lin_approx_rel_error_vdp(d, t, mu_vdp)
            err_list.append(err)

            if err > best_err:
                best_err = err
                best_axis = d

        return best_axis, err_list

    def lin_approx_rel_error_vdp(self, offset, t, mu_vdp):
        """Relative deviation of the drift from linearity along `offset`."""
        cder = van_der_pol(t, self.mu, mu_vdp)
        c_norm = np.linalg.norm(cder)
        if c_norm < 1e-12:
            return 0.0
        near = van_der_pol(t, self.mu + offset, mu_vdp) - cder
        far = 0.5 * (van_der_pol(t, self.mu + 2 * offset, mu_vdp) - cder)
        return np.linalg.norm(near - far) / c_norm


def split_component(p, d):
    """Moment-preserving split of a component into three along an arbitrary direction d.

    With d_hat = d / |d|, Sigma = M M^T, v = Sigma d_hat and s_dd = d_hat^T Sigma d_hat, the
    children share the covariance Sigma - v v^T / (2 s_dd) (the variance along d_hat is halved,
    everything Sigma-conjugate to d_hat is untouched), their means are mu and
    mu +/- 1.03332 v / sqrt(s_dd), and their weights are (1 - 2 * 0.21921) w and 0.21921 w.
    """
    n = p.mu.size
    Sigma = p.M @ p.M.T

    # no split along a (numerically) zero direction
    d_norm_val = np.linalg.norm(d)
    if d_norm_val < 1e-12:
        return [p]
    d_vec = d / d_norm_val

    v = Sigma @ d_vec
    if np.linalg.norm(v) < 1e-12:
        return [p]

    # variance along the split direction
    s_dd = d_vec.T @ v
    if s_dd < 1e-12:
        return [p]

    sigma_d = np.sqrt(s_dd)

    Sigma_final = Sigma - 0.5 * np.outer(v, v) / s_dd

    # small jitter for numerical safety; a failed factorisation cancels the split
    try:
        M_final = np.linalg.cholesky(Sigma_final + 1e-12 * np.eye(n))
    except np.linalg.LinAlgError:
        return [p]

    # displacement of the two outer children
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


def combine_components_kl_reduction(plist, Ml, Mu, lambda_thresh):
    """Greedy Gaussian-mixture reduction based on the upper bound of the KL discrimination.

    Parameters
    ----------
    plist : list of Component to reduce.
    Ml : minimum number of components after the reduction.
    Mu : maximum number of components; above it, merging is forced.
    lambda_thresh : merging stops when the smallest bound exceeds this threshold.

    Returns the reduced list of Component.
    """
    if not plist or len(plist) <= Ml:
        return plist

    def _merge_two_components(p_i, p_j):
        """Moment-preserving merge of two components (None if it fails)."""
        w_i, w_j = p_i.w, p_j.w
        mu_i, mu_j = p_i.mu, p_j.mu

        P_i = p_i.M @ p_i.M.T
        P_j = p_j.M @ p_j.M.T

        w_ij = w_i + w_j
        if w_ij == 0:
            return None

        w_i_ij = w_i / w_ij
        w_j_ij = w_j / w_ij

        mu_ij = w_i_ij * mu_i + w_j_ij * mu_j

        P_ij = (w_i_ij * P_i) + (w_j_ij * P_j) + \
               (w_i_ij * w_j_ij * np.outer(mu_i - mu_j, mu_i - mu_j))

        # square-root factor of the merged covariance (small jitter for numerical safety)
        try:
            L_ij = np.linalg.cholesky(P_ij + 1e-12 * np.eye(mu_i.size))
            return Component(w_ij, mu_ij, L_ij)
        except np.linalg.LinAlgError:
            return None

    def _calculate_kl_bound(p_i, p_j, p_merged):
        """Upper bound of the KL discrimination between the pair and its merged component."""
        if p_merged is None:
            return np.inf

        epsilon = 1e-20

        det_Pi = np.linalg.det(p_i.M @ p_i.M.T + 1e-12 * np.eye(p_i.mu.size))
        det_Pj = np.linalg.det(p_j.M @ p_j.M.T + 1e-12 * np.eye(p_j.mu.size))
        det_Pij = np.linalg.det(p_merged.M @ p_merged.M.T + 1e-12 * np.eye(p_merged.mu.size))

        if det_Pi <= 0 or det_Pj <= 0 or det_Pij <= 0:
            return np.inf

        bound = 0.5 * (p_merged.w * np.log(det_Pij + epsilon) -
                       p_i.w * np.log(det_Pi + epsilon) -
                       p_j.w * np.log(det_Pj + epsilon))
        return bound

    # 1. initialisation: components addressed by their index
    k = len(plist)
    components_dict = {i: p for i, p in enumerate(plist)}

    # 2. cost matrix B(i, j) of merging i and j (upper triangle; the rest stays inf)
    B_matrix = np.full((k, k), np.inf)
    merged_cache = {}

    indices = list(components_dict.keys())
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            idx_i, idx_j = indices[i], indices[j]
            p_i, p_j = components_dict[idx_i], components_dict[idx_j]

            p_merged = _merge_two_components(p_i, p_j)
            merged_cache[(idx_i, idx_j)] = p_merged

            B_matrix[idx_i, idx_j] = _calculate_kl_bound(p_i, p_j, p_merged)

    # 3. merge while there are too many components or the cheapest merge is below the threshold
    num_active = k
    while num_active > Mu or (num_active > Ml and np.nanmin(B_matrix) < lambda_thresh):
        if num_active <= 1:
            break

        # 4. cheapest pair (i*, j*)
        min_val = np.nanmin(B_matrix)
        if min_val == np.inf:
            break

        i_star, j_star = np.unravel_index(np.nanargmin(B_matrix), B_matrix.shape)

        # 5. merge i* and j*
        p_merged = merged_cache.get((min(i_star, j_star), max(i_star, j_star)))
        if p_merged is None:
            p_merged = _merge_two_components(components_dict[i_star], components_dict[j_star])

        if p_merged is None:
            # the merge failed: ignore this pair
            B_matrix[i_star, j_star] = np.inf
            continue

        # 6. i* becomes the merged component, j* is removed
        components_dict[i_star] = p_merged
        del components_dict[j_star]

        # 7. one component less
        num_active -= 1

        # 8. update the cost matrix: invalidate row/column j*, recompute row/column i*
        B_matrix[j_star, :] = np.inf
        B_matrix[:, j_star] = np.inf

        p_i_new = components_dict[i_star]
        active_indices = list(components_dict.keys())

        for idx_k in active_indices:
            if idx_k == i_star:
                continue

            p_k = components_dict[idx_k]
            p_merged_new = _merge_two_components(p_i_new, p_k)

            key = (min(i_star, idx_k), max(i_star, idx_k))
            merged_cache[key] = p_merged_new
            cost = _calculate_kl_bound(p_i_new, p_k, p_merged_new)
            B_matrix[i_star, idx_k] = cost
            B_matrix[idx_k, i_star] = cost

    return list(components_dict.values())


def combine_components_blocked_kl(plist, blocksize, kl_ml, kl_mu, kl_lambda):
    """Blocked mixture reduction.

    1. The components are assigned to grid blocks of size `blocksize` according to their means.
    2. The KL-bound reduction is applied independently inside every block.
    3. The reduced components of all blocks are collected.

    kl_ml / kl_mu are the minimum / maximum number of components per block and kl_lambda is the
    KL-bound threshold inside a block.
    """
    if not plist:
        return []

    dims = plist[0].mu.size

    # --- 1. grid blocks ---
    min_mu = np.min([p.mu for p in plist], axis=0)
    max_mu = np.max([p.mu for p in plist], axis=0)

    blocksize_adjusted = np.where(blocksize == 0, 1e-9, blocksize)

    n_blocks = np.ceil((max_mu - min_mu) / blocksize_adjusted).astype(int) + 1
    n_blocks = np.maximum(1, n_blocks)

    def block_id(mu):
        idx = np.floor((mu - min_mu) / blocksize_adjusted).astype(int)
        idx = np.clip(idx, 0, n_blocks - 1)
        # multi-index -> flat block id
        prod_values = [np.prod(n_blocks[:k]) if k > 0 else 1 for k in range(dims)]
        return sum(idx[k] * prod_values[k] for k in range(dims))

    blocks = {}
    for p in plist:
        blocks.setdefault(block_id(p.mu), []).append(p)

    # --- 2. KL-bound reduction inside every block ---
    new_component_list = []
    for components_in_block in blocks.values():
        if not components_in_block:
            continue

        if len(components_in_block) <= 1:
            new_component_list.extend(components_in_block)
            continue

        reduced_components = combine_components_kl_reduction(
            components_in_block,
            Ml=kl_ml,
            Mu=kl_mu,
            lambda_thresh=kl_lambda
        )
        new_component_list.extend(reduced_components)

    return new_component_list


###############################################################################
# Level-set propagation with adaptive splitting
###############################################################################
def dxMdt(t, G, k_noise, mu_vdp):
    """Right-hand side of the level-set ODE for the packed state G = [mu, M.ravel()]."""
    dim = 2
    mu = G[:dim]
    M = G[dim:].reshape(dim, dim)
    dmu_sum = np.zeros_like(mu)
    for i in range(dim):
        dmu_sum += 0.5 * (van_der_pol(t, mu + M[:, i], mu_vdp) + van_der_pol(t, mu - M[:, i], mu_vdp))
    dmu = dmu_sum / dim
    invMT = np.linalg.pinv(M.T + 1e-12 * np.eye(dim))
    dM = np.zeros_like(M)
    for i in range(dim):
        e = np.zeros(dim)
        e[i] = 1
        v_tip = van_der_pol(t, mu + M[:, i], mu_vdp)
        dM[:, i] = v_tip - dmu + 0.5 * (k_noise @ invMT @ e)
    return np.hstack((dmu, dM.ravel()))


def pack(p):
    return np.hstack((p.mu, p.M.ravel()))


def unpack(v, w):
    return Component(w, v[:2], v[2:].reshape(2, 2))


def integrate_until_split(p, t0, dt, k_noise, mu_vdp, tol=TOL, min_dt=MIN_DT):
    """Integrate a component from t0 for at most dt, stopping where it has to be split.

    Returns (propagated component, time actually integrated, split needed?).
    """
    G0 = pack(p)
    currentstep = dt
    adjustment = dt
    needsplit = True
    time_used = 0.0
    did_once = False
    max_attempts = 20
    attempts = 0
    # bisection on the integration time
    while (not did_once) or (adjustment > min_dt and attempts < max_attempts):
        did_once = True
        attempts += 1
        currentstep = max(currentstep, 1e-9)
        sol = solve_ivp(lambda t, G: dxMdt(t, G, k_noise, mu_vdp), (t0, t0 + currentstep), G0,
                        method='RK45', rtol=1e-7, atol=1e-10, t_eval=[t0 + currentstep])
        if not sol.success:
            # integration failed: retry with a shorter time
            adjustment *= 0.5
            currentstep -= adjustment
            needsplit = True
            continue
        p_mid = unpack(sol.y[:, 0], p.w)
        axis, err_list = p_mid.axis_error(t0 + currentstep, mu_vdp)
        err = max(err_list)
        if currentstep == dt and err < tol:
            # the whole step was integrated without exceeding the tolerance: no split
            needsplit = False
            time_used = currentstep
            break
        if err > tol:
            # error too large: shorten the integration time
            adjustment *= 0.5
            currentstep -= adjustment
        elif err > 0.8 * tol:
            # 0.8 tol < err < tol: integrate up to here and split
            needsplit = True
            time_used = currentstep
            break
        else:
            # error well below the tolerance: try a longer integration time
            adjustment *= 0.5
            currentstep += adjustment
            currentstep = min(currentstep, dt)
    # no split time was found: integrate at least the minimum step
    if time_used == 0:
        final_step = min(dt, max(currentstep, min_dt))
        sol = solve_ivp(lambda t, G: dxMdt(t, G, k_noise, mu_vdp), (t0, t0 + final_step), G0,
                        method='RK45', rtol=1e-7, atol=1e-10, t_eval=[t0 + final_step])
        if sol.success:
            p_mid = unpack(sol.y[:, 0], p.w)
            time_used = final_step
        else:
            # integration failed: split without changing the state
            p_mid = p
            time_used = 0.0
        needsplit = True
    return p_mid, time_used, needsplit


def propagate_adaptive(p, t0, dt, k_noise, mu_vdp, tol=TOL, min_dt=MIN_DT):
    """Propagate one component over dt, splitting recursively; returns the list of children."""
    if dt <= 0.0:
        return [p]
    t_rem = dt
    current_p = p
    propagated_components = []
    while t_rem > 1e-9:
        p_mid, t_used, needsplit = integrate_until_split(current_p, t0, t_rem, k_noise, mu_vdp, tol, min_dt)
        if t_used <= 1e-9:
            axis, _ = p_mid.axis_error(t0 + t_used, mu_vdp)
            propagated_components.extend(split_component(p_mid, axis))
            warnings.warn(f"integration failed at t = {t0:.6f}; the result is not reliable")
            break
        if not needsplit:
            propagated_components.append(p_mid)
            t_rem -= t_used
            t0 += t_used
            current_p = p_mid
            break
        else:
            axis, _ = p_mid.axis_error(t0 + t_used, mu_vdp)
            children = split_component(p_mid, axis)
            for q in children:
                propagated_components.extend(
                    propagate_adaptive(q, t0 + t_used, t_rem - t_used, k_noise, mu_vdp, tol, min_dt))
            t_rem = 0
            break
    return propagated_components


def _propagate_component_vdp(args):
    p, t0, dt, k_noise, mu_vdp, tol, min_dt = args
    return propagate_adaptive(p, t0, dt, k_noise, mu_vdp, tol, min_dt)


###############################################################################
# Square-root cubature measurement update (polar measurement, wrapped angle)
###############################################################################
def normalize_angle(angle):
    """Wrap an angle to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def srckf_measurement_update(p, z, h_fun, R):
    """Square-root cubature update of one component; returns it with an un-normalised weight."""
    n = p.mu.size
    nz = z.shape[0]
    S_pred = p.M

    # cubature points and their images in measurement space
    xi = np.sqrt(n) * np.hstack((np.eye(n), -np.eye(n)))
    X_cubature = p.mu[:, np.newaxis] + S_pred @ xi
    Z_cubature = np.array([h_fun(X_cubature[:, i]) for i in range(2 * n)]).T

    # predicted measurement; the angle is averaged on the unit circle
    r_pred = np.mean(Z_cubature[0, :])
    thetas = Z_cubature[1, :]
    mean_cos = np.mean(np.cos(thetas))
    mean_sin = np.mean(np.sin(thetas))
    theta_pred = np.arctan2(mean_sin, mean_cos)
    z_pred = np.array([r_pred, theta_pred])

    residual_for_update = z - z_pred
    residual_for_update[1] = normalize_angle(residual_for_update[1])

    # mean and square-root covariance update through a block triangularisation
    Z_centered_r = Z_cubature[0, :] - z_pred[0]
    Z_centered_theta = normalize_angle(Z_cubature[1, :] - z_pred[1])
    Z_centered = np.vstack([Z_centered_r, Z_centered_theta]) / np.sqrt(2 * n)
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

    # Kalman gain
    Wk_T = solve_triangular(T11, T21.T, lower=True, trans='T')
    W_k = Wk_T.T

    mu_up = p.mu + W_k @ residual_for_update
    M_up = T22

    # component likelihood: sum over the cubature points of N(z - h(X_j); 0, R), angle wrapped
    component_likelihood = 0.0
    for j in range(2 * n):
        mean_j = Z_cubature[:, j]
        residual_j = z - mean_j
        residual_j[1] = normalize_angle(residual_j[1])
        likelihood_at_point = multivariate_normal.pdf(residual_j, mean=np.zeros_like(residual_j), cov=R)
        component_likelihood += likelihood_at_point
    w_unnormalized = p.w * component_likelihood

    return Component(w_unnormalized, mu_up, M_up)


def _measurement_update_worker(args):
    p, z_obs, h_fun, R = args
    return srckf_measurement_update(p, z_obs, h_fun, R)


###############################################################################
# Measurement model helpers
###############################################################################
def h_fun_polar(x):
    """Polar measurement h(y, z) = (r, theta)."""
    y, z = x[0], x[1]
    r = np.sqrt(y**2 + z**2)
    theta = np.arctan2(z, y)
    return np.array([r, theta])


def make_polar_measurements(measurement_times, r_theta_values):
    """Build the {time: z} dictionary used by the filter (times rounded to 6 decimals)."""
    measurements = {}
    for t, val in zip(measurement_times, r_theta_values):
        measurements[float(round(t, 6))] = np.array(val)
    return measurements


###############################################################################
# Filter
###############################################################################
def run_amf(t_end=30.0, dt=1 / 20.0, mu_vdp=1.5, Q=None, measurements=None, h_fun=h_fun_polar,
            R=None, x0=None, P0=None, components_per_worker_min=3,
            blocksize=np.array([0.04, 0.04]), kl_ml_per_block=1, kl_mu_per_block=100, kl_lambda=0.03,
            tol=TOL, min_dt=MIN_DT, weight_thresh=1 / 10000, weight_thresh_update=1e-15,
            n_workers=None, verbose=True):
    """Run the AMF.

    Parameters
    ----------
    t_end, dt : final time and filter step.
    mu_vdp : Van der Pol parameter.
    Q : process-noise covariance (default 6e-3 I).
    measurements : dict {time (rounded to 6 decimals): z = (r, theta)} or None.
    h_fun, R : measurement function and measurement-noise covariance.
    x0, P0 : initial mean and covariance (default (2, 0) and 0.01 I).
    components_per_worker_min : the pool uses ceil(n_components / this) processes (at most n_workers).
    blocksize : grid block size of the blocked mixture reduction.
    kl_ml_per_block, kl_mu_per_block : minimum / maximum number of components per block.
    kl_lambda : KL-bound threshold of the mixture reduction.
    tol, min_dt : split tolerance on the relative linearisation error and smallest time increment
        of the bisection that locates the split time.
    weight_thresh, weight_thresh_update : components with a smaller weight are discarded after
        the propagation / after a measurement update.
    n_workers : maximum number of worker processes (None: all CPUs).

    Returns
    -------
    dict with time_hist (T,), mu_hist (T, 2), P_hist (T, 2, 2) (moment-matched mixture mean and
    covariance), count_hist (T,), components_hist (list of lists of Component) and measurements.
    The histories start at t = dt (the initial state is not stored).
    """
    x0 = np.array([2.0, 0.0]) if x0 is None else np.asarray(x0, dtype=float)
    P0 = np.diag([0.01] * 2) if P0 is None else np.asarray(P0, dtype=float)
    k_noise = 6e-3 * np.eye(2) if Q is None else np.asarray(Q, dtype=float)
    R = np.diag([0.5**2, 0.8**2]) if R is None else np.asarray(R, dtype=float)
    blocksize = np.asarray(blocksize, dtype=float)

    steps = int(np.round(t_end / dt))
    M0 = np.linalg.cholesky(P0)
    components = [Component(1.0, x0, M0)]
    mu_hist, P_hist, components_hist = [], [], []
    count_hist, time_hist = [], []
    t0 = 0.0
    max_cpu_processes = max(1, mp.cpu_count() if n_workers is None else int(n_workers))
    current_pool = None
    current_num_pool_processes = 0

    for s in range(steps):
        if verbose and s % 100 == 0:
            print(f"Step {s + 1}/{steps}: time = {t0:.2f}, components = {len(components)}")

        # resize the pool to the current number of components
        current_num_components = len(components)
        if current_num_components == 0:
            desired_num_processes = 1
        else:
            desired_num_processes = min(
                max_cpu_processes, max(1, math.ceil(current_num_components / components_per_worker_min)))
        if current_pool is None or desired_num_processes != current_num_pool_processes:
            if current_pool is not None:
                current_pool.close()
                current_pool.join()
            current_num_pool_processes = desired_num_processes
            current_pool = mp.Pool(processes=current_num_pool_processes)

        # propagate (and split) every component over dt, discard negligible weights, reduce
        task_args = [(p, t0, dt, k_noise, mu_vdp, tol, min_dt) for p in components]
        results_from_propagation = current_pool.map(_propagate_component_vdp, task_args)
        new = [item for sublist in results_from_propagation for item in sublist]
        new_parts = [p for p in new if p.w > weight_thresh]
        components = combine_components_blocked_kl(new_parts,
                                                   blocksize=blocksize,
                                                   kl_ml=kl_ml_per_block,
                                                   kl_mu=kl_mu_per_block,
                                                   kl_lambda=kl_lambda)

        current_time = round(t0 + dt, 6)

        if measurements and current_time in measurements:
            z_obs = measurements[current_time]
            if verbose:
                print(f"  -> measurement update at t = {current_time:.2f} with z(r, theta) = {z_obs}")

            if len(components) == 1:
                # single component: update mean and covariance, the weight stays 1
                p_updated_internal = srckf_measurement_update(components[0], z_obs, h_fun, R)
                components = [Component(1.0, p_updated_internal.mu, p_updated_internal.M)]

            elif len(components) > 1:
                # several components: update each one and re-weight it with its likelihood
                update_args = [(p, z_obs, h_fun, R) for p in components]
                updated_components_list = current_pool.map(_measurement_update_worker, update_args)
                components = [p for p in updated_components_list if p.w > weight_thresh_update]
                components = combine_components_blocked_kl(components,
                                                           blocksize=blocksize,
                                                           kl_ml=kl_ml_per_block,
                                                           kl_mu=kl_mu_per_block,
                                                           kl_lambda=kl_lambda)

        # normalise the weights
        total = sum(p.w for p in components)
        if total > 0.0:
            for p in components:
                p.w /= total
        else:
            # all weights vanished in the measurement update: fall back to equal weights
            warnings.warn(f"all component weights are zero at step {s + 1}; re-initialising the weights")
            if len(components) > 0:
                for p in components:
                    p.w = 1.0 / len(components)

        # moment-matched mean and covariance of the mixture
        if len(components) > 0:
            w = np.array([p.w for p in components])
            w /= w.sum()
            mu_mix = sum(w[i] * components[i].mu for i in range(len(components)))
            P_mix = sum(w[i] * (components[i].M @ components[i].M.T
                                + np.outer(components[i].mu - mu_mix, components[i].mu - mu_mix))
                        for i in range(len(components)))
        else:
            # no component left: repeat the previous moments
            mu_mix = mu_hist[-1] if mu_hist else x0
            P_mix = P_hist[-1] if P_hist else P0

        mu_hist.append(mu_mix)
        P_hist.append(P_mix)
        components_hist.append([Component(p.w, p.mu, p.M) for p in components])
        count_hist.append(len(components))
        time_hist.append(t0 + dt)
        t0 += dt

    if current_pool is not None:
        current_pool.close()
        current_pool.join()

    return {"mu_hist": np.array(mu_hist), "P_hist": np.array(P_hist), "components_hist": components_hist,
            "count_hist": np.array(count_hist), "time_hist": np.array(time_hist),
            "measurements": measurements}


###############################################################################
# Command line
###############################################################################
def parse_measurements(text):
    """Parse "t,r,theta;t,r,theta;..." into the measurement dictionary."""
    times, values = [], []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        t, r, theta = (float(v) for v in item.split(","))
        times.append(t)
        values.append([r, theta])
    return make_polar_measurements(times, values)


def select_measurements(parser, args):
    """Measurements of the paper case for --mu, or those given on the command line."""
    if args.no_measurements:
        return {}
    if args.measurements is not None:
        return parse_measurements(args.measurements)
    if args.mu in R_THETA:
        return make_polar_measurements(MEASUREMENT_TIMES, R_THETA[args.mu])
    parser.error(f"no paper measurements for mu = {args.mu}; "
                 "use --no-measurements or --measurements \"t,r,theta;t,r,theta\"")


def save_outputs(out_dir, results, save_components=False):
    """Write mean_trajectory.csv, covariance.npz and (optionally) components.csv."""
    os.makedirs(out_dir, exist_ok=True)
    table = np.column_stack([results["time_hist"], results["mu_hist"], results["count_hist"]])
    np.savetxt(os.path.join(out_dir, "mean_trajectory.csv"), table, delimiter=",",
               fmt=["%.18e", "%.18e", "%.18e", "%d"],
               header="time,mean_y,mean_z,n_components", comments="")
    np.savez(os.path.join(out_dir, "covariance.npz"), times=results["time_hist"],
             means=results["mu_hist"], covs=results["P_hist"])
    if save_components:
        rows = []
        for t, comps in zip(results["time_hist"], results["components_hist"]):
            for i, p in enumerate(comps):
                rows.append([t, i, p.w, p.mu[0], p.mu[1], p.M[0, 0], p.M[0, 1], p.M[1, 0], p.M[1, 1]])
        np.savetxt(os.path.join(out_dir, "components.csv"), np.array(rows).reshape(-1, 9),
                   delimiter=",", fmt=["%.18e", "%d"] + ["%.18e"] * 7,
                   header="time,component_index,weight,mu_y,mu_z,M_11,M_12,M_21,M_22", comments="")


def main():
    parser = argparse.ArgumentParser(
        description="AMF (adaptive Gaussian mixture filter) for the 2-D stochastic "
                    "Van der Pol oscillator (polar measurements).")
    parser.add_argument("--mu", type=float, default=1.5, help="Van der Pol parameter (default 1.5)")
    parser.add_argument("--t-end", type=float, default=30.0, help="final time (default 30)")
    parser.add_argument("--dt", type=float, default=1 / 20.0, help="filter step (default 1/20)")
    parser.add_argument("--q", type=float, default=6e-3, help="process-noise intensity, Q = q I")
    parser.add_argument("--x0", type=float, nargs=2, default=[2.0, 0.0], metavar=("Y", "Z"),
                        help="initial mean (default 2 0)")
    parser.add_argument("--p0", type=float, default=0.01, help="initial variance, P0 = p0 I")
    parser.add_argument("--sigma-r", type=float, default=0.5, help="std of the range measurement")
    parser.add_argument("--sigma-theta", type=float, default=0.8, help="std of the angle measurement")
    parser.add_argument("--measurements", default=None, metavar='"t,r,theta;..."',
                        help="explicit measurements (default: paper values for --mu at t = 10, 20)")
    parser.add_argument("--no-measurements", action="store_true", help="prediction only")
    parser.add_argument("--tol", type=float, default=TOL, help="split tolerance (default 0.05)")
    parser.add_argument("--min-dt", type=float, default=MIN_DT,
                        help="smallest time increment of the split-time search (default 0.0005)")
    parser.add_argument("--weight-thresh", type=float, default=1 / 10000,
                        help="components with a smaller weight are discarded (default 1e-4)")
    parser.add_argument("--blocksize", type=float, nargs=2, default=[0.04, 0.04], metavar=("BY", "BZ"),
                        help="grid block size of the mixture reduction (default 0.04 0.04)")
    parser.add_argument("--kl-lambda", type=float, default=0.03,
                        help="KL-bound threshold of the mixture reduction (default 0.03)")
    parser.add_argument("--kl-mu", type=int, default=100,
                        help="maximum number of components per block (default 100)")
    parser.add_argument("--kl-ml", type=int, default=1,
                        help="minimum number of components per block (default 1)")
    parser.add_argument("--workers", type=int, default=None,
                        help="maximum number of worker processes (default: all CPUs)")
    parser.add_argument("--out", default=os.path.join("results", "vdp2d_amf"), help="output directory")
    parser.add_argument("--save-components", action="store_true",
                        help="also write components.csv (all components at every step)")
    parser.add_argument("--quiet", action="store_true", help="no progress output")
    args = parser.parse_args()

    measurements = select_measurements(parser, args)

    start = time.time()
    results = run_amf(t_end=args.t_end, dt=args.dt, mu_vdp=args.mu, Q=args.q * np.eye(2),
                      measurements=measurements, h_fun=h_fun_polar,
                      R=np.diag([args.sigma_r**2, args.sigma_theta**2]),
                      x0=np.array(args.x0), P0=args.p0 * np.eye(2),
                      blocksize=np.array(args.blocksize), kl_ml_per_block=args.kl_ml,
                      kl_mu_per_block=args.kl_mu, kl_lambda=args.kl_lambda,
                      tol=args.tol, min_dt=args.min_dt, weight_thresh=args.weight_thresh,
                      n_workers=args.workers, verbose=not args.quiet)
    wall = time.time() - start

    save_outputs(args.out, results, save_components=args.save_components)
    print(f"AMF (mu = {args.mu}) finished in {wall:.2f} s wall time; "
          f"final number of components: {results['count_hist'][-1]}; results in '{args.out}'")


if __name__ == "__main__":
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass
    main()
