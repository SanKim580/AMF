#!/usr/bin/env python
"""LSKF (single-Gaussian square-root filter) for the 6-D coupled Van der Pol system.

Model (three ring-coupled Van der Pol oscillators, state s = (x1, y1, x2, y2, x3, y3)):

    dx_i = y_i dt
    dy_i = [mu (1 - x_i^2) y_i - x_i + k (x_{i-1} - x_i) + k (x_{i+1} - x_i)] dt + dW_i ,
    E[dW dW^T] = Q dt   (additive noise on all six states, Q = 6e-3 I6 in the paper case)

Measurement (per oscillator, polar):

    h(s) = (r_1, theta_1, r_2, theta_2, r_3, theta_3),  r_i = sqrt(x_i^2 + y_i^2),  theta_i = atan2(y_i, x_i)
    z = h(s) + v,  v ~ N(0, R),  R = diag(0.2^2, 0.2^2) x 3

The filter propagates the mean and a square-root factor M (P = M M^T) of a single Gaussian with an
augmented ODE (RK45) and applies a square-root cubature measurement update.

Paper case: mu = 1.5, k = 0.2, x0 = (2, 0) x 3, P0 = 0.01 I6, dt = 0.01, t in [0, 10], one measurement
at t = 5.

Usage:
    python coupled_vdp6d/lskf.py                       # paper case
    python coupled_vdp6d/lskf.py --t-end 2 --no-measurements --out results/test
    python coupled_vdp6d/lskf.py --mu 3.0 --k-coupling 0.2 --measurement "5.0,1.9,3.0"
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.linalg import cholesky, qr, solve_triangular

SYSTEM_DIM = 6
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
    """Drift of the ring-coupled Van der Pol system. state: [x1, y1, x2, y2, x3, y3]."""
    state = np.array(state)
    dim = len(state)

    x = state[0::2]
    y = state[1::2]

    # ring topology: neighbours i-1 and i+1
    x_left = np.roll(x, 1)
    x_right = np.roll(x, -1)

    # diffusive coupling term
    coupling = k_coupling * (x_left - x) + k_coupling * (x_right - x)

    dxdt = y
    dydt = mu_vdp * (1 - x**2) * y - x + coupling

    derivs = np.zeros(dim)
    derivs[0::2] = dxdt
    derivs[1::2] = dydt

    return derivs


# ==============================================================================
# 2. Augmented ODE (mean + covariance square-root propagation)
# ==============================================================================
def dxMdt_coupled(t, y, mu_vdp, k_coupling, Q):
    """ODE right-hand side that propagates the mean mu and the square-root factor M together.

    y is the flattened (dim, dim + 1) array [mu | M].
    """
    total_len = len(y)
    dim = int((-1 + np.sqrt(1 + 4 * total_len)) / 2)

    xM = y.reshape(dim, dim + 1)
    mu = xM[:, 0]
    M = xM[:, 1:]

    # mean drift: average of the drift over the symmetric points mu +/- M[:, i]
    dmu_acc = np.zeros(dim)
    for i in range(dim):
        val = 0.5 * (coupled_vdp_6d(t, mu + M[:, i], mu_vdp, k_coupling) +
                     coupled_vdp_6d(t, mu - M[:, i], mu_vdp, k_coupling))
        dmu_acc += val
    dmu = dmu_acc / dim

    # drift of the covariance factor
    try:
        MinvT = np.linalg.inv(M.T)
    except np.linalg.LinAlgError:
        MinvT = np.linalg.pinv(M.T)

    dM = np.zeros_like(M)
    for i in range(dim):
        e_i = np.zeros(dim)
        e_i[i] = 1
        v_i = coupled_vdp_6d(t, mu + M[:, i], mu_vdp, k_coupling)

        # continuous-time square-root form; Q is the process-noise covariance (intensity) matrix
        dM[:, i] = v_i - dmu + 0.5 * (Q @ MinvT @ e_i)

    return np.hstack((dmu.reshape(dim, 1), dM)).ravel()


# ==============================================================================
# 3. Measurement model and square-root cubature update
# ==============================================================================
def normalize_angle(angle):
    """Wrap an angle to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def h_fun_6d(state):
    """Polar measurement of each oscillator: [x1, y1, x2, y2, x3, y3] -> [r1, th1, r2, th2, r3, th3]."""
    n = len(state) // 2
    meas = np.zeros(len(state))

    for i in range(n):
        x = state[2 * i]
        y = state[2 * i + 1]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)

        meas[2 * i] = r
        meas[2 * i + 1] = theta

    return meas


def srckf_measurement_update_6d(mu_pred, S_pred, z_obs, h_fun, R):
    """Square-root cubature Kalman measurement update (angles at the odd measurement indices)."""
    n = len(mu_pred)
    nz = z_obs.shape[0]

    # 1. cubature points
    xi = np.sqrt(n) * np.hstack((np.eye(n), -np.eye(n)))
    X_cubature = mu_pred[:, np.newaxis] + S_pred @ xi

    # 2. propagate the points through the measurement function
    Z_cubature = np.array([h_fun(X_cubature[:, i]) for i in range(2 * n)]).T

    # 3. predicted measurement mean (circular mean for the angle components 1, 3, 5)
    z_pred = np.zeros(nz)
    for i in range(nz):
        if i % 2 == 1:
            mean_cos = np.mean(np.cos(Z_cubature[i, :]))
            mean_sin = np.mean(np.sin(Z_cubature[i, :]))
            z_pred[i] = np.arctan2(mean_sin, mean_cos)
        else:
            z_pred[i] = np.mean(Z_cubature[i, :])

    # 4. centred measurement points and innovation (angle differences wrapped)
    Z_centered = np.zeros_like(Z_cubature)
    innovation = z_obs - z_pred

    for i in range(nz):
        diff = Z_cubature[i, :] - z_pred[i]
        innov_diff = innovation[i]

        if i % 2 == 1:
            diff = normalize_angle(diff)
            innov_diff = normalize_angle(innov_diff)

        Z_centered[i, :] = diff
        innovation[i] = innov_diff

    Z_centered /= np.sqrt(2 * n)
    X_centered = (X_cubature - mu_pred[:, np.newaxis]) / np.sqrt(2 * n)

    # 5. Kalman gain from a QR decomposition
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

    # 6. state update; T22 is the square-root factor of the posterior covariance
    mu_updated = mu_pred + W_k @ innovation
    S_updated = T22

    return mu_updated, S_updated


# ==============================================================================
# 4. Filter loop
# ==============================================================================
def run_lskf(t_end=10.0, dt=0.01, mu_vdp=1.5, k_coupling=0.2, measurements=None, h_fun=h_fun_6d, R=None,
             x0=None, P0=None, Q=None, verbose=True):
    """Run the LSKF on the 6-D coupled Van der Pol system.

    measurements: dict {float(round(t, 6)): z (6,)}; None or {} -> prediction only.
    R, x0, P0, Q: measurement-noise covariance, initial mean / covariance, process-noise covariance
    (None -> paper values).

    Returns a dict with (T = number of steps + 1; the first entry is the initial condition at t = 0)
        time_hist (T,), mu_hist (T, 6), P_hist (T, 6, 6), var_hist (T, 6) [= diagonal of P_hist].
    """
    if measurements is None:
        measurements = {}
    if R is None:
        R = np.diag([0.2**2, 0.2**2] * 3)
    if x0 is None:
        x0 = np.array([2.0, 0.0, 2.0, 0.0, 2.0, 0.0])
    if P0 is None:
        P0 = np.eye(SYSTEM_DIM) * 0.01
    if Q is None:
        Q = (6e-3) * np.eye(SYSTEM_DIM)
    x0 = np.asarray(x0, dtype=float)
    dim = x0.size

    time_hist, mu_hist, var_hist, P_hist = [], [], [], []

    mu_k = x0.copy()
    M_k = cholesky(P0, lower=True)

    num_steps = int(np.round(t_end / dt))
    current_time = 0.0

    if verbose:
        print(f"Running LSKF (6-D coupled Van der Pol, mu={mu_vdp}, k={k_coupling})...")

    for k_step in range(num_steps):
        if verbose and k_step % 100 == 0:
            print(f"Step {k_step}/{num_steps}, time = {current_time:.2f}")

        # store the posterior at current_time: mean, marginal variances, full covariance
        time_hist.append(current_time)
        mu_hist.append(mu_k)
        var_hist.append(np.sum(M_k * M_k, axis=1))  # diag(M M^T)
        P_hist.append(M_k @ M_k.T)

        # time update
        y0_step = np.hstack((mu_k.reshape(-1, 1), M_k)).ravel()

        sol = solve_ivp(
            lambda t, y: dxMdt_coupled(t, y, mu_vdp, k_coupling, Q),
            (current_time, current_time + dt), y0_step,
            t_eval=[current_time + dt],
            method='RK45', rtol=1e-7, atol=1e-9
        )

        if not sol.success:
            warnings.warn(f"LSKF: integration failed at step {k_step} (t = {current_time:.4f}); stopping.")
            break

        y_pred = sol.y[:, 0]
        xM_pred = y_pred.reshape(dim, dim + 1)
        mu_pred = xM_pred[:, 0]
        M_pred = xM_pred[:, 1:]

        current_time += dt

        # measurement update
        mu_post = mu_pred
        M_post = M_pred

        current_time_rounded = float(round(current_time, 6))
        if current_time_rounded in measurements:
            if verbose:
                print(f"  -> measurement update at t = {current_time:.2f}")
            z_obs = measurements[current_time_rounded]
            mu_post, M_post = srckf_measurement_update_6d(mu_pred, M_pred, z_obs, h_fun, R)

        mu_k = mu_post
        M_k = M_post

    # final state
    time_hist.append(current_time)
    mu_hist.append(mu_k)
    var_hist.append(np.sum(M_k * M_k, axis=1))
    P_hist.append(M_k @ M_k.T)

    return {
        "time_hist": np.array(time_hist),
        "mu_hist": np.array(mu_hist),     # (T, 6)
        "P_hist": np.array(P_hist),       # (T, 6, 6)
        "var_hist": np.array(var_hist),   # (T, 6)
    }


# ==============================================================================
# 5. Command-line interface
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


def main():
    parser = argparse.ArgumentParser(
        description="LSKF for the 6-D coupled Van der Pol system (defaults: paper case).")
    parser.add_argument("--mu", type=float, default=1.5, help="Van der Pol parameter mu")
    parser.add_argument("--k-coupling", type=float, default=0.2, help="coupling strength k")
    parser.add_argument("--t-end", type=float, default=10.0, help="final time")
    parser.add_argument("--dt", type=float, default=0.01, help="filter time step")
    parser.add_argument("--no-measurements", action="store_true", help="prediction only")
    parser.add_argument("--measurement", action="append", metavar='"t,r,theta"',
                        help="measurement (same r, theta for the three oscillators) at time t; repeatable. "
                             "Default: the paper measurement at t = 5 for the given (mu, k)")
    parser.add_argument("--out", default=os.path.join("results", "coupled_vdp6d_lskf"),
                        help="output directory")
    parser.add_argument("--quiet", action="store_true", help="no progress output")
    args = parser.parse_args()

    measurements = build_measurements(args, parser)

    start = time.time()
    results = run_lskf(t_end=args.t_end, dt=args.dt, mu_vdp=args.mu, k_coupling=args.k_coupling,
                       measurements=measurements, verbose=not args.quiet)
    wall = time.time() - start

    os.makedirs(args.out, exist_ok=True)
    table = np.column_stack([results["time_hist"], results["mu_hist"]])
    pd.DataFrame(table, columns=["time"] + STATE_NAMES).to_csv(
        os.path.join(args.out, "mean_trajectory.csv"), index=False)
    np.savez(os.path.join(args.out, "covariance.npz"), times=results["time_hist"],
             means=results["mu_hist"], covs=results["P_hist"])

    print(f"Saved mean_trajectory.csv and covariance.npz to '{args.out}'")
    print(f"Wall time: {wall:.2f} s")


if __name__ == "__main__":
    main()
