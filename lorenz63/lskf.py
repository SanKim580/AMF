"""Level-set Kalman filter (LSKF) for the stochastic Lorenz-63 system.

Model (additive white-noise forcing):

    dx = sigma (y - x) dt        + sqrt(2 Gamma) dW_x
    dy = [x (rho - z) - y] dt    + sqrt(2 Gamma) dW_y
    dz = [x y - beta z] dt       + sqrt(2 Gamma) dW_z

i.e. dX = f(X) dt + sqrt(2 Gamma) dW with diffusion covariance Q = 2 Gamma I3.
Measurement: direct observation of the state, z = h(x) + v, h(x) = x, v ~ N(0, R).

The script propagates a single Gaussian (mean mu and covariance factor M, P = M M^T) with the
level-set equations, applies a square-root cubature measurement update at the measurement times
and writes the mean / covariance history.

Usage:
    python lorenz63/lskf.py                              # paper case
    python lorenz63/lskf.py --t-end 5 --no-measurements --out results/lskf_test
    python lorenz63/lskf.py --measurement "2.0,1.5,2.0,20.0" --measurement "4.0,-3.0,-2.0,15.0"
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.linalg import cholesky, qr, solve_triangular

SYSTEM_DIM = 3

# Paper case
PAPER_X0 = (1.0, 1.0, 1.0)
PAPER_P0_VAR = 0.1
PAPER_MEAS_SIGMA = 0.8
PAPER_MEASUREMENT = (10.0, -0.25, 0.02, 20.09)   # t, x, y, z


# ==============================================================================
# 1. Lorenz-63 drift (the noise covariance Q = 2 Gamma I enters the LSKF equations)
# ==============================================================================
def lorenz_drift(t, state, sigma=3.0, rho=26.5, beta=1.0):
    state = np.array(state)

    x = state[0]
    y = state[1]
    z = state[2]

    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z

    return np.array([dxdt, dydt, dzdt])


# ==============================================================================
# 2. Augmented ODE: mean + covariance square-root propagation
# ==============================================================================
def dxMdt_lorenz(t, y, sigma, rho, beta, Q):
    total_len = len(y)
    dim = int((-1 + np.sqrt(1 + 4 * total_len)) / 2)

    xM = y.reshape(dim, dim + 1)
    mu = xM[:, 0]
    M = xM[:, 1:]

    dmu_acc = np.zeros(dim)

    for i in range(dim):
        val = 0.5 * (
            lorenz_drift(t, mu + M[:, i], sigma, rho, beta)
            + lorenz_drift(t, mu - M[:, i], sigma, rho, beta)
        )
        dmu_acc += val

    dmu = dmu_acc / dim

    try:
        MinvT = np.linalg.inv(M.T)
    except np.linalg.LinAlgError:
        MinvT = np.linalg.pinv(M.T)

    dM = np.zeros_like(M)

    for i in range(dim):
        e_i = np.zeros(dim)
        e_i[i] = 1.0

        v_i = lorenz_drift(t, mu + M[:, i], sigma, rho, beta)

        dM[:, i] = v_i - dmu + 0.5 * (Q @ MinvT @ e_i)

    return np.hstack((dmu.reshape(dim, 1), dM)).ravel()


# ==============================================================================
# 3. Measurement model and square-root cubature update
# ==============================================================================
def h_fun_lorenz(state):
    return np.array(state)


def srckf_measurement_update(mu_pred, S_pred, z_obs, h_fun, R):
    n = len(mu_pred)
    nz = z_obs.shape[0]

    xi = np.sqrt(n) * np.hstack((np.eye(n), -np.eye(n)))
    X_cubature = mu_pred[:, np.newaxis] + S_pred @ xi

    Z_cubature = np.array([h_fun(X_cubature[:, i]) for i in range(2 * n)]).T

    z_pred = np.mean(Z_cubature, axis=1)

    innovation = z_obs - z_pred
    Z_centered = (Z_cubature - z_pred[:, np.newaxis]) / np.sqrt(2 * n)
    X_centered = (X_cubature - mu_pred[:, np.newaxis]) / np.sqrt(2 * n)

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

    mu_updated = mu_pred + W_k @ innovation
    S_updated = T22

    return mu_updated, S_updated


# ==============================================================================
# 4. LSKF run
# ==============================================================================
def run_lskf(
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
    verbose=False,
):
    """Run the LSKF on the stochastic Lorenz-63 system.

    Parameters
    ----------
    t_end, dt : final time and filter step.
    sigma, rho, beta : Lorenz parameters.
    gamma : forcing strength Gamma; the diffusion covariance is Q = 2 Gamma I3.
    measurements : dict {round(t, 6): z (3,)} or None (prediction only).
    h_fun, R : measurement function and noise covariance (default 0.8^2 I3).
    x0, P0 : initial mean and covariance (default (1,1,1), 0.1 I3).

    Returns
    -------
    dict with time_hist (T,), mu_hist (T,3), P_hist (T,3,3), var_hist (T,3); T = n_steps + 1
    (the initial condition at t = 0 is the first entry).
    """
    x0 = np.array(x0, dtype=float)
    if P0 is None:
        P0 = np.eye(SYSTEM_DIM) * PAPER_P0_VAR
    if R is None:
        R = np.diag([PAPER_MEAS_SIGMA ** 2] * SYSTEM_DIM)
    if measurements is None:
        measurements = {}

    # <eta_i(t) eta_j(t')> = 2 Gamma delta_ij delta(t - t')  ->  diffusion covariance Q = 2 Gamma I
    Q = (2.0 * gamma) * np.eye(SYSTEM_DIM)

    time_hist, mu_hist, var_hist, P_hist = [], [], [], []

    mu_k = x0.copy()
    M_k = cholesky(P0, lower=True)

    num_steps = int(np.round(t_end / dt))
    current_time = 0.0

    for k_step in range(num_steps):
        if verbose and k_step % 100 == 0:
            print(f"Step {k_step}/{num_steps}, time: {current_time:.3f}")

        time_hist.append(current_time)
        mu_hist.append(mu_k.copy())
        var_hist.append(np.sum(M_k * M_k, axis=1))
        P_hist.append(M_k @ M_k.T)

        y0_step = np.hstack((mu_k.reshape(-1, 1), M_k)).ravel()

        sol = solve_ivp(
            lambda t, y: dxMdt_lorenz(t, y, sigma, rho, beta, Q),
            (current_time, current_time + dt),
            y0_step,
            t_eval=[current_time + dt],
            method='RK45',
            rtol=1e-7,
            atol=1e-9
        )

        if not sol.success:
            warnings.warn(f"LSKF: integration failed at step {k_step} (t = {current_time:.6f}); stopping.")
            break

        y_pred = sol.y[:, 0]
        xM_pred = y_pred.reshape(SYSTEM_DIM, SYSTEM_DIM + 1)

        mu_pred = xM_pred[:, 0]
        M_pred = xM_pred[:, 1:]

        current_time += dt

        mu_post = mu_pred
        M_post = M_pred

        current_time_rounded = float(round(current_time, 6))

        if current_time_rounded in measurements:
            if verbose:
                print(f"  -> Measurement update at t={current_time:.3f}")
            z_obs = measurements[current_time_rounded]
            mu_post, M_post = srckf_measurement_update(
                mu_pred,
                M_pred,
                z_obs,
                h_fun,
                R
            )

        mu_k = mu_post
        M_k = M_post

    time_hist.append(current_time)
    mu_hist.append(mu_k.copy())
    var_hist.append(np.sum(M_k * M_k, axis=1))
    P_hist.append(M_k @ M_k.T)

    return {
        "time_hist": np.array(time_hist),
        "mu_hist": np.array(mu_hist),
        "P_hist": np.array(P_hist),
        "var_hist": np.array(var_hist),
    }


# ==============================================================================
# 5. Command line
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


def save_results(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(np.c_[results["time_hist"], results["mu_hist"]], columns=["time", "x", "y", "z"])
    df.to_csv(os.path.join(out_dir, "mean_trajectory.csv"), index=False)
    np.savez(
        os.path.join(out_dir, "covariance.npz"),
        times=results["time_hist"],
        means=results["mu_hist"],
        covs=results["P_hist"],
    )


def main():
    ap = argparse.ArgumentParser(description="LSKF for the stochastic Lorenz-63 system.")
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
    ap.add_argument("--out", default=os.path.join("results", "lorenz63_lskf"))
    ap.add_argument("--quiet", action="store_true", help="no progress output")
    args = ap.parse_args()

    if args.no_measurements:
        measurements = {}
    elif args.measurement:
        measurements = parse_measurements(args.measurement)
    else:
        measurements = parse_measurements([",".join(str(v) for v in PAPER_MEASUREMENT)])

    start_time = time.time()
    results = run_lskf(
        t_end=args.t_end,
        dt=args.dt,
        sigma=args.sigma,
        rho=args.rho,
        beta=args.beta,
        gamma=args.gamma,
        measurements=measurements,
        h_fun=h_fun_lorenz,
        R=np.diag([args.meas_sigma ** 2] * SYSTEM_DIM),
        verbose=not args.quiet,
    )
    wall = time.time() - start_time

    save_results(results, args.out)
    print(f"LSKF (Lorenz-63) finished: {len(results['time_hist'])} stored steps, wall time {wall:.2f} s")
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
