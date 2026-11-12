"""Level-Set Kalman Filter (LSKF) for the 2-D stochastic Van der Pol oscillator.

Model
    dy = z dt + sqrt(q) dW_1
    dz = (mu (1 - y^2) z - y) dt + sqrt(q) dW_2          (Q = q I, paper: q = 6e-3)
    measurement  h(y, z) = (r, theta) = (sqrt(y^2 + z^2), atan2(z, y)),  R = diag(0.5^2, 0.8^2)

The filter carries a single Gaussian N(mu, M M^T).  Between measurements the mean and the
square-root factor M are propagated with the level-set ODE (integrated with RK45 over each
step dt); at measurement times a square-root cubature update with a wrapped angle is applied.

The script runs the paper case (x0 = (2, 0), P0 = 0.01 I, dt = 1/20, t in [0, 30], measurements
at t = 10 and t = 20) and writes the mean trajectory and the covariance history.

Usage
    python vdp2d/lskf.py                       # paper case, mu = 1.5
    python vdp2d/lskf.py --mu 2.0 --out results/lskf_mu2
    python vdp2d/lskf.py --no-measurements --t-end 10
"""
import argparse
import os
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import cholesky, qr, solve_triangular

# (r, theta) measured at t = 10 and t = 20 in the paper, for each value of mu
MEASUREMENT_TIMES = [10.0, 20.0]
R_THETA = {
    1.0: [[2, 3.2659], [2, 0.245]],
    1.25: [[2, 3.142], [10 ** 0.5, 1.249]],
    1.5: [[5 ** 0.5, -2.0344], [2 ** 0.5, 2.3562]],
    1.75: [[2 ** 0.5, -0.7854], [2, 2.8967]],
    2.0: [[2 ** 0.5, -0.7854], [2, 3.0172]],
}


###############################################################################
# Van der Pol model and level-set ODE
###############################################################################
def van_der_pol(t, x, mu_vdp=1.0):
    """Drift of the Van der Pol oscillator: y' = z, z' = mu (1 - y^2) z - y."""
    y, z = x[0], x[1]
    dy_dt = z
    dz_dt = mu_vdp * (1 - y**2) * z - y
    return np.stack((dy_dt, dz_dt), axis=-1)


def dxMdt_vdp(t, y, mu_vdp, Q):
    """Right-hand side of the level-set ODE for the packed state [mu | M] (2 x 3)."""
    xM = y.reshape(2, 3)
    mu, M = xM[:, 0], xM[:, 1:3]

    # mean: average of the drift over the 2n level-set points mu +/- M[:, i]
    dmu = np.zeros(2)
    for i in range(2):
        mu_plus, mu_minus = mu + M[:, i], mu - M[:, i]
        dmu += 0.5 * (van_der_pol(t, mu_plus, mu_vdp) + van_der_pol(t, mu_minus, mu_vdp))
    dmu *= 1 / 2

    try:
        MinvT = np.linalg.inv(M.T)
    except np.linalg.LinAlgError:
        MinvT = np.linalg.pinv(M.T)  # pseudo-inverse if M is singular

    # square-root factor: relative drift of each level-set point plus the diffusion term
    dM = np.zeros_like(M)
    for i in range(2):
        e_i = np.zeros(2)
        e_i[i] = 1
        v_i = van_der_pol(t, mu + M[:, i], mu_vdp)
        dM[:, i] = v_i - dmu + 0.5 * (Q @ MinvT @ e_i)

    return np.hstack((dmu.reshape(2, 1), dM)).ravel()


###############################################################################
# Square-root cubature measurement update (polar measurement, wrapped angle)
###############################################################################
def normalize_angle(angle):
    """Wrap an angle to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def srckf_measurement_update(mu_pred, S_pred, z_obs, h_fun, R):
    """Square-root cubature Kalman update; the second measurement component is an angle."""
    n = len(mu_pred)
    nz = z_obs.shape[0]

    xi = np.sqrt(n) * np.hstack((np.eye(n), -np.eye(n)))
    X_cubature = mu_pred[:, np.newaxis] + S_pred @ xi
    Z_cubature = np.array([h_fun(X_cubature[:, i]) for i in range(2 * n)]).T

    # predicted measurement; the angle is averaged on the unit circle
    r_pred = np.mean(Z_cubature[0, :])
    thetas = Z_cubature[1, :]
    mean_cos = np.mean(np.cos(thetas))
    mean_sin = np.mean(np.sin(thetas))
    theta_pred = np.arctan2(mean_sin, mean_cos)
    z_pred = np.array([r_pred, theta_pred])

    # centred matrices (angle differences are wrapped)
    Z_centered_r = Z_cubature[0, :] - z_pred[0]
    Z_centered_theta = normalize_angle(Z_cubature[1, :] - z_pred[1])
    Z_centered = np.vstack([Z_centered_r, Z_centered_theta]) / np.sqrt(2 * n)

    X_centered = (X_cubature - mu_pred[:, np.newaxis]) / np.sqrt(2 * n)
    S_R = cholesky(R, lower=True)

    A_top = np.hstack([Z_centered, S_R])
    A_bottom = np.hstack([X_centered, np.zeros((n, nz))])
    A = np.vstack([A_top, A_bottom])

    # triangularisation:  T = [[T11, 0], [T21, T22]]
    _, R_qr = qr(A.T, mode='economic')
    T = R_qr.T
    T11 = T[:nz, :nz]
    T21 = T[nz:, :nz]
    T22 = T[nz:, nz:]

    # Kalman gain W = T21 T11^{-1}
    Wk_T = solve_triangular(T11, T21.T, lower=True, trans='T')
    W_k = Wk_T.T

    innovation = z_obs - z_pred
    innovation[1] = normalize_angle(innovation[1])

    mu_updated = mu_pred + W_k @ innovation
    S_updated = T22
    return mu_updated, S_updated


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
def run_lskf(t_end=30.0, dt=1 / 20.0, mu_vdp=1.5, measurements=None, h_fun=h_fun_polar, R=None,
             x0=None, P0=None, Q=None, verbose=True):
    """Run the LSKF.

    Parameters
    ----------
    t_end, dt : final time and filter step.
    mu_vdp : Van der Pol parameter.
    measurements : dict {time (rounded to 6 decimals): z = (r, theta)} or None.
    h_fun, R : measurement function and measurement-noise covariance.
    x0, P0 : initial mean and covariance (default (2, 0) and 0.01 I).
    Q : process-noise covariance (default 6e-3 I).

    Returns
    -------
    dict with time_hist (T,), mu_hist (T, 2), P_hist (T, 2, 2); the t = 0 state is included.
    """
    x0 = np.array([2.0, 0.0]) if x0 is None else np.asarray(x0, dtype=float)
    P0 = np.diag([0.01] * 2) if P0 is None else np.asarray(P0, dtype=float)
    Q = 6e-3 * np.eye(2) if Q is None else np.asarray(Q, dtype=float)
    R = np.diag([0.5**2, 0.8**2]) if R is None else np.asarray(R, dtype=float)
    measurements = {} if measurements is None else measurements

    time_hist, mu_hist, P_hist = [], [], []

    mu_k = x0
    M_k = cholesky(P0, lower=True)  # P0 = M_k @ M_k.T

    num_steps = int(round(t_end / dt))
    current_time = 0.0

    for k_step in range(num_steps):
        if verbose and k_step % 100 == 0:
            print(f"Step {k_step}/{num_steps}, time = {current_time:.2f}")

        # store the state before the prediction
        time_hist.append(current_time)
        mu_hist.append(mu_k)
        P_hist.append(M_k @ M_k.T)

        # --- time update: integrate the packed state [mu | M] over one step dt ---
        y0_step = np.hstack((mu_k.reshape(-1, 1), M_k)).ravel()
        sol = solve_ivp(
            lambda t, y: dxMdt_vdp(t, y, mu_vdp, Q),
            (current_time, current_time + dt), y0_step,
            t_eval=[current_time + dt], method='RK45',
            rtol=1e-6, atol=1e-9
        )
        y_pred = sol.y[:, 0]
        xM_pred = y_pred.reshape(2, 3)
        mu_pred = xM_pred[:, 0]
        M_pred = xM_pred[:, 1:3]

        current_time += dt

        # --- measurement update ---
        mu_posterior = mu_pred
        M_posterior = M_pred

        current_time_rounded = round(current_time, 6)
        if current_time_rounded in measurements:
            if verbose:
                print(f"  -> measurement update at t = {current_time:.2f}")
            z_obs = measurements[current_time_rounded]
            mu_posterior, M_posterior = srckf_measurement_update(mu_pred, M_pred, z_obs, h_fun, R)

        mu_k = mu_posterior
        M_k = M_posterior

    # final state
    time_hist.append(current_time)
    mu_hist.append(mu_k)
    P_hist.append(M_k @ M_k.T)

    return {"time_hist": np.array(time_hist),
            "mu_hist": np.array(mu_hist),
            "P_hist": np.array(P_hist)}


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


def save_outputs(out_dir, results):
    """Write mean_trajectory.csv and covariance.npz."""
    os.makedirs(out_dir, exist_ok=True)
    table = np.column_stack([results["time_hist"], results["mu_hist"]])
    np.savetxt(os.path.join(out_dir, "mean_trajectory.csv"), table, delimiter=",",
               header="time,mean_y,mean_z", comments="")
    np.savez(os.path.join(out_dir, "covariance.npz"), times=results["time_hist"],
             means=results["mu_hist"], covs=results["P_hist"])


def main():
    parser = argparse.ArgumentParser(
        description="LSKF for the 2-D stochastic Van der Pol oscillator (polar measurements).")
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
    parser.add_argument("--out", default=os.path.join("results", "vdp2d_lskf"), help="output directory")
    parser.add_argument("--quiet", action="store_true", help="no progress output")
    args = parser.parse_args()

    measurements = select_measurements(parser, args)

    start = time.time()
    results = run_lskf(t_end=args.t_end, dt=args.dt, mu_vdp=args.mu, measurements=measurements,
                       h_fun=h_fun_polar, R=np.diag([args.sigma_r**2, args.sigma_theta**2]),
                       x0=np.array(args.x0), P0=args.p0 * np.eye(2), Q=args.q * np.eye(2),
                       verbose=not args.quiet)
    wall = time.time() - start

    save_outputs(args.out, results)
    print(f"LSKF (mu = {args.mu}) finished in {wall:.2f} s wall time; results in '{args.out}'")


if __name__ == "__main__":
    main()
