"""Square-root continuous-discrete cubature Kalman filter (SR-CDCKF) for the 2-D stochastic
Van der Pol oscillator.

Model
    dy = z dt + sqrt(q) dW_1
    dz = (mu (1 - y^2) z - y) dt + sqrt(q) dW_2          (Q = q I, paper: q = 6e-3)
    measurement  h(y, z) = (r, theta) = (sqrt(y^2 + z^2), atan2(z, y)),  R = diag(0.5^2, 0.8^2)

The filter carries a single Gaussian N(mu, S S^T).  Each filter step dt is divided into m_sub
micro-steps; in every micro-step the cubature points are propagated with the deterministic part
of the Ito-Taylor expansion of strong order 1.5 and the square-root covariance is obtained by a
QR triangularisation.  At measurement times a square-root cubature update with a wrapped angle
is applied.

The script runs the paper case (x0 = (2, 0), P0 = 0.01 I, dt = 1/20, m_sub = 5, t in [0, 30],
measurements at t = 10 and t = 20) and writes the mean trajectory and the covariance history.

Usage
    python vdp2d/cdckf.py                      # paper case, mu = 1.5
    python vdp2d/cdckf.py --mu 2.0 --m-sub 10 --out results/cdckf_mu2
    python vdp2d/cdckf.py --no-measurements --t-end 10
"""
import argparse
import os
import time

import numpy as np
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
# Van der Pol model and Ito-Taylor (order 1.5) building blocks
###############################################################################
def vdp_dynamics(x, mu_vdp=1.0):
    """Drift f(x) of the Van der Pol oscillator: y' = z, z' = mu (1 - y^2) z - y."""
    y, z = x[0], x[1]
    dy_dt = z
    dz_dt = mu_vdp * (1 - y**2) * z - y
    return np.array([dy_dt, dz_dt])


def vdp_L0_f(x, mu_vdp, Q_c):
    """L_0 f = df/dt + sum_i f_i df/dx_i + 0.5 sum_pq Qc_pq d^2 f / (dx_p dx_q).

    x is the state (y, z), Q_c the continuous-time process-noise covariance.
    """
    y, z = x[0], x[1]

    # term 1: df/dt (autonomous system)
    term1 = np.array([0.0, 0.0])

    # term 2: J_f f(x)
    comp1_term2 = mu_vdp * (1 - y**2) * z - y
    comp2_term2 = (z * (-2 * mu_vdp * y * z - 1)) + ((mu_vdp * (1 - y**2) * z - y) * (mu_vdp * (1 - y**2)))
    term2 = np.array([comp1_term2, comp2_term2])

    # term 3: 0.5 sum_pq Q_pq d^2 f / (dx_p dx_q)
    # f1 = z has no second derivatives.  For f2 = mu (1 - y^2) z - y:
    #   d^2 f2 / dy^2 = -2 mu z,   d^2 f2 / dy dz = -2 mu y,   d^2 f2 / dz^2 = 0
    comp1_term3 = 0.0
    comp2_term3_full = (Q_c[0, 0] * (-2 * mu_vdp * z)) + ((Q_c[0, 1] + Q_c[1, 0]) * (-2 * mu_vdp * y))
    term3 = np.array([comp1_term3, 0.5 * comp2_term3_full])

    L0_f_x = term1 + term2 + term3
    return L0_f_x


def fd_deterministic_propagate(x, dt, mu_vdp, Q_c):
    """Propagate one state with the deterministic part f_d of the order-1.5 Ito-Taylor expansion."""
    f_x = vdp_dynamics(x, mu_vdp)
    L0_f_x = vdp_L0_f(x, mu_vdp, Q_c)
    x_propagated = x + f_x * dt + 0.5 * L0_f_x * dt**2
    return x_propagated


def compute_Lf_matrix(x, mu_vdp, sqrt_Q_c):
    """Lf = J_f(x) sqrt(Q_c), with J_f the Jacobian of the drift."""
    y, z = x[0], x[1]
    n = len(x)

    J_f = np.zeros((n, n))
    J_f[0, 0] = 0.0
    J_f[0, 1] = 1.0
    J_f[1, 0] = -2 * mu_vdp * y * z - 1.0
    J_f[1, 1] = mu_vdp * (1 - y**2)

    Lf = J_f @ sqrt_Q_c
    return Lf


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
def run_cdckf(t_end=30.0, dt=1 / 20.0, m_sub=5, mu_vdp=1.5, measurements=None, h_fun=h_fun_polar,
              R=None, x0=None, P0=None, Q=None, verbose=True):
    """Run the SR-CDCKF.

    Parameters
    ----------
    t_end, dt : final time and filter step.
    m_sub : number of Ito-Taylor micro-steps per filter step.
    mu_vdp : Van der Pol parameter.
    measurements : dict {time (rounded to 6 decimals): z = (r, theta)} or None.
    h_fun, R : measurement function and measurement-noise covariance.
    x0, P0 : initial mean and covariance (default (2, 0) and 0.01 I).
    Q : continuous-time process-noise covariance (default 6e-3 I).

    Returns
    -------
    dict with time_hist (T,), mu_hist (T, 2), P_hist (T, 2, 2); the t = 0 state is included.
    """
    x0 = np.array([2.0, 0.0]) if x0 is None else np.asarray(x0, dtype=float)
    P0 = np.diag([0.01] * 2) if P0 is None else np.asarray(P0, dtype=float)
    Q_c = 6e-3 * np.eye(2) if Q is None else np.asarray(Q, dtype=float)
    R = np.diag([0.5**2, 0.8**2]) if R is None else np.asarray(R, dtype=float)
    measurements = {} if measurements is None else measurements

    n = len(x0)
    delta_t_micro = dt / m_sub  # micro-step length
    sqrt_Q_c = cholesky(Q_c, lower=True)

    time_hist, mu_hist, P_hist = [], [], []
    mu_k = x0
    S_k = cholesky(P0, lower=True)

    num_major_steps = int(round(t_end / dt))
    current_time = 0.0

    for k_major in range(num_major_steps):
        if verbose and k_major % 100 == 0:
            print(f"Step {k_major}/{num_major_steps}, time = {current_time:.2f}")

        time_hist.append(current_time)
        mu_hist.append(mu_k)
        P_hist.append(S_k @ S_k.T)

        # --- time update over m_sub micro-steps ---
        mu_pred = mu_k
        S_pred = S_k

        for j_micro in range(m_sub):
            # 1. cubature points
            xi = np.sqrt(n) * np.hstack((np.eye(n), -np.eye(n)))
            X_cubature = S_pred @ xi + mu_pred[:, np.newaxis]

            # 2. propagate every cubature point with f_d
            X_propagated = np.zeros_like(X_cubature)
            for i in range(2 * n):
                X_propagated[:, i] = fd_deterministic_propagate(
                    x=X_cubature[:, i],
                    dt=delta_t_micro,
                    mu_vdp=mu_vdp,
                    Q_c=Q_c
                )

            # 3. predicted mean of the micro-step
            mu_pred_micro = np.mean(X_propagated, axis=1)

            # 4. weighted, centred matrix X* = (X_propagated - mean) / sqrt(2n)
            X_centered = (X_propagated - mu_pred_micro[:, np.newaxis]) / np.sqrt(2 * n)

            # 5. noise blocks; Lf is evaluated at the mean at the start of the micro-step
            Lf_at_pred_mean = compute_Lf_matrix(mu_pred, mu_vdp, sqrt_Q_c)

            M1 = X_centered
            M2 = np.sqrt(delta_t_micro) * (sqrt_Q_c + 0.5 * delta_t_micro * Lf_at_pred_mean)
            M3 = np.sqrt(delta_t_micro**3 / 12) * Lf_at_pred_mean

            # 6. S = Tria([M1, M2, M3]) through a QR factorisation of the transpose
            A = np.hstack([M1, M2, M3])
            _, R_qr = qr(A.T, mode='economic')
            S_pred_micro = R_qr.T

            # 7. next micro-step
            mu_pred = mu_pred_micro
            S_pred = S_pred_micro

        current_time += dt

        # --- measurement update ---
        mu_posterior = mu_pred
        S_posterior = S_pred

        current_time_rounded = round(current_time, 6)
        if current_time_rounded in measurements:
            if verbose:
                print(f"  -> measurement update at t = {current_time:.2f}")
            z_obs = measurements[current_time_rounded]
            mu_posterior, S_posterior = srckf_measurement_update(mu_pred, S_pred, z_obs, h_fun, R)

        mu_k = mu_posterior
        S_k = S_posterior

    # final state
    time_hist.append(current_time)
    mu_hist.append(mu_k)
    P_hist.append(S_k @ S_k.T)

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
        description="SR-CDCKF for the 2-D stochastic Van der Pol oscillator (polar measurements).")
    parser.add_argument("--mu", type=float, default=1.5, help="Van der Pol parameter (default 1.5)")
    parser.add_argument("--t-end", type=float, default=30.0, help="final time (default 30)")
    parser.add_argument("--dt", type=float, default=1 / 20.0, help="filter step (default 1/20)")
    parser.add_argument("--m-sub", type=int, default=5,
                        help="Ito-Taylor micro-steps per filter step (default 5)")
    parser.add_argument("--q", type=float, default=6e-3, help="process-noise intensity, Q = q I")
    parser.add_argument("--x0", type=float, nargs=2, default=[2.0, 0.0], metavar=("Y", "Z"),
                        help="initial mean (default 2 0)")
    parser.add_argument("--p0", type=float, default=0.01, help="initial variance, P0 = p0 I")
    parser.add_argument("--sigma-r", type=float, default=0.5, help="std of the range measurement")
    parser.add_argument("--sigma-theta", type=float, default=0.8, help="std of the angle measurement")
    parser.add_argument("--measurements", default=None, metavar='"t,r,theta;..."',
                        help="explicit measurements (default: paper values for --mu at t = 10, 20)")
    parser.add_argument("--no-measurements", action="store_true", help="prediction only")
    parser.add_argument("--out", default=os.path.join("results", "vdp2d_cdckf"), help="output directory")
    parser.add_argument("--quiet", action="store_true", help="no progress output")
    args = parser.parse_args()

    measurements = select_measurements(parser, args)

    start = time.time()
    results = run_cdckf(t_end=args.t_end, dt=args.dt, m_sub=args.m_sub, mu_vdp=args.mu,
                        measurements=measurements, h_fun=h_fun_polar,
                        R=np.diag([args.sigma_r**2, args.sigma_theta**2]),
                        x0=np.array(args.x0), P0=args.p0 * np.eye(2), Q=args.q * np.eye(2),
                        verbose=not args.quiet)
    wall = time.time() - start

    save_outputs(args.out, results)
    print(f"SR-CDCKF (mu = {args.mu}) finished in {wall:.2f} s wall time; results in '{args.out}'")


if __name__ == "__main__":
    main()
