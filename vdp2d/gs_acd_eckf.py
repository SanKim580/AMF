"""Gaussian-sum accurate continuous-discrete extended-cubature Kalman filter (GS-ACD-ECKF) for
the 2-D stochastic Van der Pol oscillator.

Model
    dy = z dt + sqrt(q) dW_1
    dz = (mu (1 - y^2) z - y) dt + sqrt(q) dW_2          (Q = q I, paper: q = 6e-3)
    measurement  h(y, z) = (r, theta) = (sqrt(y^2 + z^2), atan2(z, y)),  R = diag(0.5^2, 0.8^2)

The filter carries a Gaussian mixture with a fixed number of components (4 in the paper).  Every
component is propagated with the ACD-EKF time update (nested implicit Runge-Kutta scheme of
order 4 for the mean with automatic local/global error control, Mazzoni's square-root scheme for
the covariance) and updated with a square-root cubature measurement update with a wrapped
angle; the component weights are re-weighted with the measurement likelihood.

The script runs the paper case (initial density N((2, 0), 0.01 I) represented by 4 components,
dt = 1/20, t in [0, 30], measurements at t = 10 and t = 20) and writes the mean trajectory and
the covariance history of the mixture.

Usage
    python vdp2d/gs_acd_eckf.py                # paper case, mu = 1.5
    python vdp2d/gs_acd_eckf.py --mu 2.0 --out results/gs_mu2 --save-components
    python vdp2d/gs_acd_eckf.py --no-measurements --t-end 10
"""
import argparse
import os
import time

import numpy as np
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

# 4-component representation of the paper's initial density N((2, 0), 0.01 I):
# equal weights, the means below and the common covariance INIT_COV for every component.
INIT_MEANS = np.array([
    [2.03017739, 0.],
    [1.96982261, 0.],
    [2., 0.03017739],
    [2., -0.03017739]
], dtype=float)
INIT_COV = np.array([
    [0.00953656, 0.],
    [0., 0.00953656]
], dtype=float)


# ==============================================================================
# 1. System model (Van der Pol oscillator)
# ==============================================================================
def vdp_dynamics(x, mu_vdp=1.0):
    """Drift f(x) of the Van der Pol oscillator."""
    y, z = x[0], x[1]
    dy_dt = z
    dz_dt = mu_vdp * (1 - y**2) * z - y
    return np.array([dy_dt, dz_dt])


def vdp_jacobian(x, mu_vdp=1.0):
    """Jacobian J(x) of vdp_dynamics."""
    y, z = x[0], x[1]
    J = np.zeros((2, 2))
    J[0, 0] = 0.0
    J[0, 1] = 1.0
    J[1, 0] = -2 * mu_vdp * y * z - 1.0
    J[1, 1] = mu_vdp * (1 - y**2)
    return J


# ==============================================================================
# 2. ACD-ECKF time update
# ==============================================================================
def predict_mean_irk4(x_l4, f_fun, j_fun, mu_vdp, tau_l, max_iter=5, tol=1e-6):
    """Mean prediction with the nested implicit Runge-Kutta scheme of order 4.

    Returns the predicted mean and the local error estimate of the step.
    """
    n = len(x_l4)
    # NIRK4 coefficients
    a11_2 = 0.5 + 2 * np.sqrt(3) / 9
    a12_2 = 0.5 - 2 * np.sqrt(3) / 9
    d11_2 = (3 + np.sqrt(3)) / 36
    d12_2 = (-3 + np.sqrt(3)) / 36
    d22_2 = (-1) * d11_2
    d21_2 = (-1) * d12_2
    x_l_plus_1 = x_l4.copy()

    # simplified Newton iteration
    for _ in range(max_iter):
        f_x_l4 = f_fun(x_l4, mu_vdp)
        f_x_l_plus_1 = f_fun(x_l_plus_1, mu_vdp)

        # stage values
        x_l1_2l = a11_2 * x_l4 + a12_2 * x_l_plus_1 + tau_l * (d11_2 * f_x_l4 + d12_2 * f_x_l_plus_1)
        x_l2_2l = a12_2 * x_l4 + a11_2 * x_l_plus_1 + tau_l * (d21_2 * f_x_l4 + d22_2 * f_x_l_plus_1)

        # right-hand side of the Newton equation
        rhs = -x_l_plus_1 + x_l4 + (tau_l / 2) * (f_fun(x_l1_2l, mu_vdp) + f_fun(x_l2_2l, mu_vdp))

        # linear system (I - tau/4 J)^2 delta = rhs
        J_l4 = j_fun(x_l4, mu_vdp)
        A = np.eye(n) - (tau_l / 4) * J_l4
        delta_x = np.linalg.solve(A @ A, rhs)
        x_l_plus_1 += delta_x

        if np.linalg.norm(delta_x) < tol:
            break

    # local error estimate
    f_x_l4_final = f_fun(x_l4, mu_vdp)
    f_x_l1_2l_final = f_fun(x_l1_2l, mu_vdp)
    f_x_l2_2l_final = f_fun(x_l2_2l, mu_vdp)
    f_x_l_plus_1_final = f_fun(x_l_plus_1, mu_vdp)
    le_l_plus_1 = -(tau_l / 2) * (f_x_l4_final - f_x_l1_2l_final - f_x_l2_2l_final + f_x_l_plus_1_final)

    return x_l_plus_1, le_l_plus_1


def update_covariance_sqrt(S_l, x_l, x_l_plus_1, f_fun, j_fun, mu_vdp, sqrt_Qc, tau_l):
    """Square-root covariance update (Mazzoni's scheme)."""
    n = S_l.shape[0]

    # mid-point state
    J_l = j_fun(x_l, mu_vdp)
    f_l = f_fun(x_l, mu_vdp)
    x_mid = 0.5 * (x_l + x_l_plus_1 - (tau_l**2 / 4) * J_l @ f_l)

    # mid-point matrices
    J_mid = j_fun(x_mid, mu_vdp)
    inv_term = np.linalg.inv(np.eye(n) - (tau_l / 2) * J_mid)
    K_mid = inv_term
    M_mid = K_mid @ (np.eye(n) + (tau_l / 2) * J_mid)

    # triangularisation.  The SDE is dx = f(x) dt + G dW with a standard Brownian motion,
    # so G is the square root of Q_c.
    matrix_A = np.hstack([M_mid @ S_l, np.sqrt(tau_l) * K_mid @ sqrt_Qc])

    _, R_qr = qr(matrix_A.T, mode='economic')
    S_l_plus_1 = R_qr.T
    return S_l_plus_1


def single_component_time_update_acd_eckf(comp, t_start, t_end, mu_vdp, Q_c):
    """ACD-ECKF time update of one mixture component over [t_start, t_end]."""
    sqrt_Qc = cholesky(Q_c, lower=True)

    epsilon_g = 1e-5
    epsilon_loc = epsilon_g**(3 / 2)
    tau_max = 0.1

    M = 1
    while M == 1:
        # initialisation
        M = 0
        t_l = t_start
        x_l = comp['mean']
        S_l = comp['cov_sqrt']
        delta_x_l = np.zeros_like(comp['mean'])  # accumulated global error
        delta_x_max_scaled = 0.0  # largest scaled global error
        tau_l = min(1e-3, t_end - t_l)

        # main integration loop
        while t_l < t_end:
            # mean prediction and local error
            x_l_plus_1_cand, le = predict_mean_irk4(x_l, vdp_dynamics, vdp_jacobian, mu_vdp, tau_l)
            scaled_le = np.max(np.abs(le) / (np.abs(x_l_plus_1_cand) + 1.0))

            # step-size control
            tau_star = min(1.5, 0.8 * (epsilon_loc / (scaled_le + 1e-9))**(1 / 3)) * tau_l

            if scaled_le > epsilon_loc and tau_l > 1e-6:
                # reject the step
                tau_l = tau_star
                continue
            else:
                # accept the step
                x_l_plus_1 = x_l_plus_1_cand

                # global error: accumulate the local errors and track the scaled maximum
                delta_x_l += le
                scaled_delta_x = np.max(np.abs(delta_x_l) / (np.abs(x_l_plus_1) + 1.0))
                delta_x_max_scaled = max(delta_x_max_scaled, scaled_delta_x)

                # covariance update
                S_l_plus_1 = update_covariance_sqrt(S_l, x_l, x_l_plus_1, vdp_dynamics, vdp_jacobian,
                                                    mu_vdp, sqrt_Qc, tau_l)

                # next step
                t_l += tau_l
                x_l, S_l = x_l_plus_1, S_l_plus_1
                tau_l = min(tau_star, t_end - t_l, tau_max)
                if tau_l < 1e-7:
                    break

        # global error control: repeat the whole interval with a tighter local tolerance
        if delta_x_max_scaled > epsilon_g:
            M = 1
            epsilon_loc = (0.8 * epsilon_g / delta_x_max_scaled)**(1.5) * epsilon_loc
        else:
            M = 0

    return x_l, S_l


# ==============================================================================
# 3. Square-root cubature measurement update (polar measurement, wrapped angle)
# ==============================================================================
def normalize_angle(angle):
    """Wrap an angle to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def srckf_measurement_update(mu_pred, S_pred, z_obs, h_fun, R, v_bar):
    """Square-root cubature update of one component; the second measurement component is an angle.

    v_bar is the mean of the measurement noise.  Returns the updated mean, the updated
    square-root covariance and the likelihood of the measurement for this component.
    """
    n = len(mu_pred)
    nz = z_obs.shape[0]

    # cubature points and their images in measurement space
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

    # covariance square root and Kalman gain through a QR triangularisation
    X_centered = (X_cubature - mu_pred[:, np.newaxis]) / np.sqrt(2 * n)
    S_R = cholesky(R, lower=True)

    A_top = np.hstack([Z_centered, S_R])
    A_bottom = np.hstack([X_centered, np.zeros((n, nz))])
    A = np.vstack([A_top, A_bottom])

    _, R_qr = qr(A.T, mode='economic')
    T = R_qr.T

    T11, T21, T22 = T[:nz, :nz], T[nz:, :nz], T[nz:, nz:]

    Wk_T = solve_triangular(T11, T21.T, lower=True, trans='T')
    W_k = Wk_T.T

    # update with the wrapped innovation
    residual = z_obs - (z_pred + v_bar)
    residual[1] = normalize_angle(residual[1])

    mu_updated = mu_pred + W_k @ residual
    S_updated = T22

    # likelihood N(residual; 0, P_zz)
    P_zz = T11 @ T11.T
    P_zz += np.eye(nz) * 1e-9  # numerical safeguard
    likelihood = multivariate_normal.pdf(residual, mean=np.zeros_like(residual), cov=P_zz)

    return mu_updated, S_updated, likelihood


# ==============================================================================
# 4. Gaussian-sum helpers
# ==============================================================================
def prune_and_normalize(components, m_target):
    """Normalise the weights and keep the m_target components with the largest weights."""
    if not components:
        return []

    # 1. normalise
    total_weight = sum(c['weight'] for c in components)
    if total_weight < 1e-9:
        return []
    for c in components:
        c['weight'] /= total_weight

    # 2. prune
    sorted_components = sorted(components, key=lambda x: x['weight'], reverse=True)
    pruned = sorted_components[:m_target]

    # 3. re-normalise
    final_total_weight = sum(c['weight'] for c in pruned)
    if final_total_weight < 1e-9:
        return pruned
    for c in pruned:
        c['weight'] /= final_total_weight

    return pruned


def collapse_mixture(components):
    """Mean and covariance of the Gaussian mixture (moment matching)."""
    if not components:
        return np.array([0, 0]), np.eye(2)

    final_mean = sum(c['weight'] * c['mean'] for c in components)

    final_cov = np.zeros((2, 2))
    for c in components:
        P_i = c['cov_sqrt'] @ c['cov_sqrt'].T
        mean_diff = c['mean'] - final_mean
        final_cov += c['weight'] * (P_i + np.outer(mean_diff, mean_diff))

    return final_mean, final_cov


# ==============================================================================
# 5. Measurement model helpers
# ==============================================================================
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


# ==============================================================================
# 6. Filter
# ==============================================================================
def run_gs_acd_eckf(t_end=30.0, dt=1 / 20.0, n_components=4, mu_vdp=1.5, measurements=None,
                    h_fun=h_fun_polar, R=None, Q=None, init_means=None, init_cov=None, verbose=True):
    """Run the GS-ACD-ECKF.

    Parameters
    ----------
    t_end, dt : final time and filter step.
    n_components : number of mixture components kept after every step.
    mu_vdp : Van der Pol parameter.
    measurements : dict {time (rounded to 6 decimals): z = (r, theta)} or None.
    h_fun, R : measurement function and measurement-noise covariance.
    Q : continuous-time process-noise covariance (default 6e-3 I).
    init_means, init_cov : means (n_components, 2) and common covariance (2, 2) of the equally
        weighted initial components.  Default: the 4-component representation of
        N((2, 0), 0.01 I) used in the paper (INIT_MEANS, INIT_COV).

    Returns
    -------
    dict with time_hist (T,), mu_hist (T, 2), P_hist (T, 2, 2) (moment-matched mixture mean and
    covariance), count_hist (T,) and components_hist (list of the component lists; every
    component is a dict with 'weight', 'mean', 'cov_sqrt').  The t = 0 state is included.
    """
    Q_c = 6e-3 * np.eye(2) if Q is None else np.asarray(Q, dtype=float)
    R = np.diag([0.5**2, 0.8**2]) if R is None else np.asarray(R, dtype=float)
    measurements = {} if measurements is None else measurements

    if init_means is None and init_cov is None:
        assert n_components == 4, "The default initialisation assumes n_components = 4."
        means0, P_comp = INIT_MEANS, INIT_COV
    else:
        if init_means is None or init_cov is None:
            raise ValueError("init_means and init_cov must be given together")
        means0 = np.asarray(init_means, dtype=float)
        P_comp = np.asarray(init_cov, dtype=float)
        if means0.shape[0] != n_components:
            raise ValueError("init_means must contain n_components rows")

    S_comp = cholesky(P_comp, lower=True)

    components = [
        {"weight": 1.0 / n_components, "mean": means0[i].copy(), "cov_sqrt": S_comp.copy()}
        for i in range(n_components)
    ]

    # noise models of the Gaussian-sum filter (one process-noise and one measurement-noise term)
    process_noise_models = [{'weight': 1.0, 'Q_c': Q_c}]
    measurement_noise_models = [{'weight': 1.0, 'v_bar': np.zeros(R.shape[0]), 'R': R}]

    time_hist, mu_hist, P_hist, count_hist = [], [], [], []
    components_hist = []

    def store(t):
        mean, cov = collapse_mixture(components)
        time_hist.append(t)
        mu_hist.append(mean)
        P_hist.append(cov)
        count_hist.append(len(components))
        components_hist.append(components)

    current_time = 0.0
    num_major_steps = int(round(t_end / dt))

    for k_major in range(num_major_steps):
        if verbose and k_major % 100 == 0:
            print(f"Step {k_major}/{num_major_steps}, time = {current_time:.2f}")

        store(current_time)

        # --- time update ---
        predicted_components = []
        for prev_comp in components:
            for pn_model in process_noise_models:
                pred_mean, pred_cov_sqrt = single_component_time_update_acd_eckf(
                    prev_comp, current_time, current_time + dt, mu_vdp, pn_model['Q_c']
                )
                new_weight = prev_comp['weight'] * pn_model['weight']
                predicted_components.append({
                    'weight': new_weight, 'mean': pred_mean, 'cov_sqrt': pred_cov_sqrt
                })

        current_time += dt

        # --- measurement update ---
        current_time_rounded = round(current_time, 6)
        if current_time_rounded in measurements:
            if verbose:
                print(f"  -> measurement update at t = {current_time:.2f}")
            z_obs = measurements[current_time_rounded]
            posterior_components_temp = []
            for pred_comp in predicted_components:
                for mn_model in measurement_noise_models:
                    post_mean, post_cov_sqrt, likelihood = srckf_measurement_update(
                        pred_comp['mean'], pred_comp['cov_sqrt'], z_obs, h_fun, mn_model['R'],
                        mn_model['v_bar']
                    )
                    updated_weight = pred_comp['weight'] * mn_model['weight'] * likelihood
                    posterior_components_temp.append({
                        'weight': updated_weight, 'mean': post_mean, 'cov_sqrt': post_cov_sqrt
                    })
            components = prune_and_normalize(posterior_components_temp, n_components)
        else:
            components = prune_and_normalize(predicted_components, n_components)

    # final state
    store(current_time)

    return {"time_hist": np.array(time_hist),
            "mu_hist": np.array(mu_hist),
            "P_hist": np.array(P_hist),
            "count_hist": np.array(count_hist),
            "components_hist": components_hist}


# ==============================================================================
# 7. Command line
# ==============================================================================
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
            for j, comp in enumerate(comps):
                M = comp['cov_sqrt']
                rows.append([t, j, comp['weight'], comp['mean'][0], comp['mean'][1],
                             M[0, 0], M[0, 1], M[1, 0], M[1, 1]])
        np.savetxt(os.path.join(out_dir, "components.csv"), np.array(rows).reshape(-1, 9),
                   delimiter=",", fmt=["%.18e", "%d"] + ["%.18e"] * 7,
                   header="time,component_index,weight,mu_y,mu_z,M_11,M_12,M_21,M_22", comments="")


def main():
    parser = argparse.ArgumentParser(
        description="GS-ACD-ECKF for the 2-D stochastic Van der Pol oscillator (polar measurements).")
    parser.add_argument("--mu", type=float, default=1.5, help="Van der Pol parameter (default 1.5)")
    parser.add_argument("--t-end", type=float, default=30.0, help="final time (default 30)")
    parser.add_argument("--dt", type=float, default=1 / 20.0, help="filter step (default 1/20)")
    parser.add_argument("--n-components", type=int, default=4,
                        help="number of mixture components (the built-in initialisation needs 4)")
    parser.add_argument("--q", type=float, default=6e-3, help="process-noise intensity, Q = q I")
    parser.add_argument("--sigma-r", type=float, default=0.5, help="std of the range measurement")
    parser.add_argument("--sigma-theta", type=float, default=0.8, help="std of the angle measurement")
    parser.add_argument("--measurements", default=None, metavar='"t,r,theta;..."',
                        help="explicit measurements (default: paper values for --mu at t = 10, 20)")
    parser.add_argument("--no-measurements", action="store_true", help="prediction only")
    parser.add_argument("--out", default=os.path.join("results", "vdp2d_gs_acd_eckf"),
                        help="output directory")
    parser.add_argument("--save-components", action="store_true",
                        help="also write components.csv (all components at every step)")
    parser.add_argument("--quiet", action="store_true", help="no progress output")
    args = parser.parse_args()

    if args.n_components != 4:
        parser.error("the command-line interface only provides the paper's 4-component initial "
                     "mixture; call run_gs_acd_eckf(init_means=..., init_cov=...) for other sizes")
    measurements = select_measurements(parser, args)

    start = time.time()
    results = run_gs_acd_eckf(t_end=args.t_end, dt=args.dt, n_components=args.n_components,
                              mu_vdp=args.mu, measurements=measurements, h_fun=h_fun_polar,
                              R=np.diag([args.sigma_r**2, args.sigma_theta**2]),
                              Q=args.q * np.eye(2), verbose=not args.quiet)
    wall = time.time() - start

    save_outputs(args.out, results, save_components=args.save_components)
    print(f"GS-ACD-ECKF (mu = {args.mu}) finished in {wall:.2f} s wall time; results in '{args.out}'")


if __name__ == "__main__":
    main()
