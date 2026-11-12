"""Bootstrap particle filter (PF) for the 2-D stochastic Van der Pol oscillator.

Model
    dy = z dt + sqrt(q) dW_1
    dz = (mu (1 - y^2) z - y) dt + sqrt(q) dW_2          (Q = q I, paper: q = 6e-3)
    measurement  h(y, z) = (r, theta) = (sqrt(y^2 + z^2), atan2(z, y)),  R = diag(0.5^2, 0.8^2)

Every particle is propagated over a step dt with the drift and then perturbed with additive
N(0, Q dt) noise.  At measurement times the particles are re-weighted with the likelihood
(wrapped angle) and always resampled (systematic resampling).  The PF with N = 50000 particles
is the reference solution of the paper.

Two backends implement the same algorithm:
    cpu    the drift is integrated per particle with an implicit Radau solver; the particles are
           distributed over a multiprocessing pool.
    torch  vectorised over the particles (CUDA if available, otherwise CPU); the drift is
           integrated with n_sub classical RK4 sub-steps per dt.  Requires PyTorch.  When it runs
           on the CPU, set OMP_NUM_THREADS to use more than one thread.

The script runs the paper case (x0 = (2, 0), P0 = 0.01 I, dt = 1/20, t in [0, 30], measurements
at t = 10 and t = 20) and writes the mean trajectory and the sample-covariance history.

Usage
    python vdp2d/pf.py                         # paper case, mu = 1.5, cpu backend
    python vdp2d/pf.py --backend torch --n-particles 50000 --seed 1
    python vdp2d/pf.py --no-measurements --t-end 10 --n-particles 2000 --workers 8
"""
import os

# one BLAS/OpenMP thread per process: the parallelism comes from the multiprocessing pool
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import multiprocessing as mp
import time
import warnings

import numpy as np
from scipy.integrate import solve_ivp

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
# Model
###############################################################################
def van_der_pol(t, x, mu_vdp=1.0):
    """Drift of the Van der Pol oscillator: y' = z, z' = mu (1 - y^2) z - y."""
    y, z = x[0], x[1]
    dy_dt = z
    dz_dt = mu_vdp * (1 - y**2) * z - y
    return np.stack((dy_dt, dz_dt), axis=-1)


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


def normalize_angle(angle):
    """Wrap an angle to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


###############################################################################
# CPU backend
###############################################################################
class Particle:
    """State and weight of one particle."""

    def __init__(self, state, w=1.0):
        self.state = state.copy()
        self.w = float(w)


def pf_systematic_resample(weights):
    """Systematic resampling; returns the indices of the selected particles."""
    N = len(weights)
    positions = (np.random.random() + np.arange(N)) / N
    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def pf_measurement_update_polar(particles, z, h_fun, R):
    """Set the particle weights to the normalised likelihood of z (second component: angle)."""
    inv_R = np.linalg.inv(R)

    log_likelihoods = np.zeros(len(particles))

    for i, p in enumerate(particles):
        predicted_z = h_fun(p.state)
        innovation = z - predicted_z
        innovation[1] = normalize_angle(innovation[1])
        # Gaussian log-likelihood up to a constant, which cancels in the normalisation
        log_likelihoods[i] = -0.5 * (innovation @ inv_R @ innovation)

    log_weights = log_likelihoods
    max_log_w = np.max(log_weights)
    if np.isinf(max_log_w):
        max_log_w = -700

    log_weights_norm = log_weights - max_log_w
    weights_linear = np.exp(log_weights_norm)

    total_weight = np.sum(weights_linear)
    if total_weight > 1e-12:
        normalized_weights = weights_linear / total_weight
    else:
        normalized_weights = np.ones(len(particles)) / len(particles)

    for i, p in enumerate(particles):
        p.w = normalized_weights[i]


def propagate_worker(state, t0, dt, mu_vdp_val, process_noise_cov_ref):
    """Propagate one particle over dt: Radau integration of the drift + N(0, Q dt) noise."""
    try:
        sol = solve_ivp(
            lambda t, x: van_der_pol(t, x, mu_vdp_val),
            (t0, t0 + dt), state, t_eval=[t0 + dt],
            method='Radau', rtol=1e-7, atol=1e-9
        )
        if not sol.success:
            warnings.warn("particle propagation: the ODE solver did not succeed; state kept")
            return state
        propagated_state = sol.y[:, 0]
        discretized_step_cov = process_noise_cov_ref * dt
        noise = np.random.multivariate_normal(np.zeros(2), discretized_step_cov)
        return propagated_state + noise

    except Exception as e:
        warnings.warn(f"particle propagation failed ({e}); state kept")
        return state


def _seed_worker(base_seed):
    # Forked pool workers inherit the parent's numpy random state.  Without re-seeding, every
    # worker would draw the same process-noise sequence, i.e. the noise would be replicated
    # across the particles handled by different workers.  Each worker therefore gets its own
    # stream derived from (base seed, worker id).
    ident = mp.current_process()._identity
    worker_id = int(ident[0]) if ident else 0
    np.random.seed(np.random.SeedSequence([int(base_seed), worker_id]).generate_state(1)[0])


def _weighted_moments(states, weights):
    """Weighted mean and weighted sample covariance of the particle cloud."""
    total_weight = np.sum(weights)
    if total_weight > 1e-12:
        mean_state = np.sum(states * weights[:, np.newaxis], axis=0) / total_weight
        w = weights / total_weight
    else:
        # degenerate weights: plain average
        mean_state = np.mean(states, axis=0)
        w = np.ones(len(weights)) / len(weights)
    diff = states - mean_state
    cov = (diff * w[:, np.newaxis]).T @ diff
    return mean_state, cov


def _run_pf_cpu(t_end, dt, mu_vdp, measurements, h_fun, R, Q, x0, P0, n_particles, save_dt, seed,
                n_workers, store_particles, verbose):
    N = n_particles
    # seed the parent: initial particles and resampling
    np.random.seed(seed)
    particles = [Particle(np.random.multivariate_normal(x0, P0), 1.0 / N) for _ in range(N)]

    t = 0.0
    num_steps = int(round(t_end / dt))
    time_hist, mu_hist, P_hist = [], [], []
    particles_hist, weights_hist = [], []

    # results are stored every save_every_n_steps steps
    save_every_n_steps = 1
    if save_dt is not None and dt > 0:
        save_every_n_steps = int(round(save_dt / dt))
        if save_every_n_steps < 1:
            save_every_n_steps = 1

    num_cores = mp.cpu_count() if n_workers is None else int(n_workers)
    if verbose:
        print(f"PF (cpu backend): {N} particles, {num_cores} worker processes")

    with mp.Pool(processes=num_cores, initializer=_seed_worker, initargs=(seed,)) as pool:
        for s in range(num_steps):
            if verbose and s % 100 == 0:
                print(f"Step {s}/{num_steps}, time = {t:.2f}")

            # 1. propagate
            task_args = [(p.state, t, dt, mu_vdp, Q) for p in particles]
            new_states = pool.starmap(propagate_worker, task_args)

            for i, p in enumerate(particles):
                p.state = new_states[i]

            t += dt

            current_time_rounded = round(t, 6)
            if current_time_rounded in measurements:
                z_obs = measurements[current_time_rounded]
                # 2. re-weight
                pf_measurement_update_polar(particles, z_obs, h_fun, R)
                # 3. resample (always)
                weights = np.array([p.w for p in particles])
                indexes = pf_systematic_resample(weights)
                particles = [Particle(particles[idx].state, 1.0 / N) for idx in indexes]

            # 4. store
            if (s + 1) % save_every_n_steps == 0:
                current_states = np.array([p.state for p in particles])
                current_weights = np.array([p.w for p in particles])
                mean_state, cov = _weighted_moments(current_states, current_weights)

                time_hist.append(t)
                mu_hist.append(mean_state)
                P_hist.append(cov)
                if store_particles:
                    particles_hist.append(current_states)
                    weights_hist.append(current_weights)

    results = {"time_hist": np.array(time_hist), "mu_hist": np.array(mu_hist), "P_hist": np.array(P_hist)}
    if store_particles:
        results["particles_hist"] = np.array(particles_hist)
        results["weights_hist"] = np.array(weights_hist)
    return results


###############################################################################
# torch backend
###############################################################################
def _run_pf_torch(t_end, dt, mu_vdp, measurements, R, Q, x0, P0, n_particles, save_dt, seed, n_sub,
                  store_particles, verbose):
    import math
    import torch

    N = n_particles
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    d = len(x0)
    g = torch.Generator(device=dev)
    g.manual_seed(int(seed))

    def tt(a):
        return torch.tensor(np.asarray(a, dtype=float), dtype=dtype, device=dev)

    def f(s):
        y, z = s[:, 0], s[:, 1]
        return torch.stack((z, mu_vdp * (1 - y ** 2) * z - y), dim=1)

    def hfun(s):
        return torch.stack((torch.sqrt(s[:, 0] ** 2 + s[:, 1] ** 2), torch.atan2(s[:, 1], s[:, 0])), dim=1)

    angle_idx = [1]

    x = tt(x0) + torch.randn(N, d, generator=g, dtype=dtype, device=dev) @ torch.linalg.cholesky(tt(P0)).T
    Lq = torch.linalg.cholesky(tt(Q) * dt)
    Rinv = torch.linalg.inv(tt(R))
    h = dt / n_sub
    time_hist, mu_hist, P_hist = [], [], []
    particles_hist, weights_hist = [], []
    t = 0.0
    ar = torch.arange(N, dtype=dtype, device=dev)

    save_every_n_steps = 1
    if save_dt is not None and dt > 0:
        save_every_n_steps = max(1, int(round(save_dt / dt)))

    num_steps = int(round(t_end / dt))
    if verbose:
        print(f"PF (torch backend on {dev}): {N} particles, {n_sub} RK4 sub-steps per step")

    for s in range(num_steps):
        if verbose and s % 100 == 0:
            print(f"Step {s}/{num_steps}, time = {t:.2f}")

        # 1. propagate: RK4 for the drift, then additive N(0, Q dt) noise
        for _ in range(n_sub):
            k1 = f(x)
            k2 = f(x + 0.5 * h * k1)
            k3 = f(x + 0.5 * h * k2)
            k4 = f(x + h * k3)
            x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        x = x + torch.randn(N, d, generator=g, dtype=dtype, device=dev) @ Lq.T

        t += dt
        tr = float(round(t, 6))
        if tr in measurements:
            # 2. re-weight with the likelihood (wrapped angle)
            inn = tt(measurements[tr]) - hfun(x)
            for j in angle_idx:
                inn[:, j] = (inn[:, j] + math.pi) % (2 * math.pi) - math.pi
            logw = -0.5 * torch.einsum("ni,ij,nj->n", inn, Rinv, inn)
            w = torch.exp(logw - logw.max())
            w = w / w.sum()
            # 3. systematic resampling (always)
            pos = (torch.rand(1, generator=g, dtype=dtype, device=dev) + ar) / N
            idx = torch.searchsorted(torch.cumsum(w, 0), pos).clamp_(max=N - 1)
            x = x[idx]

        # 4. store (the weights are uniform after resampling)
        if (s + 1) % save_every_n_steps == 0:
            m = x.mean(0)
            dm = x - m
            time_hist.append(t)
            mu_hist.append(m.cpu().numpy())
            P_hist.append((dm.T @ dm / N).cpu().numpy())
            if store_particles:
                particles_hist.append(x.cpu().numpy())
                weights_hist.append(np.full(N, 1.0 / N))

    results = {"time_hist": np.array(time_hist), "mu_hist": np.array(mu_hist), "P_hist": np.array(P_hist)}
    if store_particles:
        results["particles_hist"] = np.array(particles_hist)
        results["weights_hist"] = np.array(weights_hist)
    return results


###############################################################################
# Filter
###############################################################################
def run_pf(t_end=30.0, dt=1 / 20.0, mu_vdp=1.5, measurements=None, h_fun=h_fun_polar, R=None, Q=None,
           x0=None, P0=None, n_particles=50000, save_dt=None, seed=0, backend="cpu", n_sub=20,
           n_workers=None, store_particles=False, verbose=True):
    """Run the bootstrap particle filter (resampling at every measurement).

    Parameters
    ----------
    t_end, dt : final time and filter step.
    mu_vdp : Van der Pol parameter.
    measurements : dict {time (rounded to 6 decimals): z = (r, theta)} or None.
    h_fun, R : measurement function and measurement-noise covariance.  The torch backend only
        implements the polar measurement h_fun_polar.
    Q : process-noise covariance (default 6e-3 I); the noise added per step is N(0, Q dt).
    x0, P0 : mean and covariance of the initial particle cloud (default (2, 0) and 0.01 I).
    n_particles : number of particles.
    save_dt : interval between stored results (None: every step).
    seed : seed of the random number generators.
    backend : "cpu" (Radau per particle, multiprocessing pool) or "torch" (vectorised RK4).
    n_sub : RK4 sub-steps per dt (torch backend only).
    n_workers : number of worker processes of the cpu backend (None: all CPUs).
    store_particles : also return the particle clouds.

    Returns
    -------
    dict with time_hist (T,), mu_hist (T, 2) (weighted mean), P_hist (T, 2, 2) (weighted sample
    covariance) and, with store_particles, particles_hist (T, N, 2) and weights_hist (T, N).
    The histories start at t = dt (the initial cloud is not stored).

    Note: with the cpu backend the worker streams are seeded reproducibly, but the assignment of
    particles to workers is decided by the pool at run time, so two runs with the same seed
    agree only statistically.  The torch backend is reproducible on a given device.
    """
    x0 = np.array([2.0, 0.0]) if x0 is None else np.asarray(x0, dtype=float)
    P0 = np.diag([0.01, 0.01]) if P0 is None else np.asarray(P0, dtype=float)
    Q = 6e-3 * np.eye(2) if Q is None else np.asarray(Q, dtype=float)
    R = np.diag([0.5**2, 0.8**2]) if R is None else np.asarray(R, dtype=float)
    measurements = {} if measurements is None else measurements

    if backend == "cpu":
        return _run_pf_cpu(t_end, dt, mu_vdp, measurements, h_fun, R, Q, x0, P0, n_particles, save_dt,
                           seed, n_workers, store_particles, verbose)
    if backend == "torch":
        if h_fun is not h_fun_polar:
            raise ValueError("the torch backend only implements the polar measurement h_fun_polar")
        return _run_pf_torch(t_end, dt, mu_vdp, measurements, R, Q, x0, P0, n_particles, save_dt,
                             seed, n_sub, store_particles, verbose)
    raise ValueError(f"unknown backend '{backend}' (use 'cpu' or 'torch')")


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
    """Write mean_trajectory.csv, covariance.npz and (if present) particles.npz."""
    os.makedirs(out_dir, exist_ok=True)
    table = np.column_stack([results["time_hist"], results["mu_hist"]])
    np.savetxt(os.path.join(out_dir, "mean_trajectory.csv"), table, delimiter=",",
               header="time,mean_y,mean_z", comments="")
    np.savez(os.path.join(out_dir, "covariance.npz"), times=results["time_hist"],
             means=results["mu_hist"], covs=results["P_hist"])
    if "particles_hist" in results:
        np.savez_compressed(os.path.join(out_dir, "particles.npz"), times=results["time_hist"],
                            states=results["particles_hist"], weights=results["weights_hist"])


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap particle filter for the 2-D stochastic Van der Pol oscillator "
                    "(polar measurements).")
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
    parser.add_argument("--n-particles", type=int, default=50000, help="number of particles (default 50000)")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    parser.add_argument("--backend", choices=["cpu", "torch"], default="cpu",
                        help="cpu: Radau per particle in a process pool; torch: vectorised RK4")
    parser.add_argument("--n-sub", type=int, default=20,
                        help="RK4 sub-steps per step of the torch backend (default 20)")
    parser.add_argument("--save-dt", type=float, default=None,
                        help="interval between stored results (default: every step)")
    parser.add_argument("--workers", type=int, default=None,
                        help="worker processes of the cpu backend (default: all CPUs)")
    parser.add_argument("--out", default=os.path.join("results", "vdp2d_pf"), help="output directory")
    parser.add_argument("--save-particles", action="store_true",
                        help="also write particles.npz (all particles at every stored step; large)")
    parser.add_argument("--quiet", action="store_true", help="no progress output")
    args = parser.parse_args()

    measurements = select_measurements(parser, args)

    start = time.time()
    results = run_pf(t_end=args.t_end, dt=args.dt, mu_vdp=args.mu, measurements=measurements,
                     h_fun=h_fun_polar, R=np.diag([args.sigma_r**2, args.sigma_theta**2]),
                     Q=args.q * np.eye(2), x0=np.array(args.x0), P0=args.p0 * np.eye(2),
                     n_particles=args.n_particles, save_dt=args.save_dt, seed=args.seed,
                     backend=args.backend, n_sub=args.n_sub, n_workers=args.workers,
                     store_particles=args.save_particles, verbose=not args.quiet)
    wall = time.time() - start

    save_outputs(args.out, results)
    print(f"PF (mu = {args.mu}, {args.n_particles} particles, {args.backend} backend) finished in "
          f"{wall:.2f} s wall time; results in '{args.out}'")


if __name__ == "__main__":
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass
    main()
