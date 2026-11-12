#!/usr/bin/env python
"""Bootstrap particle filter (PF) for the 6-D coupled Van der Pol system.

Model (three ring-coupled Van der Pol oscillators, state s = (x1, y1, x2, y2, x3, y3)):

    dx_i = y_i dt
    dy_i = [mu (1 - x_i^2) y_i - x_i + k (x_{i-1} - x_i) + k (x_{i+1} - x_i)] dt + dW_i ,
    E[dW dW^T] = Q dt   (additive noise on all six states, Q = 6e-3 I6 in the paper case)

Measurement (per oscillator, polar):

    h(s) = (r_1, theta_1, r_2, theta_2, r_3, theta_3),  r_i = sqrt(x_i^2 + y_i^2),  theta_i = atan2(y_i, x_i)
    z = h(s) + v,  v ~ N(0, R),  R = diag(0.2^2, 0.2^2) x 3

Algorithm: every particle is propagated over dt with the deterministic drift and then perturbed by
additive N(0, Q dt) noise; at a measurement time the particles are re-weighted with the Gaussian
likelihood (angle differences wrapped) and resampled (systematic resampling).

Two backends:
    cpu    one RK45 integration (scipy) per particle and step, in a multiprocessing pool
    torch  vectorised over the particles, RK4 with n_sub sub-steps per dt; runs on CUDA if
           available, otherwise on the CPU (PyTorch is only imported for this backend)

Paper case: mu = 1.5, k = 0.2, x0 = (2, 0) x 3, P0 = 0.01 I6, dt = 0.01, t in [0, 10], one measurement
at t = 5, N = 300000 particles.

Usage:
    python coupled_vdp6d/pf.py --backend torch         # paper case, vectorised
    python coupled_vdp6d/pf.py                         # paper case, cpu backend (slow)
    python coupled_vdp6d/pf.py --n-particles 2000 --t-end 1 --no-measurements --workers 4 --out results/test
"""
import argparse
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
# 1. Model
# ==============================================================================
def coupled_vdp_6d(t, state, mu, k):
    """Drift of three ring-coupled Van der Pol oscillators (oscillator i is coupled to i-1 and i+1,
    periodic). state: [x1, y1, x2, y2, x3, y3]."""
    state = np.array(state)
    N = 3
    derivs = np.zeros(2 * N)

    for i in range(N):
        idx_x = 2 * i
        idx_y = 2 * i + 1

        x = state[idx_x]
        y = state[idx_y]

        idx_prev = 2 * ((i - 1) % N)
        idx_next = 2 * ((i + 1) % N)

        x_prev = state[idx_prev]
        x_next = state[idx_next]

        coupling = k * (x_next - x) + k * (x_prev - x)

        derivs[idx_x] = y
        derivs[idx_y] = mu * (1 - x**2) * y - x + coupling

    return derivs


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


# ==============================================================================
# 2. Particle filter building blocks (cpu backend)
# ==============================================================================
class Particle_pf:
    def __init__(self, state, w=1.0):
        self.state = state.copy()
        self.w = float(w)


def pf_systematic_resample(weights):
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


def pf_measurement_update_6d(particles, z, h_fun, R):
    """Re-weight the particles with the likelihood of z = [r1, th1, r2, th2, r3, th3]."""
    inv_R = np.linalg.inv(R)
    meas_dim = len(z)

    log_likelihoods = np.zeros(len(particles))

    for i, p in enumerate(particles):
        predicted_z = h_fun(p.state)
        innovation = z - predicted_z

        # wrap the angle components (1, 3, 5)
        for k in range(1, meas_dim, 2):
            innovation[k] = normalize_angle(innovation[k])

        log_likelihoods[i] = -0.5 * (innovation @ inv_R @ innovation)

    max_log_w = np.max(log_likelihoods)
    if np.isinf(max_log_w):
        normalized_weights = np.ones(len(particles)) / len(particles)
    else:
        weights_linear = np.exp(log_likelihoods - max_log_w)
        total_weight = np.sum(weights_linear)
        if total_weight > 1e-12:
            normalized_weights = weights_linear / total_weight
        else:
            normalized_weights = np.ones(len(particles)) / len(particles)

    for i, p in enumerate(particles):
        p.w = normalized_weights[i]


def propagate_worker_6d(state, t0, dt, mu, k, process_noise_cov):
    """Propagate one particle over dt (RK45) and add the process noise N(0, Q dt)."""
    try:
        sol = solve_ivp(
            lambda t, x: coupled_vdp_6d(t, x, mu, k),
            (t0, t0 + dt), state, t_eval=[t0 + dt],
            method='RK45', rtol=1e-7, atol=1e-9
        )

        if not sol.success:
            return state

        propagated_state = sol.y[:, 0]

        dim = len(state)
        discretized_step_cov = process_noise_cov * dt
        noise = np.random.multivariate_normal(np.zeros(dim), discretized_step_cov)

        return propagated_state + noise

    except Exception as e:
        warnings.warn(f"PF: propagation of a particle failed ({e}); the particle is kept unchanged.")
        return state


def _seed_worker(base_seed):
    # Forked pool workers inherit the numpy RNG state of the parent process; without re-seeding,
    # every worker would draw the same process-noise sequence. Give each worker its own stream,
    # derived from the base seed and the worker index.
    ident = mp.current_process()._identity
    worker_index = int(ident[0]) if ident else 0
    np.random.seed(np.random.SeedSequence([int(base_seed), worker_index]).generate_state(1)[0])


def _append_particles_csv(path, step, t, states, weights, write_header):
    df = pd.DataFrame(states, columns=STATE_NAMES)
    df.insert(0, 'weight', weights)
    df.insert(0, 'time', t)
    df.insert(0, 'time_step', step)
    df.to_csv(path, mode='a', header=write_header, index=False)


# ==============================================================================
# 3. Filter loops
# ==============================================================================
def _run_pf_cpu(t_end, dt, mu, k_coupling, measurements, R, Q, x0, P0, N, save_every_n_steps, seed,
                n_workers, particle_csv_path, verbose):
    np.random.seed(seed)
    particles = [Particle_pf(np.random.multivariate_normal(x0, P0), 1.0 / N) for _ in range(N)]

    t = 0.0
    num_steps = int(round(t_end / dt))

    time_hist, mu_hist, P_hist = [], [], []
    mean_hist = []
    write_header = True

    num_cores = mp.cpu_count() if n_workers is None else int(n_workers)
    if verbose:
        print(f"Running PF (6-D coupled Van der Pol, mu={mu}, k={k_coupling}): "
              f"N = {N}, cpu backend, {num_cores} processes")

    with mp.Pool(processes=num_cores, initializer=_seed_worker, initargs=(seed,)) as pool:
        for s in range(num_steps):
            if verbose and s % 100 == 0:
                print(f"Step {s}/{num_steps}, time = {t:.2f}")

            # 1) propagate
            task_args = [(p.state, t, dt, mu, k_coupling, Q) for p in particles]
            new_states = pool.starmap(propagate_worker_6d, task_args)

            for i, p in enumerate(particles):
                p.state = new_states[i]

            t += dt
            current_time_rounded = float(round(t, 6))

            # 2) measurement update and resampling
            if current_time_rounded in measurements:
                z_obs = measurements[current_time_rounded]
                if verbose:
                    print(f"  -> measurement update at t = {current_time_rounded:.2f}")

                pf_measurement_update_6d(particles, z_obs, h_fun_6d, R)

                weights = np.array([p.w for p in particles])
                indexes = pf_systematic_resample(weights)
                particles = [Particle_pf(particles[idx].state, 1.0 / N) for idx in indexes]

            # 3) store the mean and the weighted sample covariance
            if (s + 1) % save_every_n_steps == 0:
                current_states = np.array([p.state for p in particles])
                mean_state = np.mean(current_states, axis=0)
                mean_hist.append((t, mean_state))

                weights = np.array([p.w for p in particles])
                w_norm = weights / weights.sum()
                dev = current_states - mean_state
                time_hist.append(t)
                mu_hist.append(mean_state)
                P_hist.append((dev * w_norm[:, None]).T @ dev)

                if particle_csv_path is not None:
                    _append_particles_csv(particle_csv_path, s + 1, t, current_states, weights, write_header)
                    write_header = False

    return {"time_hist": np.array(time_hist), "mu_hist": np.array(mu_hist), "P_hist": np.array(P_hist),
            "mean_hist": mean_hist}


def _run_pf_torch(t_end, dt, mu, k_coupling, measurements, R, Q, x0, P0, N, save_every_n_steps, seed,
                  n_sub, particle_csv_path, verbose):
    """Vectorised bootstrap PF: RK4 with n_sub sub-steps per dt for the drift, additive N(0, Q dt)
    noise once per dt, likelihood re-weighting and systematic resampling at measurement times."""
    import math
    import torch

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    g = torch.Generator(device=dev)
    g.manual_seed(int(seed))

    def tt(a):
        return torch.tensor(np.asarray(a, dtype=float), dtype=dtype, device=dev)

    x = tt(x0) + torch.randn(N, 6, generator=g, dtype=dtype, device=dev) @ torch.linalg.cholesky(tt(P0)).T
    Lq = torch.linalg.cholesky(tt(Q) * dt)
    Rinv = torch.linalg.inv(tt(R))

    def f(s):
        xx = s[:, 0::2]
        yy = s[:, 1::2]
        coup = k_coupling * (torch.roll(xx, 1, 1) - xx) + k_coupling * (torch.roll(xx, -1, 1) - xx)
        d = torch.empty_like(s)
        d[:, 0::2] = yy
        d[:, 1::2] = mu * (1 - xx ** 2) * yy - xx + coup
        return d

    h = dt / n_sub
    time_hist, mu_hist, P_hist = [], [], []
    mean_hist = []
    write_header = True
    t = 0.0
    ar = torch.arange(N, dtype=dtype, device=dev)
    num_steps = int(round(t_end / dt))

    if verbose:
        print(f"Running PF (6-D coupled Van der Pol, mu={mu}, k={k_coupling}): "
              f"N = {N}, torch backend on {dev}, n_sub = {n_sub}")

    for s in range(num_steps):
        if verbose and s % 100 == 0:
            print(f"Step {s}/{num_steps}, time = {t:.2f}")

        # 1) propagate: RK4 drift + process noise
        for _ in range(n_sub):
            k1 = f(x)
            k2 = f(x + 0.5 * h * k1)
            k3 = f(x + 0.5 * h * k2)
            k4 = f(x + h * k3)
            x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        x = x + torch.randn(N, 6, generator=g, dtype=dtype, device=dev) @ Lq.T

        t += dt
        current_time_rounded = float(round(t, 6))

        # 2) measurement update and systematic resampling
        if current_time_rounded in measurements:
            if verbose:
                print(f"  -> measurement update at t = {current_time_rounded:.2f}")
            z = tt(measurements[current_time_rounded])
            xx = x[:, 0::2]
            yy = x[:, 1::2]
            hz = torch.empty_like(x)
            hz[:, 0::2] = torch.sqrt(xx ** 2 + yy ** 2)
            hz[:, 1::2] = torch.atan2(yy, xx)
            inn = z - hz
            inn[:, 1::2] = (inn[:, 1::2] + math.pi) % (2 * math.pi) - math.pi
            logw = -0.5 * torch.einsum("ni,ij,nj->n", inn, Rinv, inn)
            w = torch.exp(logw - logw.max())
            w = w / w.sum()
            pos = (torch.rand(1, generator=g, dtype=dtype, device=dev) + ar) / N
            idx = torch.searchsorted(torch.cumsum(w, 0), pos).clamp_(max=N - 1)
            x = x[idx]

        # 3) store the mean and the sample covariance (uniform weights after resampling)
        if (s + 1) % save_every_n_steps == 0:
            m = x.mean(0)
            dm = x - m
            mean_state = m.cpu().numpy()
            time_hist.append(t)
            mu_hist.append(mean_state)
            P_hist.append((dm.T @ dm / N).cpu().numpy())
            mean_hist.append((t, mean_state))

            if particle_csv_path is not None:
                _append_particles_csv(particle_csv_path, s + 1, t, x.cpu().numpy(),
                                      np.full(N, 1.0 / N), write_header)
                write_header = False

    return {"time_hist": np.array(time_hist), "mu_hist": np.array(mu_hist), "P_hist": np.array(P_hist),
            "mean_hist": mean_hist}


def run_pf(t_end=10.0, dt=0.01, mu_vdp=1.5, k_coupling=0.2, measurements=None, R=None, Q=None,
           x0=None, P0=None, n_particles=300000, save_dt=None, seed=0, backend="cpu", n_sub=10,
           n_workers=None, particle_csv_path=None, verbose=True):
    """Run the bootstrap PF on the 6-D coupled Van der Pol system.

    measurements: dict {float(round(t, 6)): z (6,)}; None or {} -> prediction only.
    R, Q, x0, P0: measurement-noise covariance, process-noise covariance, initial mean / covariance
    (None -> paper values).
    n_particles: number of particles; save_dt: storage interval (None -> every step); seed: RNG seed.
    backend: "cpu" (RK45 per particle, multiprocessing pool of n_workers processes, None ->
    mp.cpu_count()) or "torch" (vectorised RK4 with n_sub sub-steps per dt).
    particle_csv_path: if given, the particles of every stored step are appended to this CSV file.

    Returns a dict with (T = number of stored steps; no t = 0 entry)
        time_hist (T,), mu_hist (T, 6), P_hist (T, 6, 6) [weighted sample covariance],
        mean_hist [list of (t, mean)].
    """
    if measurements is None:
        measurements = {}
    if R is None:
        R = np.diag([0.2**2, 0.2**2] * 3)
    if Q is None:
        Q = (6e-3) * np.eye(SYSTEM_DIM)
    if x0 is None:
        x0 = np.array([2.0, 0.0, 2.0, 0.0, 2.0, 0.0])
    if P0 is None:
        P0 = np.eye(SYSTEM_DIM) * 0.01

    save_every_n_steps = 1
    if save_dt is not None and dt > 0:
        save_every_n_steps = int(round(save_dt / dt))
        if save_every_n_steps < 1:
            save_every_n_steps = 1

    if backend == "cpu":
        return _run_pf_cpu(t_end, dt, mu_vdp, k_coupling, measurements, R, Q, x0, P0, n_particles,
                           save_every_n_steps, seed, n_workers, particle_csv_path, verbose)
    if backend == "torch":
        return _run_pf_torch(t_end, dt, mu_vdp, k_coupling, measurements, R, Q, x0, P0, n_particles,
                             save_every_n_steps, seed, n_sub, particle_csv_path, verbose)
    raise ValueError(f'unknown backend "{backend}" (use "cpu" or "torch")')


# ==============================================================================
# 4. Command-line interface
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
        description="Bootstrap PF for the 6-D coupled Van der Pol system (defaults: paper case).")
    parser.add_argument("--mu", type=float, default=1.5, help="Van der Pol parameter mu")
    parser.add_argument("--k-coupling", type=float, default=0.2, help="coupling strength k")
    parser.add_argument("--t-end", type=float, default=10.0, help="final time")
    parser.add_argument("--dt", type=float, default=0.01, help="filter time step")
    parser.add_argument("--no-measurements", action="store_true", help="prediction only")
    parser.add_argument("--measurement", action="append", metavar='"t,r,theta"',
                        help="measurement (same r, theta for the three oscillators) at time t; repeatable. "
                             "Default: the paper measurement at t = 5 for the given (mu, k)")
    parser.add_argument("--n-particles", type=int, default=300000, help="number of particles")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--backend", choices=["cpu", "torch"], default="cpu",
                        help="cpu: RK45 per particle in a process pool; "
                             "torch: vectorised RK4 (GPU if available)")
    parser.add_argument("--n-sub", type=int, default=10, help="torch backend: RK4 sub-steps per dt")
    parser.add_argument("--save-dt", type=float, default=None, help="storage interval (default: every step)")
    parser.add_argument("--workers", type=int, default=None,
                        help="cpu backend: pool size (default: all CPUs)")
    parser.add_argument("--save-particles", action="store_true",
                        help="also write the particles of every stored step to particles.csv (large)")
    parser.add_argument("--out", default=os.path.join("results", "coupled_vdp6d_pf"), help="output directory")
    parser.add_argument("--quiet", action="store_true", help="no progress output")
    args = parser.parse_args()

    measurements = build_measurements(args, parser)

    os.makedirs(args.out, exist_ok=True)
    particle_csv_path = None
    if args.save_particles:
        particle_csv_path = os.path.join(args.out, "particles.csv")
        if os.path.exists(particle_csv_path):
            os.remove(particle_csv_path)

    start = time.time()
    results = run_pf(t_end=args.t_end, dt=args.dt, mu_vdp=args.mu, k_coupling=args.k_coupling,
                     measurements=measurements, n_particles=args.n_particles, save_dt=args.save_dt,
                     seed=args.seed, backend=args.backend, n_sub=args.n_sub, n_workers=args.workers,
                     particle_csv_path=particle_csv_path, verbose=not args.quiet)
    wall = time.time() - start

    table = np.column_stack([results["time_hist"], results["mu_hist"]])
    pd.DataFrame(table, columns=["time"] + STATE_NAMES).to_csv(
        os.path.join(args.out, "mean_trajectory.csv"), index=False)
    np.savez(os.path.join(args.out, "covariance.npz"), times=results["time_hist"],
             means=results["mu_hist"], covs=results["P_hist"])
    saved = "mean_trajectory.csv and covariance.npz"
    if args.save_particles:
        saved += " and particles.csv"

    print(f"Saved {saved} to '{args.out}'")
    print(f"Wall time: {wall:.2f} s")


if __name__ == "__main__":
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    main()
