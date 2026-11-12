"""Bootstrap particle filter (PF) for the stochastic Lorenz-63 system.

Model (additive white-noise forcing):

    dx = sigma (y - x) dt        + sqrt(2 Gamma) dW_x
    dy = [x (rho - z) - y] dt    + sqrt(2 Gamma) dW_y
    dz = [x y - beta z] dt       + sqrt(2 Gamma) dW_z

i.e. dX = f(X) dt + sqrt(2 Gamma) dW; the noise increment over one filter step dt has covariance
2 Gamma dt I3.  Measurement: direct observation of the state, z = h(x) + v, h(x) = x, v ~ N(0, R).

Every particle is propagated over dt with the deterministic drift and then perturbed by an
N(0, 2 Gamma dt I3) increment.  At measurement times the particles are re-weighted by the
likelihood and resampled (systematic resampling).  Two backends are available:

    cpu    one adaptive RK45 solve per particle, distributed over a multiprocessing pool
    torch  vectorised over all particles, fixed-step RK4 with n_sub sub-steps per dt
           (uses CUDA if available, otherwise the CPU; requires PyTorch)

Usage:
    python lorenz63/pf.py                                   # paper case, N = 500000 (expensive)
    python lorenz63/pf.py --backend torch --n-particles 500000
    python lorenz63/pf.py --t-end 1 --n-particles 2000 --workers 4 --no-measurements --out results/pf_test
"""
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

SYSTEM_DIM = 3

# Paper case
PAPER_X0 = (1.0, 1.0, 1.0)
PAPER_P0_VAR = 0.1
PAPER_MEAS_SIGMA = 0.8
PAPER_MEASUREMENT = (10.0, -0.25, 0.02, 20.09)   # t, x, y, z
PAPER_N_PARTICLES = 500000


# --------------------------------------------------------------------------
# 0. Lorenz-63 drift
# --------------------------------------------------------------------------
def lorenz_drift(t, state, sigma=3.0, rho=26.5, beta=1.0):
    state = np.array(state)

    x = state[0]
    y = state[1]
    z = state[2]

    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z

    return np.array([dxdt, dydt, dzdt])


def h_fun_lorenz(state):
    return np.array(state)


class Particle_pf:
    def __init__(self, state, w=1.0):
        self.state = state.copy()
        self.w = float(w)


def pf_systematic_resample(weights):
    N = len(weights)
    positions = (np.random.random() + np.arange(N)) / N
    indexes = np.zeros(N, dtype=int)
    cumulative_sum = np.cumsum(weights)

    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1

    return indexes


def pf_measurement_update(particles, z, h_fun, R):
    """Set the particle weights to the normalised Gaussian likelihood of the measurement z."""
    inv_R = np.linalg.inv(R)
    log_likelihoods = np.zeros(len(particles))

    for i, p in enumerate(particles):
        predicted_z = h_fun(p.state)
        innovation = z - predicted_z
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


# --------------------------------------------------------------------------
# 1. Propagation worker (cpu backend)
# --------------------------------------------------------------------------
def _seed_worker(base_seed):
    # Forked pool workers inherit the parent's numpy RNG state; without re-seeding, every worker
    # would draw the same process-noise sequence (noise replicated across workers).  Each worker
    # therefore gets its own stream derived from (base_seed, worker index).
    ident = mp.current_process()._identity
    np.random.seed(
        np.random.SeedSequence([int(base_seed), int(ident[0]) if ident else 0]).generate_state(1)[0]
    )


def propagate_worker(
    state,
    t0,
    dt,
    sigma,
    rho,
    beta,
    gamma
):
    """Propagate one particle over dt with the drift, then add the N(0, 2 Gamma dt I) increment."""
    try:
        sol = solve_ivp(
            lambda t, x: lorenz_drift(t, x, sigma, rho, beta),
            (t0, t0 + dt),
            state,
            t_eval=[t0 + dt],
            method="RK45",
            rtol=1e-7,
            atol=1e-9,
        )

        if not sol.success:
            return state

        propagated_state = sol.y[:, 0].copy()

        noise = np.sqrt(2.0 * gamma * dt) * np.random.randn(3)
        propagated_state += noise

        return propagated_state

    except Exception as e:
        warnings.warn(f"PF: propagation failed at t = {t0:.6f} ({e}); particle left unchanged.")
        return state


# --------------------------------------------------------------------------
# 2. PF simulation, cpu backend
# --------------------------------------------------------------------------
def _run_pf_cpu(t_end, dt, sigma, rho, beta, gamma, measurements, h_fun, R, x0, P0, N, save_dt,
                seed, n_workers, verbose):
    np.random.seed(seed)

    particles = [
        Particle_pf(np.random.multivariate_normal(x0, P0), 1.0 / N)
        for _ in range(N)
    ]

    t = 0.0
    num_steps = int(round(t_end / dt))

    time_hist, mu_hist, P_hist = [], [], []

    save_every_n_steps = 1
    if save_dt is not None and dt > 0:
        save_every_n_steps = int(round(save_dt / dt))
        if save_every_n_steps < 1:
            save_every_n_steps = 1

    num_cores = mp.cpu_count() if n_workers is None else int(n_workers)
    if verbose:
        print(f"Starting Lorenz-63 PF (cpu backend): N = {N}, {num_cores} worker processes.")

    with mp.Pool(processes=num_cores, initializer=_seed_worker, initargs=(seed,)) as pool:
        for s_idx in range(num_steps):
            if verbose and s_idx % 100 == 0:
                print(f"Step {s_idx}/{num_steps}, time: {t:.3f}")

            # 1) Propagate
            task_args = [
                (p.state, t, dt, sigma, rho, beta, gamma)
                for p in particles
            ]

            new_states = pool.starmap(
                propagate_worker,
                task_args
            )

            for i, p in enumerate(particles):
                p.state = new_states[i]

            t += dt
            current_time_rounded = float(round(t, 6))

            # 2) Update & resample
            if current_time_rounded in measurements:
                z_obs = measurements[current_time_rounded]
                if verbose:
                    print(f"  --> Measurement update at t={current_time_rounded:.3f}")

                pf_measurement_update(particles, z_obs, h_fun, R)

                weights = np.array([p.w for p in particles])
                indexes = pf_systematic_resample(weights)

                particles = [
                    Particle_pf(particles[idx].state, 1.0 / N)
                    for idx in indexes
                ]

            # 3) Store mean and weighted sample covariance
            if (s_idx + 1) % save_every_n_steps == 0:
                current_states = np.array([p.state for p in particles])
                mean_state = np.mean(current_states, axis=0)

                weights = np.array([p.w for p in particles])
                weights = weights / weights.sum()
                centered = current_states - mean_state
                cov_state = (centered * weights[:, None]).T @ centered

                time_hist.append(t)
                mu_hist.append(mean_state)
                P_hist.append(cov_state)

    return {
        "time_hist": np.array(time_hist),
        "mu_hist": np.array(mu_hist),
        "P_hist": np.array(P_hist),
        "particles": np.array([p.state for p in particles]),
        "weights": np.array([p.w for p in particles]),
    }


# --------------------------------------------------------------------------
# 3. PF simulation, torch backend (vectorised; same algorithm)
# --------------------------------------------------------------------------
def _run_pf_torch(t_end, dt, sigma, rho, beta, gamma, measurements, R, x0, P0, N, save_dt,
                  seed, n_sub, verbose):
    """Bootstrap PF vectorised over the particles: RK4 with n_sub sub-steps per dt for the drift,
    additive N(0, Q dt) noise, likelihood re-weighting + systematic resampling at measurement times."""
    import torch

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    d = SYSTEM_DIM

    g = torch.Generator(device=dev)
    g.manual_seed(int(seed))

    def tt(a):
        return torch.tensor(np.asarray(a, dtype=float), dtype=dtype, device=dev)

    def f(s):
        x, y, z = s[:, 0], s[:, 1], s[:, 2]
        return torch.stack((sigma * (y - x), x * (rho - z) - y, x * y - beta * z), dim=1)

    Q = 2.0 * gamma * np.eye(d)

    x = tt(x0) + torch.randn(N, d, generator=g, dtype=dtype, device=dev) @ torch.linalg.cholesky(tt(P0)).T
    Lq = torch.linalg.cholesky(tt(Q) * dt)
    Rinv = torch.linalg.inv(tt(R))

    h = dt / n_sub
    t = 0.0
    ar = torch.arange(N, dtype=dtype, device=dev)
    num_steps = int(round(t_end / dt))

    save_every_n_steps = 1
    if save_dt is not None and dt > 0:
        save_every_n_steps = max(1, int(round(save_dt / dt)))

    time_hist, mu_hist, P_hist = [], [], []

    if verbose:
        print(f"Starting Lorenz-63 PF (torch backend on {dev}): N = {N}, n_sub = {n_sub}.")

    for s_idx in range(num_steps):
        if verbose and s_idx % 100 == 0:
            print(f"Step {s_idx}/{num_steps}, time: {t:.3f}")

        # 1) Propagate: RK4 drift + additive noise
        for _ in range(n_sub):
            k1 = f(x)
            k2 = f(x + 0.5 * h * k1)
            k3 = f(x + 0.5 * h * k2)
            k4 = f(x + h * k3)
            x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        x = x + torch.randn(N, d, generator=g, dtype=dtype, device=dev) @ Lq.T

        t += dt
        current_time_rounded = float(round(t, 6))

        # 2) Update & resample (direct measurement h(x) = x)
        if current_time_rounded in measurements:
            if verbose:
                print(f"  --> Measurement update at t={current_time_rounded:.3f}")
            inn = tt(measurements[current_time_rounded]) - x
            logw = -0.5 * torch.einsum("ni,ij,nj->n", inn, Rinv, inn)
            w = torch.exp(logw - logw.max())
            w = w / w.sum()

            pos = (torch.rand(1, generator=g, dtype=dtype, device=dev) + ar) / N
            idx = torch.searchsorted(torch.cumsum(w, 0), pos).clamp_(max=N - 1)
            x = x[idx]

        # 3) Store mean and sample covariance (uniform weights after resampling)
        if (s_idx + 1) % save_every_n_steps == 0:
            m = x.mean(0)
            dm = x - m
            time_hist.append(t)
            mu_hist.append(m.cpu().numpy())
            P_hist.append((dm.T @ dm / N).cpu().numpy())

    return {
        "time_hist": np.array(time_hist),
        "mu_hist": np.array(mu_hist),
        "P_hist": np.array(P_hist),
        "particles": x.cpu().numpy(),
        "weights": np.full(N, 1.0 / N),
    }


# --------------------------------------------------------------------------
# 4. Public entry point
# --------------------------------------------------------------------------
def run_pf(
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
    n_particles=PAPER_N_PARTICLES,
    save_dt=None,
    seed=0,
    backend="cpu",
    n_sub=10,
    n_workers=None,
    verbose=False,
):
    """Run the bootstrap PF on the stochastic Lorenz-63 system.

    Parameters
    ----------
    t_end, dt : final time and filter step.
    sigma, rho, beta : Lorenz parameters.
    gamma : forcing strength Gamma; the noise increment per step has covariance 2 Gamma dt I3.
    measurements : dict {round(t, 6): z (3,)} or None (prediction only).
    h_fun, R : measurement function and noise covariance (default 0.8^2 I3).  The torch backend
        implements the direct measurement h(x) = x only.
    x0, P0 : mean and covariance of the initial particle cloud (default (1,1,1), 0.1 I3).
    n_particles : number of particles.
    save_dt : interval between stored steps (None -> every step).
    seed : seed of the random number generator(s).
    backend : "cpu" (RK45 per particle, multiprocessing pool) or "torch" (vectorised RK4).
    n_sub : RK4 sub-steps per dt (torch backend only).
    n_workers : pool size of the cpu backend (None -> mp.cpu_count()).

    Returns
    -------
    dict with time_hist (T,), mu_hist (T,3), P_hist (T,3,3) (no entry for t = 0), and the final
    particle cloud: particles (N,3), weights (N,).

    Note: with the cpu backend the result is statistically, but not bit-wise, reproducible for a
    given seed, because the assignment of particles to pool workers is not deterministic.
    """
    x0 = np.array(x0, dtype=float)
    if P0 is None:
        P0 = np.eye(SYSTEM_DIM) * PAPER_P0_VAR
    if R is None:
        R = np.diag([PAPER_MEAS_SIGMA ** 2] * SYSTEM_DIM)
    if measurements is None:
        measurements = {}

    if backend == "cpu":
        return _run_pf_cpu(t_end, dt, sigma, rho, beta, gamma, measurements, h_fun, R, x0, P0, n_particles,
                           save_dt, seed, n_workers, verbose)
    if backend == "torch":
        if h_fun is not h_fun_lorenz:
            raise ValueError("the torch backend only supports the direct measurement h(x) = x")
        return _run_pf_torch(t_end, dt, sigma, rho, beta, gamma, measurements, R, x0, P0, n_particles,
                             save_dt, seed, n_sub, verbose)
    raise ValueError(f"unknown backend '{backend}' (use 'cpu' or 'torch')")


# --------------------------------------------------------------------------
# 5. Command line
# --------------------------------------------------------------------------
def parse_measurements(specs):
    """["t,x,y,z", ...] -> {round(t, 6): array([x, y, z])}."""
    measurements = {}
    for spec in specs:
        vals = [float(v) for v in spec.split(",")]
        if len(vals) != 1 + SYSTEM_DIM:
            raise ValueError(f"--measurement expects 't,x,y,z', got '{spec}'")
        measurements[float(round(vals[0], 6))] = np.array(vals[1:])
    return measurements


def save_results(results, out_dir, save_particles=False):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(np.c_[results["time_hist"], results["mu_hist"]], columns=["time", "x", "y", "z"])
    df.to_csv(os.path.join(out_dir, "mean_trajectory.csv"), index=False)

    np.savez(
        os.path.join(out_dir, "covariance.npz"),
        times=results["time_hist"],
        means=results["mu_hist"],
        covs=results["P_hist"],
    )

    if save_particles:
        # Final particle cloud
        states = results["particles"]
        pd.DataFrame({
            "particle_id": np.arange(len(states)),
            "weight": results["weights"],
            "x": states[:, 0],
            "y": states[:, 1],
            "z": states[:, 2],
        }).to_csv(os.path.join(out_dir, "particles.csv"), index=False)


def main():
    ap = argparse.ArgumentParser(description="Bootstrap particle filter for the stochastic Lorenz-63 system.")
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
    ap.add_argument("--n-particles", type=int, default=PAPER_N_PARTICLES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", choices=["cpu", "torch"], default="cpu")
    ap.add_argument("--n-sub", type=int, default=10, help="RK4 sub-steps per dt (torch backend)")
    ap.add_argument("--workers", type=int, default=None,
                    help="pool size of the cpu backend (default: all CPUs)")
    ap.add_argument("--out", default=os.path.join("results", "lorenz63_pf"))
    ap.add_argument("--save-particles", action="store_true",
                    help="also write the final particle cloud to particles.csv")
    ap.add_argument("--quiet", action="store_true", help="no progress output")
    args = ap.parse_args()

    if args.no_measurements:
        measurements = {}
    elif args.measurement:
        measurements = parse_measurements(args.measurement)
    else:
        measurements = parse_measurements([",".join(str(v) for v in PAPER_MEASUREMENT)])

    start_time = time.time()
    results = run_pf(
        t_end=args.t_end,
        dt=args.dt,
        sigma=args.sigma,
        rho=args.rho,
        beta=args.beta,
        gamma=args.gamma,
        measurements=measurements,
        h_fun=h_fun_lorenz,
        R=np.diag([args.meas_sigma ** 2] * SYSTEM_DIM),
        n_particles=args.n_particles,
        seed=args.seed,
        backend=args.backend,
        n_sub=args.n_sub,
        n_workers=args.workers,
        verbose=not args.quiet,
    )
    wall = time.time() - start_time

    save_results(results, args.out, save_particles=args.save_particles)
    print(f"PF (Lorenz-63, {args.backend} backend, N = {args.n_particles}) finished: "
          f"{len(results['time_hist'])} stored steps, wall time {wall:.2f} s")
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass

    main()
