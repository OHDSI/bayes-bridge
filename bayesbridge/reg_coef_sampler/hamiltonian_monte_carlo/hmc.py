import numpy as np
import math
import time
from .stepsize_adapter import HamiltonianBasedStepsizeAdapter, initialize_stepsize
from .util import warn_message_only
from .dynamics import HamiltonianDynamics


dynamics = HamiltonianDynamics()
integrator = dynamics.integrate
compute_hamiltonian = dynamics.compute_hamiltonian
draw_momentum = dynamics.draw_momentum


def generate_samples(
        f, q0, n_burnin, n_sample, nstep_range, dt_range=None,
        seed=None, n_update=0, adapt_stepsize=False, target_accept_prob=.9,
        final_adaptsize=.05):
    """ Run HMC and return samples and some additional info. """

    if seed is not None:
        np.random.seed(seed)

    q = q0
    logp, grad = f(q)

    if np.isscalar(dt_range):
        dt_range = np.array(2 * [dt_range])

    elif dt_range is None:
        p = draw_momentum(len(q))
        logp_joint0 = - compute_hamiltonian(logp, p)
        dt = initialize_stepsize(
            lambda dt: compute_onestep_accept_prob(dt, f, q, p, grad, logp_joint0)
        )
        dt_range = dt * np.array([.8, 1.0])
        adapt_stepsize = True

    if np.isscalar(nstep_range):
        nstep_range = np.array(2 * [nstep_range])

    max_stepsize_adapter = HamiltonianBasedStepsizeAdapter(
        init_stepsize=1., target_accept_prob=target_accept_prob,
        reference_iteration=n_burnin, adaptsize_at_reference=final_adaptsize
    )

    if n_update > 0:
        n_per_update = math.ceil((n_burnin + n_sample) / n_update)
    else:
        n_per_update = float('inf')

    samples = np.zeros((len(q), n_sample + n_burnin))
    logp_samples = np.zeros(n_sample + n_burnin)
    accept_prob = np.zeros(n_sample + n_burnin)

    tic = time.time()  # Start clock
    use_averaged_stepsize = False
    for i in range(n_sample + n_burnin):
        dt = np.random.uniform(dt_range[0], dt_range[1])
        dt *= max_stepsize_adapter.get_current_stepsize(use_averaged_stepsize)
        nstep = np.random.randint(nstep_range[0], nstep_range[1] + 1)
        q, info = generate_next_state(
            f, dt, nstep, q, logp0=logp, grad0=grad
        )
        logp, grad, pathlen, accept_prob[i] = (
            info[key] for key in ['logp', 'grad', 'n_grad_evals', 'accept_prob']
        )
        if i < n_burnin and adapt_stepsize:
            max_stepsize_adapter.adapt_stepsize(info['hamiltonian_error'])
        elif i == n_burnin - 1:
            use_averaged_stepsize = True
        samples[:, i] = q
        logp_samples[i] = logp
        if (i + 1) % n_per_update == 0:
            print('{:d} iterations have been completed.'.format(i + 1))

    toc = time.time()
    time_elapsed = toc - tic

    return samples, logp_samples, accept_prob, time_elapsed


def compute_onestep_accept_prob(dt, f, q0, p0, grad0, logp_joint0):
    _, p, logp, _ = integrator(f, dt, q0, p0, grad0)
    logp_joint = - compute_hamiltonian(logp, p)
    accept_prob = np.exp(logp_joint - logp_joint0)
    return accept_prob


def generate_next_state(
        f, dt, n_step, q0,
        p0=None, logp0=None, grad0=None, hamiltonian_tol=100.):

    n_grad_evals = 0

    if (logp0 is None) or (grad0 is None):
        logp0, grad0 = f(q0)
        n_grad_evals += 1

    if p0 is None:
        p0 = draw_momentum(len(q0))

    log_joint0 = - compute_hamiltonian(logp0, p0)

    q, p, logp, grad, simulation_info = simulate_dynamics(
        f, dt, n_step, q0, p0, logp0, grad0, hamiltonian_tol
    )
    n_grad_evals += simulation_info['n_grad_evals']
    instability_detected = simulation_info['instability_detected']

    if instability_detected:
        acceptprob = 0.
        hamiltonian_error = - float('inf')
    else:
        log_joint = - compute_hamiltonian(logp, p)
        hamiltonian_error = log_joint - log_joint0
        acceptprob = min(1, np.exp(hamiltonian_error))

    accepted = acceptprob > np.random.rand()
    if not accepted:
        q = q0
        logp = logp0
        grad = grad0

    info = {
        'logp': logp,
        'grad': grad,
        'accepted': accepted,
        'accept_prob': acceptprob,
        'hamiltonian_error': hamiltonian_error,
        'instability_detected': instability_detected,
        'n_grad_evals': n_grad_evals
    }

    return q, info


def simulate_dynamics(f, dt, n_step, q0, p0, logp0, grad0, hamiltonian_tol=float('inf')):

    n_grad_evals = 0
    instability_detected = False

    # Keep track of Hamiltonians along the trajectory.
    hamiltonians = np.full(n_step + 1, float('nan'))
    hamiltonian = compute_hamiltonian(logp0, p0)
    hamiltonians[0] = hamiltonian
    min_h, max_h = 2 * [hamiltonian]

    q, p, logp, grad = q0, p0, logp0, grad0
    if n_step == 0:
        warn_message_only("The number of integration steps was set to be 0.")

    for i in range(n_step):
        q, p, logp, grad \
            = integrator(f, dt, q, p, grad)
        hamiltonian = compute_hamiltonian(logp, p)
        hamiltonians[i + 1] = hamiltonian
        min_h, max_h = update_running_minmax(min_h, max_h, hamiltonian)
        n_grad_evals += 1
        instability_detected \
            = math.isinf(logp) or (max_h - min_h) > hamiltonian_tol
        if instability_detected:
            warn_message_only(
                "Numerical integration became unstable while simulating the "
                "HMC trajectory."
            )
            break

    info = {
        'energy_trajectory': hamiltonians,
        'n_grad_evals': n_grad_evals,
        'instability_detected': instability_detected,
    }

    return q, p, logp, grad, info


def update_running_minmax(running_min, running_max, curr_val):
    running_min = min(running_min, curr_val)
    running_max = max(running_max, curr_val)
    return running_min, running_max
