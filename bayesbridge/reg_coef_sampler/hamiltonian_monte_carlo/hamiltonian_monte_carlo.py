import numpy as np
import math
import time
from bayesbridge.reg_coef_sampler.hamiltonian_monte_carlo.stepsize_adapter import HmcStepsizeAdapter
from bayesbridge.util import warn_message_only


def generate_samples(
        f, theta0, dt_range, nstep_range, n_burnin, n_sample,
        seed=None, n_update=0, adapt_stepsize=False, target_accept_prob=.9,
        final_adaptsize=.05):
    """ Run HMC and return samples and some additional info. """

    np.random.seed(seed)

    if np.isscalar(dt_range):
        dt_range = np.array(2 * [dt_range])

    if np.isscalar(nstep_range):
        nstep_range = np.array(2 * [nstep_range])

    max_stepsize_adapter = HmcStepsizeAdapter(
        init_stepsize=1., target_accept_prob=target_accept_prob,
        reference_iteration=n_burnin, adaptsize_at_reference=final_adaptsize
    )

    theta = theta0
    if n_update > 0:
        n_per_update = math.ceil((n_burnin + n_sample) / n_update)
    else:
        n_per_update = float('inf')
    pathlen_ave = 0
    samples = np.zeros((len(theta), n_sample + n_burnin))
    logp_samples = np.zeros(n_sample + n_burnin)
    accept_prob = np.zeros(n_sample + n_burnin)

    tic = time.time()  # Start clock
    logp, grad = f(theta)
    use_averaged_stepsize = False
    for i in range(n_sample + n_burnin):
        dt = np.random.uniform(dt_range[0], dt_range[1])
        dt *= max_stepsize_adapter.get_current_stepsize(use_averaged_stepsize)
        nstep = np.random.randint(nstep_range[0], nstep_range[1] + 1)
        theta, info = generate_next_state(
            f, dt, nstep, theta, logp0=logp, grad0=grad
        )
        logp, grad, pathlen, accept_prob[i] = (
            info[key] for key in ['logp', 'grad', 'n_grad_evals', 'accept_prob']
        )
        if i < n_burnin and adapt_stepsize:
            max_stepsize_adapter.adapt_stepsize(info['hamiltonian_error'])
        elif i == n_burnin - 1:
            use_averaged_stepsize = True
        pathlen_ave = i / (i + 1) * pathlen_ave + 1 / (i + 1) * pathlen
        samples[:, i] = theta
        logp_samples[i] = logp
        if (i + 1) % n_per_update == 0:
            print('{:d} iterations have been completed.'.format(i + 1))

    toc = time.time()
    time_elapsed = toc - tic

    return samples, logp_samples, accept_prob, time_elapsed


def generate_next_state(
        f, dt, n_step, theta0,
        p0=None, logp0=None, grad0=None, hamiltonian_tol=100.):

    n_grad_evals = 0

    if (logp0 is None) or (grad0 is None):
        logp0, grad0 = f(theta0)
        n_grad_evals += 1

    if p0 is None:
        p0 = draw_momentum(len(theta0))

    log_joint0 = - compute_hamiltonian(logp0, p0)

    theta, p, logp, grad, simulation_info = simulate_dynamics(
        f, dt, n_step, theta0, p0, logp0, grad0, hamiltonian_tol
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
        theta = theta0
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

    return theta, info


def simulate_dynamics(f, dt, n_step, theta0, p0, logp0, grad0, hamiltonian_tol):

    n_grad_evals = 0
    instability_detected = False

    # Keep track of Hamiltonians along the trajectory.
    hamiltonians = np.full(n_step + 1, float('nan'))
    hamiltonian = compute_hamiltonian(logp0, p0)
    hamiltonians[0] = hamiltonian
    min_h, max_h = 2 * [hamiltonian]

    # First integration step.
    if n_step == 0:
        warn_message_only("The number of integration steps was set to be 0.")
        theta, p, logp, grad = theta0, p0, logp0, grad0
    else:
        theta, p, logp, grad \
            = integrator(f, dt, theta0, p0, grad0)
        hamiltonian = compute_hamiltonian(logp, p)
        hamiltonians[1] = hamiltonian
        min_h, max_h = update_running_minmax(min_h, max_h, hamiltonian)
        n_grad_evals += 1
        if math.isinf(logp) or (max_h - min_h) > hamiltonian_tol:
            instability_detected = True

    for i in range(1, n_step):

        theta, p, logp, grad \
            = integrator(f, dt, theta, p, grad)
        hamiltonian = compute_hamiltonian(logp, p)
        hamiltonians[i + 1] = hamiltonian
        min_h, max_h = update_running_minmax(min_h, max_h, hamiltonian)
        n_grad_evals += 1

        if math.isinf(logp) or (max_h - min_h) > hamiltonian_tol:
            instability_detected = True
            break

    info = {
        'logp_trajectory': - hamiltonians,
        'n_grad_evals': n_grad_evals,
        'instability_detected': instability_detected,
    }

    return theta, p, logp, grad, info


def update_running_minmax(running_min, running_max, curr_val):
    running_min = min(running_min, curr_val)
    running_max = max(running_max, curr_val)
    return running_min, running_max


def integrator(f, dt, theta, p, grad):

    p = p + 0.5 * dt * grad
    theta = theta + dt * p
    logp, grad = f(theta)
    p = p + 0.5 * dt * grad

    return theta, p, logp, grad


def compute_hamiltonian(logp, p):
    return - logp + 0.5 * np.dot(p, p)


def draw_momentum(n_param):
    return np.random.randn(n_param)