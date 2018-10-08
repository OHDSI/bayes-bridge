import numpy as np
import math
import time


def generate_samples(
        f, theta0, dt_range, nstep_range, n_burnin, n_sample,
        seed=None, n_update=0):
    """ Run HMC and return samples and some additional info. """

    np.random.seed(seed)

    if np.isscalar(dt_range):
        dt_range = np.array(2 * [dt_range])

    if np.isscalar(nstep_range):
        nstep_range = np.array(2 * [nstep_range])

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
    for i in range(n_sample + n_burnin):
        dt = np.random.uniform(dt_range[0], dt_range[1])
        nstep = np.random.randint(nstep_range[0], nstep_range[1] + 1)
        theta, info \
            = generate_next_state(f, dt, nstep, theta, logp, grad)
        logp, grad, pathlen, accept_prob[i] = (
            info[key] for key in ['logp', 'grad', 'n_grad_evals', 'accept_prob']
        )
        pathlen_ave = i / (i + 1) * pathlen_ave + 1 / (i + 1) * pathlen
        samples[:, i] = theta
        logp_samples[i] = logp
        if (i + 1) % n_per_update == 0:
            print('{:d} iterations have been completed.'.format(i + 1))

    toc = time.time()
    time_elapsed = toc - tic

    return samples, logp_samples, accept_prob, time_elapsed


def generate_next_state(f, dt, n_step, theta0, logp0=None, grad0=None):

    n_grad_evals = 0

    if (logp0 is None) or (grad0 is None):
        logp0, grad0 = f(theta0)
        n_grad_evals += 1

    p = draw_momentum(len(theta0))
    joint0 = - compute_hamiltonian(logp0, p)

    theta, p, grad, logp, n_grad_evals \
            = simulate_dynamics(f, dt, n_step, theta0, p, grad0)

    if math.isinf(logp):
        acceptprob = 0.
    else:
        joint = - compute_hamiltonian(logp, p)
        acceptprob = min(1, np.exp(joint - joint0))

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
        'n_grad_evals': n_grad_evals
    }

    return theta, info


def simulate_dynamics(f, dt, n_step, theta0, p, grad0):

    n_grad_evals = 0
    theta, grad = theta0.copy(), grad0.copy()
    theta, p, grad, logp \
        = integrator(f, dt, theta, p, grad)
    n_grad_evals += 1
    for i in range(1, n_step):
        if math.isinf(logp):
            break
        theta, p, grad, logp \
            = integrator(f, dt, theta, p, grad)
        n_grad_evals += 1

    return theta, p, grad, logp, n_grad_evals


def integrator(f, dt, theta, p, grad):

    p = p + 0.5 * dt * grad
    theta = theta + dt * p
    logp, grad = f(theta)
    p = p + 0.5 * dt * grad

    return theta, p, grad, logp


def compute_hamiltonian(logp, p):
    return - logp + 0.5 * np.dot(p, p)


def draw_momentum(n_param):
    return np.random.randn(n_param)