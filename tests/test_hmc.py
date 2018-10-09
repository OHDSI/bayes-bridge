import sys
sys.path.append(".") # needed if pytest called from the parent directory
sys.path.append("..") # needed if pytest called from this directory.

import numpy as np
import bayesbridge.reg_coef_sampler.hamiltonian_monte_carlo as hmc
from tests.distributions import BivariateGaussian



""" Regression test. """

data_folder = 'regression_test_saved_outputs/'
test_samples_file = data_folder + 'hmc_bivariate_gaussian_samples.npy'


def test_hmc():
    samples = run_hmc()
    prev_output = np.load(test_samples_file)
    assert np.allclose(samples[:, -1], prev_output, atol=1e-10, rtol=1e-10)


def run_hmc():

    dt = np.array([.55, .63])
    n_step = np.array([3, 4])
    n_burnin = 10
    n_sample = 100

    theta0 = np.zeros(2)
    f = BivariateGaussian().compute_logp_and_gradient

    samples = hmc.generate_samples(
        f, theta0, dt, n_step, n_burnin, n_sample, seed=0)[0]

    return samples


# Update the saved HMC outputs, if called as a script with option 'update'.
if __name__ == '__main__':
    option = sys.argv[-1]
    if option == 'update':
        samples = run_hmc()
        np.save(test_samples_file, samples[:, -1])


""" Other tests. """

def test_early_termination_symmetry():

    bi_gauss = BivariateGaussian(rho=.9, sigma=np.array([1., 2.]))
    f = bi_gauss.compute_logp_and_gradient
    dt = .99 * bi_gauss.get_stepsize_stability_limit()
    n_step = 100
    theta0 = np.array([0., 0.])
    p0 = np.array([1., 1.])

    # Expect early termination.
    hamiltonian_tol = 5.7
    logp0, grad0 = f(theta0)
    theta, p, logp, info, info_reverse = simulate_forward_and_backward(
        f, dt, n_step, theta0, p0, logp0, grad0, hamiltonian_tol
    )

    assert (
        info['instability_detected']
        == info_reverse['instability_detected']
        == True
    )

    # Expect NO early termination.
    hamiltonian_tol = 5.8
    logp0, grad0 = f(theta0)
    theta, p, logp, info, info_reverse = simulate_forward_and_backward(
        f, dt, n_step, theta0, p0, logp0, grad0, hamiltonian_tol
    )

    assert info['instability_detected'] == info_reverse['instability_detected'] == False
    assert np.allclose(logp0, logp, atol=1e-10)
    assert np.allclose(theta0, theta, atol=1e-10)
    assert np.allclose(p0, p, atol=1e-10)


def simulate_forward_and_backward(f, dt, n_step, theta0, p0, logp0, grad0,
                                  hamiltonian_tol):
    # Forward dynamics.
    theta, p, logp, grad, info = hmc.simulate_dynamics(
        f, dt, n_step, theta0, p0, logp0, grad0, hamiltonian_tol
    )

    # Reverse dynamics.
    theta, p, logp, grad, reverse_info = hmc.simulate_dynamics(
        f, dt, n_step, theta, -p, logp, grad, hamiltonian_tol
    )
    p = -p

    return theta, p, logp, info, reverse_info