import sys
sys.path.append(".") # needed if pytest called from the parent directory
sys.path.append("..") # needed if pytest called from this directory.

import numpy as np
import bayesbridge.reg_coef_sampler.hamiltonian_monte_carlo as hmc


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
    f = define_bivariate_gaussian_target()

    samples = hmc.generate_samples(
        f, theta0, dt, n_step, n_burnin, n_sample, seed=0)[0]

    return samples

def define_bivariate_gaussian_target(rho=.9, sigma=np.array([1.0, 2.0])):

    Sigma = np.array([
        [sigma[0] ** 2, rho * np.prod(sigma)],
        [rho * np.prod(sigma), sigma[1] ** 2]
    ])
    Phi = np.linalg.inv(Sigma)

    def bivariate_gaussian_target(x):
        logp = - np.inner(x, Phi.dot(x)) / 2
        grad = - Phi.dot(x)
        return logp, grad

    return bivariate_gaussian_target


if __name__ == '__main__':
    option = sys.argv[-1]
    if option == 'update':
        samples = run_hmc()
        np.save(test_samples_file, samples[:, -1])
