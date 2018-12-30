import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../..')

import numpy as np
from hamiltonian_monte_carlo import hmc
from hamiltonian_monte_carlo.tests.distributions import BivariateGaussian

data_folder = 'saved_outputs'
test_samples_file = 'hmc_bivariate_gaussian_samples.npy'


def test_hmc(request):
    samples = run_hmc()
    filepath = '/'.join([
        request.fspath.dirname, data_folder, test_samples_file
    ])
    prev_output = np.load(filepath)
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
        filepath = '/'.join([data_folder, test_samples_file])
        np.save(filepath, samples[:, -1])