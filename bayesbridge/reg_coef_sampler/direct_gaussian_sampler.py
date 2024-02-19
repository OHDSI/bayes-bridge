import numpy as np
import scipy as sp

def generate_gaussian_with_weight(design, obs_prec, prior_prec_sqrt, z, rand_gen=None):
    """
    Generate a multi-variate Gaussian with covariance Sigma
        Sigma^{-1} = X diag(obs_prec) X + diag(prior_prec_sqrt) ** 2
    and mean = Sigma z, where X is the `design` matrix.

    Parameters
    ----------
        obs_prec : 1-d numpy array
        prior_prec_sqrt : 1-d numpy array
    """

    diag = prior_prec_sqrt ** 2 \
           + design.compute_fisher_info(weight=obs_prec, diag_only=True)
    jacobi_precond_scale = 1 / np.sqrt(diag)
    Prec_precond = compute_precond_post_prec(
        design, obs_prec, prior_prec_sqrt, jacobi_precond_scale
    )
    Prec_precond_chol = sp.linalg.cholesky(Prec_precond, jacobi_precond_scale)
    mean_precond = sp.linalg.cho_solve(
        (Prec_precond_chol, False), jacobi_precond_scale * z
    )
    if rand_gen is None:
        gaussian_vec = np.random.randn(design.shape[1])
    else:
        gaussian_vec = rand_gen.np_random.randn(design.shape[1])
    sample_precond = mean_precond
    sample_precond += sp.linalg.solve_triangular(
        Prec_precond_chol, gaussian_vec, lower=False
    )
    sample = jacobi_precond_scale * sample_precond

    return sample

def compute_precond_post_prec(design, obs_prec, prior_prec_sqrt, precond_scale):
    Prec_precond = \
        precond_scale[:, np.newaxis] \
        * design.compute_fisher_info(obs_prec) \
        * precond_scale[np.newaxis, :]
    Prec_precond += np.diag((precond_scale * prior_prec_sqrt) ** 2)
    return Prec_precond
