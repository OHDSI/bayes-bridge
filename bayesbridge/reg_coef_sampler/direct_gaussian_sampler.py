import numpy as np
import scipy as sp

def generate_gaussian_with_weight(design, obs_prec, prior_prec_sqrt, z, rand_gen=None):
    """
    Generate a multi-variate Gaussian with
        mean = Sigma z,
        covariance^{-1} = X' diag(obs_prec) X + diag(prior_prec_sqrt) ** 2,
    where X is the `design` matrix.

    Parameters
    ----------
        obs_prec : 1-d numpy array
        prior_prec_sqrt : 1-d numpy array
    """

    Prec_precond, precond_scale \
        = compute_jacobi_precond_post_prec(design, obs_prec, prior_prec_sqrt)
    Prec_precond_chol = sp.linalg.cholesky(Prec_precond, precond_scale)
    mean_precond = sp.linalg.cho_solve(
        (Prec_precond_chol, False), precond_scale * z
    )
    if rand_gen is None:
        gaussian_vec = np.random.randn(design.shape[1])
    else:
        gaussian_vec = rand_gen.np_random.randn(design.shape[1])
    sample_precond = mean_precond
    sample_precond += sp.linalg.solve_triangular(
        Prec_precond_chol, gaussian_vec, lower=False
    )
    sample = precond_scale * sample_precond

    return sample

def compute_jacobi_precond_post_prec(design, obs_prec, prior_prec_sqrt):
    diag = prior_prec_sqrt ** 2 \
        + design.compute_fisher_info(weight=obs_prec, diag_only=True)
    precond_scale = 1 / np.sqrt(diag)
    Prec_precond = \
        precond_scale[:, np.newaxis] \
        * design.compute_fisher_info(obs_prec) \
        * precond_scale[np.newaxis, :]
    Prec_precond += np.diag((precond_scale * prior_prec_sqrt) ** 2)
    return Prec_precond, precond_scale
