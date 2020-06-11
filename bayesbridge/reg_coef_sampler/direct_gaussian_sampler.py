import numpy as np
import scipy as sp
import scipy.sparse

def generate_gaussian_with_weight(X, obs_prec, prior_prec_sqrt, z, rand_gen=None):
    """
    Generate a multi-variate Gaussian with the mean mu and covariance Sigma of the form
        mu = Sigma z,
        Sigma^{-1} = X' diag(obs_prec) X + diag(prior_prec_sqrt) ** 2,

    Parameters
    ----------
        obs_prec : 1-d numpy array
        prior_prec_sqrt : 1-d numpy array
    """

    diag_sqrt = prior_prec_sqrt.copy()
    fisher_info = X.compute_fisher_info(weight=obs_prec, diag_only=True)
    fisher_info[fisher_info < 0.] = 0.
    fisher_info_sqrt = np.sqrt(fisher_info)
    has_pos_prior_prec = (prior_prec_sqrt > 0)
    has_zero_prior_prec = np.logical_not(has_pos_prior_prec)
    diag_sqrt[has_pos_prior_prec] *= np.sqrt(
        1. + (fisher_info_sqrt[has_pos_prior_prec] / diag_sqrt[has_pos_prior_prec]) ** 2
    )
    diag_sqrt[has_zero_prior_prec] = fisher_info_sqrt[has_zero_prior_prec]
    inv_sqrt_diag_scale = 1 / diag_sqrt
    Phi_scaled = inv_sqrt_diag_scale[:, np.newaxis] \
        * X.compute_fisher_info(obs_prec) * inv_sqrt_diag_scale[np.newaxis, :]
    Phi_scaled += np.diag((inv_sqrt_diag_scale * prior_prec_sqrt) ** 2)
    Phi_scaled_chol = sp.linalg.cholesky(Phi_scaled)
    mu = sp.linalg.cho_solve((Phi_scaled_chol, False), inv_sqrt_diag_scale * z)
    if rand_gen is None:
        gaussian_vec = np.random.randn(X.shape[1])
    else:
        gaussian_vec = rand_gen.np_random.randn(X.shape[1])
    beta_scaled = mu + sp.linalg.solve_triangular(
        Phi_scaled_chol, gaussian_vec, lower=False
    )
    beta = inv_sqrt_diag_scale * beta_scaled

    return beta
