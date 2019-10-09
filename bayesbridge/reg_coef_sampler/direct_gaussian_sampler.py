import numpy as np
import scipy as sp
import scipy.sparse

def generate_gaussian_with_weight(X, omega, D, z, rand_gen=None):
    """
    Generate a multi-variate Gaussian with the mean mu and covariance Sigma of the form
       Sigma^{-1} = X' diag(omega) X + D^2, mu = Sigma z
    where D is assumed to be diagonal.

    Param:
    ------
        omega : vector
        D : vector
    """

    diag_sqrt = D.copy()
    fisher_info = X.compute_fisher_info(weight=omega, diag_only=True)
    fisher_info[fisher_info < 0.] = 0.
    fisher_info_sqrt = np.sqrt(fisher_info)
    diag_sqrt *= np.sqrt(1 + (fisher_info_sqrt / diag_sqrt) ** 2)
    diag_sqrt[D == 0] = fisher_info_sqrt[D == 0]
    inv_sqrt_diag_scale = 1 / diag_sqrt
    Phi_scaled = inv_sqrt_diag_scale[:, np.newaxis] \
        * X.compute_fisher_info(omega) * inv_sqrt_diag_scale[np.newaxis, :]
    Phi_scaled += np.diag((inv_sqrt_diag_scale * D) ** 2)
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
