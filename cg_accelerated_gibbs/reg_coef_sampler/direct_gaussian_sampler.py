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

    omega_sqrt = omega ** (1 / 2)
    weighted_X = X.matmul_by_diag(omega_sqrt, from_='left')
    diag = D ** 2 + weighted_X.sqnorm(axis=0)
    inv_sqrt_diag_scale = 1 / np.sqrt(diag)
    weighted_X_scaled = \
        weighted_X.matmul_by_diag(inv_sqrt_diag_scale,
                                  from_='right', order='col_major')

    Phi_scaled = weighted_X_scaled.transpose().matdot(weighted_X_scaled)
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
