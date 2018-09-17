from ..sparse_dense_matrix_operators \
    import elemwise_power, left_matmul_by_diag, right_matmul_by_diag
import numpy as np
import scipy as sp
import scipy.sparse


def generate_gaussian_with_weight(X_row_major, omega, D, z, rand_gen=None):
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
    weighted_X = left_matmul_by_diag(omega_sqrt, X_row_major)

    inv_sqrt_diag_scale = jacobi_precondition(D, omega, X_row_major)
    weighted_X_scaled = right_matmul_by_diag(weighted_X, inv_sqrt_diag_scale)

    Phi_scaled = weighted_X_scaled.T.dot(weighted_X_scaled)
    if sp.sparse.issparse(X_row_major):
        Phi_scaled = Phi_scaled.toarray()
    Phi_scaled += np.diag((inv_sqrt_diag_scale * D) ** 2)
    Phi_scaled_chol = sp.linalg.cholesky(Phi_scaled)
    mu = sp.linalg.cho_solve((Phi_scaled_chol, False), inv_sqrt_diag_scale * z)
    if rand_gen is None:
        gaussian_vec = np.random.randn(X_row_major.shape[1])
    else:
        gaussian_vec = rand_gen.np_random.randn(X_row_major.shape[1])
    beta_scaled = mu + sp.linalg.solve_triangular(
        Phi_scaled_chol, gaussian_vec, lower=False
    )
    beta = inv_sqrt_diag_scale * beta_scaled

    return beta


def jacobi_precondition(D, omega, X_row_major):

    diag = D ** 2 + np.squeeze(np.asarray(
        left_matmul_by_diag(
            omega, elemwise_power(X_row_major, 2)
        ).sum(axis=0)
    ))
    precond_scale = 1 / np.sqrt(diag)

    return precond_scale