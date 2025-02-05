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

def generate_gaussian_via_woodbury(design, obs_prec, prior_prec_sqrt, z):
    """
    Sample from a multi-variate Gaussian using the identity
        (D + X' W X)^{-1}
            = D^{-1} - D^{-1} X' (W^{-1} + X D^{-1} X')^{-1} X D^{-1}.
    """
    if np.any(prior_prec_sqrt[design.intercept_added:] == 0):
        raise NotImplementedError(
            "Woodbury sampler currently does not support flat prior on fixed effects."
        )

    # Draw a "target" vector, right-hand side of the linear system to be solved.
    randn_vec_1 = np.random.randn(design.shape[0])
    randn_vec_2 = np.random.randn(design.shape[1])
    v = design.Tdot(obs_prec ** (1 / 2) * randn_vec_1) \
        + prior_prec_sqrt * randn_vec_2
    rhs_target_vec = (z + v)
    sample = matvec_by_post_prec_inverse_via_woodbury(
        design, obs_prec, prior_prec_sqrt, rhs_target_vec
    )
    return sample

def matvec_by_post_prec_inverse_via_woodbury(design, obs_prec, prior_prec_sqrt, x):
    D_inv = prior_prec_sqrt[design.intercept_added:] ** -2
    # Matrix W^{-1} + X D^{-1} X', which is to be inverted.
    reduced_mat = \
        np.diag(obs_prec ** - 1) \
        + design.compute_transposed_fisher_info(weight=D_inv)
    precond_scale = 1 / np.diag(reduced_mat)
    precond_reduced_mat_cho_factor = sp.linalg.cho_factor(
        precond_scale[:, np.newaxis] * reduced_mat * precond_scale[np.newaxis, :]
    )

    def matvec_by_post_prec_inverse_via_reduced_mat_cho_factor(x):
        result = precond_scale * design.main_dot(D_inv * x)
        result = sp.linalg.cho_solve(precond_reduced_mat_cho_factor, result)
        result *= precond_scale
        result = D_inv * design.main_Tdot(result)
        result = D_inv * x - result
        return result

    if design.intercept_added:
        return matvec_via_block_inverse(
            prior_prec_sqrt[0] ** 2 + np.sum(obs_prec),
            design.main_Tdot(obs_prec),
            matvec_by_post_prec_inverse_via_reduced_mat_cho_factor,
            x
        )
    else:
        return matvec_by_post_prec_inverse_via_reduced_mat_cho_factor(x)

def matvec_via_block_inverse(block11, block12, block22_inv_matvec, x):
    """
    Apply the inverse of a 2x2 symmetic block matrix with scalar `block11` to
    a vector `x` using the block inversion formula.

    The result of a block matrix inversion
        [B11  B12]^{-1} [x1]
        [B21  B22]      [x2]
    is computed via
        res1 = S^{-1} (x1 - B12.T @ B22^{-1} x2)
        res2 = B22^{-1} (x2 - B21 res1)
    where S is the Schur complement given as
        S = B11 - B12.T @ B22^{-1} @ B12

    Parameters
    ----------
    block11 : float
        The (1,1) block "B11" with block size assumed 1x1.
    block12 : ndarray
        The (1,2) block "B12"
    block22_inv_matvec : callable
        A function that returns `B22^{-1} @ v` for any input vector `v`.
    x : ndarray
        Right-hand side vector, partitioned as `x = [x1; x2]`.

    Returns
    -------
    result : ndarray
    """
    x1, x2 = x[:1], x[1:]
    block21 = block12  # Symmetric assumption: B21 = B12.T

    block22_inv_x2 = block22_inv_matvec(x2)
    block22_inv_block21 = block22_inv_matvec(block21)
    schur_complement = block11 - block21.T @ block22_inv_block21

    result1 = (x1 - block12.T @ block22_inv_x2) / schur_complement
    result2 = block22_inv_x2 - block22_inv_block21 * result1

    return np.concatenate([result1, result2])