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
    if np.any(prior_prec_sqrt == 0):
        raise NotImplementedError(
            "Woodbury sampler currently does not support flat prior on intercept "
            "nor on fixed effects."
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
    D_inv = prior_prec_sqrt ** -2
    to_be_inverted = \
        np.diag(obs_prec ** - 1) \
        + design.compute_transposed_fisher_info(weight=D_inv, include_intrcpt=True)
    result = solve_via_chol(to_be_inverted, design.dot(D_inv * x))
    result = D_inv * design.Tdot(result)
    result = D_inv * x - result
    return result

def solve_via_chol(pos_def_mat, x):
    # Use Jacobi preconditioner to improve numerical stability.
    precond_scale = 1 / np.diag(pos_def_mat)
    precond_mat = \
        precond_scale[:, np.newaxis] \
        * pos_def_mat \
        * precond_scale[np.newaxis, :]
    result = precond_scale * x
    result = sp.linalg.cho_solve(sp.linalg.cho_factor(precond_mat), result)
    result *= precond_scale
    return result