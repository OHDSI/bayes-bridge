import numpy as np
import scipy as sp
import scipy.stats
import scipy.linalg
import scipy.sparse
import math
import warnings
from pypolyagamma import PyPolyaGamma
pg = PyPolyaGamma()


def gibbs(y, X, n_burnin, n_post_burnin, thin, tau_fixed=False,
          init={}, link='gaussian'):
    """
    MCMC implementation for the Bayesian lasso.

    Model: y = X\beta + \epslion, \epsilon \sim N(0, \sigma^2) 
           \beta_j \sim N(0, \sigma^2 \lambda_j^2 \tau^2)
           \lambda_j^2 \sim Exp(2),
           \tau \sim Half-Cauchy(0, global_scale^2),
           \pi(\sigma^2) \sim 1 / \sigma^2


    Input: y = response, a n * 1 vector
           X = matrix of covariates, dimension n * p
           n_burnin = number of burnin MCMC samples
           n_post_burnin = number of posterior draws to be saved
           thin = thinning parameter of the chain
           tau_fixed = if true, the penalty parameter will not be updated.
           tau0 = the initial value for MCMC
    """

    n_iter = n_burnin + n_post_burnin
    n_sample = math.floor(n_post_burnin / thin)  # Number of samples to keep
    n, p = np.shape(X)
    if link == 'logit':
        n_trial = np.ones(n)
    else:
        n_trial = None
    X = sp.sparse.hstack((np.ones((n, 1)), X))  # Add an intercept term.
    X_csr = X.tocsr()
    X_csc = X.tocsc()

    # Hyper-parameters
    global_scale = 1 # scale of the half-Cauchy prior on 'tau'

    # Initial state of the Markov chain
    beta, sigma_sq, omega, lam, tau  = \
        initialize_chain(init, p, link, n_trial)

    # Pre-allocate
    samples = {
        'beta': np.zeros((p + 1, n_sample)),
        'lambda': np.zeros((p, n_sample)),
        'tau': np.zeros(n_sample)
    }
    if link == 'gaussian':
        samples['sigma_sq'] = np.zeros(n_sample)
    elif link == 'logit':
        samples['omega'] = np.zeros((n, n_sample))

    # Start Gibbs sampling
    for i in range(1, n_iter + 1):

        # Update beta and related parameters.
        if link == 'gaussian':
            omega = np.ones(n) / sigma_sq
            beta = update_beta(y, X_csr, X_csc, omega, tau, lam)
            resid = y - X_csr.dot(beta)
            scale = np.sum(resid ** 2) / 2
            sigma_sq = scale / np.random.gamma(n / 2, 1)
        elif link == 'logit':
            pg.pgdrawv(n_trial, X_csr.dot(beta), omega)
            beta = update_beta((y - n_trial / 2) / omega, X_csr, X_csc,
                               omega, tau, lam)
        else:
            raise NotImplementedError(
                'The specified link function is not supported.')

        # Draw from \tau | \beta and then \lambda | \tau, \beta. (The order matters.)
        if not tau_fixed:
            tau = update_global_shrinkage(tau, beta[1:], global_scale)

        lam_sq = 1 / np.random.wald(mean=np.abs(tau / beta[1:]), scale=1)
        lam = np.sqrt(lam_sq)

        if i > n_burnin and (i - n_burnin) % thin == 0:
            index = math.floor((i - n_burnin) / thin) - 1
            samples['beta'][:, index] = beta
            samples['lambda'][:, index] = lam
            samples['tau'][index] = tau
            if link == 'gaussian':
                samples['sigma_sq'][index] = sigma_sq
            elif link == 'logit':
                samples['omega'][:, index] = omega

    return samples


def initialize_chain(init, p, link, n_trial):
    # Choose the user-specified state if provided, the default ones otherwise.

    if 'beta' in init:
        beta = init['beta']
    else:
        beta = np.zeros(p + 1)
        if 'intercept' in init:
            beta[0] = init['intercept']

    if 'sigma' in init:
        sigma_sq = init['sigma'] ** 2
    else:
        sigma_sq = 1

    if 'omega' in init:
        omega = np.ascontiguousarray(np.init['omega'])
            # Cython requires a C-contiguous array.
    elif link == 'logit':
        omega = n_trial / 2
    else:
        omega = None

    if 'lambda' in init:
        lam = init['lambda']
    else:
        lam = np.ones(p)
        
    if 'tau' in init:
        tau = init['tau']
    else:
        tau = 1
        
    return beta, sigma_sq, omega, lam, tau


def sample_gaussian_posterior(y, X_csr, X_csc, prior_sd, omega):
    """
    Sample from a Gaussian with a precision Phi and mean mu such that
        Phi = X.T * diag(omega) * X + diag(0, prior_sd) ** -2
        mu = inv(Phi) * X.T * omega * y
    For numerical stability, the code first sample from the scaled parameter
    beta / precond_scale.
    """

    X = X_csr
    XT = X_csc.T
    n = X.shape[0]
    p = X.shape[1] - 1 #
    precond_scale = np.concatenate(([np.sum(omega) ** (- 1 / 2)], prior_sd))
    precond_scale_mat = sp.sparse.dia_matrix((precond_scale, 0), (p + 1, p + 1))
    omega_mat = sp.sparse.dia_matrix((omega, 0), (n, n))
    X_scaled = X.dot(precond_scale_mat)
    XT_scaled = precond_scale_mat.dot(XT)
    Phi_scaled = XT_scaled.dot(omega_mat.dot(X_scaled)).toarray() \
            + np.diag(np.concatenate(([0], np.ones(p))))
    Phi_scaled_chol = sp.linalg.cholesky(Phi_scaled)
    mu = sp.linalg.cho_solve((Phi_scaled_chol, False),
                             XT_scaled.dot(omega * y))
    beta_scaled = mu \
        + sp.linalg.solve_triangular(Phi_scaled_chol, np.random.randn(p + 1), lower=False)
    beta = precond_scale * beta_scaled

    return beta

def update_beta(y, X_csr, X_csc, omega, tau, lam):

    n = X_csr.shape[0]

    prior_sd = np.concatenate(([float('inf')], tau * lam))
        # Flat prior for intercept
    omega_sqrt = omega ** (1 / 2)
    omega_sqrt_mat = sp.sparse.dia_matrix((omega_sqrt, 0), (n, n))
    weighted_X = omega_sqrt_mat.dot(X_csr).tocsc()

    beta = generate_gaussian(weighted_X, 1 / prior_sd, X_csc.T.dot(omega * y))
    return beta


def generate_gaussian(X_csc, D, v):
    """
    Generate a multi-variate Gaussian with the mean mu and covariance Sigma of the form
       Sigma^{-1} = X' X + D^2, mu = Sigma * v
    where D is assumed to be diagonal.

    Param:
    ------
        D : vector
    """

    p = X_csc.shape[1] - 1

    Phi_diag = D ** 2 + np.squeeze(np.asarray(
        X_csc.power(2).sum(axis=0)
    ))
    precond_scale = 1 / np.sqrt(Phi_diag)
    precond_scale_mat = \
        sp.sparse.dia_matrix((precond_scale, 0), (p + 1, p + 1))
    X_scaled = X_csc.dot(precond_scale_mat)

    Phi_scaled = X_scaled.T.dot(X_scaled).toarray() \
          + np.diag((precond_scale * D) ** 2)
    Phi_scaled_chol = sp.linalg.cholesky(Phi_scaled)
    mu = sp.linalg.cho_solve((Phi_scaled_chol, False), precond_scale * v)
    beta_scaled = mu + sp.linalg.solve_triangular(
        Phi_scaled_chol, np.random.randn(p + 1), lower=False
    )

    beta = precond_scale * beta_scaled
    return beta


def update_global_shrinkage(tau, beta, global_scale):
    """ Update the global shrinkage parameter with slice sampling. """

    n_update = 10 # Slice sample for multiple iterations to ensure good mixing.

    # Initialize a gamma distribution object.
    shape = beta.size + 1
    scale = 1 / np.sum(np.abs(beta))
    gamma_rv = sp.stats.gamma(shape, scale=scale)

    # Slice sample phi = 1 / tau.
    phi = 1 / tau
    for i in range(n_update):
        u = np.random.uniform() / (1 + (global_scale * phi) ** 2)
        upper = np.sqrt(1 / u - 1) / global_scale  # Invert the half-Cauchy density.
        phi = gamma_rv.ppf(gamma_rv.cdf(upper) * np.random.uniform())
    tau = 1 / phi

    return tau


def generate_gaussian_with_weight(y, X_csr, omega, D):
    """
    Generate a multi-variate Gaussian with the mean mu and covariance Sigma of the form
       Sigma^{-1} = X' diag(omega) X + D^2, mu = Sigma X' (omega * y)
    where D is assumed to be diagonal.

    Param:
    ------
        omega : vector
        D : vector
    """

    n = X_csr.shape[0]
    p = X_csr.shape[1] - 1

    omega_sqrt = omega ** (1 / 2)
    omega_sqrt_mat = sp.sparse.dia_matrix((omega_sqrt, 0), (n, n))
    weighted_X = omega_sqrt_mat.dot(X_csr).tocsc()
    Phi_diag = D ** 2 + np.squeeze(np.asarray(
        weighted_X.power(2).sum(axis=0)
    ))
    precond_scale = 1 / np.sqrt(Phi_diag)
    precond_scale_mat = \
        sp.sparse.dia_matrix((precond_scale, 0), (p + 1, p + 1))

    weighted_X_scaled = weighted_X.dot(precond_scale_mat)
    Phi_scaled = weighted_X_scaled.T.dot(weighted_X_scaled).toarray() \
          + np.diag((precond_scale * D) ** 2)
    Phi_scaled_chol = sp.linalg.cholesky(Phi_scaled)
    mu = sp.linalg.cho_solve((Phi_scaled_chol, False),
                             weighted_X_scaled.T.dot(omega_sqrt * y))
    beta_scaled = mu + sp.linalg.solve_triangular(
        Phi_scaled_chol, np.random.randn(p + 1), lower=False
    )
    beta = precond_scale * beta_scaled
    return beta


def generate_gaussian_alla_anirban(y, X, D, A=None, is_chol=False):
    """
    Generate a multi-variate Gaussian with the mean mu and covariance Sigma of the form
       Sigma = (X'X + D^{-1})^{-1}, mu = Sigma X' y
    where D is assumed to be diagonal.

    Param:
    ------
        A : matrix
            Equals the matrix (XDX' + I) or, if is_chol == true, its cholesky decomposition
        D : vector
    """

    n, p = np.shape(X)
    if n > p:
        Phi = np.dot(X.T, X) + np.diag(D ** -1)
        Phi_chol = sp.linalg.cholesky(Phi)
        mu = sp.linalg.cho_solve((Phi_chol, False), np.dot(X.T, y))
        x = mu + sp.linalg.solve_triangular(Phi_chol, np.random.randn(p),
                                            lower=False)
    else:
        if A is None:
            A = np.dot(X, D[:, np.newaxis] * X.T) + np.eye(n)
        u = np.sqrt(D) * np.random.randn(p)
        v = np.dot(X, u) + np.random.randn(n)
        if is_chol:
            w = sp.linalg.cho_solve((A, False), y - v)
        else:
            w = sp.linalg.solve(A, y - v, sym_pos=True)
        x = u + D * np.dot(X.T, w)
    return x

