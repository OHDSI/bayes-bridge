import numpy as np
import scipy as sp
import scipy.stats
import scipy.linalg
import math
import warnings
from pypolyagamma import PyPolyaGamma
pg = PyPolyaGamma()


def gibbs(y, X, n_burnin, n_post_burnin, thin, tau_fixed=False,
          tau0=None, lam0=None, link='gaussian'):
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
    n_sample = math.ceil(n_post_burnin / thin)  # Number of samples to keep
    n, p = np.shape(X)
    if link == 'logit':
        n_trial = np.ones(n)
        kappa = y - n_trial / 2
        omega = n_trial / 2
    X = np.hstack((np.ones((n, 1)), X))  # Add an intercept term.

    # Hyper-parameters
    global_scale = 1 # scale of the half-Cauchy prior on 'tau'

    # Initial state of the Markov chain
    beta = np.zeros(p + 1)
    sigma_sq = 1
    if lam0 is not None:
        lam = lam0
    else:
        lam = np.ones(p)
    if tau0 is not None:
        tau = tau0
    else:
        tau = 1

    # Pre-allocate
    samples = {
        'beta': np.zeros((p + 1, n_sample)),
        'lambda': np.zeros((p, n_sample)),
        'tau': np.zeros(n_sample)
    }
    if link == 'gaussian':
        samples['sigma_sq'] = np.zeros(n_sample)

    # Start Gibbs sampling
    for i in range(n_iter):

        # Update beta and related parameters.
        if link == 'gaussian':
            beta = sample_gaussian_posterior(y, X, lam, tau, np.ones(n) / sigma_sq)
            resid = y - np.dot(X, beta)
            scale = np.sum(resid ** 2) / 2
            sigma_sq = scale / np.random.gamma(n / 2, 1)
        elif link == 'logit':
            pg.pgdrawv(n_trial, np.dot(X, beta), omega)
            beta = sample_gaussian_posterior(kappa / omega, X, lam, tau, omega)
        else:
            raise NotImplementedError(
                'The specified link function is not supported.')

        # Draw from \tau | \beta and then \lambda | \tau, \beta. (The order matters.)
        if not tau_fixed:
            shape = p + 1
            scale = 1 / np.sum(np.abs(beta[1:]))
            tau = update_global_shrinkage(tau, beta[1:], global_scale)

        lam_sq = 1 / np.random.wald(mean=np.abs(tau / beta[1:]), scale=1)
        lam = np.sqrt(lam_sq)

        if i >= n_burnin and i % thin == 0:
            index = math.floor((i - n_burnin) / thin)
            samples['beta'][:, index] = beta
            samples['lambda'][:, index] = lam
            samples['tau'][index] = tau
            if link == 'gaussian':
                samples['sigma_sq'][index] = sigma_sq

    return samples

def sample_gaussian_posterior(y, X, lam, tau, omega):
    """
    Sample from a Gaussian with a precision Phi and mean mu such that
        Phi = X.T * diag(omega) * X + (tau * diag(float('inf'), lam)) ** -2
        mu = inv(Phi) * X.T * omega * y
    For numerical stability, the code first sample from the scaled parameter
    beta / precond_scale.
    """

    p = lam.size
    lam_aug = np.concatenate(([np.sum(omega) ** (- 1 / 2) / tau], lam))
    X_lam = X * lam_aug[np.newaxis, :]
    Phi_scaled = np.diag(np.concatenate(([0], np.ones(p)))) \
                  + tau ** 2 * np.dot(X_lam.T, omega[:, np.newaxis] * X_lam)
    Phi_scaled_chol = sp.linalg.cholesky(Phi_scaled)
    mu = sp.linalg.cho_solve((Phi_scaled_chol, False),
                             tau * lam_aug * np.dot(X.T, omega * y))
    beta_scaled = mu \
        + sp.linalg.solve_triangular(Phi_scaled_chol, np.random.randn(p + 1), lower=False)
    beta = tau * lam_aug * beta_scaled

    return beta

def generate_gaussian(y, X, D, A=None, is_chol=False):
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