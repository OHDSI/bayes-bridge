import numpy as np
import scipy as sp
import scipy.linalg
import math
import warnings
from pypolyagamma import PyPolyaGamma
pg = PyPolyaGamma()

def grad_marginal(sampler, log_tau, lam0, n_burnin, n_samples):
    """
    Returns a Monte Carlo estimate of the gradient of log marginal
    distribution of 'tau'.

    Params
    ------
      sampler(tau, lam0, n_burnin, n_samples): callable
          Returns the samples of 'beta' and 'lam' drawn from the posterior
          conditional distribution.
      tau: float
          The value at which the gradient will be evaluated.
      lam0: vector
          The initial value of local shrinkage parameters for the Gibbs
          sampler.
    """

    # TODO: Update the code to accommodate other values of horseshoe
    # hyper-parameters other than all 1's.

    tau = math.exp(log_tau)
    samples = sampler(tau, lam0, n_burnin, n_samples)
    p = np.size(samples['beta'], 0)

    # Contributions from the likelihood \pi(\beta | \lam, \tau)
    sq_norm_samples = np.sum((samples['beta'] / samples['lambda']) ** 2, 0)
    grad_samples = sq_norm_samples / tau ** 2 - p
    grad = np.mean(grad_samples)
    cov_grad = np.var(grad_samples)
    hess = - 2 / tau ** 2 * np.mean(sq_norm_samples) + cov_grad

    # Contributions from the half-Cauchy prior on 'tau'.
    grad += (1 - tau ** 2) / (1 + tau ** 2)
    hess += - 4 * tau ** 2 / (1 + tau ** 2) ** 2

    # Return a sample of lam to feed into the next iteration.
    lam = samples['lambda'][:, -1]

    return grad, cov_grad, hess, lam


def mcem(sampler, log_tau, lam0, n_burnin, n_samples, include_prior=True):
    """
    Update tau via Monte Carlo EM.

    Params
    ------
      sampler(tau, lam0, n_burnin, n_samples): callable
          Returns the samples of 'beta' and 'lam' drawn from the posterior
          conditional distribution.
      tau: float
          The value with respect to which the incomplete log-likelihood is
          computed.
      lam0: vector
          The initial value of local shrinkage parameters for the Gibbs
          sampler.
      include_prior: bool
          If False, the algorithm tries to maximize only the marginal
          likelihood ignoring the half-Cauchy prior.
    """

    # TODO: Update the code to accommodate other values of horseshoe
    # hyper-parameters other than all 1's.

    tau = math.exp(log_tau)
    samples = sampler(tau, lam0, n_burnin, n_samples)
    p = np.size(samples['beta'], 0)
    sq_norm_samples = np.sum((samples['beta'] / samples['lambda']) ** 2, 0)

    if include_prior:
        tau = math.sqrt(np.mean(sq_norm_samples) / p)
    else:
        a = 1 + p
        b = - (1 - p + np.mean(sq_norm_samples))
        c = - np.mean(sq_norm_samples)
        tau_sq = (- b + math.sqrt(b ** 2 - 4 * a * c)) / 2 / a
        tau = math.sqrt(tau_sq)

    log_tau = math.log(tau)
    lam = samples['lambda'][:, -1]
    return log_tau, lam


def gibbs(y, X, n_burnin, n_post_burnin, thin, tau_fixed=False,
          tau0=None, lam0=None, link='gaussian'):
    """
    MCMC implementation for Bayesian Lasso. Note that the names of 'lambda'
    (local parameter here) and 'tau' (global parameter here) are often switched
    in the literature.

    Model: y = X\beta + \epslion, \epsilon \sim N(0, \sigma^2) 
           \beta_j \sim N(0, \sigma^2 \lambda_j^2 \tau^2)
           \lambda_j \sim Half-Cauchy(0, 1),
           \tau \sim Half-Cauchy (0, 1), 
           \pi(\sigma^2) \sim 1 / \sigma^2


    Input: y = response, a n * 1 vector
           X = matrix of covariates, dimension n * p
           n_burnin = number of burnin MCMC samples
           n_post_burnin = number of posterior draws to be saved
           thin = thinning parameter of the chain
           fixed_penalty = if true, the penalty parameter will not be updated.
           tau0 = the initial value for MCMC
    """

    n_iter = n_burnin + n_post_burnin
    n_sample = math.ceil(n_post_burnin / thin)  # Number of samples to keep
    n, p = np.shape(X)
    if link == 'logit':
        n_trial = np.ones(n)
        kappa = y - n_trial / 2
        omega = n_trial / 2

    # Hyper-parameters
    df_local = 1
    df_global = 1
    global_scale = 1 # scale of the t-distribution prior on 'tau'

    # Initial state of the Markov chain
    beta = np.zeros(p)
    sigma_sq = 1
    if lam0 is not None:
        lam = lam0
    else:
        lam = np.ones(p)
    nu = np.random.gamma((df_local + 1) / 2, 1, size=p) \
         / (1 + df_local / lam ** 2)
        # an auxiliary parameter for sampling 'lam'
    if tau0 is not None:
        tau = tau0
    else:
        tau = 1
    xi = np.random.gamma((df_global + 1) / 2, 1) \
         / (1 + global_scale ** 2 * df_global / tau ** 2)
        # an auxiliary parameter for sampling 'tau'

    # Pre-allocate
    samples = {
        'beta': np.zeros((p, n_sample)),
        'lambda': np.zeros((p, n_sample)),
        'tau': np.zeros(n_sample)
    }
    if link == 'gaussian':
        samples['sigma_sq'] = np.zeros(n_sample)

    # Start Gibbs sampling
    for i in range(n_iter):

        # Update beta and related parameters.
        if link == 'gaussian':
            D = (tau * lam) ** 2 / sigma_sq
            beta = math.sqrt(sigma_sq) \
                   * generate_gaussian(y / math.sqrt(sigma_sq), X, D)
            resid = y - np.dot(X, beta)
            scale = np.sum(resid ** 2) / 2
            sigma_sq = scale / np.random.gamma(n / 2, 1)
        elif link == 'logit':
            pg.pgdrawv(n_trial, np.dot(X, beta), omega)
            D = (tau * lam) ** 2
            beta = generate_gaussian(
                kappa / np.sqrt(omega), np.sqrt(omega)[:, np.newaxis] * X, D)

        else:
            raise NotImplementedError(
                'The specified link function is not supported.')

        # Update local shrinkage parameters via parameter expansion
        # TODO: this data augmentation strategy seems to not-so-good mixing
        # property. Run several iterations or replace with slice sampler.
        scale = df_local * nu + beta ** 2 / 2 / tau ** 2
        lam_sq = scale / np.random.gamma((df_local + 1) / 2, 1, size=p)
        nu = np.random.gamma((df_local + 1) / 2, 1, size=p) \
             / (1 + df_local / lam_sq)
        lam = np.sqrt(lam_sq)

        # Update the global shrinkage parameter
        if not tau_fixed:
            scale = global_scale ** 2 * df_global * xi \
                    + np.sum((beta / lam) ** 2) / 2  # inverse-gamma scale
            tau_sq = scale / np.random.gamma((df_global + p) / 2, 1)
            xi = np.random.gamma((df_global + 1) / 2, 1) \
                 / (1 + global_scale ** 2 * df_global / tau_sq)
            tau = math.sqrt(tau_sq)

        if i >= n_burnin and i % thin == 0:
            index = math.floor((i - n_burnin) / thin)
            samples['beta'][:, index] = beta
            samples['lambda'][:, index] = lam
            samples['tau'][index] = tau
            if link == 'gaussian':
                samples['sigma_sq'][index] = sigma_sq

    return samples


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
