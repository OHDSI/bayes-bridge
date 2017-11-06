import numpy as np
import scipy as sp
import scipy.linalg
import math
import warnings

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

    tau = math.exp(log_tau)
    beta_samples, sigma_sq_samples, lam_samples, _ = \
        sampler(tau, lam0, n_burnin, n_samples)
    p = np.size(beta_samples, 0)

    # Contributions from the likelihood \pi(\beta | \lam, \tau)
    sq_norm_samples = np.sum((beta_samples / lam_samples) ** 2, 0) \
                      / sigma_sq_samples
    grad_samples = sq_norm_samples / tau ** 2 - p
    grad = np.mean(grad_samples)
    cov_grad = np.var(grad_samples)
    hess = - 2 / tau ** 2 * np.mean(sq_norm_samples) + cov_grad

    # Return a sample of lam to feed into the next iteration.
    lam = lam_samples[:, -1]

    return grad, cov_grad, hess, lam


def mcem(sampler, log_tau, lam0, n_burnin, n_samples):
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
    """

    tau = math.exp(log_tau)
    beta_samples, sigma_sq_samples, lam_samples, _ = \
        sampler(tau, lam0, n_burnin, n_samples)
    p = np.size(beta_samples, 0)
    sq_norm_samples = np.sum((beta_samples / lam_samples) ** 2, 0) / sigma_sq_samples
    tau = math.sqrt(np.mean(sq_norm_samples) / p)
    log_tau = math.log(tau)
    lam = lam_samples[:, -1]
    return log_tau, lam


def gibbs(y, X, n_burnin, n_post_burnin, thin, fixed_tau=False,
          tau0=None, lam0=None):
    """
    MCMC implementation for Bayesian linear regression with the horseshoe prior.
    Based on code by Antik Chakraborty, Anirban Bhattacharya, and James Johndrow
    Modified and ported to Python by Akihiko Nishimura

    Model: y = X\beta + \epslion, \epsilon \sim N(0, \sigma^2) 
           \beta_j \sim N(0, \sigma^2 \lambda_j^2 \tau^2) 
           \lambda_j \sim Half-Cauchy(0, 1),
           \tau \sim Half-Cauchy (0, 1), 
           \pi(\sigma^2) \sim 1/\sigma^2 


    This function implements the algorithm proposed in "Scalable MCMC for Bayes
    Shrinkage Priors" by Johndrow and Orenstein (2017). The local scale
    parameters are updated via a slice sampling scheme given in the online
    supplement of "The Bayesian Bridge" by Polson et. al. (2011).

    Input: y = response, a n * 1 vector
           X = matrix of covariates, dimension n * p
           n_burnin = number of burnin MCMC samples
           n_post_burnin = number of posterior draws to be saved
           thin = thinning parameter of the chain
           fixed_tau = if true, tau will not be updated.
           lam0 = the initial value for MCMC
    """


    n_iter = n_burnin + n_post_burnin
    n_sample = math.ceil(n_post_burnin / thin)  # Number of samples to keep
    n, p = np.shape(X)

    # Hyper-params on the prior for sigma_sq. Jeffrey's prior would be a0 = b0 = 0.
    a0 = .5
    b0 = .5

    # The value proposed as a good default in Johndrow and Orenstein (2017).
    metropolis_sd = .8

    # Initial state of the Markov chain
    beta = np.zeros(p)  # Unused with the current gibbs update order.
    sigma_sq = 1  # Unused with the current gibbs update order.
    if lam0 is not None:
        lam = lam0
    else:
        lam = np.ones(p)
    if tau0 is not None:
        tau = tau0
    else:
        tau = 1
    xi = tau ** -2
    eta = lam ** -2

    # Pre-allocate
    beta_samples = np.zeros((p, n_sample))
    lam_samples = np.zeros((p, n_sample))
    tau_samples = np.zeros(n_sample)
    sigma_sq_samples = np.zeros(n_sample)
    accepted = np.zeros(n_post_burnin + n_burnin)
    I_n = np.eye(n)

    # Start Gibbs sampling
    for i in range(n_iter):

        LX = lam[:, np.newaxis] ** 2 * X.T
        XLX = np.dot(X, LX)

        # Update tau
        M = I_n + (1 / xi) * XLX
        M_chol = sp.linalg.cholesky(M)
        if not fixed_tau:
            prop_xi = xi * math.exp(metropolis_sd * np.random.randn())
            try:
                M_chol_prop = sp.linalg.cholesky(I_n + (1 / prop_xi) * XLX)
            except:
                warnings.warn(
                    'Proposed value yields a non-positive-definite matrix, '
                    + 'rejectiong.', RuntimeWarning)
                break
            logp_curr = compute_beta_logp(M_chol, y, xi, a0, b0)
            logp_prop = compute_beta_logp(M_chol_prop, y, prop_xi, a0, b0)

            accepted[i] = np.random.rand() < \
                math.exp(logp_prop - logp_curr + math.log(prop_xi) - math.log(xi))
            if accepted[i]:
                xi = prop_xi
                M_chol = M_chol_prop

        tau = 1 / math.sqrt(xi)

        # Update sigma_sq with beta maginalized out
        ssr = np.dot(y, sp.linalg.cho_solve((M_chol, False), y))
        sigma_sq = 1 / np.random.gamma((n + a0) / 2, 2 / (ssr + b0))

        # Alternative update of sigma_sq conditional on beta
        """
        if trunc
            E_1 = max((y - X * Beta)' * (y - X * Beta), (1e-10)) # for numerical stability
            E_2 = max(sum(Beta.^2 ./ ((tau * lam)).^2), (1e-10))
        else
            E_1 = (y - X * Beta)' * (y - X * Beta) E_2 = sum(Beta.^2 ./ ((tau * lam)).^2)
        end

        sigma_sq = 1 / gamrnd((n + p + a0) / 2, 2 / (b0 + E_1 + E_2))
        """

        # Update beta
        D = (tau * lam) ** 2
        beta = math.sqrt(sigma_sq) \
               * generate_gaussian(y / math.sqrt(sigma_sq), X, D, M_chol, True)

        # Update lam_j's using slice sampling
        u = np.random.rand(p) / (eta + 1)
        gamma_rate = (beta ** 2) * xi / (2 * sigma_sq)
        upper_bd = (1 - u) / u
        eta = quantile_truncated_exp(np.random.rand(p), gamma_rate, upper_bd) # inverse CDF method
        if np.any(eta <= 0):
            print("Eta underflowed, replacing with machine epsilon.")
            eta[eta <= 0] = np.finfo(float).eps
        lam = 1 / np.sqrt(eta)

        # (theoretically) equivalent way to sample lam_j's, but supposedly
        # not as numerically stable.
        """
        eta = 1 ./ (lam.^2)
        upsi = unifrnd(0, 1 ./ (1 + eta))
        tempps = beta.^2 / (2 * sigma_sq * tau^2)
        ub = (1 - upsi) ./ upsi

        # now sample eta from exp(tempv) truncated between 0 & upsi / (1 - upsi)
        Fub = 1 - exp( - tempps .* ub) # exp cdf at ub
        Fub(Fub < (1e-4)) = 1e-4  # for numerical stability
        up = unifrnd(0, Fub)
        eta = -log(1 - up) ./ tempps
        lam = 1 ./ sqrt(eta)
        """

        if i >= n_burnin and i % thin == 0:
            index = math.floor((i - n_burnin) / thin)
            beta_samples[:, index] = beta
            lam_samples[:, index] = lam
            tau_samples[index] = tau
            sigma_sq_samples[index] = sigma_sq

    return beta_samples, sigma_sq_samples, lam_samples, tau_samples


def compute_beta_logp(M_chol, y, xi, a0, b0):
    """ Computes the log posterior conditional of 'beta' given 'lam'. """
    n = len(y)
    x = sp.linalg.cho_solve((M_chol, False), y)
    ssr = np.dot(y, x) + b0
    ldetM = 2 * np.sum(np.log(np.diag(M_chol)))
    loglik = - .5 * ldetM - ((n + a0) / 2) * math.log(ssr)
    log_prior = - np.log(math.sqrt(xi) * (1 + xi))
    logp = loglik + log_prior

    return logp


def generate_gaussian(y, X, D, A, is_chol):
    """
    Generate a multi-variate Gaussian with the mean mu and covariance Sigma of the form
       Sigma = (X'X + D^{-1})^{-1}, mu = Sigma X' y
    where D is assumed to be diagonal.

    Param:
    ------
        A : matrix
            Equals the matrix (XDX' + I) or, if is_chol == true, its cholesky decomposition
    """

    n, p = np.shape(X)
    u = np.sqrt(D) * np.random.randn(p)
    v = np.dot(X, u) + np.random.randn(n)
    if is_chol:
        w = sp.linalg.cho_solve((A, False), y - v)
    else:
        w = np.linalg.solve(A, y - v)
    x = u + D * np.dot(X.T, w)
    return x


def quantile_truncated_exp(x, rate, upper_bd):
    """ Computes the inverse cdf function of a truncated exponential
    distribution. Coordinates with small rates are treated separately for
    numerical stability. """

    small = np.abs(upper_bd * rate) < np.finfo(float).eps
    not_small = np.logical_not(small)
    tmp = np.zeros(len(rate)) # intermediate value
    tmp[small] = np.expm1( -upper_bd[small] * rate[small]) * x[small]
    tmp[not_small] = \
        (np.exp( -upper_bd[not_small] * rate[not_small]) - 1) * x[not_small]

    small = np.abs(tmp) < np.finfo(float).eps
    not_small = np.logical_not(small)
    q = np.zeros(len(rate))
    q[small] = - np.log1p(tmp[small]) / rate[small]
    q[not_small] = - np.log(1 + tmp[not_small]) / rate[not_small]

    return q

