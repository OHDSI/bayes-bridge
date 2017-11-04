import numpy as np
import scipy as sp
import scipy.linalg
import math
import warnings


def gibbs(y, X, n_burnin, n_post_burnin, thin, fixed_tau, tau, lam0=None):
    """
    Function to impelement Horseshoe shrinkage prior (http://faculty.chicagobooth.edu/nicholas.polson/research/papers/Horse.pdf)
    in Bayesian Linear Regression.
    Based on code by James Johndrow (johndrow@stanford.edu)
    Modified and ported to Python by Akihiko Nishimura

    Model: y = X\beta + \epslion, \epsilon \sim N(0, \sigma^2) 
           \beta_j \sim N(0, \sigma^2 \lambda_j^2 \tau^2) 
           \lambda_j \sim Half-Cauchy(0, 1), \tau \sim Half-Cauchy (0, 1) 
           \pi(\sigma^2) \sim 1/\sigma^2 


    This function implements the algorithm proposed in "Scalable MCMC for
    Bayes Shrinkage Priors" by Johndrow and Orenstein (2017).
    The global local scale parameters are updated via a Slice sampling scheme given in the online supplement
    of "The Bayesian Bridge" by Polson et. al. (2011). Setting ab = true
    implements the algorithm of Bhattacharya et al. Setting ab = false
    implements the algorith of Johndrow and Orenstein, which uses a block
    update for \tau, \sigma^2, \beta

    Input: y = response, a n * 1 vector 
           X = matrix of covariates, dimension n * p 
           n_burnin = number of burnin MCMC samples 
           n_post_burnin = number of posterior draws to be saved 
           thin = thinning parameter of the chain 
           and lower bound on MH proposals usually make this 1 and just make
           scl_ub = scl_lb only use for particularly challenging cases
           fixed_tau = if true, tau will not be updated.
           lam0 = the initial value for MCMC
    """
    

    n_iter = n_burnin + n_post_burnin
    n_sample = math.ceil(n_post_burnin / thin)  # Number of samples to keep
    n, p = np.shape(X)

    # Hyper-params on the prior for sigma_sq. Jeffrey's prior would be a0 = b0 = 0.
    a0 = .5
    b0 = .5

    # Stepsize of Metropolis. Apparently, .8 is a good default (Johndrow and Orenstein 2017).
    std_MH = .8

    # Initial state of the Markov chain
    beta = np.zeros(p)  # Unused with the current gibbs update order.
    sigma_sq = 1  # Unused with the current gibbs update order.
    if lam0 is not None:
        lam = lam0
    else:
        lam = np.ones(p)
    if not fixed_tau:
        tau = 1
    xi = tau ** -2
    eta = lam ** -2

    # Pre-allocate
    beta_samples = np.zeros((p, n_sample))
    lam_samples = np.zeros((p, n_sample))
    tau_samples = np.zeros(n_sample)
    sigma_sq_samples = np.zeros(n_sample)
    accept_prob = np.zeros(n_post_burnin + n_burnin)
    I_n = np.eye(n)

    # start Gibbs sampling #
    for i in range(n_iter):

        LX = lam[:, np.newaxis] ** 2 * X.T
        XLX = np.dot(X, LX)

        # update tau
        M = I_n + (1 / xi) * XLX
        M_chol = sp.linalg.cholesky(M)
        if not fixed_tau:
            prop_xi = xi * math.exp(std_MH * np.random.randn())
            try:
                M_chol_prop = sp.linalg.cholesky(I_n + (1 / prop_xi) * XLX)
            except:
                warnings.warn(
                    'Proposal rejected because of a non-positive-definite matrix.',
                    RuntimeWarning)
                break
            lrat_curr = lmh_ratio(M_chol, y, xi, n, a0, b0)
            lrat_prop = lmh_ratio(M_chol_prop, y, prop_xi, n, a0, b0)
            log_acc_rat = (lrat_prop - lrat_curr) + (
            math.log(prop_xi) - math.log(xi))

            accept_prob[i] = min(1, math.exp(log_acc_rat))
            if accept_prob[i] > np.random.rand():  # if accepted, update
                xi = prop_xi
                M_chol = M_chol_prop

        tau = 1 / math.sqrt(xi)

        # update sigma_sq marginal of beta #
        xtmp = sp.linalg.cho_solve((M_chol, False), y)
        ssr = np.dot(y, xtmp)
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

        # update beta #
        D = (tau * lam) ** 2
        beta = math.sqrt(sigma_sq) \
               * generate_gaussian(y / math.sqrt(sigma_sq), X, D, M_chol, True)

        # update lam_j's in a block using slice sampling #
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


def lmh_ratio(M_chol, y, xi, n, a0, b0):
    """ Computes the log posterior conditional of 'beta' given 'lam'. """
    try:
        x = sp.linalg.cho_solve((M_chol, False), y)
        ssr = np.dot(y, x) + b0
        ldetM = 2 * np.sum(np.log(np.diag(M_chol)))
        loglik = - .5 * ldetM - ((n + a0) / 2) * math.log(ssr)
        log_prior = - np.log(math.sqrt(xi) * (1 + xi))
        logp = loglik + log_prior
    except:
        logp = - np.float('inf')
        warnings.warn(
            'Proposal rejected because I + XDX was not positive-definite.',
            RuntimeWarning)

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

