import numpy as np
import scipy as sp
import scipy.stats
import scipy.linalg
import scipy.sparse
import math
import warnings
import pdb
from pypolyagamma import PyPolyaGamma
pg = PyPolyaGamma()


def gibbs(y, X, n_burnin, n_post_burnin, thin, tau_fixed=False,
          init={}, link='gaussian', mvnorm_method='pcg', include_intercept=True):
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
           mvnorm_method = {'dense', 'pcg'}

    """

    n_iter = n_burnin + n_post_burnin
    n, p = np.shape(X)
    if link == 'logit':
        n_trial = np.ones(n)
    else:
        n_trial = None
    if include_intercept:
        X = sp.sparse.hstack((np.ones((n, 1)), X))  # Add an intercept term.
    X_csr = X.tocsr()
    X_csc = X.tocsc()

    # Hyper & tuning parameters
    global_scale = 1 # scale of the half-Cauchy prior on 'tau'
    n_averaged = 0

    # Initial state of the Markov chain
    beta, sigma_sq, omega, lam, tau  = \
        initialize_chain(init, X, p, link, n_trial)
    beta_runmean = beta
    beta_scaled_runmean = None

    # Pre-allocate
    samples = {}
    pre_allocate(samples, n, p, X, n_post_burnin, thin, link)

    # Start Gibbs sampling
    for i in range(1, n_iter + 1):

        # Update beta and related parameters.
        if link == 'gaussian':
            omega = np.ones(n) / sigma_sq
            beta = update_beta(y, X_csr, X_csc, omega, tau, lam, beta)
            resid = y - X_csr.dot(beta)
            scale = np.sum(resid ** 2) / 2
            sigma_sq = scale / np.random.gamma(n / 2, 1)
        elif link == 'logit':
            pg.pgdrawv(n_trial, X_csr.dot(beta), omega)
            beta = update_beta(
                (y - n_trial / 2) / omega, X_csr, X_csc, omega, tau, lam,
                beta_runmean, mvnorm_method, include_intercept
            )
        else:
            raise NotImplementedError(
                'The specified link function is not supported.')

        # Draw from \tau | \beta and then \lambda | \tau, \beta. (The order matters.)
        if not tau_fixed:
            tau = update_global_shrinkage(tau, beta[1:], global_scale)

        lam = 1 / np.sqrt(np.random.wald(mean=np.abs(tau / beta[1:]), scale=1))
        # TODO: Pick the lower and upper bound more carefully.
        if np.any(lam == 0):
            warnings.warn("Local shrinkage parameter under-flowed. Replacing with a small number.")
            lam[lam == 0] = 10e-16
        elif np.any(np.isinf(lam)):
            warnings.warn("Local shrinkage parameter under-flowed. Replacing with a large number.")
            lam[np.isinf(lam)] = 2.0 / tau

        store_current_state(samples, i, n_burnin, thin, link,
                            beta, lam, tau, sigma_sq, omega)

        # Compute the running mean of
        #     beta[1:, iter] / tau[iter - 1] / lam[:, iter - 1]
        if i == 1:
            beta_runmean = beta.copy()
            beta_apriori_scale = tau * lam
        else:
            beta_scaled_runmean = \
                compute_scaled_runmean(beta, beta_apriori_scale,
                                       beta_scaled_runmean, n_averaged)
            n_averaged += 1
            beta_apriori_scale = tau * lam
            beta_runmean = beta_scaled_runmean.copy()
            beta_runmean[1:] *= beta_apriori_scale

    return samples


def pre_allocate(samples, n, p, X, n_post_burnin, thin, link):

    n_sample = math.floor(n_post_burnin / thin)  # Number of samples to keep
    samples['beta'] = np.zeros((X.shape[1], n_sample))
    samples['lambda'] =  np.zeros((p, n_sample))
    samples['tau'] = np.zeros(n_sample)
    if link == 'gaussian':
        samples['sigma_sq'] = np.zeros(n_sample)
    elif link == 'logit':
        samples['omega'] = np.zeros((n, n_sample))

    return

def initialize_chain(init, X, p, link, n_trial):
    # Choose the user-specified state if provided, the default ones otherwise.

    if 'beta' in init:
        beta = init['beta']
        if not len(beta) == X.shape[1]:
            raise ValueError('An invalid initial state.')
    else:
        beta = np.zeros(X.shape[1])
        if 'intercept' in init:
            beta[0] = init['intercept']

    if 'sigma' in init:
        sigma_sq = init['sigma'] ** 2
    else:
        sigma_sq = 1

    if 'omega' in init:
        omega = np.ascontiguousarray(init['omega'])
            # Cython requires a C-contiguous array.
        if not len(omega) == X.shape[0]:
            raise ValueError('An invalid initial state.')
    elif link == 'logit':
        omega = n_trial / 2
    else:
        omega = None

    if 'lambda' in init:
        lam = init['lambda']
        if not len(lam) == p:
            raise ValueError('An invalid initial state.')
    else:
        lam = np.ones(p)

    if 'tau' in init:
        tau = init['tau']
    else:
        tau = 1
        
    return beta, sigma_sq, omega, lam, tau


def update_beta(y, X_csr, X_csc, omega, tau, lam, beta_init=None,
                method='pcg', include_intercept=True):
    """
    Param:
    ------
        beta_init: vector
            Used when when method == 'pcg' as the starting value of the
            preconditioned conjugate gradient algorithm.
        method: {'dense', 'pcg'}
            If 'dense', a sample is generated using a direct method based on the
            dense linear algebra. If 'pcg', the preconditioned conjugate gradient
            sampler is used.

    """

    prior_sd = tau * lam
    if include_intercept:
        # Flat prior for intercept
        prior_sd = np.concatenate(([float('inf')], prior_sd))

    v = X_csc.T.dot(omega * y)
    prec_sqrt = 1 / prior_sd

    if method == 'dense':
        beta = generate_gaussian_with_weight(X_csr, omega, prec_sqrt, v, include_intercept)

    elif method == 'pcg':
        # TODO: incorporate an automatic calibration of 'maxiter' and 'atol' to
        # control the error in the MCMC output.
        beta = pcg_gaussian_sampler(
            X_csr, X_csc, omega, prec_sqrt, v,
            beta_init_1=beta_init, beta_init_2=None,
            precond_by='prior', maxiter=500, atol=10e-4,
            include_intercept=include_intercept
        )
    else:
        raise NotImplementedError()

    return beta


def generate_gaussian_with_weight(X_csr, omega, D, z, precond_by='diag',
                                  include_intercept=True):
    """
    Generate a multi-variate Gaussian with the mean mu and covariance Sigma of the form
       Sigma^{-1} = X' diag(omega) X + D^2, mu = Sigma z
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

    precond_scale = choose_preconditioner(D, omega, X_csr, precond_by, include_intercept)
    precond_scale_mat = \
        sp.sparse.dia_matrix((precond_scale, 0), (p + 1, p + 1))

    weighted_X_scaled = weighted_X.dot(precond_scale_mat)
    Phi_scaled = weighted_X_scaled.T.dot(weighted_X_scaled).toarray() \
          + np.diag((precond_scale * D) ** 2)
    Phi_scaled_chol = sp.linalg.cholesky(Phi_scaled)
    mu = sp.linalg.cho_solve((Phi_scaled_chol, False), precond_scale * z)
    beta_scaled = mu + sp.linalg.solve_triangular(
        Phi_scaled_chol, np.random.randn(p + 1), lower=False
    )
    beta = precond_scale * beta_scaled
    return beta


def pcg_gaussian_sampler(X_csr, X_csc, omega, D, z,
                         beta_init_1=None, beta_init_2=None,
                         precond_by='diag', maxiter=None, atol=10e-6,
                         seed=None, include_intercept=True):
    """
    Generate a multi-variate Gaussian with the mean mu and covariance Sigma of the form
       Sigma^{-1} = X' Omega X + D^2, mu = Sigma v
    where D is assumed to be diagonal. For numerical stability, the code first sample
    from the scaled parameter beta / precond_scale.

    Param:
    ------
        D : vector
        atol : float
            The absolute tolerance on the residual norm at the termination
            of CG iterations.
    """

    X = X_csr
    X_T = X_csc.T
    n = X.shape[0]
    p = X.shape[1] - 1

    if seed is not None:
        np.random.seed(seed)

    # Compute the diagonal (sqrt) preconditioner.
    if precond_by == 'prior':
        precond_scale = D ** -1
        if include_intercept:
            # TODO: Consider a better preconditioner for the intercept such
            # as a posterior standard deviation.
            precond_scale[0] = 1 # np.sum(omega) ** (- 1 / 2)
    elif precond_by == 'diag':
        omega_mat = sp.sparse.dia_matrix((omega, 0), (n, n))
        diag = D ** 2 + np.squeeze(np.asarray(
            omega_mat.dot(X_csr.power(2)).sum(axis=0)
        ))
        precond_scale = 1 / np.sqrt(diag)
    elif precond_by is None:
        precond_scale = np.ones(p + 1)
    else:
        raise NotImplementedError()

    # Define a preconditioned linear operator.
    D_scaled_sq = (precond_scale * D) ** 2
    def Phi(x):
        Phi_x = D_scaled_sq * x \
                + precond_scale * X_T.dot(omega * X.dot(precond_scale * x))
        return Phi_x
    A = sp.sparse.linalg.LinearOperator((p + 1, p + 1), matvec=Phi)

    # Draw a target vector.
    v = X_T.dot(omega ** (1 / 2) * np.random.randn(n)) \
        + D * np.random.randn(p + 1)
    b = precond_scale * (z + v)

    # Choose the best linear combination of the two candidates for CG.
    if beta_init_1 is not None:
        beta_init_1 = beta_init_1.copy() / precond_scale
    if beta_init_2 is not None:
        beta_init_2 = beta_init_2.copy() / precond_scale
    beta_scaled_init = optimize_cg_objective(A, b, beta_init_1, beta_init_2)

    rtol = atol / np.linalg.norm(b)
    beta_scaled, info = sp.sparse.linalg.cg(A, b, x0=beta_scaled_init,
                                            maxiter=maxiter, tol=rtol)
    if info != 0:
        warnings.warn(
            "The conjugate gradient algorithm did not achieve the requested " +
            "tolerance level. You may increase the maxiter or use the dense " +
            "linear algebra instead."
        )
    beta = precond_scale * beta_scaled
    # beta_init = precond_scale * beta_scaled_init

    return beta # , info, beta_init, A, b, precond_scale


def optimize_cg_objective(A, b, x1, x2=None):
    # Minimize the function f(x) = x'Ax / 2 - x'b along the line connecting
    # x1 and x2.
    if x2 is None:
        x = x1
    else:
        v = x2 - x1
        Av = A(v)
        denom = v.dot(Av)
        if denom == 0:
            t_argmin = 0
        else:
            t_argmin = (x1.dot(Av) - b.dot(v)) / denom
        x = x1 - t_argmin * v
    return x


def choose_preconditioner(D, omega, X_csr, precond_by='diag'):
    # Compute the diagonal (sqrt) preconditioner.

    include_intercept = True
        # In case we want to change the behavior in the future

    n = X_csr.shape[0]
    p = X_csr.shape[1] - 1

    if precond_by == 'prior':
        precond_scale = D ** -1
        if include_intercept:
            precond_scale[0] = np.sum(omega) ** (- 1 / 2)

    elif precond_by == 'diag':
        omega_mat = sp.sparse.dia_matrix((omega, 0), (n, n))
        diag = D ** 2 + np.squeeze(np.asarray(
            omega_mat.dot(X_csr.power(2)).sum(axis=0)
        ))
        precond_scale = 1 / np.sqrt(diag)

    elif precond_by is None:
        precond_scale = np.ones(p + 1)

    else:
        raise NotImplementedError()

    return precond_scale


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


def compute_scaled_runmean(beta, beta_apriori_scale,
                           prev_scaled_runmean, n_averaged):
    # Computes the running mean of beta / (tau * lam) and rescale it with the
    # current values of tau and lam.

    beta_scaled = beta.copy()
    beta_scaled[1:] *= 1 / beta_apriori_scale
    if n_averaged == 0:
        beta_scaled_runmean = beta_scaled
    else:
        weight = 1 / (1 + n_averaged)
        beta_scaled_runmean = \
            weight * beta_scaled + (1 - weight) * prev_scaled_runmean

    return beta_scaled_runmean

def store_current_state(samples, mcmc_iter, n_burnin, thin, link,
                        beta, lam, tau, sigma_sq, omega):

    if mcmc_iter > n_burnin and (mcmc_iter - n_burnin) % thin == 0:
        index = math.floor((mcmc_iter - n_burnin) / thin) - 1
        samples['beta'][:, index] = beta
        samples['lambda'][:, index] = lam
        samples['tau'][index] = tau
        if link == 'gaussian':
            samples['sigma_sq'][index] = sigma_sq
        elif link == 'logit':
            samples['omega'][:, index] = omega

    return


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

