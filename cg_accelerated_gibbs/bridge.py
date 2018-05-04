import numpy as np
import scipy as sp
import scipy.stats
import scipy.linalg
import scipy.sparse
import math
import warnings
from inspect import currentframe, getframeinfo
import pdb
from pypolyagamma import PyPolyaGamma
from tilted_stable_dist.rand_exp_tilted_stable import ExpTiltedStableDist

class BayesBridge():

    def __init__(self, y, X, n_trial=None, link='gaussian',
                 n_coef_without_shrinkage=0, add_intercept=True):
        """
        Params
        ------
        n_coef_without_shrinkage : int
            The number of predictors whose coefficients are to be estimated
            without any shrinkage (a.k.a. regularization).
        """

        if add_intercept:
            if sp.sparse.issparse(X):
                hstack = sp.sparse.hstack
            else:
                hstack = np.hstack
            X = hstack((np.ones((X.shape[0], 1)), X))
            n_coef_without_shrinkage += 1

        if sp.sparse.issparse(X):
            X = X.tocsr()

        if link == 'logit':
            if n_trial is None:
                self.n_trial = np.ones(len(y))
                self.warn_message_only(
                    "The numbers of trials were not specified. The binary "
                    "outcome is assumed."
                )
            else:
                self.n_trial = n_trial

        self.n_coef_wo_shrinkage = n_coef_without_shrinkage
        self.link = link
        self.y = y
        if sp.sparse.issparse(X):
            X = X.tocsr()
            self.X_col_major = X.tocsc()
        else:
            self.X_col_major = None
        self.X_row_major = X
        self.X = X
        self.n_obs = X.shape[0]
        self.n_pred = X.shape[1]
        self.prior_type = {}
        self.prior_param = {}
        self.set_default_priors(self.prior_type, self.prior_param)
        self.pg = None
        self.tilted_stable = None

    def set_default_priors(self, prior_type, prior_param):
        prior_type['tau'] = 'jeffreys'
        # prior_type['tau'] = 'half-cauchy'
        # prior_param['tau'] = {'scale': 1.0}
        return prior_type, prior_param

    def elemwise_power(self, X, exponent):
        """ Wrapper function that works with both dense and sparse matrices. """
        if sp.sparse.issparse(X):
            return X.power(exponent)
        else:
            return X ** exponent

    def left_matmul_by_diag(self, v, A):
        """ Computes dot(diag(v), A) for a vector 'v' and matrix 'A'. """
        if sp.sparse.issparse(A):
            v_mat = sp.sparse.dia_matrix((v, 0), (len(v), len(v)))
            return v_mat.dot(A)
        else:
            return v[:, np.newaxis] * A

    def right_matmul_by_diag(self, A, v):
        """ Computes dot(A, diag(v)) for a matrix 'A' and vector 'v'. """
        if sp.sparse.issparse(A):
            v_mat = sp.sparse.dia_matrix((v, 0), (len(v), len(v)))
            return A.dot(v_mat)
        else:
            return A * v[np.newaxis, :]

    def choose_optimal_format_for_matvec(self, X_row_major, X_col_major):
        X = X_row_major
        if X_col_major is not None:
            X_T = X_col_major.T
        else:
            X_T = X_row_major.T
        return X, X_T

    def warn_message_only(self, message, category=UserWarning):
        frameinfo = getframeinfo(currentframe())
        warnings.showwarning(
            message, category, frameinfo.filename, frameinfo.lineno,
            file=None, line=''
        ) # line='' supresses printing the line from codes.

    def gibbs(self, n_burnin, n_post_burnin, thin, reg_exponent=.5,
              tau_fixed=False, init={}, mvnorm_method='pcg', seed=None):
        """
        MCMC implementation for the Bayesian bridge.

        Model: y = X \beta + \epslion, \epsilon \sim N(0, \sigma^2)
               \beta_j \sim N(0, \sigma^2 \lambda_j^2 \tau^2)
               \lambda_j^{-2} \sim \lambda_j^{-1} g(\lambda_j^{-2})
                   where g is the density of positive alpha-stable random variable
                   with index of stability reg_exponent / 2
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

        self.set_seed(seed)

        if self.link not in ('gaussian', 'logit'):
            raise NotImplementedError()

        n_iter = n_burnin + n_post_burnin

        # Initial state of the Markov chain
        beta, sigma_sq, omega, lam, tau  = \
            self.initialize_chain(init)

        # Variables for sequentially updating the running average of 'beta'.
        n_averaged = 0
        beta_runmean = beta
        beta_scaled_runmean = None
        beta_shrinkage_scale = None

        # Pre-allocate
        samples = {}
        self.pre_allocate(samples, n_post_burnin, thin)

        # Outputs of the algorim useful for research purposes; a user does not need to see this.
        self.cg_iter = []

        # Start Gibbs sampling
        for mcmc_iter in range(1, n_iter + 1):

            if self.link == 'gaussian':
                omega = np.ones(self.n_obs) / sigma_sq
            beta = self.update_beta(omega, tau, lam, beta_runmean, mvnorm_method)
            omega, sigma_sq = self.update_obs_precision(beta, omega)

            # Draw from \tau | \beta and then \lambda | \tau, \beta. (The order matters.)
            if not tau_fixed:
                tau = self.update_global_shrinkage(
                    tau, beta[self.n_coef_wo_shrinkage:], reg_exponent)

            lam = self.update_local_shrinkage(
                tau, beta[self.n_coef_wo_shrinkage:], reg_exponent)

            self.store_current_state(samples, mcmc_iter, n_burnin, thin,
                                beta, lam, tau, sigma_sq, omega)

            beta_runmean, beta_scaled_runmean, beta_shrinkage_scale, n_averaged \
                = self.update_beta_runmean(
                beta, tau, lam, beta_scaled_runmean, beta_shrinkage_scale, n_averaged
            )

        return samples

    def set_seed(self, seed):
        np.random.seed(seed)
        pg_seed = np.random.random_integers(np.iinfo(np.uint32).max)
        ts_seed = np.random.random_integers(np.iinfo(np.uint32).max)
        self.pg = PyPolyaGamma(seed=pg_seed)
        self.tilted_stable = ExpTiltedStableDist(seed=ts_seed)

    def pre_allocate(self, samples, n_post_burnin, thin):

        n_sample = math.floor(n_post_burnin / thin)  # Number of samples to keep
        samples['beta'] = np.zeros((self.n_pred, n_sample))
        samples['lambda'] = np.zeros((self.n_pred - self.n_coef_wo_shrinkage, n_sample))
        samples['tau'] = np.zeros(n_sample)
        if self.link == 'gaussian':
            samples['sigma_sq'] = np.zeros(n_sample)
        elif self.link == 'logit':
            samples['omega'] = np.zeros((self.n_obs, n_sample))

        return

    def initialize_chain(self, init):
        # Choose the user-specified state if provided, the default ones otherwise.

        if 'beta' in init:
            beta = init['beta']
            if not len(beta) == self.n_pred:
                raise ValueError('An invalid initial state.')
        else:
            beta = np.zeros(self.n_pred)
            if 'intercept' in init:
                beta[0] = init['intercept']

        if 'sigma' in init:
            sigma_sq = init['sigma'] ** 2
        else:
            sigma_sq = np.mean((self.y - self.X.dot(beta)) ** 2)

        if 'omega' in init:
            omega = np.ascontiguousarray(init['omega'])
                # Cython requires a C-contiguous array.
            if not len(omega) == self.n_obs:
                raise ValueError('An invalid initial state.')
        elif self.link == 'logit':
            predicted_prob = 1 / (1 + np.exp( - self.X.dot(beta)))
            hess_neg_loglik = self.n_trial * predicted_prob * (1 - predicted_prob)
            omega = hess_neg_loglik
        else:
            omega = None

        if 'lambda' in init:
            lam = init['lambda']
            if not len(lam) == (self.n_pred - self.n_coef_wo_shrinkage):
                raise ValueError('An invalid initial state.')
        else:
            lam = np.ones(self.n_pred - self.n_coef_wo_shrinkage)

        if 'tau' in init:
            tau = init['tau']
        else:
            tau = 1

        return beta, sigma_sq, omega, lam, tau

    def update_beta(self, omega, tau, lam, beta_runmean, mvnorm_method):

        if self.link == 'gaussian':
            y_gaussian = self.y
        elif self.link == 'logit':
            y_gaussian = (self.y - self.n_trial / 2) / omega

        prior_sd = np.concatenate((
            [float('inf')] * self.n_coef_wo_shrinkage,
            tau * lam
        ))
        beta = self.sample_gaussian_posterior(
            y_gaussian, self.X_row_major, self.X_col_major, omega, prior_sd,
            beta_runmean, mvnorm_method
        )
        return beta

    def sample_gaussian_posterior(
            self, y, X_row_major, X_col_major, omega, prior_sd, beta_init=None,
            method='pcg'):
        """
        Param:
        ------
            X_col_major: None if X is dense, sparse csc matrix otherwise
            beta_init: vector
                Used when when method == 'pcg' as the starting value of the
                preconditioned conjugate gradient algorithm.
            method: {'dense', 'pcg'}
                If 'dense', a sample is generated using a direct method based on the
                dense linear algebra. If 'pcg', the preconditioned conjugate gradient
                sampler is used.

        """
        #TODO: Comment on the form of the posterior.

        _, X_T = self.choose_optimal_format_for_matvec(X_row_major, X_col_major)
        v = X_T.dot(omega * y)
        prec_sqrt = 1 / prior_sd

        if method == 'dense':
            beta = self.generate_gaussian_with_weight(
                X_row_major, omega, prec_sqrt, v)

        elif method == 'pcg':
            # TODO: incorporate an automatic calibration of 'maxiter' and 'atol' to
            # control the error in the MCMC output.
            beta, cg_info = self.pcg_gaussian_sampler(
                X_row_major, X_col_major, omega, prec_sqrt, v,
                beta_init_1=beta_init, beta_init_2=None,
                precond_by='prior', maxiter=500, atol=10e-4
            )
            self.cg_iter.append(cg_info['n_iter'])
        else:
            raise NotImplementedError()

        return beta

    def generate_gaussian_with_weight(self, X_row_major, omega, D, z,
                                      precond_by='diag'):
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
        weighted_X = self.left_matmul_by_diag(omega_sqrt, X_row_major)
        if sp.sparse.issparse(X_row_major):
            weighted_X = weighted_X.tocsc()

        precond_scale = self.choose_diag_preconditioner(D, omega, X_row_major, precond_by)
        weighted_X_scaled = self.right_matmul_by_diag(weighted_X, precond_scale)

        Phi_scaled = weighted_X_scaled.T.dot(weighted_X_scaled)
        if sp.sparse.issparse(X_row_major):
            Phi_scaled = Phi_scaled.toarray()
        Phi_scaled += np.diag((precond_scale * D) ** 2)
        Phi_scaled_chol = sp.linalg.cholesky(Phi_scaled)
        mu = sp.linalg.cho_solve((Phi_scaled_chol, False), precond_scale * z)
        beta_scaled = mu + sp.linalg.solve_triangular(
            Phi_scaled_chol, np.random.randn(X_row_major.shape[1]), lower=False
        )
        beta = precond_scale * beta_scaled
        return beta

    def pcg_gaussian_sampler(self, X_row_major, X_col_major, omega, D, z,
                             beta_init_1=None, beta_init_2=None,
                             precond_by='diag', precond_blocksize=0, maxiter=None, atol=10e-6,
                             seed=None, iter_list=None):
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

        X, X_T = self.choose_optimal_format_for_matvec(X_row_major, X_col_major)

        if seed is not None:
            np.random.seed(seed)

        # Define a preconditioned linear operator.
        Phi_precond_op, precond_scale, block_precond_op = \
            self.precondition_linear_system(
                D, omega, X_row_major, X_col_major, precond_by, precond_blocksize
            )

        # Draw a target vector.
        v = X_T.dot(omega ** (1 / 2) * np.random.randn(X.shape[0])) \
            + D * np.random.randn(X.shape[1])
        b = precond_scale * (z + v)

        # Pick the initial vector for CG iteration
        beta_scaled_init = self.choose_best_linear_comb(
            beta_init_1, beta_init_2, Phi_precond_op, precond_scale, b
        )

        # Callback function to count the number of PCG iterations.
        cg_info = {'n_iter': 0}
        def cg_callback(x): cg_info['n_iter'] += 1

        # Run PCG.
        rtol = atol / np.linalg.norm(b)
        beta_scaled, info = sp.sparse.linalg.cg(
            Phi_precond_op, b, x0=beta_scaled_init, maxiter=maxiter, tol=rtol,
            M=block_precond_op, callback=cg_callback
        )

        if info != 0:
            self.warn_message_only(
                "The conjugate gradient algorithm did not achieve the requested " +
                "tolerance level. You may increase the maxiter or use the dense " +
                "linear algebra instead."
            )

        beta = precond_scale * beta_scaled
        cg_info['valid_input'] = (info >= 0)
        cg_info['converged'] = (info == 0)

        return beta, cg_info

    def precondition_linear_system(
            self, D, omega, X_row_major, X_col_major, precond_by, precond_blocksize):

        X, X_T = self.choose_optimal_format_for_matvec(X_row_major, X_col_major)

        # Compute the preconditioners.
        precond_scale, block_precond_op = self.choose_preconditioner(
            D, omega, X_row_major, X_col_major, precond_by, precond_blocksize
        )

        # Define a preconditioned linear operator.
        D_scaled_sq = (precond_scale * D) ** 2
        def Phi_precond(x):
            Phi_x = D_scaled_sq * x \
                    + precond_scale * X_T.dot(omega * X.dot(precond_scale * x))
            return Phi_x
        Phi_precond_op = sp.sparse.linalg.LinearOperator(
            (X.shape[1], X.shape[1]), matvec=Phi_precond
        )
        return Phi_precond_op, precond_scale, block_precond_op


    def choose_preconditioner(self, D, omega, X_row_major, X_col_major,
                              precond_by, precond_blocksize):

        precond_scale = self.choose_diag_preconditioner(D, omega, X_row_major, precond_by)

        block_precond_op = None
        if precond_by == 'prior+block' and precond_blocksize > 0:
            pred_importance = precond_scale.copy()
            pred_importance[:self.n_coef_wo_shrinkage] = float('inf')
            subset_indices = np.argsort(pred_importance)[-precond_blocksize:]
            subset_indices = np.sort(subset_indices)
            block_precond_op = self.compute_block_preconditioner(
                omega, X_row_major, X_col_major, D, precond_scale, subset_indices
            )

        return precond_scale, block_precond_op

    def choose_diag_preconditioner(self, D, omega, X_row_major, precond_by='diag'):
        # Compute the diagonal (sqrt) preconditioner.

        if precond_by in ('prior', 'prior+block'):
            precond_scale = D ** -1
            if self.n_coef_wo_shrinkage > 0:
                # TODO: Consider a better preconditioner for the intercept such
                # as a posterior standard deviation.
                precond_scale[:self.n_coef_wo_shrinkage] = 1  # np.sum(omega) ** (- 1 / 2)

        elif precond_by == 'diag':
            diag = D ** 2 + np.squeeze(np.asarray(
                self.left_matmul_by_diag(
                    omega, self.elemwise_power(X_row_major, 2)
                ).sum(axis=0)
            ))
            precond_scale = 1 / np.sqrt(diag)

        elif precond_by is None:
            precond_scale = np.ones(X_row_major.shape[1])

        else:
            raise NotImplementedError()

        return precond_scale

    def compute_block_preconditioner(
            self, omega, X_row_major, X_col_major, D, precond_scale, indices):

        if X_col_major is not None:
            X = X_col_major
        else:
            X = X_row_major

        weighted_X_subset = \
            self.left_matmul_by_diag(omega ** (1 / 2), X[:, indices])
        if sp.sparse.issparse(weighted_X_subset):
            weighted_X_subset = weighted_X_subset.tocsc()

        weighted_X_subset_scaled = self.right_matmul_by_diag(weighted_X_subset, precond_scale[indices])
        if sp.sparse.issparse(weighted_X_subset_scaled):
            weighted_X_subset_scaled = weighted_X_subset_scaled.toarray() 
        B = weighted_X_subset_scaled.T.dot(weighted_X_subset_scaled) \
            + np.diag((D[indices] * precond_scale[indices]) ** 2)

        B_cho_factor = sp.linalg.cho_factor(B)
        def B_inv_on_indices(x):
            x = x.copy() # TODO: Check if a shallow copy is OK.
            x[indices] = sp.linalg.cho_solve(B_cho_factor, x[indices])
            return x
        block_preconditioner_op = sp.sparse.linalg.LinearOperator(
            (X.shape[1], X.shape[1]), matvec=B_inv_on_indices
        )

        return block_preconditioner_op

    def choose_best_linear_comb(
            self, beta_init_1, beta_init_2, Phi_precond_op, precond_scale, b):

        if beta_init_1 is not None:
            beta_init_1 = beta_init_1.copy() / precond_scale
        if beta_init_2 is not None:
            beta_init_2 = beta_init_2.copy() / precond_scale
        beta_scaled_init = self.optimize_cg_objective(
            Phi_precond_op, b, beta_init_1, beta_init_2)

        return beta_scaled_init

    def optimize_cg_objective(self, A, b, x1, x2=None):
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

    def update_obs_precision(self, beta, omega):

        sigma_sq = None
        if self.link == 'gaussian':
            resid = self.y - self.X_row_major.dot(beta)
            scale = np.sum(resid ** 2) / 2
            sigma_sq = scale / np.random.gamma(self.n_obs / 2, 1)
        elif self.link == 'logit':
            self.pg.pgdrawv(self.n_trial, self.X_row_major.dot(beta), omega)

        return omega, sigma_sq

    def update_global_shrinkage(self, tau, beta_with_shrinkage, reg_exponent):

        if self.prior_type['tau'] == 'jeffreys':

            # Conjugate update for phi = 1 / tau ** reg_exponent
            shape = beta_with_shrinkage.size / reg_exponent
            scale = 1 / np.sum(np.abs(beta_with_shrinkage) ** reg_exponent)
            phi = np.random.gamma(shape, scale=scale)
            tau = 1 / phi ** (1 / reg_exponent)

        elif self.prior_type['tau'] == 'half-cauchy':

            tau = self.slice_sample_global_shrinkage(
                tau, beta_with_shrinkage, self.prior_param['tau']['scale'], reg_exponent
            )
        else:
            raise NotImplementedError()

        return tau

    def slice_sample_global_shrinkage(
            self, tau, beta_with_shrinkage, global_scale, reg_exponent):
        """ Slice sample phi = 1 / tau ** reg_exponent. """

        n_update = 10 # Slice sample for multiple iterations to ensure good mixing.

        # Initialize a gamma distribution object.
        shape = (beta_with_shrinkage.size + 1) / reg_exponent
        scale = 1 / np.sum(np.abs(beta_with_shrinkage) ** reg_exponent)
        gamma_rv = sp.stats.gamma(shape, scale=scale)

        phi = 1 / tau
        for i in range(n_update):
            u = np.random.uniform() \
                / (1 + (global_scale * phi ** (1 / reg_exponent)) ** 2)
            upper = (np.sqrt(1 / u - 1) / global_scale) ** reg_exponent
                # Invert the half-Cauchy density.
            phi = gamma_rv.ppf(gamma_rv.cdf(upper) * np.random.uniform())
            if np.isnan(phi):
                # Inverse CDF method can fail if the current conditional
                # distribution is drastically different from the previous one.
                # In this case, ignore the slicing variable and just sample from
                # a Gamma.
                phi = gamma_rv.rvs()
        tau = 1 / phi ** (1 / reg_exponent)

        return tau


    def update_local_shrinkage(self, tau, beta_with_shrinkage, reg_exponent):

        lam_sq = 1 / np.array([
            2 * self.tilted_stable.rv(reg_exponent / 2, (beta_j / tau) ** 2)
            for beta_j in beta_with_shrinkage
        ])
        lam = np.sqrt(lam_sq)

        # TODO: Pick the lower and upper bound more carefully.
        if np.any(lam == 0):
            self.warn_message_only(
                "Local shrinkage parameter under-flowed. Replacing with a small number.")
            lam[lam == 0] = 10e-16
        elif np.any(np.isinf(lam)):
            self.warn_message_only(
                "Local shrinkage parameter under-flowed. Replacing with a large number.")
            lam[np.isinf(lam)] = 2.0 / tau

        return lam

    def update_beta_runmean(self, beta, tau, lam, beta_scaled_runmean,
                            beta_shrinkage_scale, n_averaged):
        if n_averaged == 0:
            beta_runmean = beta.copy()
            beta_shrinkage_scale = tau * lam
        else:
            beta_scaled_runmean = \
                self.compute_scaled_runmean(
                    beta, beta_shrinkage_scale, beta_scaled_runmean, n_averaged)
            n_averaged += 1
            beta_shrinkage_scale = tau * lam
            beta_runmean = beta_scaled_runmean.copy()
            beta_runmean[self.n_coef_wo_shrinkage:] *= beta_shrinkage_scale

        return beta_runmean, beta_scaled_runmean, beta_shrinkage_scale, n_averaged

    def compute_scaled_runmean(self, beta, beta_shrinkage_scale,
                               prev_scaled_runmean, n_averaged):
        # Computes the running mean of beta / (tau * lam) and rescale it with the
        # current values of tau and lam.

        beta_scaled = beta.copy()
        beta_scaled[self.n_coef_wo_shrinkage:] *= 1 / beta_shrinkage_scale
        if n_averaged == 0:
            beta_scaled_runmean = beta_scaled
        else:
            weight = 1 / (1 + n_averaged)
            beta_scaled_runmean = \
                weight * beta_scaled + (1 - weight) * prev_scaled_runmean

        return beta_scaled_runmean

    def store_current_state(self, samples, mcmc_iter, n_burnin, thin,
                            beta, lam, tau, sigma_sq, omega):

        if mcmc_iter > n_burnin and (mcmc_iter - n_burnin) % thin == 0:
            index = math.floor((mcmc_iter - n_burnin) / thin) - 1
            samples['beta'][:, index] = beta
            samples['lambda'][:, index] = lam
            samples['tau'][index] = tau
            if self.link == 'gaussian':
                samples['sigma_sq'][index] = sigma_sq
            elif self.link == 'logit':
                samples['omega'][:, index] = omega

        return

