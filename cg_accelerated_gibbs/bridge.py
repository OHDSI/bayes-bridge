import numpy as np
import scipy as sp
import scipy.stats
import scipy.linalg
import scipy.sparse
import math
import warnings
from inspect import currentframe, getframeinfo
import time
import pdb
from .sparse_dense_matrix_operators \
    import elemwise_power, left_matmul_by_diag, right_matmul_by_diag, \
    choose_optimal_format_for_matvec
from pypolyagamma import PyPolyaGamma
from tilted_stable_dist.rand_exp_tilted_stable import ExpTiltedStableDist
from .cg_sampler import ConjugateGradientSampler

class BayesBridge():

    def __init__(self, y, X, n_trial=None, link='gaussian',
                 n_coef_without_shrinkage=0, prior_sd_for_unshrunk=float('inf'),
                 add_intercept=True):
        """
        Params
        ------
        n_coef_without_shrinkage : int
            The number of predictors whose coefficients are to be estimated
            without any shrinkage (a.k.a. regularization).
        prior_sd_for_unshrunk : float, numpy array
            If an array, the length must be the same as n_coef_without_shrinkage.
        """

        # TODO: Make each MCMC run more "independent" i.e. not rely on the
        # previous instantiation of the class. The initial run of the Gibbs
        # sampler probably depends too much the stuffs here.

        if not (np.isscalar(prior_sd_for_unshrunk)
                or n_coef_without_shrinkage == len(prior_sd_for_unshrunk)):
            raise ValueError('Invalid array size for prior sd.')

        if add_intercept:
            if sp.sparse.issparse(X):
                hstack = sp.sparse.hstack
            else:
                hstack = np.hstack
            X = hstack((np.ones((X.shape[0], 1)), X))
            n_coef_without_shrinkage += 1
            if not np.isscalar(prior_sd_for_unshrunk):
                prior_sd_for_unshrunk = np.concatenate((
                    [float('inf')], prior_sd_for_unshrunk
                ))

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

        if np.isscalar(prior_sd_for_unshrunk):
            self.prior_sd_for_unshrunk = prior_sd_for_unshrunk \
                                         * np.ones(n_coef_without_shrinkage)
        else:
            self.prior_sd_for_unshrunk = prior_sd_for_unshrunk
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

    def warn_message_only(self, message, category=UserWarning):
        frameinfo = getframeinfo(currentframe())
        warnings.showwarning(
            message, category, frameinfo.filename, frameinfo.lineno,
            file=None, line=''
        ) # line='' supresses printing the line from codes.

    def gibbs_additional_iter(
            self, mcmc_output, n_iter, merge=False, deallocate=False):
        """
        Continue running the Gibbs sampler from the previous state. To continue
        the random number stream of the previous Gibbs sampler iterations, the
        same instance of the BayesBridge class must be used.

        Parameter
        ---------
        mcmc_output : the output of the 'gibbs' method.
        """

        if merge and deallocate:
            self.warn_message_only(
                "To merge the outputs, the previous one cannot be deallocated.")
            deallocate = False

        np.random.set_state(mcmc_output['_random_gen_state'])
        init = {
            key: np.take(val, -1, axis=-1).copy()
            for key, val in mcmc_output['samples'].items()
        }
        if 'precond_blocksize' in mcmc_output:
            precond_blocksize = mcmc_output['precond_blocksize']
        else:
            precond_blocksize = 0

        thin, reg_exponent, mvnorm_method, global_shrinkage_update = (
            mcmc_output[key] for key in
            ['thin', 'reg_exponent', 'mvnorm_method', 'global_shrinkage_update']
        )
        if deallocate:
            mcmc_output.clear()

        next_mcmc_output = self.gibbs(
            0, n_iter, thin, reg_exponent, init, mvnorm_method=mvnorm_method,
            precond_blocksize=precond_blocksize,
            global_shrinkage_update=global_shrinkage_update,
            _add_iter_mode=True
        )
        if merge:
            next_mcmc_output \
                = self.merge_outputs(mcmc_output, next_mcmc_output)

        return next_mcmc_output

    def merge_outputs(self, mcmc_output, next_mcmc_output):

        samples = mcmc_output['samples']
        next_samples = next_mcmc_output['samples']
        next_mcmc_output['samples'] = {
            key : np.concatenate(
                (samples[key], next_samples[key]), axis=-1
            ) for key in samples.keys()
        }
        next_mcmc_output['n_post_burnin'] += mcmc_output['n_post_burnin']
        next_mcmc_output['runtime'] += mcmc_output['runtime']

        return next_mcmc_output

    def gibbs(self, n_burnin, n_post_burnin, thin, reg_exponent=.5,
              init={}, mvnorm_method='pcg', precond_blocksize=0, seed=None,
              global_shrinkage_update='sample', _add_iter_mode=False):
        """
        MCMC implementation for the Bayesian bridge.

        Parameters
        ----------
        y : vector
        X : numpy array
        n_burnin : int
            number of burn-in samples to be discarded
        n_post_burnin : int
            number of posterior draws to be saved
        mvnorm_method : str, {'dense', 'pcg'}
        precond_blocksize : int
            size of the block preconditioner
        global_shrinkage_update : str, {'sample', 'optimize', None}

        """

        if not _add_iter_mode:
            self.set_seed(seed)

        if self.link not in ('gaussian', 'logit'):
            raise NotImplementedError()

        n_iter = n_burnin + n_post_burnin

        if mvnorm_method == 'pcg':
            self.cg_sampler = ConjugateGradientSampler(self.n_coef_wo_shrinkage)

        # Initial state of the Markov chain
        beta, sigma_sq, omega, lam, tau, init = \
            self.initialize_chain(init)

        # Object for keeping track of running average.
        if not _add_iter_mode:
            self.averager = self.runmeanUpdater(self.scale_beta(beta, tau, lam))

        # Pre-allocate
        samples = {}
        self.pre_allocate(samples, n_post_burnin, thin)
        n_pcg_iter = np.zeros(n_iter)

        # Start Gibbs sampling
        start_time = time.time()
        for mcmc_iter in range(1, n_iter + 1):

            if self.link == 'gaussian':
                omega = np.ones(self.n_obs) / sigma_sq
            beta_runmean = self.scale_back_beta(
                self.averager.runmean['beta_scaled'], tau, lam)
            beta_post_sd = self.averager.estimate_post_sd()

            beta, n_pcg_iter[mcmc_iter - 1] = self.update_beta(
                omega, tau, lam, beta_runmean, mvnorm_method,
                precond_blocksize, beta_post_sd
            )
            self.averager.update_runmean(self.scale_beta(beta, tau, lam))

            omega, sigma_sq = self.update_obs_precision(beta)

            # Draw from \tau | \beta and then \lambda | \tau, \beta. (The order matters.)
            tau = self.update_global_shrinkage(
                tau, beta[self.n_coef_wo_shrinkage:], reg_exponent, global_shrinkage_update)

            lam = self.update_local_shrinkage(
                tau, beta[self.n_coef_wo_shrinkage:], reg_exponent)

            self.store_current_state(samples, mcmc_iter, n_burnin, thin,
                                beta, lam, tau, sigma_sq, omega, reg_exponent)

        runtime = time.time() - start_time
        mcmc_output = {
            'samples': samples,
            'init': init,
            'n_burnin': n_burnin,
            'n_post_burnin': n_post_burnin,
            'thin': thin,
            'seed': seed,
            'n_coef_wo_shrinkage': self.n_coef_wo_shrinkage,
            'prior_sd_for_unshrunk': self.prior_sd_for_unshrunk,
            'reg_exponent': reg_exponent,
            'mvnorm_method': mvnorm_method,
            'runtime': runtime,
            'global_shrinkage_update': global_shrinkage_update,
            '_random_gen_state': np.random.get_state()
        }
        if mvnorm_method == 'pcg':
            mcmc_output['n_pcg_iter'] = n_pcg_iter
            if precond_blocksize > 0:
                mcmc_output['precond_blocksize'] = precond_blocksize

        return mcmc_output

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
        samples['logp'] = np.zeros(n_sample)

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
            omega = self.compute_polya_gamma_mean(self.n_trial, self.X.dot(beta))
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

        init = {
            'beta': beta,
            'sigma_sq': sigma_sq,
            'omega': omega,
            'lambda': lam,
            'tau': tau
        }

        return beta, sigma_sq, omega, lam, tau, init

    def compute_polya_gamma_mean(self, shape, tilt):
        min_magnitude = 1e-5
        pg_mean = shape.copy() / 2
        is_nonzero = (np.abs(tilt) > min_magnitude)
        pg_mean[is_nonzero] \
            *= 1 / tilt[is_nonzero] \
               * (np.exp(tilt[is_nonzero]) - 1) / (np.exp(tilt[is_nonzero]) + 1)
        return pg_mean

    def scale_beta(self, beta, tau, lam):
        beta_scaled = beta.copy()
        beta_scaled[self.n_coef_wo_shrinkage:] /= tau * lam
        return beta_scaled

    def scale_back_beta(self, beta_scaled, tau, lam):
        beta = beta_scaled.copy()
        beta[self.n_coef_wo_shrinkage:] *= tau * lam
        return beta

    def update_beta(self, omega, tau, lam, beta_runmean,
                    mvnorm_method, precond_blocksize, beta_scaled_sd):

        if self.link == 'gaussian':
            y_gaussian = self.y
        elif self.link == 'logit':
            y_gaussian = (self.y - self.n_trial / 2) / omega

        prior_sd = np.concatenate((
            self.prior_sd_for_unshrunk, tau * lam
        ))
        beta = self.sample_gaussian_posterior(
            y_gaussian, self.X_row_major, self.X_col_major, omega, prior_sd,
            beta_runmean, mvnorm_method, precond_blocksize, beta_scaled_sd
        )
        return beta

    def sample_gaussian_posterior(
            self, y, X_row_major, X_col_major, omega, prior_sd, beta_init=None,
            method='pcg', precond_blocksize=0, beta_scaled_sd=None):
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

        _, X_T = choose_optimal_format_for_matvec(X_row_major, X_col_major)
        v = X_T.dot(omega * y)
        prec_sqrt = 1 / prior_sd

        if method == 'dense':
            beta = self.generate_gaussian_with_weight(
                X_row_major, omega, prec_sqrt, v)
            n_pcg_iter = np.nan

        elif method == 'pcg':
            # TODO: incorporate an automatic calibration of 'maxiter' and 'atol' to
            # control the error in the MCMC output.
            beta, cg_info = self.cg_sampler.sample(
                X_row_major, X_col_major, omega, prec_sqrt, v,
                beta_init_1=beta_init, beta_init_2=None,
                precond_by='prior+block', precond_blocksize=precond_blocksize,
                beta_scaled_sd=beta_scaled_sd,
                maxiter=500, atol=10e-4
            )
            n_pcg_iter = cg_info['n_iter']

        else:
            raise NotImplementedError()

        return beta, n_pcg_iter

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
        weighted_X = left_matmul_by_diag(omega_sqrt, X_row_major)

        precond_scale = self.choose_diag_preconditioner(D, omega, X_row_major, precond_by)
        weighted_X_scaled = right_matmul_by_diag(weighted_X, precond_scale)

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

    def update_obs_precision(self, beta):

        sigma_sq = None
        omega = None
        if self.link == 'gaussian':
            resid = self.y - self.X_row_major.dot(beta)
            scale = np.sum(resid ** 2) / 2
            sigma_sq = scale / np.random.gamma(self.n_obs / 2, 1)
        elif self.link == 'logit':
            omega = np.zeros(self.X.shape[0])
            self.pg.pgdrawv(self.n_trial, self.X_row_major.dot(beta), omega)

        return omega, sigma_sq

    def update_global_shrinkage(
            self, tau, beta_with_shrinkage, reg_exponent, method='sample'):
        # :param method: {"sample", "optimize", None}

        if method == 'optimize':
            tau = self.monte_carlo_em_global_shrinkage(
                beta_with_shrinkage, reg_exponent)

        elif method == 'sample':

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

    def monte_carlo_em_global_shrinkage(
            self, beta_with_shrinkage, reg_exponent):
        phi = len(beta_with_shrinkage) / reg_exponent \
              / np.sum(np.abs(beta_with_shrinkage) ** reg_exponent)
        tau = phi ** - (1 / reg_exponent)
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

    def store_current_state(self, samples, mcmc_iter, n_burnin, thin,
                            beta, lam, tau, sigma_sq, omega, reg_exponent):

        if mcmc_iter > n_burnin and (mcmc_iter - n_burnin) % thin == 0:
            index = math.floor((mcmc_iter - n_burnin) / thin) - 1
            samples['beta'][:, index] = beta
            samples['lambda'][:, index] = lam
            samples['tau'][index] = tau
            if self.link == 'gaussian':
                samples['sigma_sq'][index] = sigma_sq
            elif self.link == 'logit':
                samples['omega'][:, index] = omega
            samples['logp'][index] = \
                self.compute_posterior_logprob(beta, tau, sigma_sq, reg_exponent)

        return

    def compute_posterior_logprob(self, beta, tau, sigma_sq, reg_exponent):

        prior_logp = 0

        if self.link == 'logit':
            predicted_prob = 1 / (1 + np.exp( - self.X.dot(beta)))
            machine_prec = 2. ** - 53
            within_bd = np.logical_and(
                predicted_prob > machine_prec,
                predicted_prob < 1. - machine_prec
            )
            loglik = np.sum(
                self.y[within_bd] * np.log(predicted_prob[within_bd]) \
                + (self.n_trial - self.y)[within_bd] 
                    * np.log(1 - predicted_prob[within_bd])
            )
        elif self.link == 'gaussian':
            loglik = - len(self.y) * math.log(sigma_sq) / 2 \
                     - np.sum((self.y - self.X.dot(beta)) ** 2) / sigma_sq
            prior_logp += - math.log(sigma_sq) / 2

        n_shrunk_coef = len(beta) - self.n_coef_wo_shrinkage

        # Contribution from beta | tau.
        prior_logp += \
            - n_shrunk_coef * math.log(tau) \
            - np.sum(np.abs(beta[self.n_coef_wo_shrinkage:] / tau) ** reg_exponent)

        # for coefficients without shrinkage.
        prior_logp += - 1 / 2 * np.sum(
            (beta[:self.n_coef_wo_shrinkage] / self.prior_sd_for_unshrunk) ** 2
        )
        prior_logp += - np.sum(np.log(
            self.prior_sd_for_unshrunk[self.prior_sd_for_unshrunk < float('inf')]
        ))
        if self.prior_type['tau'] == 'jeffreys':
            prior_logp += - math.log(tau)
        else:
            raise NotImplementedError()

        logp = loglik + prior_logp

        return logp


    class runmeanUpdater():

        def __init__(self, beta_scaled, sd_prior_samplesize=5):
            """

            Params
            ------
            init: dict
            sd_prior_samplesize: int
                Weight on the initial estimate of the posterior standard
                deviation; the estimate is treated as if it is an average of
                'prior_samplesize' previous values.
            """
            self.sd_prior_samplesize = sd_prior_samplesize
            self.sd_prior_guess = np.ones(len(beta_scaled))
            self.n_averaged = 0
            self.runmean = {
                'beta_scaled': np.zeros(len(beta_scaled)),
                'beta_scaled_sq': np.ones(len(beta_scaled))
            }

        def update_runmean(self, beta_scaled):

            weight = 1 / (1 + self.n_averaged)
            self.runmean['beta_scaled'] = (
                weight * beta_scaled + (1 - weight) * self.runmean['beta_scaled']
            )
            self.runmean['beta_scaled_sq'] = (
                weight * beta_scaled ** 2
                    + (1 - weight) * self.runmean['beta_scaled_sq']
            )
            self.n_averaged += 1

        def estimate_post_sd(self):

            beta_scaled_mean = self.runmean['beta_scaled']
            beta_scaled_sq_mean = self.runmean['beta_scaled_sq']

            if self.n_averaged > 1:
                var_estimator = self.n_averaged / (self.n_averaged - 1) * (
                    beta_scaled_sq_mean - beta_scaled_mean ** 2
                )
                estimator_weight = (self.n_averaged - 1) \
                    / (self.n_averaged - 1 + self.sd_prior_samplesize)
                beta_scaled_sd = np.sqrt(
                    estimator_weight * var_estimator \
                        + (1 - estimator_weight) * self.sd_prior_guess ** 2
                )
            else:
                beta_scaled_sd = self.sd_prior_guess

            return beta_scaled_sd