import numpy as np
import scipy as sp
import scipy.stats
import scipy.linalg
import scipy.sparse
import math
import time
import pdb
from .util.simple_warnings import warn_message_only
from .random import BasicRandom
from .reg_coef_sampler import SparseRegressionCoefficientSampler
from .design_matrix import SparseDesignMatrix, DenseDesignMatrix
from .model import LinearModel, LogisticModel


class BayesBridge():

    def __init__(self, y, X, n_trial=None, model='linear',
                 n_coef_without_shrinkage=0, prior_sd_for_unshrunk=float('inf'),
                 add_intercept=True):
        """
        Params
        ------
        y : vector
        X : numpy array or scipy sparse matrix
        n_trial : vector
            Used for the logistic model for binomial outcomes.
        model : str, {'linear', 'logit'}
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
            X, n_coef_without_shrinkage, prior_sd_for_unshrunk = \
                self.add_intercept(X, n_coef_without_shrinkage, prior_sd_for_unshrunk)

        X = SparseDesignMatrix(X) if sp.sparse.issparse(X) else DenseDesignMatrix(X)

        if model == 'linear':
            self.model = LinearModel(y, X)
        elif model == 'logit':
            self.model = LogisticModel(y, X, n_trial)
        else:
            raise NotImplementedError()

        if np.isscalar(prior_sd_for_unshrunk):
            self.prior_sd_for_unshrunk = prior_sd_for_unshrunk \
                                         * np.ones(n_coef_without_shrinkage)
        else:
            self.prior_sd_for_unshrunk = prior_sd_for_unshrunk
        self.n_unshrunk = n_coef_without_shrinkage
        self.y = y
        self.X = X
        self.n_obs = X.shape[0]
        self.n_pred = X.shape[1]
        self.prior_type = {}
        self.prior_param = {}
        self.set_default_priors(self.prior_type, self.prior_param)
        self.rg = BasicRandom()

    def add_intercept(self, X, n_coef_without_shrinkage, prior_sd_for_unshrunk):
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
        return X, n_coef_without_shrinkage, prior_sd_for_unshrunk

    def set_default_priors(self, prior_type, prior_param):
        prior_type['global_shrinkage'] = 'jeffreys'
        # prior_type['global_shrinkage'] = 'half-cauchy'
        # prior_param['global_shrinkage'] = {'scale': 1.0}
        return prior_type, prior_param

    def gibbs_additional_iter(
            self, mcmc_output, n_iter, merge=False, deallocate=False):
        """
        Continue running the Gibbs sampler from the previous state.

        Parameter
        ---------
        mcmc_output : the output of the 'gibbs' method.
        """

        if merge and deallocate:
            warn_message_only(
                "To merge the outputs, the previous one cannot be deallocated.")
            deallocate = False

        self.rg.set_state(mcmc_output['_random_gen_state'])

        init = {
            key: np.take(val, -1, axis=-1).copy()
            for key, val in mcmc_output['samples'].items()
        }
        if 'precond_blocksize' in mcmc_output:
            precond_blocksize = mcmc_output['precond_blocksize']
        else:
            precond_blocksize = 0

        thin, shrinkage_exponent, sampling_method, global_shrinkage_update = (
            mcmc_output[key] for key in
            ['thin', 'shrinkage_exponent', 'sampling_method', 'global_shrinkage_update']
        )

        # Initalize the regression coefficient sampler with the previous state.
        self.reg_coef_sampler = SparseRegressionCoefficientSampler(
            init, self.prior_sd_for_unshrunk, sampling_method
        )
        self.reg_coef_sampler.set_internal_state(mcmc_output['_reg_coef_sampler_state'])

        if deallocate:
            mcmc_output.clear()

        next_mcmc_output = self.gibbs(
            0, n_iter, thin, shrinkage_exponent, init, sampling_method=sampling_method,
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

    def gibbs(self, n_burnin, n_post_burnin, thin=1, shrinkage_exponent=.5,
              init={}, sampling_method='cg', precond_blocksize=0, seed=None,
              global_shrinkage_update='sample', _add_iter_mode=False):
        """
        MCMC implementation for the Bayesian bridge.

        Parameters
        ----------
        n_burnin : int
            number of burn-in samples to be discarded
        n_post_burnin : int
            number of posterior draws to be saved
        sampling_method : str, {'direct', 'cg', 'hmc'}
        precond_blocksize : int
            size of the block preconditioner
        global_shrinkage_update : str, {'sample', 'optimize', None}

        """

        if not _add_iter_mode:
            self.rg.set_seed(seed)

        n_iter = n_burnin + n_post_burnin

        # Initial state of the Markov chain
        beta, obs_prec, lshrink, gshrink, init = \
            self.initialize_chain(init)

        if not _add_iter_mode:
            self.reg_coef_sampler = SparseRegressionCoefficientSampler(
                init, self.prior_sd_for_unshrunk, sampling_method
            )

        # Pre-allocate
        samples = {}
        self.pre_allocate(samples, n_post_burnin, thin)
        n_cg_iter = np.zeros(n_iter)

        # Start Gibbs sampling
        start_time = time.time()
        for mcmc_iter in range(1, n_iter + 1):

            beta, info = self.update_beta(
                beta, obs_prec, gshrink, lshrink, sampling_method, precond_blocksize
            )
            if 'n_cg_iter' in info.keys():
                n_cg_iter[mcmc_iter - 1] = info['n_cg_iter']

            obs_prec = self.update_obs_precision(beta)

            # Draw from gshrink | \beta and then lshrink | gshrink, \beta.
            # (The order matters.)
            gshrink = self.update_global_shrinkage(
                gshrink, beta[self.n_unshrunk:], shrinkage_exponent, global_shrinkage_update)

            lshrink = self.update_local_shrinkage(
                gshrink, beta[self.n_unshrunk:], shrinkage_exponent)

            self.store_current_state(samples, mcmc_iter, n_burnin, thin,
                                     beta, lshrink, gshrink, obs_prec, shrinkage_exponent)

        runtime = time.time() - start_time
        mcmc_output = {
            'samples': samples,
            'init': init,
            'n_burnin': n_burnin,
            'n_post_burnin': n_post_burnin,
            'thin': thin,
            'seed': seed,
            'n_coef_wo_shrinkage': self.n_unshrunk,
            'prior_sd_for_unshrunk': self.prior_sd_for_unshrunk,
            'shrinkage_exponent': shrinkage_exponent,
            'sampling_method': sampling_method,
            'runtime': runtime,
            'global_shrinkage_update': global_shrinkage_update,
            '_random_gen_state': self.rg.get_state(),
            '_reg_coef_sampler_state': self.reg_coef_sampler.get_internal_state()
        }
        if sampling_method == 'cg':
            mcmc_output['n_cg_iter'] = n_cg_iter
            if precond_blocksize > 0:
                mcmc_output['precond_blocksize'] = precond_blocksize

        return mcmc_output

    def pre_allocate(self, samples, n_post_burnin, thin):

        n_sample = math.floor(n_post_burnin / thin)  # Number of samples to keep
        samples['beta'] = np.zeros((self.n_pred, n_sample))
        samples['local_shrinkage'] = np.zeros((self.n_pred - self.n_unshrunk, n_sample))
        samples['global_shrinkage'] = np.zeros(n_sample)
        if self.model.name == 'linear':
            samples['obs_prec'] = np.zeros(n_sample)
        elif self.model.name == 'logit':
            samples['obs_prec'] = np.zeros((self.n_obs, n_sample))
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

        if 'obs_prec' in init:
            obs_prec = np.ascontiguousarray(init['obs_prec'])
                # Cython requires a C-contiguous array.
            if not len(obs_prec) == self.n_obs:
                raise ValueError('An invalid initial state.')
        elif self.model.name == 'linear':
            obs_prec = np.mean((self.y - self.X.dot(beta)) ** 2) ** -1
        elif self.model.name == 'logit':
            obs_prec = self.compute_polya_gamma_mean(self.model.n_trial, self.X.dot(beta))
        else:
            obs_prec = None

        if 'local_shrinkage' in init:
            lshrink = init['local_shrinkage']
            if not len(lshrink) == (self.n_pred - self.n_unshrunk):
                raise ValueError('An invalid initial state.')
        else:
            lshrink = np.ones(self.n_pred - self.n_unshrunk)

        if 'global_shrinkage' in init:
            gshrink = init['global_shrinkage']
        else:
            gshrink = .01

        init = {
            'beta': beta,
            'obs_prec': obs_prec,
            'local_shrinkage': lshrink,
            'global_shrinkage': gshrink
        }

        return beta, obs_prec, lshrink, gshrink, init

    def compute_polya_gamma_mean(self, shape, tilt):
        min_magnitude = 1e-5
        pg_mean = shape.copy() / 2
        is_nonzero = (np.abs(tilt) > min_magnitude)
        pg_mean[is_nonzero] \
            *= 1 / tilt[is_nonzero] \
               * (np.exp(tilt[is_nonzero]) - 1) / (np.exp(tilt[is_nonzero]) + 1)
        return pg_mean

    def update_beta(self, beta, obs_prec, gshrink, lshrink, sampling_method, precond_blocksize):

        if sampling_method in ('direct', 'cg'):

            if self.model.name == 'linear':
                y_gaussian = self.y
                obs_prec = obs_prec * np.ones(self.n_obs)
            elif self.model.name == 'logit':
                y_gaussian = (self.y - self.model.n_trial / 2) / obs_prec

            beta, info = self.reg_coef_sampler.sample_gaussian_posterior(
                y_gaussian, self.X, obs_prec, gshrink, lshrink,
                sampling_method, precond_blocksize
            )

        elif sampling_method == 'hmc':
            beta, info = self.reg_coef_sampler.sample_by_hmc(
                self.y, self.X, beta, gshrink, lshrink, self.model
            )

        else:
            raise NotImplementedError()

        return beta, info

    def update_obs_precision(self, beta):

        obs_prec = None
        if self.model.name == 'linear':
            resid = self.y - self.X.dot(beta)
            scale = np.sum(resid ** 2) / 2
            obs_var = scale / self.rg.np_random.gamma(self.n_obs / 2, 1)
            obs_prec = 1 / obs_var
        elif self.model.name == 'logit':
            obs_prec = self.rg.polya_gamma(
                self.model.n_trial, self.X.dot(beta),self.X.shape[0])

        return obs_prec

    def update_global_shrinkage(
            self, gshrink, beta_with_shrinkage, shrinkage_exponent, method='sample'):
        # :param method: {"sample", "optimize", None}

        lower_bd = min(10e-4, .1 / len(beta_with_shrinkage))
            # TODO: make it an optional parameter.

        if method == 'optimize':
            gshrink = self.monte_carlo_em_global_shrinkage(
                beta_with_shrinkage, shrinkage_exponent)

        elif method == 'sample':

            if self.prior_type['global_shrinkage'] == 'jeffreys':

                # Conjugate update for phi = 1 / gshrink ** shrinkage_exponent
                shape = beta_with_shrinkage.size / shrinkage_exponent
                if np.count_nonzero(beta_with_shrinkage) == 0:
                    gshrink = 0
                else:
                    scale = 1 / np.sum(np.abs(beta_with_shrinkage) ** shrinkage_exponent)
                    phi = self.rg.np_random.gamma(shape, scale=scale)
                    gshrink = 1 / phi ** (1 / shrinkage_exponent)

                gshrink = max(gshrink, lower_bd)

            elif self.prior_type['global_shrinkage'] == 'half-cauchy':

                gshrink = self.slice_sample_global_shrinkage(
                    gshrink, beta_with_shrinkage, self.prior_param['global_shrinkage']['scale'], shrinkage_exponent
                )
            else:
                raise NotImplementedError()

        return gshrink

    def monte_carlo_em_global_shrinkage(
            self, beta_with_shrinkage, shrinkage_exponent):
        phi = len(beta_with_shrinkage) / shrinkage_exponent \
              / np.sum(np.abs(beta_with_shrinkage) ** shrinkage_exponent)
        gshrink = phi ** - (1 / shrinkage_exponent)
        return gshrink

    def slice_sample_global_shrinkage(
            self, gshrink, beta_with_shrinkage, global_scale, shrinkage_exponent):
        """ Slice sample phi = 1 / gshrink ** shrinkage_exponent. """

        n_update = 10 # Slice sample for multiple iterations to ensure good mixing.

        # Initialize a gamma distribution object.
        shape = (beta_with_shrinkage.size + 1) / shrinkage_exponent
        scale = 1 / np.sum(np.abs(beta_with_shrinkage) ** shrinkage_exponent)
        gamma_rv = sp.stats.gamma(shape, scale=scale)

        phi = 1 / gshrink
        for i in range(n_update):
            u = self.rg.np_random.uniform() \
                / (1 + (global_scale * phi ** (1 / shrinkage_exponent)) ** 2)
            upper = (np.sqrt(1 / u - 1) / global_scale) ** shrinkage_exponent
                # Invert the half-Cauchy density.
            phi = gamma_rv.ppf(gamma_rv.cdf(upper) * self.rg.np_random.uniform())
            if np.isnan(phi):
                # Inverse CDF method can fail if the current conditional
                # distribution is drastically different from the previous one.
                # In this case, ignore the slicing variable and just sample from
                # a Gamma.
                phi = gamma_rv.rvs()
        gshrink = 1 / phi ** (1 / shrinkage_exponent)

        return gshrink


    def update_local_shrinkage(self, gshrink, beta_with_shrinkage, shrinkage_exponent):

        lshrink_sq = 1 / np.array([
            2 * self.rg.tilted_stable(shrinkage_exponent / 2, (beta_j / gshrink) ** 2)
            for beta_j in beta_with_shrinkage
        ])
        lshrink = np.sqrt(lshrink_sq)

        # TODO: Pick the lower and upper bound more carefully.
        if np.any(lshrink == 0):
            warn_message_only(
                "Local shrinkage parameter under-flowed. Replacing with a small number.")
            lshrink[lshrink == 0] = 10e-16
        elif np.any(np.isinf(lshrink)):
            warn_message_only(
                "Local shrinkage parameter under-flowed. Replacing with a large number.")
            lshrink[np.isinf(lshrink)] = 2.0 / gshrink

        return lshrink

    def store_current_state(self, samples, mcmc_iter, n_burnin, thin,
                            beta, lshrink, gshrink, obs_prec, shrinkage_exponent):

        if mcmc_iter > n_burnin and (mcmc_iter - n_burnin) % thin == 0:
            index = math.floor((mcmc_iter - n_burnin) / thin) - 1
            samples['beta'][:, index] = beta
            samples['local_shrinkage'][:, index] = lshrink
            samples['global_shrinkage'][index] = gshrink
            if self.model.name == 'linear':
                samples['obs_prec'][index] = obs_prec
            elif self.model.name == 'logit':
                samples['obs_prec'][:, index] = obs_prec
            samples['logp'][index] = \
                self.compute_posterior_logprob(beta, gshrink, obs_prec, shrinkage_exponent)

        return

    def compute_posterior_logprob(self, beta, gshrink, obs_prec, shrinkage_exponent):

        prior_logp = 0

        if self.model.name == 'logit':
            loglik, _ = self.model.compute_loglik_and_gradient(beta, loglik_only=True)
        elif self.model.name == 'linear':
            loglik = len(self.y) * math.log(obs_prec) / 2 \
                     - obs_prec * np.sum((self.y - self.X.dot(beta)) ** 2)
            prior_logp += math.log(obs_prec) / 2

        n_shrunk_coef = len(beta) - self.n_unshrunk

        # Contribution from beta | gshrink.
        prior_logp += \
            - n_shrunk_coef * math.log(gshrink) \
            - np.sum(np.abs(beta[self.n_unshrunk:] / gshrink) ** shrinkage_exponent)

        # for coefficients without shrinkage.
        prior_logp += - 1 / 2 * np.sum(
            (beta[:self.n_unshrunk] / self.prior_sd_for_unshrunk) ** 2
        )
        prior_logp += - np.sum(np.log(
            self.prior_sd_for_unshrunk[self.prior_sd_for_unshrunk < float('inf')]
        ))
        if self.prior_type['global_shrinkage'] == 'jeffreys':
            prior_logp += - math.log(gshrink)
        else:
            raise NotImplementedError()

        logp = loglik + prior_logp

        return logp
