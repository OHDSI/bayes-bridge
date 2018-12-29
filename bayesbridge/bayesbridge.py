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
from .model import LinearModel, LogisticModel, CoxModel


class BayesBridge():

    def __init__(self, outcome, X, model='linear',
                 n_coef_without_shrinkage=0, prior_sd_for_unshrunk=float('inf'),
                 add_intercept=True):
        """
        Params
        ------
        outcome : vector if model == 'linear' else tuple
            (n_success, n_trial) if model == 'logistic'.
                The outcome is assumed binary if n_trial is None.
            (event_time, censoring_time) if model == 'cox'
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

        if model != 'cox' and add_intercept:
            X, n_coef_without_shrinkage, prior_sd_for_unshrunk = \
                self.add_intercept(X, n_coef_without_shrinkage, prior_sd_for_unshrunk)

        if model == 'cox':

            event_time, censoring_time = outcome
            event_time, censoring_time, X = CoxModel.preprocess_data(
                event_time, censoring_time, X
            )

        X = SparseDesignMatrix(X) if sp.sparse.issparse(X) else DenseDesignMatrix(X)

        if model == 'linear':
            self.model = LinearModel(outcome, X)
        elif model == 'logit':
            n_success, n_trial = outcome
            self.model = LogisticModel(n_success, X, n_trial)
        elif model == 'cox':
            self.model = CoxModel(event_time, censoring_time, X)
        else:
            raise NotImplementedError()

        if np.isscalar(prior_sd_for_unshrunk):
            self.prior_sd_for_unshrunk = \
                prior_sd_for_unshrunk * np.ones(n_coef_without_shrinkage)
        else:
            self.prior_sd_for_unshrunk = prior_sd_for_unshrunk
        self.n_unshrunk = n_coef_without_shrinkage
        self.n_obs = X.shape[0]
        self.n_pred = X.shape[1]
        self.prior_type = {}
        self.prior_param = {}
        self.set_default_priors(self.prior_type, self.prior_param)
        self.rg = BasicRandom()
        self.manager = MarkovChainManager(
            self.n_obs, self.n_pred, self.n_unshrunk, model
        )

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

        init = mcmc_output['_markov_chain_state']
        if 'precond_blocksize' in mcmc_output:
            precond_blocksize = mcmc_output['precond_blocksize']
        else:
            precond_blocksize = 0

        thin, shrinkage_exponent, sampling_method, global_shrinkage_update = (
            mcmc_output[key] for key in [
                'thin', 'shrinkage_exponent', 'sampling_method', 'global_shrinkage_update'
            ]
        )
        params_to_save = mcmc_output['samples'].keys()

        # Initalize the regression coefficient sampler with the previous state.
        self.reg_coef_sampler = SparseRegressionCoefficientSampler(
            self.n_pred, self.prior_sd_for_unshrunk, sampling_method
        )
        self.reg_coef_sampler.set_internal_state(mcmc_output['_reg_coef_sampler_state'])

        if deallocate:
            mcmc_output.clear()

        next_mcmc_output = self.gibbs(
            0, n_iter, thin, shrinkage_exponent, init,
            sampling_method=sampling_method,
            precond_blocksize=precond_blocksize,
            global_shrinkage_update=global_shrinkage_update,
            params_to_save=params_to_save,
            _add_iter_mode=True
        )
        if merge:
            next_mcmc_output \
                = self.manager.merge_outputs(mcmc_output, next_mcmc_output)

        return next_mcmc_output

    def gibbs(self, n_burnin, n_post_burnin, thin=1, shrinkage_exponent=.5,
              init={}, sampling_method='cg', precond_blocksize=0, seed=None,
              global_shrinkage_update='sample', params_to_save=None,
              n_init_optim_step=10, _add_iter_mode=False):
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
        params_to_save : {None, 'all', list of str}
        n_init_optim_step : int
            If > 0, the Markov chain will be run after the specified number of
            optimization steps in which the regression coefficients are
            optimized conditionally on the shrinkage parameters. During the
            optimization, the global shrinkage parameter is fixed while the
            local ones are sampled.

        """

        if _add_iter_mode:
            n_init_optim_step = 0
        else:
            self.rg.set_seed(seed)
            self.reg_coef_sampler = SparseRegressionCoefficientSampler(
                self.n_pred, self.prior_sd_for_unshrunk, sampling_method
            )

        if params_to_save == 'all':
            params_to_save = [
                'beta', 'local_shrinkage', 'global_shrinkage', 'logp'
            ]
            if self.model != 'cox':
                params_to_save.append('obs_prec')
        elif params_to_save is None:
            params_to_save = ['beta', 'global_shrinkage', 'logp']

        n_iter = n_burnin + n_post_burnin

        # Initial state of the Markov chain
        beta, obs_prec, lshrink, gshrink, init, initial_optim_info = \
            self.initialize_chain(init, shrinkage_exponent, n_init_optim_step)

        # Pre-allocate
        samples = {}
        sampling_info = {}
        self.manager.pre_allocate(
            samples, sampling_info, n_post_burnin, thin, params_to_save, sampling_method
        )

        # Start Gibbs sampling
        start_time = time.time()
        for mcmc_iter in range(1, n_iter + 1):

            beta, info = self.update_beta(
                beta, obs_prec, gshrink, lshrink, sampling_method, precond_blocksize
            )

            obs_prec = self.update_obs_precision(beta)

            # Draw from gshrink | \beta and then lshrink | gshrink, \beta.
            # (The order matters.)
            gshrink = self.update_global_shrinkage(
                gshrink, beta[self.n_unshrunk:], shrinkage_exponent, global_shrinkage_update)

            lshrink = self.update_local_shrinkage(
                gshrink, beta[self.n_unshrunk:], shrinkage_exponent)

            logp = self.compute_posterior_logprob(
                beta, gshrink, obs_prec, shrinkage_exponent
            )

            self.manager.store_current_state(
                samples, mcmc_iter, n_burnin, thin, beta, lshrink, gshrink,
                obs_prec, logp, params_to_save
            )
            self.manager.store_sampling_info(
                sampling_info, info, mcmc_iter, n_burnin, thin, sampling_method
            )

        runtime = time.time() - start_time

        _markov_chain_state = \
            self.manager.pack_parameters(beta, obs_prec, lshrink, gshrink)

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
            'initial_optimization_info': initial_optim_info,
            'reg_coef_sampling_info': sampling_info,
            '_markov_chain_state': _markov_chain_state,
            '_random_gen_state': self.rg.get_state(),
            '_reg_coef_sampler_state': self.reg_coef_sampler.get_internal_state()
        }
        if sampling_method == 'cg' and precond_blocksize > 0:
            mcmc_output['precond_blocksize'] = precond_blocksize

        return mcmc_output

    def initialize_chain(self, init, shrinkage_exponent, n_optim):
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
            obs_prec = np.mean((self.model.y - self.model.X.dot(beta)) ** 2) ** -1
        elif self.model.name == 'logit':
            obs_prec = LogisticModel.compute_polya_gamma_mean(
                self.model.n_trial, self.model.X.dot(beta)
            )
        else:
            obs_prec = None

        lshrink, gshrink = self.initialize_shrinkage_parameters(init, shrinkage_exponent)

        info_keys = ['is_success', 'n_design_matvec', 'n_iter']
        optim_info = {
            key: np.zeros(n_optim, dtype=np.int) for key in info_keys
        }
        optim_info['n_optim'] = n_optim
        for i in range(n_optim):
            beta, info = self.reg_coef_sampler.search_mode(
                beta, lshrink, gshrink, obs_prec, self.model
            )
            for key in info_keys:
                optim_info[key][i] = info[key]
            obs_prec = self.update_obs_precision(beta)
            lshrink = self.update_local_shrinkage(
                gshrink, beta[self.n_unshrunk:], shrinkage_exponent
            )
            
        init = {
            'beta': beta,
            'obs_prec': obs_prec,
            'local_shrinkage': lshrink,
            'global_shrinkage': gshrink
        }

        return beta, obs_prec, lshrink, gshrink, init, optim_info

    def initialize_shrinkage_parameters(
            self, init, shrinkage_exponent, apriori_coef_magnitude=.01):
        """
        Current options allow specifying 1) shrinkage parameters directly,
        2) regression coefficients only, and 3) none, which defaults to
        initialization based on 'apriori_coef_magnitude'.
        """

        if 'local_shrinkage' in init and  'global_shrinkage' in init:
            lshrink = init['local_shrinkage']
            gshrink = init['global_shrinkage']
            if not len(lshrink) == (self.n_pred - self.n_unshrunk):
                raise ValueError('An invalid initial state.')

        elif 'beta' in init:
            gshrink = self.update_global_shrinkage(
                None, init['beta'][self.n_unshrunk:], shrinkage_exponent,
                method='optimize'
            )
            lshrink = self.update_local_shrinkage(
                gshrink, init['beta'][self.n_unshrunk:], shrinkage_exponent
            )
        else:
            power_exponential_mean = (
                math.gamma(2 / shrinkage_exponent)
                / math.gamma(1 / shrinkage_exponent)
            )
            gshrink = apriori_coef_magnitude / power_exponential_mean
            lshrink = power_exponential_mean * np.ones(self.n_pred - self.n_unshrunk)

        return lshrink, gshrink

    def update_beta(self, beta, obs_prec, gshrink, lshrink, sampling_method, precond_blocksize):

        if sampling_method in ('direct', 'cg'):

            if self.model.name == 'linear':
                y_gaussian = self.model.y
                obs_prec = obs_prec * np.ones(self.n_obs)
            elif self.model.name == 'logit':
                y_gaussian = (self.model.n_success - self.model.n_trial / 2) / obs_prec

            beta, info = self.reg_coef_sampler.sample_gaussian_posterior(
                y_gaussian, self.model.X, obs_prec, gshrink, lshrink,
                sampling_method, precond_blocksize
            )

        elif sampling_method == 'hmc':
            beta, info = self.reg_coef_sampler.sample_by_hmc(
                beta, gshrink, lshrink, self.model
            )

        else:
            raise NotImplementedError()

        return beta, info

    def update_obs_precision(self, beta):

        obs_prec = None
        if self.model.name == 'linear':
            resid = self.model.y - self.model.X.dot(beta)
            scale = np.sum(resid ** 2) / 2
            obs_var = scale / self.rg.np_random.gamma(self.n_obs / 2, 1)
            obs_prec = 1 / obs_var
        elif self.model.name == 'logit':
            obs_prec = self.rg.polya_gamma(
                self.model.n_trial, self.model.X.dot(beta), self.model.X.shape[0])

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

    def compute_posterior_logprob(self, beta, gshrink, obs_prec, shrinkage_exponent):

        prior_logp = 0

        params = [beta] if self.model.name != 'linear' else [beta, obs_prec]
        loglik, _ = self.model.compute_loglik_and_gradient(*params, loglik_only=True)

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


class MarkovChainManager():

    def __init__(self, n_obs, n_pred, n_unshrunk, model_name):
        self.n_obs = n_obs
        self.n_pred = n_pred
        self.n_unshrunk = n_unshrunk
        self.model_name = model_name

    def merge_outputs(self, mcmc_output, next_mcmc_output):

        for output_key in ['samples']:
            curr_output = mcmc_output[output_key]
            next_output = next_mcmc_output[output_key]
            next_mcmc_output[output_key] = {
                key : np.concatenate(
                    (curr_output[key], next_output[key]), axis=-1
                ) for key in curr_output.keys()
            }

        next_mcmc_output['n_post_burnin'] += mcmc_output['n_post_burnin']
        next_mcmc_output['runtime'] += mcmc_output['runtime']

        return next_mcmc_output

    def pre_allocate(self, samples, sampling_info, n_post_burnin, thin, params_to_save, sampling_method):

        n_sample = math.floor(n_post_burnin / thin)  # Number of samples to keep

        if 'beta' in params_to_save:
            samples['beta'] = np.zeros((self.n_pred, n_sample))

        if 'local_shrinkage' in params_to_save:
            samples['local_shrinkage'] = np.zeros((self.n_pred - self.n_unshrunk, n_sample))

        if 'global_shrinkage' in params_to_save:
            samples['global_shrinkage'] = np.zeros(n_sample)

        if 'obs_prec' in params_to_save:
            if self.model_name == 'linear':
                samples['obs_prec'] = np.zeros(n_sample)
            elif self.model_name == 'logit':
                samples['obs_prec'] = np.zeros((self.n_obs, n_sample))

        if 'logp' in params_to_save:
            samples['logp'] = np.zeros(n_sample)

        for key in self.get_sampling_info_keys(sampling_method):
            sampling_info[key] = np.zeros(n_sample)

    def get_sampling_info_keys(self, sampling_method):
        if sampling_method == 'cg':
            keys = ['n_cg_iter']
        elif sampling_method == 'hmc':
            keys = [
                'n_integrator_step', 'accepted', 'accept_prob', 'stepsize',
                'n_hessian_matvec', 'stability_limit_est', 'stability_adjustment_factor'
            ]
        else:
            keys = []
        return keys

    def store_current_state(
            self, samples, mcmc_iter, n_burnin, thin, beta, lshrink,
            gshrink, obs_prec, logp, params_to_save):

        if mcmc_iter <= n_burnin or (mcmc_iter - n_burnin) % thin != 0:
            return

        index = math.floor((mcmc_iter - n_burnin) / thin) - 1

        if 'beta' in params_to_save:
            samples['beta'][:, index] = beta

        if 'local_shrinkage' in params_to_save:
            samples['local_shrinkage'][:, index] = lshrink

        if 'global_shrinkage' in params_to_save:
            samples['global_shrinkage'][index] = gshrink

        if 'obs_prec' in params_to_save:
            if self.model_name == 'linear':
                samples['obs_prec'][index] = obs_prec
            elif self.model_name == 'logit':
                samples['obs_prec'][:, index] = obs_prec

        if 'logp' in params_to_save:
            samples['logp'][index] = logp

    def store_sampling_info(
            self, sampling_info, info, mcmc_iter, n_burnin, thin, sampling_method):

        if mcmc_iter <= n_burnin or (mcmc_iter - n_burnin) % thin != 0:
            return

        index = math.floor((mcmc_iter - n_burnin) / thin) - 1
        for key in self.get_sampling_info_keys(sampling_method):
            sampling_info[key][index] = info[key]

    def pack_parameters(self, beta, obs_prec, lshrink, gshrink):
        state = {
            'beta': beta,
            'local_shrinkage': lshrink,
            'global_shrinkage': gshrink,
        }
        if self.model_name in ('linear', 'logit'):
            state['obs_prec'] = obs_prec
        return state