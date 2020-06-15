import math
import time
from warnings import warn
import numpy as np


class SamplerOptions():

    def __init__(self, regress_coef_sampler,
                 global_scale_update='sample',
                 hmc_curvature_est_stabilized=False):
        """
        Parameters
        ----------
        regress_coef_sampler : {'cholesky', 'cg', 'hmc'}
        global_scale_update : str, {'sample', 'optimize', None}
        hmc_curvature_est_stabilized : bool
        """
        if regress_coef_sampler not in ('cholesky', 'cg', 'hmc'):
            raise ValueError("Unsupported regression coefficient sampler.")
        self.coef_sampler_type = regress_coef_sampler
        self.gscale_update = global_scale_update
        self.curvature_est_stabilized = hmc_curvature_est_stabilized

    def get_info(self):
        return {
            'regress_coef_sampler': self.coef_sampler_type,
            'global_scale_update': self.gscale_update,
            'hmc_curvature_est_stabilized': self.curvature_est_stabilized
        }

    @staticmethod
    def create(regress_coef_sampler, options, model_name, n_obs, n_pred):
        """ Initialize class with, if unspecified, an appropriate default
        sampling method based on the type and size of model.
        """
        if options is None:
            options = {}

        if 'regress_coef_sampler' in options:
            if regress_coef_sampler is not None:
                warn("Duplicate specification of method for sampling "
                     "regression coefficient. Will use the dictionary one.")
            regress_coef_sampler = options['regress_coef_sampler']

        if model_name in ('linear', 'logit'):

            # TODO: Make the choice between Cholesky and CG more carefully.
            MATMUL_COST_THRESHOLD = 10 ** 12
            # TODO: Implement Woodbury-based Gaussian sampler.
            if n_pred > n_obs:
                warn("Sampler has not been optimized for 'small n' problem.")

            smaller_dim_size = min(n_obs, n_pred)
            larger_dim_size = max(n_obs, n_pred)
            matmul_cost = smaller_dim_size ** 2 * larger_dim_size
            direct_linalg_preferred = (matmul_cost < MATMUL_COST_THRESHOLD)

            if regress_coef_sampler is None:
                if direct_linalg_preferred:
                    regress_coef_sampler = 'cholesky'
                else:
                    regress_coef_sampler = 'cg'
            else:
                if regress_coef_sampler == 'cg' and direct_linalg_preferred:
                    warn("Design matrix may be too small to benefit from the "
                         "conjugate gradient sampler.")

        else:
            if regress_coef_sampler != 'hmc':
                warn("Specified sampler is not supported for the {:s} "
                     "model. Will use HMC instead.".format(model_name))
            regress_coef_sampler = 'hmc'

        options['regress_coef_sampler'] = regress_coef_sampler
        return SamplerOptions(**options)


class MarkovChainManager():

    def __init__(self, n_obs, n_pred, n_unshrunk, model_name):
        self.n_obs = n_obs
        self.n_pred = n_pred
        self.n_unshrunk = n_unshrunk
        self.model_name = model_name
        self._prev_timestamp = None # For status update during Gibbs
        self._curr_timestamp = None

    def merge_outputs(self, mcmc_output, next_mcmc_output):

        for output_key in ['samples', '_reg_coef_sampling_info']:
            curr_output = mcmc_output[output_key]
            next_output = next_mcmc_output[output_key]
            next_mcmc_output[output_key] = {
                key : np.concatenate(
                    (curr_output[key], next_output[key]), axis=-1
                ) for key in curr_output.keys()
            }

        next_mcmc_output['n_post_burnin'] += mcmc_output['n_post_burnin']
        next_mcmc_output['runtime'] += mcmc_output['runtime']

        for output_key in ['initial_optimization_info', 'seed']:
            next_mcmc_output[output_key] = mcmc_output[output_key]

        return next_mcmc_output

    def pre_allocate(self, samples, sampling_info, n_post_burnin, thin, params_to_save, sampling_method):

        n_sample = math.floor(n_post_burnin / thin)  # Number of samples to keep

        if 'coef' in params_to_save:
            samples['coef'] = np.zeros((self.n_pred, n_sample))

        if 'local_scale' in params_to_save:
            samples['local_scale'] = np.zeros((self.n_pred - self.n_unshrunk, n_sample))

        if 'global_scale' in params_to_save:
            samples['global_scale'] = np.zeros(n_sample)

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
        elif sampling_method in ['hmc', 'nuts']:
            keys = [
                'stepsize', 'n_hessian_matvec', 'n_grad_evals',
                'stability_limit_est', 'stability_adjustment_factor',
                'instability_detected'
            ]
            if sampling_method == 'hmc':
                keys += ['n_integrator_step', 'accepted', 'accept_prob']
            else:
                keys += ['tree_height', 'ave_accept_prob']
        else:
            keys = []
        return keys

    def store_current_state(
            self, samples, mcmc_iter, n_burnin, thin, coef, lscale,
            gscale, obs_prec, logp, params_to_save):

        if mcmc_iter <= n_burnin or (mcmc_iter - n_burnin) % thin != 0:
            return

        index = math.floor((mcmc_iter - n_burnin) / thin) - 1

        if 'coef' in params_to_save:
            samples['coef'][:, index] = coef

        if 'local_scale' in params_to_save:
            samples['local_scale'][:, index] = lscale

        if 'global_scale' in params_to_save:
            samples['global_scale'][index] = gscale

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

    def pack_parameters(self, coef, obs_prec, lscale, gscale):
        state = {
            'coef': coef,
            'local_scale': lscale,
            'global_scale': gscale,
        }
        if self.model_name in ('linear', 'logit'):
            state['obs_prec'] = obs_prec
        return state

    def stamp_time(self, curr_time):
        self._prev_timestamp = curr_time

    def print_status(self, n_status_update, mcmc_iter, n_iter,
                     msg_type='sampling', time_format='minute'):

        if n_status_update == 0:
            return
        n_iter_per_update = int(n_iter / n_status_update)
        if mcmc_iter % n_iter_per_update != 0:
            return

        self._curr_timestamp = time.time()

        time_elapsed = self._curr_timestamp - self._prev_timestamp
        if time_format == 'second':
            time_str = "{:.3g} seconds".format(time_elapsed)
        elif time_format == 'minute':
            time_str = "{:.3g} minutes".format(time_elapsed / 60)
        else:
            raise ValueError()

        if msg_type == 'optim':
            msg = "Initial optimization took " + time_str + "."
        else:
            msg = " ".join((
                "{:d} Gibbs iterations complete:".format(mcmc_iter),
                time_str, "has elasped since the last update."
            ))
        print(msg)
        self._prev_timestamp = self._curr_timestamp