import math
import time
from warnings import warn
import numpy as np


class SamplerOptions():

    def __init__(self, coef_sampler_type,
                 global_scale_update='sample',
                 hmc_curvature_est_stabilized=False):
        """
        Parameters
        ----------
        coef_sampler_type : {'cholesky', 'cg', 'hmc'}
        global_scale_update : str, {'sample', 'optimize', None}
        hmc_curvature_est_stabilized : bool
        """
        if coef_sampler_type not in ('cholesky', 'cg', 'hmc'):
            raise ValueError("Unsupported regression coefficient sampler.")
        self.coef_sampler_type = coef_sampler_type
        self.gscale_update = global_scale_update
        self.curvature_est_stabilized = hmc_curvature_est_stabilized

    def get_info(self):
        return {
            'coef_sampler_type': self.coef_sampler_type,
            'global_scale_update': self.gscale_update,
            'hmc_curvature_est_stabilized': self.curvature_est_stabilized
        }

    @staticmethod
    def pick_default_and_create(coef_sampler_type, options, model_name, design):
        """ Initialize class with, if unspecified, an appropriate default
        sampling method based on the type and size of model.
        """
        if options is None:
            options = {}

        if 'coef_sampler_type' in options:
            if coef_sampler_type is not None:
                warn("Duplicate specification of method for sampling "
                     "regression coefficient. Will use the dictionary one.")
            coef_sampler_type = options['coef_sampler_type']

        if coef_sampler_type not in (None, 'cholesky', 'cg', 'hmc'):
            raise ValueError("Unsupported sampler type.")

        if model_name in ('linear', 'logit'):

            n_obs, n_pred = design.shape
            if not design.is_sparse:
                preferred_method = 'cholesky'
            else:
                # TODO: Make more informed choice between Cholesky and CG.
                frac = design.nnz / (n_obs * n_pred)
                fisher_info_cost = frac ** 2 * n_obs * n_pred ** 2
                cg_cost = design.nnz * 100.
                preferred_method = 'cg' if cg_cost < fisher_info_cost \
                    else 'cholesky'

            # TODO: Implement Woodbury-based Gaussian sampler.
            if n_pred > n_obs:
                warn("Sampler has not been optimized for 'small n' problem.")

            if coef_sampler_type is None:
                coef_sampler_type = preferred_method
            elif coef_sampler_type not in ('hmc', preferred_method):
                warn("Specified sampler may not be optimal. Worth experimenting "
                     "with the '{:s}' option.".format(preferred_method))

        else:
            if coef_sampler_type != 'hmc':
                warn("Specified sampler type is not supported for the {:s} "
                     "model. Will use HMC instead.".format(model_name))
            coef_sampler_type = 'hmc'

        options['coef_sampler_type'] = coef_sampler_type
        return SamplerOptions(**options)


class MarkovChainManager():

    def __init__(self, n_obs, n_pred, n_unshrunk, model_name):
        self.n_obs = n_obs
        self.n_pred = n_pred
        self.n_unshrunk = n_unshrunk
        self.model_name = model_name
        self._prev_timestamp = None # For status update during Gibbs
        self._curr_timestamp = None

    def merge_outputs(self, prev_samples, prev_mcmc_info, new_samples, new_mcmc_info):

        new_samples = {
            key: np.concatenate(
                (prev_samples[key], new_samples[key]), axis=-1
            ) for key in new_samples.keys()
        }

        for output_key in ['_reg_coef_sampling_info']:
            prev_output = prev_mcmc_info[output_key]
            next_output = new_mcmc_info[output_key]
            new_mcmc_info[output_key] = {
                key : np.concatenate(
                    (prev_output[key], next_output[key]), axis=-1
                ) for key in prev_output.keys()
            }

        new_mcmc_info['n_iter'] += prev_mcmc_info['n_iter']
        new_mcmc_info['runtime'] += prev_mcmc_info['runtime']

        for output_key in ['_init_optim_info', 'seed']:
            new_mcmc_info[output_key] = prev_mcmc_info[output_key]

        return new_samples, new_mcmc_info

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
                     time_format='minute'):

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

        msg = " ".join((
            "{:d} Gibbs iterations complete:".format(mcmc_iter),
            time_str, "has elasped since the last update."
        ))
        print(msg)
        self._prev_timestamp = self._curr_timestamp