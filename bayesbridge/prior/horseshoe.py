import math
from warnings import warn
import numpy as np
import scipy as sp
from ..random.local_scale_sampler import sample_horseshoe_local_scale
from .transfer_learning_helper import compute_horseshoe_lscale

class HorseshoePrior():

    def __init__(
            self,
            n_fixed_effect=0,
            sd_for_intercept=float('inf'),
            sd_for_fixed_effect=float('inf'),
            regularizing_slab_size=float('inf'),
            skew_mean=0.,
            skew_sd=1.,
            global_scale_prior=None
    ):
        """ Encapisulate horseshoe prior information for BayesBridge.

        Parameters
        ----------
        n_fixed_effect : int
            Number of predictors --- other than intercept and placed at the
            first columns of the design matrices --- whose coefficients are
            estimated with Gaussian priors of pre-specified standard
            deviation(s).
        sd_for_intercept : float
            Standard deviation of Gaussian prior on the intercept. `Inf`
            corresponds to an uninformative flat prior.
        sd_for_fixed_effect : float, numpy array
            Standard deviation(s) of Gaussian prior(s) on fixed effects.
            If an array, the length must be the same as `n_fixed_effect`.
            `Inf` corresponds to an uninformative flat prior.
        regularizing_slab_size : float
            Standard deviation of the Gaussian tail-regularizer on
            the bridge prior. Used to impose soft prior constraints on a
            range of regression coefficients in case the data provides limited
            information (e.g. when complete separation occurs). One may, for
            example, set the slab size by first choosing a value which
            regression coefficients are very unlikely to exceed in magnitude and
            then dividing the value by 1.96.
        skew_mean : float, numpy array
        skew_sd : float, numpy array
        global_scale_prior : callable, None
        """
        if not (np.isscalar(sd_for_fixed_effect)
                or n_fixed_effect == len(sd_for_fixed_effect)):
            raise ValueError(
                "Prior sd for fixed effects must be specified either by a "
                "scalar or array of the same length as n_fixed_effect."
            )

        if np.isscalar(sd_for_fixed_effect):
            sd_for_fixed_effect = sd_for_fixed_effect * np.ones(n_fixed_effect)
        self.sd_for_intercept = sd_for_intercept
        self.sd_for_fixed = sd_for_fixed_effect
        self.slab_size = regularizing_slab_size
        self.n_fixed = n_fixed_effect
        self.gscale_prior = global_scale_prior
        self.name = "horseshoe"
        self.skew_mean = skew_mean
        self.skew_sd = skew_sd
        self.bridge_exp = None
        self._gscale_paramet = None
        if self.gscale_prior is None:
            self.param = {
                'gscale_neg_power': {'shape': 0., 'rate': 0.},
                # Reference prior for a scale family.
                'gscale': None
            }

    def update_local_scale(self, gscale, coef, beta_mean_prior, beta_sd_prior, rg):
        # rg: random generator
        #     we might want to switch to the rg for sampling

        # TODO: Implemement, probably starting with the unskewed version.
        # the unskewed version uses the bridge prior
        # lscale = sample_horseshoe_local_scale(coef, gscale)

        # switch for the function that works for both skewed and centered version
        lscale, acc_count = compute_horseshoe_lscale(coef, gscale, self.skew_mean, self.skew_sd)

        # ar = 1 / np.mean(acc_count)

        # TODO: Pick the lower and upper bound more carefully.
        if np.any(lscale == 0):
            warn(
                "Local scale parameter under-flowed. Replacing with a small number.")
            lscale[lscale == 0] = 10e-16
        elif np.any(np.isinf(lscale)):
            warn(
                "Local scale parameter over-flowed. Replacing with a large number.")
            lscale[np.isinf(lscale)] = 2.0 / gscale

        return lscale
