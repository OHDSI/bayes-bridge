import math
from warnings import warn
import numpy as np
import scipy as sp
from scipy.special import polygamma as scipy_polygamma

class RegressionCoefPrior():

    def __init__(
            self,
            bridge_exponent=.5,
            n_fixed_effect=0,
            sd_for_intercept=float('inf'),
            sd_for_fixed_effect=float('inf'),
            regularizing_slab_size=float('inf'),
            global_scale_prior_hyper_param=None,
            _global_scale_parametrization='coef_magnitude'
        ):
        """ Encapisulate prior information for BayesBridge.

        Parameters
        ----------
        bridge_exponent : float < 2
            Exponent of the bridge prior on regression coefficients. For example,
            the value of 2 (albeit unsupported) would correspond to Gaussian prior
            and of 1 double-exponential as in Bayesian Lasso.
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
        global_scale_prior_hyper_param : dict, None
            Should contain pair of keys 'log10_mean' and 'log10_sd',
            specifying the prior mean and standard deviation of
            log10(global_scale). If None, the default reference prior for a
            scale parameter is used.

        Other Parameters
        ----------------
        _global_scale_parametrization: str, {'raw', 'coef_magnitude'}
            If 'coef_magnitude', scale the local and global scales so that the
            global scale parameter coincide with the prior expected
            magnitude of regression coefficients.
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
        self.bridge_exp = bridge_exponent
        self._gscale_paramet = _global_scale_parametrization
        if global_scale_prior_hyper_param is None:
            self.param = {
                'gscale_neg_power': {'shape': 0., 'rate': 0.},
                    # Reference prior for a scale family.
                'gscale': None
            }

        else:
            keys = global_scale_prior_hyper_param.keys()
            if not ({'log10_mean', 'log10_sd'} <= keys):
                raise ValueError(
                    "Dictionary should contain keys 'log10_mean' and 'log10_sd.'"
                )
            log10_mean = global_scale_prior_hyper_param['log10_mean']
            log10_sd = global_scale_prior_hyper_param['log10_sd']
            shape, rate = self.solve_for_gscale_prior_hyperparam(
                log10_mean, log10_sd, bridge_exponent, self._gscale_paramet
            )
            self.param = {
                'gscale_neg_power': {'shape': shape, 'rate': rate},
                'gscale': {'log10_mean': log10_mean, 'log10_sd': log10_sd}
            }   # Hyper-parameters on the negative power are specified in
                # terms of the 'raw' parametrization.

    def get_info(self):
        sd_for_fixed = self.sd_for_fixed
        if len(sd_for_fixed) > 0 and np.all(sd_for_fixed == sd_for_fixed[0]):
            sd_for_fixed = sd_for_fixed[0]
        info = {
            'bridge_exponent': self.bridge_exp,
            'n_fixed_effect': self.n_fixed,
            'sd_for_intercept': self.sd_for_intercept,
            'sd_for_fixed_effect': sd_for_fixed,
            'regularizing_slab_size': self.slab_size,
            'global_scale_prior_hyper_param': self.param['gscale'],
            '_global_scale_parametrization': self._gscale_paramet
        }
        return info

    def clone(self, **kwargs):
        """ Make a clone with only specified attributes modified. """
        info = self.get_info()
        if '_global_scale_parametrization' in kwargs:
            raise ValueError("Change of parametrization is not supported.")
        for key in kwargs.keys():
            if key in info:
                info[key] = kwargs[key]
            else:
                warn("'{:s} is not a valid keyward argument.".format(key))
        return RegressionCoefPrior(**info)

    def adjust_scale(self, gscale, lscale, to):
        unit_bridge_magnitude \
            = self.compute_power_exp_ave_magnitude(self.bridge_exp, 1.)
        if to == 'raw':
            gscale /= unit_bridge_magnitude
            lscale *= unit_bridge_magnitude
        elif to == 'coef_magnitude':
            gscale *= unit_bridge_magnitude
            lscale /= unit_bridge_magnitude
        else:
            raise ValueError()
        return gscale, lscale

    def solve_for_gscale_prior_hyperparam(
            self, log10_mean, log10_sd, bridge_exp, gscale_paramet):
        log_mean = self.change_log_base(log10_mean, from_=10., to=math.e)
        log_sd = self.change_log_base(log10_sd, from_=10., to=math.e)
        if gscale_paramet == 'coef_magnitude':
            unit_bridge_magnitude \
                = self.compute_power_exp_ave_magnitude(bridge_exp, 1.)
            log_mean -= math.log(unit_bridge_magnitude)
        shape, rate = self.solve_for_gamma_param(
            log_mean, log_sd, bridge_exp
        )
        return shape, rate

    @staticmethod
    def compute_power_exp_ave_magnitude(exponent, scale=1.):
        """ Returns the expected absolute value of a random variable with
        density proportional to exp( - |x / scale|^exponent ).
        """
        return scale * math.gamma(2 / exponent) / math.gamma(1 / exponent)

    @staticmethod
    def change_log_base(val, from_=math.e, to=10.):
        return val * math.log(from_) / math.log(to)

    def solve_for_gamma_param(self, log_mean, log_sd, bridge_exp):
        """ Find hyper-parameters matching specified mean and sd in log scale.

        Determine the shape and rate parameters of a Gamma prior on
            phi = gscale ** (- 1 / bridge_exp)
        so that the mean and sd of log(phi) coincide with log_mean and log_sd.
        The calculations are done in the 'raw' parametrization of gscale,
        as opposed to the 'coef_magnitude' parametrization.
        """

        f = lambda log_shape: (
            math.sqrt(self._polygamma(1, math.exp(log_shape))) / bridge_exp
            - log_sd
        ) # Function whose root coincides with the desired log-shape parameter.
        lower_lim = -10.  # Any sufficiently small number is fine.
        if log_sd < 0:
            raise ValueError("Variance has to be positive.")
        elif log_sd > 10 ** 8:
            raise ValueError("Specified prior variance is too large.")
        lower, upper = self._find_root_bounds(f, lower_lim)

        try:
            log_shape = sp.optimize.brentq(f, lower, upper)
        except BaseException as error:
            print('Solving for the global scale gamma prior hyper-parameters '
                  'failed; {}'.format(error))
        shape = math.exp(log_shape)
        rate = math.exp(
            self._polygamma(0, shape) + bridge_exp * log_mean
        )
        return shape, rate

    @staticmethod
    def _polygamma(n, x):
        """ Wrap the scipy function so that it returns a scalar. """
        return scipy_polygamma([n], x)[0]

    @staticmethod
    def _find_root_bounds(f, init_lower_lim, increment=5., max_lim=None):
        if max_lim is None:
            max_lim = init_lower_lim + 10 ** 4
        if f(init_lower_lim) < 0:
            raise ValueError(
                "Objective function must have positive value "
                "at the lower limit."
            )
        lower_lim = init_lower_lim
        while f(lower_lim + increment) > 0 and lower_lim < max_lim:
            lower_lim += increment
        if lower_lim >= max_lim:
            raise Exception()  # Replace with a warning.
        upper_lim = lower_lim + increment
        return (lower_lim, upper_lim)