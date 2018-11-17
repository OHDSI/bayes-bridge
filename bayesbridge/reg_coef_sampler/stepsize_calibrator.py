from math import exp, log, sqrt
from bayesbridge.util import warn_message_only

class StepsizeCalibrator():

    def __init__(self, init_stepsize, target_accept_prob=.9,
                 method='robbins-monro', rm_options={}):
        """

        Parameters
        ----------
        method: str, {'robbins-monro', 'dual-average'}
        rm_options: dict with keys 'decay_scale' or ('final', 'n_total_step')
            Contains optional parameters for Robbins-Monro algorithm.
        """
        self.method = method
        if init_stepsize <= 0:
            raise ValueError("The initial stepsize must be positive.")
        log_init_stepsize = log(init_stepsize)
        self.log_stepsize = log_init_stepsize
        self.log_stepsize_averaged = log_init_stepsize
        self.latent_stat = 0. # Used for the dual-averaging algorithm.
        self.n_averaged = 0
        self.target_accept_prob = target_accept_prob

        if method == 'robbins-monro':
            self.rm_stepsize = RobbinsMonroStepsize(
                init=1., decay_exponent=2/3, options=rm_options
            )

        # Parameters for the dual-averaging algorithm.
        self.stepsize_averaging_log_decay_rate = 0.75
        self.latent_prior_samplesize = 10
        multiplier = 2. # > 1 to err on the side of shrinking toward a larger value.
        self.log_stepsize_shrinkage_mean = log(multiplier) + log_init_stepsize
        self.log_stepsize_shrinkage_strength = 0.05
            # Variable name is not quite accurate since this parameter interacts with latent_prior_samplesize.

    def get_current_stepsize(self, averaged=False):
        if averaged:
            return exp(self.log_stepsize_averaged)
        else:
            return exp(self.log_stepsize)

    def update_stepsize(self, accept_prob):
        self.n_averaged += 1

        if self.method == 'robbins-monro':

            rm_stepsize = self.rm_stepsize.next()
            self.log_stepsize += rm_stepsize * (accept_prob - self.target_accept_prob)
            weight = 1 / self.n_averaged
            self.log_stepsize_averaged = (
                weight * self.log_stepsize
                + (1 - weight) * self.log_stepsize_averaged
            )

        elif self.method == 'dual-average':

            self.latent_stat = self.update_latent_stat(
                accept_prob, self.target_accept_prob, self.latent_stat
            )
            self.log_stepsize, self.log_stepsize_averaged = self.dual_average_stepsize(
                self.latent_stat, self.log_stepsize_averaged
            )

        else:
            raise NotImplementedError()

        return exp(self.log_stepsize)

    def update_latent_stat(self, accept_prob, target_accept_prob, latent_stat):
        weight_latent = (self.n_averaged + self.latent_prior_samplesize) ** -1
        latent_stat = (1 - weight_latent) * latent_stat \
                      + weight_latent * (target_accept_prob - accept_prob)
        return latent_stat

    def dual_average_stepsize(self, latent_stat, log_stepsize_optimized):
        log_stepsize = (
            self.log_stepsize_shrinkage_mean
            - sqrt(self.n_averaged) / self.log_stepsize_shrinkage_strength * latent_stat
        )
        weight = self.n_averaged ** - self.stepsize_averaging_log_decay_rate
        log_stepsize_optimized = \
            (1 - weight) * log_stepsize_optimized + weight * log_stepsize
        return log_stepsize, log_stepsize_optimized


class RobbinsMonroStepsize():

    def __init__(self, init=1., decay_exponent=2./3., options={}):
        """
        Parameters
        ----------
        options: dict with keys ('decay_exponent', 'decay_scale', 'final', 'n_total_step')
            Either 'decay_scale' or ('final', 'n_total_step') can be specified
        """

        if 'decay_exponent' in options:
            decay_exponent = options['decay_exponent']

        self.n_iter = 0
        self.init = init
        self.exponent = decay_exponent
        self.scale = self.determine_decay_scale(init, decay_exponent, options)

    def determine_decay_scale(self, init, decay_exponent, options):

        if 'decay_scale' in options:
            decay_scale = options['decay_scale']
        elif ('final' in options) and ('n_total_step' in options):
            if 'decay_scale' in options:
                warn_message_only(
                    "Decay scale and final stepsize cannot be both specified."
                    "The decay scale parameter is ignored."
                )
            decay_scale = options['n_total_step'] / (
                (init / options['final']) ** (1 / decay_exponent) - 1
            )
        else:
            warn_message_only(
                'The default stepsize sequence tends to decay too quicky; '
                'consider manually setting the decay scale.'
            )
            decay_scale = 1.

        return decay_scale

    def next(self):
        stepsize = self.init / (1 + self.n_iter / self.scale) ** self.exponent
        self.n_iter += 1
        return stepsize
