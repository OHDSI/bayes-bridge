from math import exp, log, log10, sqrt, copysign
from .util import warn_message_only
import scipy.stats as stats


class HamiltonianBasedStepsizeAdapter():
    """
    Updates the stepsize of an HMC integrator so that the average Hamiltonian
    error matches a pre-specified target value.
    """

    def __init__(self, init_stepsize, target_accept_prob=.9,
                 init_adaptsize=1., adapt_decay_exponent=1.,
                 reference_iteration=500, adaptsize_at_reference=.05):
        """
        Parameters
        ----------
        reference_iteration & adaptsize_at_reference:
            Stepsize sequence of Robbins-Monro algorithm will be set so that it
            decreases to `adaptsize_at_refrence` after `reference_iteration`.
        """
        if init_stepsize <= 0:
            raise ValueError("The initial stepsize must be positive.")
        log_init_stepsize = log(init_stepsize)
        self.log_stepsize = log_init_stepsize
        self.log_stepsize_averaged = log_init_stepsize
        self.n_averaged = 0
        self.target_accept_prob = target_accept_prob
        self.target_log10_hamiltonian_error \
            = self.convert_to_log_hamiltonian_error(target_accept_prob)

        self.rm_stepsizer = RobbinsMonroStepsizer(
            init=init_adaptsize,
            decay_exponent=adapt_decay_exponent,
            reference_iteration=reference_iteration,
            size_at_reference=adaptsize_at_reference
        )

    @staticmethod
    def convert_to_log_hamiltonian_error(target_accept_prob):
        """ Calculate the target squared Hamiltonian error in the log scale.

        Under a high-dimensional limit of i.i.d. parameters, the Hamiltonian
        error is distributed as
            Normal(mean = - delta / 2, var = delta),
        and the corresponding average acceptance rate is
            2 GausssianCDF(- sqrt(delta) / 2).
        So we solve for `delta` that theoretically achieves the target acceptance
        rate and try to calibrate the average square error of the Hamiltonian
        to be the theoretical value (delta^2 / 4 + delta).
        """
        if target_accept_prob <= 0 or target_accept_prob >= 1:
            raise ValueError("Target probability must be within (0, 1).")
        delta = 4 * stats.norm.ppf(target_accept_prob / 2) ** 2
        target_log10_hamiltonian_error = .5 * log10(delta + delta ** 2 / 4)
        return target_log10_hamiltonian_error

    def get_current_stepsize(self, averaged=False):
        if averaged:
            return exp(self.log_stepsize_averaged)
        else:
            return exp(self.log_stepsize)

    def reinitialize(self, init_stepsize):
        log_init_stepsize = log(init_stepsize)
        self.log_stepsize = log_init_stepsize
        self.log_stepsize_averaged = log_init_stepsize
        self.n_averaged = 0

    def adapt_stepsize(self, hamiltonian_error):
        rm_stepsize = self.rm_stepsizer.calculate_stepsize(self.n_averaged)
        self.n_averaged += 1
        adaptsize = self.transform_to_adaptsize(hamiltonian_error)
        self.log_stepsize += rm_stepsize * adaptsize
        weight = 1 / self.n_averaged
        self.log_stepsize_averaged = (
            weight * self.log_stepsize
            + (1 - weight) * self.log_stepsize_averaged
        )
        return exp(self.log_stepsize)

    def transform_to_adaptsize(
            self, error, upper_bound=1., trans_type='piecewise'):
        """
        Parameters
        ----------
        trans_type: str, {'log-linear', 'sign', 'piecewise'}
        """

        if trans_type == 'probability':
            accept_prob = min(1, exp(error))
            adapt_size = accept_prob - self.target_accept_prob
            return adapt_size

        if error == 0.:
            log10_error = - float('inf')
        else:
            log10_error = log10(abs(error))

        target = self.target_log10_hamiltonian_error
        if trans_type == 'log-linear':
            adapt_size = target - log10_error

        elif trans_type == 'sign':
            adapt_size = copysign(1., target - log10_error)

        elif trans_type == 'piecewise':
            # Increase the adjustment when the error is larger than the target.
            if log10_error > target:
                adapt_size = (target - log10_error) / .301 # Convert to log2 scale.
            else:
                adapt_size = (target - log10_error) / 3 # Convert to log1000 scale.

        else:
            raise NotImplementedError()

        if abs(adapt_size) > upper_bound:
            adapt_size = copysign(1., adapt_size)

        return adapt_size


def initialize_stepsize(compute_acceptprob, dt=1.0):
    """ Heuristic for choosing an initial value of dt

    Parameters
    ----------
    compute_acceptprob: callable
        Computes the acceptance probability of the proposal one-step HMC proposal.
    """

    # Figure out what direction we should be moving dt.
    acceptprob = compute_acceptprob(dt)
    direc = 2 * int(acceptprob > 0.5) - 1

    # Keep moving dt in that direction until acceptprob crosses 0.5.
    while acceptprob == 0 or (2 * acceptprob) ** direc > 1:
        dt = dt * (2 ** direc)
        acceptprob = compute_acceptprob(dt)
        if acceptprob == 0 and direc == 1:
            # The last doubling of stepsize was too much.
            dt /= 2
            break

    return dt


class RobbinsMonroStepsizer():

    def __init__(self, init=1., decay_exponent=1.,
                 reference_iteration=None, size_at_reference=None):
        self.init = init
        self.exponent = decay_exponent
        self.scale = self.determine_decay_scale(
            init, decay_exponent, reference_iteration, size_at_reference
        )

    def determine_decay_scale(self, init, decay_exponent, ref_iter, size_at_ref):

        if (ref_iter is not None) and (size_at_ref is not None):
            decay_scale = \
                ref_iter / ((init / size_at_ref) ** (1 / decay_exponent) - 1)
        else:
            warn_message_only(
                'The default stepsize sequence tends to decay too quicky; '
                'consider manually setting the decay scale.'
            )
            decay_scale = 1.

        return decay_scale

    def __iter__(self):
        self.n_iter = 0
        return self

    def __next__(self):
        stepsize = self.calculate_stepsize(self.n_iter)
        self.n_iter += 1
        return stepsize

    def calculate_stepsize(self, n_iter):
        stepsize = self.init / (1 + n_iter / self.scale) ** self.exponent
        return stepsize


class RobbinsMonroStepsizeAdapter():

    def __init__(self, init_stepsize, target_accept_prob=.9,
                 init_adaptsize=1., adapt_decay_exponent=1.,
                 reference_iteration=100, adaptsize_at_reference=.05):
        """
        Parameters
        ----------
        reference_iteration & adaptsize_at_reference:
            Stepsize sequence of Robbins-Monro algorithm will be set so that it
            decreases to `adaptsize_at_refrence` after `reference_iteration`.
        """
        if init_stepsize <= 0:
            raise ValueError("The initial stepsize must be positive.")
        log_init_stepsize = log(init_stepsize)
        self.log_stepsize = log_init_stepsize
        self.log_stepsize_averaged = log_init_stepsize
        self.n_averaged = 0
        self.target_accept_prob = target_accept_prob

        self.rm_stepsizer = iter(RobbinsMonroStepsizer(
            init=init_adaptsize,
            decay_exponent=adapt_decay_exponent,
            reference_iteration=reference_iteration,
            size_at_reference=adaptsize_at_reference
        ))

    def get_current_stepsize(self, averaged=False):
        if averaged:
            return exp(self.log_stepsize_averaged)
        else:
            return exp(self.log_stepsize)

    def adapt_stepsize(self, accept_prob):
        self.n_averaged += 1
        rm_stepsize = next(self.rm_stepsizer)
        adaptsize = \
            self.transform_to_adaptsize(accept_prob, self.target_accept_prob)
        self.log_stepsize += rm_stepsize * adaptsize
        weight = 1 / self.n_averaged
        self.log_stepsize_averaged = (
            weight * self.log_stepsize
            + (1 - weight) * self.log_stepsize_averaged
        )
        return exp(self.log_stepsize)

    def transform_to_adaptsize(
            self, accept_prob, target, trans_type='linear'):
        """
        Parameters
        ----------
        trans_type: str, {'linear', 'sign', 'penalize-high-prob'}
        """

        if trans_type == 'linear':
            adapt_size = accept_prob - target

        elif trans_type == 'sign':
            adapt_size = copysign(1., accept_prob - target)

        elif trans_type == 'penalize-high-prob':
            # Transforms accept_prob -> adapt_size so that it roughly interpolate
            # the points (0, -1), (target, 0), and (1, 1). Transformation is
            # linear near accept_prob = target but quickly goes up to
            # adapt_size = 1 as (1 - accecpt_prob) becomes an order of manitude
            # smaller than (1 - target).
            if accept_prob <= target:
                adapt_size = (accept_prob - target) / target
            else:
                epsilon = 2. ** -52
                magnitude_diff = log10(
                    (1. - (accept_prob - epsilon)) / (1 - target)
                )
                if magnitude_diff == 0:
                    w = 0.
                else:
                    w = exp(magnitude_diff ** - 1)
                adapt_size = (
                    (1 - w) * (accept_prob - target) / target
                    - w * magnitude_diff
                )
                adapt_size = min(1., adapt_size)

        else:
            raise NotImplementedError()

        return adapt_size


class DualAverageStepsizeAdapter():

    def __init__(self, init_stepsize, target_accept_prob=.9):

        if init_stepsize <= 0:
            raise ValueError("The initial stepsize must be positive.")
        log_init_stepsize = log(init_stepsize)
        self.log_stepsize = log_init_stepsize
        self.log_stepsize_averaged = log_init_stepsize
        self.n_averaged = 0
        self.target_accept_prob = target_accept_prob
        self.latent_stat = 0.  # Used for dual-averaging.

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

    def adapt_stepsize(self, accept_prob):
        self.n_averaged += 1
        self.latent_stat = self.update_latent_stat(
            accept_prob, self.target_accept_prob, self.latent_stat
        )
        self.log_stepsize, self.log_stepsize_averaged = self.dual_average_stepsize(
            self.latent_stat, self.log_stepsize_averaged
        )
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