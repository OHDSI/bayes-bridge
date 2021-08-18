# cython: cdivision = True
from libc.math cimport exp, log, sqrt, fabs, M_PI
import random
import cython
import numpy as np
cimport numpy as np
from numpy.random import PCG64
from numpy.random.bit_generator cimport BitGenerator
from scipy_ndtr cimport log_ndtr as normal_logcdf
from bayesbridge.random.normal.normal cimport random_normal
from bayesbridge.random.uniform.uniform cimport random_uniform



cdef class PolyaGammaDist():
    # Threshold below (and above) which the target density is bounded by inverse
    # Gaussian (and exponential) and have different analytical series expressions.
    cdef double THRESHOLD
    # Number of terms in the infinite alternating series beyond which to truncate.
    cdef int MAX_SERIES_TERMS
    cdef BitGenerator bitgen

    def __init__(self, seed=None):
        self.set_seed(seed)
        self.THRESHOLD = 2.0 / M_PI
        self.MAX_SERIES_TERMS = 100
        self.bitgen = PCG64(seed)

    def set_seed(self, seed):
        self.bitgen = PCG64(seed)

    def get_state(self):
        return self.bitgen.state

    def set_state(self, state):
        self.bitgen.state = state

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def rand_polyagamma(self, shape, tilt):
        """
        Sample from exponentially tilted Polya-Gamma distribution
            p(x | shape, tilt) \propto \exp(- tilt^2 / 2 * x) p(x | shape, tilt=0)
        via Devroye's alternatig series method.

        Parameters
        ----------
        shape : numpy array of integers
            Distribution is defined for non-integer values but the implemented
            algorithm only support integer values.
        tilt : numpy array of doubles
            Sqrt of twice the negative tilting parameter, actually.
        """

        if not isinstance(shape, np.ndarray) and isinstance(tilt, np.ndarray):
            raise TypeError('Input must be numpy arrays.')
        if not shape.size == tilt.size:
            raise ValueError('Input arrays must be of the same length.')
        if not np.issubdtype(shape.dtype, np.integer):
            raise ValueError('Shape parameter must be integers.')
        shape = shape.astype(np.intc)
        tilt = tilt.astype(np.double)
        result = np.zeros(shape.size, dtype=np.double)

        cdef int[:] shape_view = shape
        cdef double[:] tilt_view = tilt
        cdef double[:] result_view = result
        cdef long n_samples = shape_view.size
        cdef Py_ssize_t index, j
        for index in range(n_samples):
            for j in range(shape_view[index]):
                result_view[index] \
                    += self.rand_scalar_unit_shape_polyagamma(tilt_view[index])
        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def rand_unit_shape_polyagamma(self, tilt):

        if not isinstance(tilt, np.ndarray):
            raise TypeError('Input must be numpy arrays.')
        tilt = tilt.astype(np.double)
        result = np.empty(tilt.size, dtype=np.double)

        cdef double[:] tilt_view = tilt
        cdef double[:] result_view = result
        cdef long n_samples = tilt_view.size
        cdef Py_ssize_t index
        for index in range(n_samples):
            result_view[index] \
                = self.rand_scalar_unit_shape_polyagamma(tilt_view[index])
        return result

    cdef double rand_scalar_unit_shape_polyagamma(self, double tilt):
        return .25 * self.rand_tilted_jocobi(.5 * fabs(tilt))

    cdef double rand_tilted_jocobi(self, double tilt):
        """
        Sample from tilted Jacobi distribution
            p(x | tilt) \propto \exp(- tilt^2 / 2 * x) p(x | 0)
        via Devroye's alternatig series method.
        """
        cdef double X, U, proposal_density
        cdef bint accepted = False

        # Main sampling loop page 130 of the Windle PhD thesis
        while not accepted:
            X, proposal_density = self.rand_proposal(tilt)
            U = random_uniform(self.bitgen) * proposal_density
            accepted = self.decide_acceptability(U, X, proposal_density)

        return X

    cdef (double, double) rand_proposal(self, double tilt):
        # Many quantities here can be cached and reused in case of rejection, but
        # the acceptance rate is so high that it does not matter.
        cdef double exp_rate = .5 * tilt ** 2 + .125 * M_PI ** 2
        cdef double prob_to_right = self.calc_prob_to_right(tilt, exp_rate)
        if random_uniform(self.bitgen) < prob_to_right:
            X = self.rand_left_truncated_exp(1. / exp_rate, self.THRESHOLD)
        else:
            X = self.rand_right_truncated_unit_shape_invgauss(tilt, self.THRESHOLD)
        proposal_density = self.calc_next_term_in_series(0, X)
        return X, proposal_density

    cdef double calc_prob_to_right(self, double tilt, double exp_rate):
        cdef double log_mass_expo \
            = - log(exp_rate) - exp_rate * self.THRESHOLD + log(.25 * M_PI)
        cdef double log_mass_invg_1 \
            = - tilt + normal_logcdf(
                (self.THRESHOLD * tilt - 1.) / sqrt(self.THRESHOLD))
        cdef double log_mass_invg_2 \
            = tilt + normal_logcdf(
                - (self.THRESHOLD * tilt + 1.) / sqrt(self.THRESHOLD))
        cdef double mass_ratio = (
            exp(log_mass_invg_1 - log_mass_expo)
            + exp(log_mass_invg_2 - log_mass_expo)
        )
        return 1.0 / (1.0 + mass_ratio)

    # Equations (12) and (13) of Polson, Scott, and Windle (2013)
    cdef double calc_next_term_in_series(self, int n, double x):
        cdef double log_result = log(M_PI * (n + 0.5))
        if x <= self.THRESHOLD:
            log_result += - 1.5 * log(.5 * x * M_PI) - 2 * (n + 0.5) ** 2 / x
        else:
            log_result += - 0.5 * x * M_PI ** 2 * (n + 0.5) ** 2
        return exp(log_result)

    cdef bint decide_acceptability(self, double U, double X, double zeroth_term):

        cdef double partial_sum = zeroth_term
        cdef int n_summed = 1
        cdef int sign = -1 # Sign of the next term in the alternating sequence
        cdef bint acceted
        cdef bint is_determinate = False

        while not is_determinate:
            partial_sum += sign * self.calc_next_term_in_series(n_summed, X)
            n_summed += 1
            if sign == -1:
                if U <= partial_sum:
                    accepted = True
                    is_determinate = True
            else: # sign == 1
                if U > partial_sum:
                    accepted = False
                    is_determinate = True
                elif n_summed >= self.MAX_SERIES_TERMS:
                    acceted = True # Take the partial sum lower-bound as the target
                    is_determinate = True
            sign = - sign

        return accepted

    cdef double rand_left_truncated_exp(self, double scale, double trunc):
        return trunc - scale * log(1.0 - random_uniform(self.bitgen))

    # Ref: "Simulation of truncated gamma variables" by Younshik Chung
    # Korean Journal of Computational & Applied Mathematics, 1998
    cdef double rand_left_truncated_chisq(self, double trunc):
        cdef double X, density_ratio
        cdef bint accepted = False
        while not accepted:
            X = self.rand_left_truncated_exp(2., trunc)
            density_ratio = sqrt(0.5 * M_PI / X)
            accepted = (random_uniform(self.bitgen) <= density_ratio)
        return X


    cdef double rand_right_truncated_unit_shape_invgauss(self, double rate, double trunc):
        # Shape parameter is assumed to be one.
        cdef double X
        cdef double mean = 1. / rate
        cdef bint accepted = False

        # Choose a better sampler depending on the input parameters
        if mean > trunc:
            # Algorithm 3 in Windle's PhD thesis, page 128
            while not accepted:
                X = 1.0 / self.rand_left_truncated_chisq(.5 * M_PI)
                accepted = (log(random_uniform(self.bitgen)) < - 0.5 * X * rate ** 2)
        else:
            while not accepted:
                X = self.rand_unit_shape_invgauss(mean)
                accepted = (X < trunc)
        return X

    cdef double rand_unit_shape_invgauss(self, double mean):
        cdef double V = random_normal(self.bitgen) ** 2
        cdef double X = mean + 0.5 * mean * (
            mean * V - sqrt(4.0 * mean * V + mean ** 2 * V ** 2)
        )
        if random_uniform(self.bitgen) > mean / (mean + X):
            X = mean ** 2 / X
        return X

