cimport cython
from libc.math cimport exp as exp_c
from libc.math cimport fabs, pow, log, sqrt, sin, floor, INFINITY, M_PI
import random
import numpy as np
cimport numpy as np
from numpy.random import PCG64
from numpy.random.bit_generator cimport BitGenerator
from bayesbridge.random.normal.normal cimport random_normal
from bayesbridge.random.uniform.uniform cimport random_uniform


cdef double MAX_EXP_ARG = 709  # ~ log(2 ** 1024)
ctypedef np.uint8_t np_uint8
ctypedef double (*rand_generator)()


cdef double exp(double x):
    if x > MAX_EXP_ARG:
        val = INFINITY
    elif x < - MAX_EXP_ARG:
        val = 0.
    else:
        val = exp_c(x)
    return val


@cython.cdivision(True)
cdef double sinc(double x):
    cdef double x_sq
    if fabs(x) < .01:
        x_sq = x * x
        val = 1. - x_sq / 6. * (1 - x_sq / 20.)
            # Taylor approximation with an error bounded by 2e-16
    else:
        val = sin(x) / x
    return val


cdef double python_builtin_next_double():
    return <double>random.random()


cdef class ExpTiltedStableDist():
    cdef rand_generator next_double
    cdef double TILT_POWER_THRESHOLD # For deciding the faster of two algorithms
    cdef BitGenerator bitgen

    def __init__(self, seed=None):
        self.set_seed(seed)
        self.bitgen = PCG64(seed)
        self.TILT_POWER_THRESHOLD = 2.

    def set_seed(self, seed):
        self.bitgen = PCG64(seed)

    def get_state(self):
        return self.bitgen.state

    def set_state(self, state):
        self.bitgen.state = state

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def sample(self, char_exponent, tilt, method=None):
        """
        Generate a random variable from a stable distribution with
            characteristic exponent =  char_exponent < 1
            skewness = 1
            scale = cos(char_exponent * pi / 2) ** (1 / char_exponent)
            location = 0
            exponential tilting = tilt
        (The density p(x) is tilted by exp(- tilt * x).)

        The cost of the divide-conquer algorithm increases as a function of
        'tilt ** char_exp'. While the cost of double-rejection algorithm is
        bounded, the divide-conquer algorithm is simpler and faster for small
        'tilt ** char_exp'.

        References:
        -----------
        Implementation is mostly based on the algorithm descriptions in
            'Sampling Exponentially Tilted Stable Distributions' by Hofert (2011)
        Ideas behind and details on the double-rejection sampling is better
        described in
            'Random variate generation for exponentially and polynomially tilted
            stable distributions' by Devroye (2009)
        """
        if not isinstance(tilt, np.ndarray):
            raise TypeError('Tilt parameter must be a numpy array.')
        if isinstance(char_exponent, (np.floating, float)):
            char_exponent = np.tile(char_exponent, tilt.size)
        elif isinstance(char_exponent, np.ndarray):
            if not char_exponent.size == tilt.size:
                raise ValueError('Input arrays must be of the same length.')
        else:
            raise TypeError('Characteristic exponent must be float or numpy array.')
        if not np.all(char_exponent < 1):
            raise ValueError('Characteristic exponent must be smaller than 1.')
        if not np.all(tilt > 0):
            raise ValueError('Tilting parameter must be positive.')

        if method is None:
            # Choose a likely faster method.
            divide_conquer_cost = tilt ** char_exponent
            double_rejection_cost = self.TILT_POWER_THRESHOLD
                # The relative costs are implementation & architecture dependent.
            use_divide_conquer = (divide_conquer_cost < double_rejection_cost)
        elif method in ['divide-conquer', 'double-rejection']:
            use_divide_conquer = np.tile(method == 'divide-conquer', tilt.size)
        else:
            raise ValueError("Unrecognized method name.")

        char_exponent = char_exponent.astype(np.double)
        tilt = tilt.astype(np.double)
        use_divide_conquer = use_divide_conquer.astype(np.uint8)
        result = np.zeros(tilt.size, dtype=np.double)

        cdef double[:] char_exponent_view = char_exponent
        cdef double[:] tilt_view = tilt
        cdef np_uint8[:] use_divide_conquer_view = use_divide_conquer
        cdef double[:] result_view = result
        cdef long n_sample = tilt.size
        cdef Py_ssize_t i

        for i in range(n_sample):
            if use_divide_conquer_view[i]:
                result_view[i] = self.sample_by_divide_and_conquer(
                    char_exponent_view[i], tilt_view[i]
                )
            else:
                result_view[i] = self.sample_by_double_rejection(
                    char_exponent_view[i], tilt_view[i]
                )
        return result

    cdef double sample_by_divide_and_conquer(self, double char_exp, double tilt):
        cdef double X, c
        cdef long partition_size = max(1, <long>floor(pow(tilt, char_exp)))
        X = 0.
        c = pow(1. / partition_size, 1. / char_exp)
        for i in range(partition_size):
            X += self.sample_divided_rv(char_exp, tilt, c)
        return X

    cdef double sample_divided_rv(self, double char_exp, double tilt, double c):
        cdef bint accepted = False
        while not accepted:
            S = c * self.sample_non_tilted_rv(char_exp)
            accept_prob = exp(- tilt * S)
            accepted = (random_uniform(self.bitgen) < accept_prob)
        return S

    cdef double sample_non_tilted_rv(self, double char_exp):
        cdef double S = pow(
            - self.zolotarev_function(M_PI * random_uniform(self.bitgen), char_exp)
                / log(random_uniform(self.bitgen)),
            (1. - char_exp) / char_exp
        )
        return S

    cdef double sample_by_double_rejection(self, double char_exp, double tilt):

        cdef double U, V, X, z, log_accept_prob
        cdef double tilt_power = pow(tilt, char_exp)

        # Start double-rejection sampling.
        cdef bint accepted = False
        while not accepted:
            U, V, z = self.sample_aux_rv(char_exp, tilt_power)
            X, log_accept_prob = \
                self.sample_reference_rv(U, char_exp, tilt_power, z)
            accepted = (log_accept_prob > log(V))

        return pow(X, - (1. - char_exp) / char_exp)

    cdef (double, double, double) \
        sample_aux_rv(self, double char_exp, double tilt_power):
        """
        Samples an auxiliary random variable for the double-rejection algorithm.
        Returns:
            U : auxiliary random variable for the double-rejection algorithm
            V : uniform random variable independent of U, X
            z : scalar quantity used later
        """
        cdef double U, V, z, accept_prob
        cdef double gamma, xi, psi
            # Intermediate quantities; could be computed outside the funciton
            # and reused in case of rejection
        gamma = tilt_power * char_exp * (1. - char_exp)
        xi = (1. + sqrt(2. * gamma) * (2. + sqrt(.5 * M_PI))) / M_PI
        psi = sqrt(gamma / M_PI) * (2. + sqrt(.5 * M_PI)) \
            * exp(- gamma * M_PI * M_PI / 8.)
        cdef bint accepted = False
        while not accepted:
            U = self.sample_aux2_rv(xi, psi, gamma)
            if U > M_PI:
                continue

            zeta = sqrt(self.zolotarev_pdf_exponentiated(U, char_exp))
            z = 1. / (1. - pow(1. + char_exp * zeta / sqrt(gamma), -1. / char_exp))
            accept_prob = self.compute_aux2_accept_prob(
                U, xi, psi, zeta, z, tilt_power, gamma
            )
            if accept_prob > 0.:
                V = random_uniform(self.bitgen) / accept_prob
                accepted = (U < M_PI and V <= 1.)

        return U, V, z

    cdef double sample_aux2_rv(self,
            double xi, double psi, double gamma):
        """
        Sample the 2nd level auxiliary random variable (i.e. the additional
        auxiliary random variable used to sample the auxilary variable for
        double-rejection algorithm.)
        """

        w1 = sqrt(.5 * M_PI / gamma) * xi
        w2 = 2. * sqrt(M_PI) * psi
        w3 = xi * M_PI
        V = random_uniform(self.bitgen)
        if gamma >= 1:
            if V < w1 / (w1 + w2):
                U = fabs(random_normal(self.bitgen)) / sqrt(gamma)
            else:
                W = random_uniform(self.bitgen)
                U = M_PI * (1. - W * W)
        else:
            W = random_uniform(self.bitgen)
            if V < w3 / (w2 + w3):
                U = M_PI * W
            else:
                U = M_PI * (1. - W * W)

        return U

    cdef double compute_aux2_accept_prob(self,
            double U, double xi, double psi, double zeta, double z,
            double tilt_power, double gamma
        ):
        inverse_accept_prob = M_PI * exp(-tilt_power * (1. - 1. / (zeta * zeta))) \
              / ((1. + sqrt(.5 * M_PI)) * sqrt(gamma) / zeta + z)
        d = 0.
        if U >= 0. and gamma >= 1:
            d += xi * exp(-gamma * U * U / 2.)
        if U > 0. and U < M_PI:
            d += psi / sqrt(M_PI - U)
        if U >= 0. and U <= M_PI and gamma < 1.:
            d += xi
        inverse_accept_prob *= d
        accept_prob = 1 / inverse_accept_prob
        return accept_prob

    cdef (double, double) sample_reference_rv(self,
            double U, double char_exp, double tilt_power, double z):
        """
        Generate a sample from the reference (augmented) distribution conditional
        on U for the double-rejection algorithm. The algorithm use a rejection
        sampler with half-Gaussian, uniform, and truncated exponential to the
        left, middle, and right of a partitioned real-line.

        Returns:
        --------
            X : random variable from the reference distribution
            N, E : random variables used later for computing the acceptance prob
        """
        cdef double a, left_thresh, right_thresh, expo_scale, \
            mass_left, mass_mid, mass_right, mass_total, X, V, N, E
        a = self.zolotarev_function(U, char_exp)
        left_thresh = pow((1. - char_exp) / char_exp / a, char_exp) * tilt_power
        right_thresh = left_thresh + sqrt(left_thresh * char_exp / a)
        expo_scale = z / a
        mass_left = (right_thresh - left_thresh) * sqrt(.5 * M_PI)
        mass_mid = (right_thresh - left_thresh)
        mass_right = expo_scale
        mass_total = mass_left + mass_mid + mass_right
        V = random_uniform(self.bitgen)
        N = 0.
        E = 0.
        # Divided into three pieces at left_thresh and (left_thresh + mid_width)
        if V < mass_left / mass_total:
            N = random_normal(self.bitgen)
            X = left_thresh - (right_thresh - left_thresh) * fabs(N)
        elif V < (mass_left + mass_mid) / mass_total:
            X = left_thresh + (right_thresh - left_thresh) * random_uniform(self.bitgen)
        else:
            E = - log(random_uniform(self.bitgen))
            X = right_thresh + E * mass_right

        log_accept_prob = self.compute_log_accept_prob(
            X, N, E, left_thresh, right_thresh, a, char_exp, tilt_power
        )
        return X, log_accept_prob

    cdef double compute_log_accept_prob(self,
            double X, double N, double E, double left_thresh, double right_thresh,
            double a, double char_exp, double tilt_power
        ):
        cdef double char_exp_odds = (1. - char_exp) / char_exp
        if X < 0:
            log_accept_prob = - INFINITY
        else:
            log_accept_prob = - (
                a * (X - left_thresh)
                + exp(log(tilt_power) / char_exp - char_exp_odds * log(left_thresh))
                * (pow(left_thresh / X, char_exp_odds) - 1.)
            )
            if X < left_thresh:
                log_accept_prob += N * N / 2.
            elif X > right_thresh:
                log_accept_prob += E

        return log_accept_prob

    cdef double zolotarev_pdf_exponentiated(self, double x, double char_exp):
        """
        Evaluates a function proportional to a power of the Zolotarev density.
        """
        cdef double denominator, numerator
        denominator = pow(sinc(char_exp * x), char_exp) \
                      * pow(sinc((1. - char_exp) * x), (1. - char_exp))
        numerator = sinc(x)
        return numerator / denominator

    cdef double zolotarev_function(self, double x, double char_exp):
        cdef double val = pow(
            pow((1. - char_exp) * sinc((1. - char_exp) * x), (1. - char_exp))
            * pow(char_exp * sinc(char_exp * x), char_exp)
            / sinc(x)
        , 1. / (1. - char_exp))
        return val
