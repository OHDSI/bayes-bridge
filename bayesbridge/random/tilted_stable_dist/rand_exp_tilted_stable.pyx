cimport cython
from libc.math cimport exp as exp_c
from libc.math cimport fabs, pow, log, sqrt, sin, floor
from libc.math cimport INFINITY, M_PI
import math
import random
cdef double MAX_EXP_ARG = 709  # ~ log(2 ** 1024)
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

    def __init__(self, seed=None):
        random.seed(seed)
        self.next_double = python_builtin_next_double

    def get_state(self):
        return random.getstate()

    def set_state(self, state):
        random.setstate(state)

    def rv(self, char_exponent, tilt, method=None):
        """
        Generate a random variable from a stable distribution with
            characteristic exponent =  char_exponent < 1
            skewness = 1
            scale = cos(char_exponent * pi / 2) ** (1 / char_exponent)
            location = 0
            exponential tilting = tilt
        (The density p(x) is tilted by exp(- tilt * x).)

        The cost of the divide-conquer algorithm increases as a function of
        'tilt ** alpha'. While the cost of double-rejection algorithm is
        bounded, the divide-conquer algorithm is simpler and faster for small
        'tilt ** alpha'.

        References:
        -----------
        Implementation is mostly based on the algorithm descriptions in
            'Sampling Exponentially Tilted Stable Distributions' by Hofert (2011)
        Ideas behind and details on the double-rejection sampling is better
        described in
            'Random variate generation for exponentially and polynomially tilted
            stable distributions' by Devroye (2009)
        """

        if method is None:
            # Choose a likely faster method.
            divide_conquer_cost = pow(tilt, char_exponent)
            double_rejection_cost = 5.0
                # The relative costs are implementation & architecture dependent.
            if divide_conquer_cost < double_rejection_cost:
                method = 'divide-conquer'
            else:
                method = 'double-rejection'

        if method == 'divide-conquer':
            X = self.sample_by_divide_and_conquer(char_exponent, tilt)
        elif method == 'double-rejection':
            X = self.sample_by_double_rejection(char_exponent, tilt)
        else:
            raise NotImplementedError()

        return X

    cdef double sample_by_divide_and_conquer(self, double alpha, double lam):
        cdef double X, c
        cdef long partition_size = max(1, <long>floor(pow(lam, alpha)))
        X = 0.
        c = pow(1. / partition_size, 1. / alpha)
        for i in range(partition_size):
            X += self.sample_divided_rv(alpha, lam, c)
        return X

    cdef double sample_divided_rv(self, double alpha, double lam, double c):
        cdef bint accepted = False
        while not accepted:
            S = c * self.sample_non_tilted_rv(alpha)
            accept_prob = exp(- lam * S)
            accepted = (self.next_double() < accept_prob)
        return S

    cdef double sample_non_tilted_rv(self, double alpha):
        cdef double V, E, S
        V = self.next_double()
        E = - log(self.next_double())
        S = pow(
            self.zolotarev_function(M_PI * V, alpha) / E
        , (1. - alpha) / alpha)
        return S

    cdef double sample_by_double_rejection(self, double alpha, double lam):

        cdef double b, lam_alpha, gamma, sqrt_gamma, c1, c2, c3, xi, psi, \
            U, Z, z, X, N, E, a, m, delta, log_accept_prob

        # Pre-compute a bunch of quantities.
        b = (1. - alpha) / alpha
        lam_alpha = pow(lam, alpha)
        gamma = lam_alpha * alpha * (1. - alpha)
        sqrt_gamma = sqrt(gamma)
        c1 = sqrt(M_PI / 2.)
        c2 = 2. + c1
        c3 = c2 * sqrt_gamma
        xi = (1. + sqrt(2.) * c3) / M_PI
        psi = c3 * exp(-gamma * M_PI * M_PI / 8.) / sqrt(M_PI)

        # Start double-rejection sampling.
        cdef bint accepted = False
        while not accepted:
            U, Z, z = self.sample_aux_rv(c1, xi, psi, gamma, sqrt_gamma, alpha, lam_alpha)
            X, N, E, a, m, delta = \
                self.sample_reference_rv(U, alpha, lam_alpha, b, c1, z)
            log_accept_prob = \
                self.compute_log_accept_prob(X, N, E, a, m, alpha, lam_alpha, b, delta)
            accepted = (log_accept_prob > log(Z))

        return pow(X, -b)

    cdef sample_aux_rv(self,
            double c1, double xi, double psi, double gamma, double sqrt_gamma,
            double alpha, double lam_alpha
        ):
        """
        Samples an auxiliary random variable for the double-rejection algorithm.
        Returns:
            U : auxiliary random variable for the double-rejection algorithm
            Z : uniform random variable independent of U, X
            z : scalar quantity used later
        """
        cdef double U, Z, z, accept_prob
        cdef bint accepted = False
        while not accepted:
            U = self.sample_aux2_rv(c1, xi, psi, gamma, sqrt_gamma)
            if U > M_PI:
                accept_prob = 0.
            else:
                zeta = sqrt(self.zolotarev_pdf_exponentiated(U, alpha))
                z = 1. / (1. - pow(1. + alpha * zeta / sqrt_gamma, -1. / alpha))
                accept_prob = self.compute_aux2_accept_prob(
                    U, c1, xi, psi, zeta, z, lam_alpha, gamma, sqrt_gamma)
            if accept_prob == 0.:
                accepted = False
            else:
                Z = self.next_double() / accept_prob
                accepted = (U < M_PI and Z <= 1.)

        return U, Z, z

    cdef double sample_aux2_rv(self,
            double c1, double xi, double psi, double gamma, double sqrt_gamma):
        """
        Sample the 2nd level auxiliary random variable (i.e. the additional
        auxiliary random variable used to sample the auxilary variable for
        double-rejection algorithm.)
        """

        w1 = c1 * xi / sqrt_gamma
        w2 = 2. * sqrt(M_PI) * psi
        w3 = xi * M_PI
        V = self.next_double()
        if gamma >= 1:
            if V < w1 / (w1 + w2):
                U = fabs(self.rand_standard_normal()) / sqrt_gamma
            else:
                W = self.next_double()
                U = M_PI * (1. - W * W)
        else:
            W = self.next_double()
            if V < w3 / (w2 + w3):
                U = M_PI * W
            else:
                U = M_PI * (1. - W * W)

        return U

    cdef double compute_aux2_accept_prob(self,
            double U, double c1, double xi, double psi, double zeta, double z,
            double lam_alpha, double gamma, double sqrt_gamma
        ):
        inverse_accept_prob = M_PI * exp(-lam_alpha * (1. - 1. / (zeta * zeta))) \
              / ((1. + c1) * sqrt_gamma / zeta + z)
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

    cdef sample_reference_rv(self,
            double U, double alpha, double lam_alpha, double b, double c1, double z):
        """
        Generate a sample from the reference (augmented) distribution conditional
        on U for the double-rejection algorithm

        Returns:
        --------
            X : random variable from the reference distribution
            N, E : random variables used later for computing the acceptance prob
            a, m, delta: scalar quantities used later
        """
        a = self.zolotarev_function(U, alpha)
        m = pow(b / a, alpha) * lam_alpha
        delta = sqrt(m * alpha / a)
        a1 = delta * c1
        a3 = z / a
        s = a1 + delta + a3
        V2 = self.next_double()
        N = 0.
        E = 0.
        if V2 < a1 / s:
            N = self.rand_standard_normal()
            X = m - delta * fabs(N)
        elif V2 < (a1 + delta) / s:
            X = m + delta * self.next_double()
        else:
            E = - log(self.next_double())
            X = m + delta + E * a3
        return X, N, E, a, m, delta

    cdef double compute_log_accept_prob(self,
            double X, double N, double E, double a, double m,
            double alpha, double lam_alpha, double b, double delta
        ):
        if X < 0:
            log_accept_prob = - INFINITY
        else:
            log_accept_prob = - (
                a * (X - m)
                + exp((1. / alpha) * log(lam_alpha) - b * log(m)) * (pow(m / X, b) - 1.)
            )
            if X < m:
                log_accept_prob += N * N / 2.
            elif X > m + delta:
                log_accept_prob += E

        return log_accept_prob

    cdef double zolotarev_pdf_exponentiated(self, double x, double alpha):
        """
        Evaluates a function proportional to a power of the Zolotarev density.
        """
        cdef double denominator, numerator
        denominator = pow(sinc(alpha * x), alpha) \
                      * pow(sinc((1. - alpha) * x), (1. - alpha))
        numerator = sinc(x)
        return numerator / denominator

    cdef double zolotarev_function(self, double x, double alpha):
        cdef double val = pow(
            pow((1. - alpha) * sinc((1. - alpha) * x), (1. - alpha))
            * pow(alpha * sinc(alpha * x), alpha)
            / sinc(x)
        , 1. / (1. - alpha))
        return val

    cdef double rand_standard_normal(self):
        # Sample via Polar method
        cdef double X, Y, sq_norm
        sq_norm = 1. # Placeholder value to pass through the first loop
        while sq_norm >= 1. or sq_norm == 0.:
          X = 2. * self.next_double() - 1.
          Y = 2. * self.next_double() - 1.
          sq_norm = X * X + Y * Y
        return sqrt(-2. * log(sq_norm) / sq_norm) * Y
