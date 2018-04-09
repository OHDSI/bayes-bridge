import math
from math import sqrt, log, pow, sin
from numpy import exp # For handling over & underflow
import numpy as np

class ExpTiltedStableDist():

    def __init__(self, seed=None):
        np.random.seed(seed)
        self.unif_rv = np.random.uniform
        self.normal_rv = np.random.normal

    def rv(self, alpha, lam):
        """
        Generate a random variable from a stable distribution with
            characteristic exponent =  alpha < 1
            skewness = 1
            scale = cos(alpha * pi / 2) ** (1 / alpha)
            location = 0
            exponential tilting = lam
        (The density p(x) is tilted by exp(-lam * x).)
        """

        b = (1. - alpha) / alpha
        lam_alpha = pow(lam, alpha)
        gamma = lam_alpha * alpha * (1. - alpha)
        sqrt_gamma = sqrt(gamma)
        c1 = sqrt(math.pi / 2.)
        c2 = 2. + c1
        c3 = c2 * sqrt_gamma
        xi = (1. + sqrt(2.) * c3) / math.pi
        psi = c3 * exp(-gamma * math.pi * math.pi / 8.) / sqrt(math.pi)

        accepted = False
        while not accepted:
            U, Z, z = self.sample_aux_rv(c1, xi, psi, gamma, sqrt_gamma, alpha, lam_alpha)
            a = pow(self.A_3(U, alpha), 1. / (1. - alpha))
            m = pow(b / a, alpha) * lam_alpha
            delta = sqrt(m * alpha / a)
            a1 = delta * c1
            a3 = z / a
            s = a1 + delta + a3
            V2 = self.unif_rv()
            N = 0.
            E1 = 0.
            if V2 < a1 / s:
                N = self.normal_rv(0., 1.)
                X = m - delta * abs(N)
            else:
                if V2 < (a1 + delta) / s:
                    X = m + delta * self.unif_rv()
                else:
                    E1 = - log(self.unif_rv())
                    X = m + delta + E1 * a3
            if X > 0:
                E2 = -log(Z)
                c = a * (X - m) + exp((1. / alpha) * log(lam_alpha) - b * log(m)) * (pow(m / X, b) - 1.)
                    #/**< Marius Hofert: numerically more stable for small alpha */
                if X < m:
                    c -= N * N / 2.
                elif X > m + delta:
                    c -= E1
            accepted = (X >= 0. and c <= E2)

        return pow(X, -b)

    def sample_aux_rv(self, c1, xi, psi, gamma, sqrt_gamma, alpha, lam_alpha):
        """
        Samples an auxiliary random variable for the double-rejection algorithm.
        Returns:
            U : auxiliary random variable for the double-rejection algorithm
            Z : uniform random variable independent of U, X
            z : scalar quantity used later
        """

        accepted = False
        while not accepted:
            U = self.sample_aux2_rv(c1, xi, psi, gamma, sqrt_gamma)
            zeta = sqrt(self.BdB0(U, alpha))
            z = 1. / (1. - pow(1. + alpha * zeta / sqrt_gamma, -1. / alpha))
            rho = self.compute_aux2_accept_prob(
                U, c1, xi, psi, zeta, z, lam_alpha, gamma, sqrt_gamma)
            Z = self.unif_rv() * rho
            accepted = (U < math.pi and Z <= 1.)

        return U, Z, z

    def sample_aux2_rv(self, c1, xi, psi, gamma, sqrt_gamma):
        """
        Sample the 2nd level auxiliary random variable (i.e. the additional
        auxiliary random variable used to sample the auxilary variable for
        double-rejection algorithm.)
        """

        w1 = c1 * xi / sqrt_gamma
        w2 = 2. * sqrt(math.pi) * psi
        w3 = xi * math.pi
        V = self.unif_rv()
        if gamma >= 1:
            if V < w1 / (w1 + w2):
                U = abs(self.normal_rv(0., 1.)) / sqrt_gamma
            else:
                W = self.unif_rv()
                U = math.pi * (1. - W * W)
        else:
            W = self.unif_rv()
            if V < w3 / (w2 + w3):
                U = math.pi * W
            else:
                U = math.pi * (1. - W * W)

        return U

    def compute_aux2_accept_prob(self, U, c1, xi, psi, zeta, z, lam_alpha, gamma, sqrt_gamma):
        rho = math.pi * exp(-lam_alpha * (1. - 1. / (zeta * zeta))) \
              / ((1. + c1) * sqrt_gamma / zeta + z)
        d = 0.
        if U >= 0. and gamma >= 1:
            d += xi * exp(-gamma * U * U / 2.)
        if U > 0. and U < math.pi:
            d += psi / sqrt(math.pi - U)
        if U >= 0. and U <= math.pi and gamma < 1.:
            d += xi
        rho *= d
        return rho


    def BdB0(self, x, alpha):
        denominator = pow(self.sinc(alpha * x), alpha) \
                      * pow(self.sinc((1. - alpha) * x), (1. - alpha))
        numerator = self.sinc(x)
        return numerator / denominator

    def A_3(self, x, alpha):
        return pow((1. - alpha) * self.sinc((1. - alpha) * x), (1. - alpha)) * \
               pow(alpha * self.sinc(alpha * x), alpha) / self.sinc(x)

    def sinc(self, x):
        if x == 0.:
            val = 1.
        elif abs(x) < 2e-4:
            val = 1. - pow(x, 2) / 6.
        elif abs(x) < 0.006:
            x_sq = pow(x, 2)
            val = 1. - x_sq / 6. * (1 - x_sq / 20.)
        else:
            val = sin(x) / x
        return val
