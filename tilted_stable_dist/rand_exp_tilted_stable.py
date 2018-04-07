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

        Ialpha = 1. - alpha
        b = Ialpha / alpha
        lambda_alpha = lam ** alpha
        M_PI = math.pi
        M_SQRT_PI = sqrt(math.pi)
        M_SQRT2 = sqrt(2.)

        gamma = lambda_alpha * alpha * Ialpha;
        sgamma = sqrt(gamma);
        c1 = sqrt(M_PI / 2)
        c2 = 2. + c1
        c3 = c2 * sgamma;
        xi = (1. + M_SQRT2 * c3) / M_PI;
        psi = c3 * exp(-gamma * M_PI * M_PI / 8.) / M_SQRT_PI;
        w1 = c1 * xi / sgamma;
        w2 = 2. * M_SQRT_PI * psi;
        w3 = xi * M_PI;

        accepted = False
        aug_accepted = False
        while not accepted:

            while not aug_accepted:
                V = self.unif_rv()
                if gamma >= 1:
                    if(V < w1/(w1+w2)):
                        U = abs(self.normal_rv(0, 1))/sgamma
                    else:
                        W_ = self.unif_rv();
                        U = M_PI*(1.-W_*W_);
                else:
                    W_ = self.unif_rv();
                    if(V < w3/(w2+w3)):
                        U = M_PI*W_;
                    else:
                        U = M_PI*(1.-W_*W_);
                W = self.unif_rv();
                zeta = sqrt(self.BdB0(U,alpha));
                z = 1/(1.-pow(1+alpha*zeta/sgamma,-1/alpha));
                rho = M_PI * exp(-lambda_alpha*(1. - 1./(zeta*zeta))) \
                    / ((1.+c1)*sgamma/zeta + z);
                d = 0.;
                if(U >= 0 and gamma >= 1):
                    d += xi*exp(-gamma*U*U/2.);
                if(U > 0 and U < M_PI):
                    d += psi/sqrt(M_PI-U);
                if(U >= 0 and U <= M_PI and gamma < 1):
                    d += xi;
                rho *= d;
                Z = W*rho;
                aug_accepted = (U < M_PI and Z <= 1.);

            a = pow(self.A_3(U,alpha,Ialpha), 1./Ialpha)
            m = pow(b/a,alpha)*lambda_alpha
            delta = sqrt(m*alpha/a)
            a1 = delta*c1
            a3 = z/a
            s = a1+delta+a3;
            V_ = self.unif_rv()
            N_ = 0.
            E_ = 0.
            if(V_ < a1/s):
                N_ = self.normal_rv(0, 1);
                X = m-delta* abs(N_);
            else:
                if(V_ < (a1+delta)/s):
                    X = m+delta*self.unif_rv();
                else:
                    E_ = - log(self.unif_rv());
                    X = m+delta+E_*a3;
            if X > 0:
                E = -log(Z);
                c = a*(X-m)+exp((1/alpha)*log(lambda_alpha)-b*log(m))*(pow(m/X,b)-1);
                    #/**< Marius Hofert: numerically more stable for small alpha */
                if(X < m):
                    c -= N_*N_/2.;
                elif (X > m+delta):
                    c -= E_;
            accepted = (X >= 0 and c <= E)

        return X ** -b

    def BdB0(self, x, alpha):
        denominator = pow(self.sinc(alpha * x), alpha) \
                      * pow(self.sinc((1 - alpha) * x), (1 - alpha))
        numerator = self.sinc(x)
        return numerator / denominator

    def A_3(self, _x, _alpha_, _I_alpha):
        return pow(_I_alpha * self.sinc(_I_alpha * _x), _I_alpha) * \
            pow(_alpha_ * self.sinc(_alpha_ * _x), _alpha_) / self.sinc(_x)

    def sinc(self, x):
        ax = abs(x);
        if(ax < 0.006):
            if (x == 0.):
                return 1;
            x2 = x * x;
            if(ax < 2e-4):
                 return 1. - x2/6.;
            return 1. - x2/6 * (1 - x2/20);
        return sin(x) / x