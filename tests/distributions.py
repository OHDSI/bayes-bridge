import numpy as np
from scipy.stats import norm, skewnorm

class BivariateGaussian():

    def __init__(self, rho=.9, sigma=np.array([1.0, 2.0])):
        Sigma = np.array([
            [sigma[0] ** 2, rho * np.prod(sigma)],
            [rho * np.prod(sigma), sigma[1] ** 2]
        ])
        self.sigma = sigma
        self.Sigma = Sigma
        self.Phi = np.linalg.inv(Sigma)

    def compute_logp_and_gradient(self, x):
        grad = - self.Phi.dot(x)
        logp = np.inner(x, grad) / 2
        return logp, grad

    def compute_marginal_pdf(self, x, axis):
        pdf = 1 / np.sqrt(2 * np.pi) / self.sigma[axis] \
              * np.exp(- x ** 2 / 2 / self.sigma[axis] ** 2)
        return pdf

    def get_principal_components(self):
        """ Returns principal component variances and directions. """
        eigvals, eigvec = np.linalg.eig(self.Phi)
        return 1 / eigvals, eigvec

    def get_stepsize_stability_limit(self):
        pc_scale = np.sqrt(
            self.get_principal_components()[0]
        )
        return 2 * np.min(pc_scale)


"""
Defines decoraters to take the log density and gradient functions for a
random variable X and returns the corresponding functions for a random
variable Y = np.dot(Q, X) for a rotation matrix Q.
"""

def rotate_logp(rotation, compute_logp):
    Q = rotation
    def compute_rotated_logp(y):
        return compute_logp(Q.T.dot(y))
    return compute_rotated_logp


def rotate_gradient(rotation, compute_gradient):
    Q = rotation
    def compute_rotated_gradient(y):
        return Q.dot(compute_gradient(Q.T.dot(y)))
    return compute_rotated_gradient


class BivariateSkewNormal():
    """
    Product of independent skew-normals are rotated (i.e. linearly trasformed
    through an orthogonal matrix).
    """

    def __init__(self, shape=np.array([1., 4.]), rotation=None):

        if rotation is None:
            rotation = np.array([
                [1., 1.],
                [1., -1.]
            ]) / np.sqrt(2)

        self.shape = shape
        self.rotation = rotation

        product_dist = ProductSkewNormal(self.shape)
        self.compute_logp = rotate_logp(
            self.rotation, product_dist.compute_logpdf
        )
        self.compute_gradient = rotate_gradient(
            self.rotation, product_dist.compute_gradient
        )

    def compute_logp_and_gradient(self, x):
        return (
            self.compute_logp(x),
            self.compute_gradient(x)
        )

    def compute_marginal_pdf(self, x, y):
        logp = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                logp[i, j] = self.compute_logp(np.array([x[i], y[j]]))
        marginal_pdf = [
            np.trapz(np.exp(logp), [y, x][axis], axis=axis - 1)
            for axis in [0, 1]
        ]
        return marginal_pdf

    def decorrelate(self, x_samples):
        """
        Rotate back the coordinates so that the distributions become independent.
        Params:
        ------
        x_samples: numpy array of size (n_param, ...)
        """
        Q = self.rotation
        return Q.T.dot(x_samples)



class ProductSkewNormal():

    def __init__(self, shape):
        self.shape = shape

    def compute_logpdf(self, x):
        logp = np.sum(
            skewnorm.logpdf(x, self.shape)
        )
        return logp

    def compute_gradient(self, x):
        grad = - x + self.shape * np.exp(
            norm.logpdf(self.shape * x) - norm.logcdf(self.shape * x)
        )
        return grad
